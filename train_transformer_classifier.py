#!/usr/bin/env python3
"""
Train a Transformer-based multi-class transformation type predictor.

Architecture (Hybrid):
  Text path:   Serialized record → DistilBERT → [CLS] representation (768-d)
  Numeric path: [entropy, pk, n_src, ...] → MLP → (64-d)
  Fusion:       concat → LayerNorm → MLP → num_classes logits

Text serialization format:
  "first_name [string] , last_name [string] -> full_name [string]"

Why this is better than the lightweight model:
  - Pre-trained on billions of words: understands "hashed"≈"hash", "monthly"≈"month"
  - Contextual attention: relates source and target names together
  - Handles any column name naturally — no OOV problem

Usage (GPU recommended):
  pip install torch transformers
  python train_transformer_classifier.py --data unified_transformation_training_data.jsonl

  # Smaller/faster model:
  python train_transformer_classifier.py --model prajjwal1/bert-small --batch-size 64

  # Full DistilBERT (best quality):
  python train_transformer_classifier.py --model distilbert-base-uncased --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
DEFAULT_DATA = BASE / "unified_transformation_training_data.jsonl"
DEFAULT_SAVE = BASE / "transformer_classifier_checkpoints"
NUM_NUMERIC = 7  # mean_src_ent, max_src_ent, tgt_ent, n_src, has_pk, ent_delta, tok_overlap


# ---------------------------------------------------------------------------
# Text serialization — converts a JSON record into natural language for BERT
# ---------------------------------------------------------------------------

def _col_text(col: dict) -> str:
    """Serialize one column: 'column_name [type]'."""
    name = col.get("name", "unknown")
    ctype = col.get("type", "string")
    return f"{name} [{ctype}]"


def record_to_text(record: dict) -> str:
    """
    Serialize a record into text for the transformer.
    Format: "src_col1 [type] , src_col2 [type] -> tgt_col [type]"
    """
    src_parts = []
    for col in record.get("source_columns", []):
        src_parts.append(_col_text(col))
    src_text = " , ".join(src_parts) if src_parts else "unknown [string]"
    tgt_text = _col_text(record.get("target_column", {}))
    return f"{src_text} -> {tgt_text}"


def extract_numeric(record: dict) -> List[float]:
    """Extract numeric features from a record."""
    src_cols = record.get("source_columns", [])
    tcol = record.get("target_column", {})
    src_ents = [c.get("entropy", 0.0) for c in src_cols]
    tgt_ent = tcol.get("entropy", 0.0)
    mean_s = sum(src_ents) / len(src_ents) if src_ents else 0.0
    max_s = max(src_ents) if src_ents else 0.0
    n_src = float(len(src_cols))
    pk = 1.0 if any(c.get("is_pk", False) for c in src_cols) else 0.0
    delta = tgt_ent - mean_s

    # Token overlap between source and target names
    src_toks = set()
    for c in src_cols:
        for w in re.split(r"\W+", (c.get("name") or "").lower().replace("_", " ")):
            if w:
                src_toks.add(w)
    tgt_toks = set()
    for w in re.split(r"\W+", (tcol.get("name") or "").lower().replace("_", " ")):
        if w:
            tgt_toks.add(w)
    overlap = float(len(src_toks & tgt_toks))

    return [mean_s, max_s, tgt_ent, n_src, pk, delta, overlap]


# ---------------------------------------------------------------------------
# Class vocabulary
# ---------------------------------------------------------------------------

def build_class_vocab(records: List[dict]) -> Tuple[Dict[str, int], List[str]]:
    counts = Counter()
    for r in records:
        tt = (r.get("transform_type") or "").strip()
        if tt:
            counts[tt] += 1
    ordered = [tt for tt, _ in counts.most_common()]
    vocab = {tt: i for i, tt in enumerate(ordered)}
    return vocab, ordered


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TransformDataset(Dataset):
    def __init__(self, records, tokenizer, class_vocab, max_len=128):
        self.records = records
        self.tokenizer = tokenizer
        self.class_vocab = class_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        text = record_to_text(r)
        enc = self.tokenizer(
            text, max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        nums = extract_numeric(r)
        class_id = self.class_vocab.get((r.get("transform_type") or "").strip(), 0)
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "numeric": torch.tensor(nums, dtype=torch.float32),
            "label": class_id,
        }


# ---------------------------------------------------------------------------
# Model: Transformer + Numeric hybrid
# ---------------------------------------------------------------------------

class TransformerClassifier(nn.Module):
    """
    DistilBERT (or any HuggingFace model) + numeric features → multi-class.
    """
    def __init__(self, model_name: str, num_classes: int, num_numeric: int = NUM_NUMERIC,
                 dropout: float = 0.2, freeze_layers: int = 0):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        bert_dim = self.bert.config.hidden_size  # 768 for distilbert-base

        # Optionally freeze early layers for faster training
        if freeze_layers > 0 and hasattr(self.bert, "transformer"):
            for i, layer in enumerate(self.bert.transformer.layer):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

        # Numeric features path
        self.num_proj = nn.Sequential(
            nn.Linear(num_numeric, 64),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Classification head
        fusion_dim = bert_dim + 64
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_ids, attention_mask, numeric):
        # Text path: [CLS] token representation
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = outputs.last_hidden_state[:, 0]  # (B, bert_dim)

        # Numeric path
        num_rep = self.num_proj(numeric)  # (B, 64)

        # Fuse and classify
        combined = torch.cat([cls_rep, num_rep], dim=1)  # (B, bert_dim + 64)
        return self.classifier(combined)  # (B, num_classes)


# ---------------------------------------------------------------------------
# Label-smoothed CrossEntropy
# ---------------------------------------------------------------------------

class LabelSmoothCE(nn.Module):
    def __init__(self, num_classes, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.num_classes = num_classes

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        smooth = -log_probs.mean(dim=-1)
        return (self.confidence * nll + self.smoothing * smooth).mean()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, scaler, scheduler,
                    grad_accum_steps=1):
    model.train()
    total_loss = n = 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        nums = batch["numeric"].to(device)
        y = batch["label"].to(device)

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            logits = model(ids, mask, nums)
            loss = criterion(logits, y) / grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps * ids.size(0)
        n += ids.size(0)

    return total_loss / n if n else 0.0


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    total_loss = n = correct_top1 = correct_top3 = 0
    criterion = nn.CrossEntropyLoss()
    class_tp = [0] * num_classes
    class_fp = [0] * num_classes
    class_fn = [0] * num_classes

    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        nums = batch["numeric"].to(device)
        y = batch["label"].to(device)

        logits = model(ids, mask, nums)
        loss = criterion(logits, y)
        total_loss += loss.item() * ids.size(0)
        n += ids.size(0)

        pred = logits.argmax(dim=1)
        correct_top1 += (pred == y).sum().item()

        top3 = logits.topk(min(3, num_classes), dim=1).indices
        for j in range(y.size(0)):
            if y[j] in top3[j]:
                correct_top3 += 1

        for p, t in zip(pred.cpu().tolist(), y.cpu().tolist()):
            if p == t:
                class_tp[t] += 1
            else:
                class_fp[p] += 1
                class_fn[t] += 1

    acc1 = correct_top1 / n if n else 0.0
    acc3 = correct_top3 / n if n else 0.0

    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        tp, fp, fn = class_tp[c], class_fp[c], class_fn[c]
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        if (tp + fp + fn) > 0:
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)

    mp = sum(precisions) / len(precisions) if precisions else 0.0
    mr = sum(recalls) / len(recalls) if recalls else 0.0
    mf = sum(f1s) / len(f1s) if f1s else 0.0

    return total_loss / n, acc1, acc3, mp, mr, mf, class_tp, class_fp, class_fn


def print_per_class_report(class_names, class_tp, class_fp, class_fn, top_n=15):
    rows = []
    for c, name in enumerate(class_names):
        tp, fp, fn = class_tp[c], class_fp[c], class_fn[c]
        total = tp + fn
        if total == 0:
            continue
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        rows.append((name, total, tp, fp, fn, p, r, f))
    rows.sort(key=lambda x: x[7])
    print(f"\n{'Type':<35} {'Tot':>5} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>6} {'R':>6} {'F1':>6}")
    print("-" * 90)
    for name, total, tp, fp, fn, p, r, f in rows[:top_n]:
        print(f"{name:<35} {total:>5} {tp:>4} {fp:>4} {fn:>4} {p:>6.3f} {r:>6.3f} {f:>6.3f}")
    if len(rows) > top_n:
        print(f"  ... and {len(rows) - top_n} more classes")
    perfect = sum(1 for row in rows if row[7] >= 0.999)
    above_95 = sum(1 for row in rows if row[7] >= 0.95)
    print(f"\nPerfect F1: {perfect}/{len(rows)} | F1>=95%: {above_95}/{len(rows)}")


@torch.no_grad()
def calibrate_temperature(model, val_loader, device):
    """Find optimal temperature for confidence calibration."""
    model.eval()
    all_logits, all_targets = [], []
    for batch in val_loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        nums = batch["numeric"].to(device)
        logits = model(ids, mask, nums)
        all_logits.append(logits.cpu())
        all_targets.append(batch["label"])
    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)

    best_temp, best_nll = 1.0, float("inf")
    criterion = nn.CrossEntropyLoss()
    for temp in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]:
        nll = criterion(all_logits / temp, all_targets).item()
        conf = torch.softmax(all_logits / temp, dim=-1).max(dim=-1).values.mean().item()
        print(f"  T={temp:.1f}  NLL={nll:.4f}  mean_conf={conf:.3f}")
        if nll < best_nll:
            best_nll = nll
            best_temp = temp
    return best_temp


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_records(path: Path) -> List[dict]:
    recs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            recs.append(json.loads(line))
    return recs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Train transformer transformation classifier")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE)
    ap.add_argument("--model", type=str, default="distilbert-base-uncased",
                    help="HuggingFace model name (distilbert-base-uncased, prajjwal1/bert-small, etc.)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5,
                    help="Peak learning rate (2e-5 is standard for BERT fine-tuning)")
    ap.add_argument("--max-len", type=int, default=96,
                    help="Max token length (column names are short, 96 is plenty)")
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-accum", type=int, default=2,
                    help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    ap.add_argument("--freeze-layers", type=int, default=0,
                    help="Freeze first N transformer layers (0=finetune all)")
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true", default=True,
                    help="Use mixed precision (auto-disabled on CPU)")
    ap.add_argument("--no-fp16", action="store_true", help="Disable fp16")
    args = ap.parse_args()

    if args.no_fp16:
        args.fp16 = False

    # Seed everything
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = args.fp16 and device.type == "cuda"
    print(f"Device: {device} | FP16: {use_fp16} | Model: {args.model}")

    # ------------------------------------------------------------------
    # Load and split data
    # ------------------------------------------------------------------
    all_records = load_records(args.data)
    records = [r for r in all_records if r.get("label") == 1]
    print(f"Total records: {len(all_records)}, positive (used): {len(records)}")

    random.shuffle(records)
    n_val = max(1, len(records) // 5)
    train_recs, val_recs = records[n_val:], records[:n_val]

    class_vocab, class_names = build_class_vocab(train_recs)
    num_classes = len(class_names)
    counts = Counter(class_vocab.get((r.get("transform_type") or "").strip(), -1) for r in train_recs)
    print(f"Train: {len(train_recs)}  Val: {len(val_recs)}  Classes: {num_classes}")
    print(f"Samples/class: min={min(counts.values())}, max={max(counts.values())}, "
          f"avg={len(train_recs)//num_classes}")

    # ------------------------------------------------------------------
    # Tokenizer and datasets
    # ------------------------------------------------------------------
    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Show a sample serialization
    sample_text = record_to_text(train_recs[0])
    print(f"Sample text: '{sample_text}'")
    sample_enc = tokenizer(sample_text, max_length=args.max_len, truncation=True)
    print(f"Sample tokens: {tokenizer.convert_ids_to_tokens(sample_enc['input_ids'][:20])}...")

    train_ds = TransformDataset(train_recs, tokenizer, class_vocab, args.max_len)
    val_ds = TransformDataset(val_recs, tokenizer, class_vocab, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=0, pin_memory=device.type == "cuda")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print(f"\nLoading model: {args.model}")
    model = TransformerClassifier(
        model_name=args.model,
        num_classes=num_classes,
        freeze_layers=args.freeze_layers,
        dropout=0.2,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # ------------------------------------------------------------------
    # Optimizer, scheduler, loss
    # ------------------------------------------------------------------
    # Separate BERT params (lower LR) from classifier params (higher LR)
    bert_params = [p for n, p in model.named_parameters() if "bert" in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if "bert" not in n]
    optimizer = torch.optim.AdamW([
        {"params": bert_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * 10},  # 10x LR for classification head
    ], weight_decay=args.weight_decay)

    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    criterion = LabelSmoothCE(num_classes, smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None

    effective_batch = args.batch_size * args.grad_accum
    print(f"\nEffective batch size: {effective_batch}")
    print(f"Total steps: {total_steps} | Warmup: {warmup_steps}")
    print(f"Label smoothing: {args.label_smoothing}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_top1 = -1.0
    best_epoch = 0
    patience_left = args.patience

    print(f"\n{'Ep':>3} {'LR':>9} {'TLoss':>7} {'VLoss':>7} "
          f"{'Top1':>6} {'Top3':>6} {'MPr':>6} {'MRe':>6} {'MF1':>6} {'Time':>6}")
    print("-" * 82)

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        t_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                 device, scaler, scheduler, args.grad_accum)
        v_loss, top1, top3, mp, mr, mf, c_tp, c_fp, c_fn = evaluate(
            model, val_loader, device, num_classes)
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(f"{ep:3d} {lr:9.2e} {t_loss:7.4f} {v_loss:7.4f} "
              f"{top1:6.4f} {top3:6.4f} {mp:6.4f} {mr:6.4f} {mf:6.4f} {elapsed:5.0f}s")

        if top1 > best_top1:
            best_top1 = top1
            best_epoch = ep
            patience_left = args.patience
            # Save model + tokenizer + metadata
            torch.save({
                "epoch": ep,
                "model_state": model.state_dict(),
                "model_name": args.model,
                "class_names": class_names,
                "class_vocab": class_vocab,
                "num_classes": num_classes,
                "max_len": args.max_len,
                "val_top1": top1, "val_top3": top3,
                "val_macro_f1": mf, "val_macro_prec": mp, "val_macro_rec": mr,
            }, args.save_dir / "best.pt")
            tokenizer.save_pretrained(str(args.save_dir / "tokenizer"))
            best_tp, best_fp, best_fn = c_tp, c_fp, c_fn
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"\nEarly stop at epoch {ep}")
                break

    print(f"\nBest epoch {best_epoch}: top1={best_top1:.4f}")

    # Per-class report
    print("\n" + "=" * 90)
    print("PER-CLASS REPORT (worst first):")
    print_per_class_report(class_names, best_tp, best_fp, best_fn, top_n=20)

    # Temperature calibration
    print("\n" + "=" * 90)
    print("TEMPERATURE CALIBRATION:")
    ckpt = torch.load(args.save_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    opt_temp = calibrate_temperature(model, val_loader, device)
    ckpt["temperature"] = opt_temp
    torch.save(ckpt, args.save_dir / "best.pt")
    print(f"Optimal temperature: {opt_temp:.2f}")
    print(f"\nCheckpoints saved to: {args.save_dir}")
    print("Files: best.pt, tokenizer/")


if __name__ == "__main__":
    main()
