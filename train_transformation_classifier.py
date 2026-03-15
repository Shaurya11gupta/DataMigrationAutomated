#!/usr/bin/env python3
"""
Train transformation classifier — multi-feature architecture with source/target interaction.

Architecture:
  1. Source-name tokens → Embedding → mean-pool → src_rep
  2. Target-name tokens → Embedding → mean-pool → tgt_rep
  3. Interaction: element-wise product src*tgt, token-overlap count
  4. Categorical: transform_type emb, source_types emb, target_type emb
  5. Numeric: entropies, n_src, pk flag, entropy delta
  6. Fusion: concat all → MLP (with BatchNorm, GELU, Dropout) → binary logit

Key: source and target name paths are SEPARATE so the model can compare them.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

BASE = Path(__file__).resolve().parent
DEFAULT_DATA = BASE / "unified_transformation_training_data.jsonl"
DEFAULT_SAVE = BASE / "transformation_classifier_checkpoints"

TYPE_VOCAB = {"<unk>": 0, "string": 1, "int": 2, "decimal": 3, "date": 4, "boolean": 5}
NUM_TYPES = len(TYPE_VOCAB)
MAX_SRC_TOKS = 32
MAX_TGT_TOKS = 16
NUM_NUMERIC = 7  # mean_src_ent, max_src_ent, tgt_ent, n_src, pk, delta, tok_overlap


def type_id(t: str) -> int:
    return TYPE_VOCAB.get((t or "").lower().strip(), 0)


def _name_tokens(name: str) -> List[str]:
    return [w for w in re.split(r"\W+", (name or "").lower().replace("_", " ").replace("-", " ")) if w]


def extract_features(record: dict) -> Tuple[List[str], List[str], List[float]]:
    """Returns (src_tokens, tgt_tokens, numeric_features)."""
    src_toks = []
    for col in record.get("source_columns", []):
        src_toks.extend(_name_tokens(col.get("name", "")))
    tcol = record.get("target_column", {})
    tgt_toks = _name_tokens(tcol.get("name", ""))

    src_cols = record.get("source_columns", [])
    src_ents = [c.get("entropy", 0.0) for c in src_cols]
    tgt_ent = tcol.get("entropy", 0.0)
    mean_s = sum(src_ents) / len(src_ents) if src_ents else 0.0
    max_s = max(src_ents) if src_ents else 0.0
    n_src = float(len(src_cols))
    pk = 1.0 if any(c.get("is_pk", False) for c in src_cols) else 0.0
    delta = tgt_ent - mean_s
    overlap = float(len(set(src_toks) & set(tgt_toks)))
    return src_toks, tgt_toks, [mean_s, max_s, tgt_ent, n_src, pk, delta, overlap]


class Vocab:
    PAD_ID = 0
    UNK_ID = 1

    def __init__(self):
        self._idx = {"<pad>": 0, "<unk>": 1}
        self._counts = {}

    def add(self, tokens):
        for t in tokens:
            self._counts[t] = self._counts.get(t, 0) + 1

    def build(self, min_count=2):
        for t, c in sorted(self._counts.items(), key=lambda x: -x[1]):
            if c >= min_count and t not in self._idx:
                self._idx[t] = len(self._idx)

    def encode(self, tokens, max_len):
        ids = [self._idx.get(t, self.UNK_ID) for t in tokens[:max_len]]
        ids += [self.PAD_ID] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self._idx)

    def ordered(self):
        out = [""] * len(self._idx)
        for t, i in self._idx.items():
            out[i] = t
        return out


def build_tt_vocab(records):
    v = {"<unk>": 0}
    for r in records:
        tt = (r.get("transform_type") or "").strip()
        if tt and tt not in v:
            v[tt] = len(v)
    return v


if TORCH_AVAILABLE:

    class TxDataset(Dataset):
        def __init__(self, records, vocab, tt_vocab):
            self.records = records
            self.vocab = vocab
            self.tt_vocab = tt_vocab

        def __len__(self):
            return len(self.records)

        def __getitem__(self, i):
            r = self.records[i]
            src_toks, tgt_toks, nums = extract_features(r)
            src_ids = self.vocab.encode(src_toks, MAX_SRC_TOKS)
            tgt_ids = self.vocab.encode(tgt_toks, MAX_TGT_TOKS)
            src_types = [type_id(c.get("type")) for c in r.get("source_columns", [])][:4]
            src_types += [0] * (4 - len(src_types))
            tgt_type = type_id((r.get("target_column") or {}).get("type"))
            tt = self.tt_vocab.get((r.get("transform_type") or "").strip(), 0)
            y = int(r.get("label", 0))
            return (
                torch.tensor(src_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long),
                torch.tensor(src_types, dtype=torch.long),
                torch.tensor(tgt_type, dtype=torch.long),
                torch.tensor(tt, dtype=torch.long),
                torch.tensor(nums, dtype=torch.float32),
                y,
            )

    class TransformClassifier(nn.Module):
        def __init__(self, vocab_size, num_tt, embed_dim=64, hidden=192, dropout=0.3):
            super().__init__()
            self.tok_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

            # Separate projections for source and target
            self.src_proj = nn.Linear(embed_dim, hidden // 2)
            self.tgt_proj = nn.Linear(embed_dim, hidden // 2)

            # Categorical
            self.src_type_embed = nn.Embedding(NUM_TYPES, 16)
            self.tgt_type_embed = nn.Embedding(NUM_TYPES, 16)
            self.tt_embed = nn.Embedding(max(num_tt, 1), 48)

            # Numeric
            self.num_proj = nn.Linear(NUM_NUMERIC, 32)

            # Fusion: src_rep(H/2) + tgt_rep(H/2) + interaction(H/2) + src_types(64) + tgt_type(16) + tt(48) + num(32)
            H2 = hidden // 2
            fusion_dim = H2 + H2 + H2 + 64 + 16 + 48 + 32
            self.classifier = nn.Sequential(
                nn.BatchNorm1d(fusion_dim),
                nn.Linear(fusion_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )

        def _pool(self, ids):
            mask = (ids != 0).float().unsqueeze(-1)
            e = self.tok_embed(ids) * mask
            return e.sum(1) / mask.sum(1).clamp(min=1e-9)

        def forward(self, src_ids, tgt_ids, src_types, tgt_type, tt_id, nums):
            # Separate source and target token representations
            src_raw = self._pool(src_ids)
            tgt_raw = self._pool(tgt_ids)
            src_rep = torch.relu(self.src_proj(src_raw))
            tgt_rep = torch.relu(self.tgt_proj(tgt_raw))
            # Interaction: element-wise product captures name similarity
            interact = src_rep * tgt_rep

            src_emb = self.src_type_embed(src_types).view(src_types.size(0), -1)
            tgt_emb = self.tgt_type_embed(tgt_type)
            tt_emb = self.tt_embed(tt_id)
            num_rep = torch.relu(self.num_proj(nums))

            fused = torch.cat([src_rep, tgt_rep, interact, src_emb, tgt_emb, tt_emb, num_rep], dim=1)
            return self.classifier(fused).squeeze(-1)


def load_records(path):
    recs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            recs.append(json.loads(line))
    return recs


def train_one_epoch(model, loader, opt, criterion, device, sched=None):
    model.train()
    total_loss = n = 0
    for si, ti, st, tt_type, tt, nums, y in loader:
        si, ti = si.to(device), ti.to(device)
        st, tt_type, tt = st.to(device), tt_type.to(device), tt.to(device)
        nums, y = nums.to(device), y.to(device).float()
        opt.zero_grad()
        logit = model(si, ti, st, tt_type, tt, nums)
        loss = criterion(logit, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if sched:
            sched.step()
        total_loss += loss.item() * si.size(0)
        n += si.size(0)
    return total_loss / n if n else 0.0


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = n = correct = tp = fp = tn = fn = 0
    for si, ti, st, tt_type, tt, nums, y in loader:
        si, ti = si.to(device), ti.to(device)
        st, tt_type, tt = st.to(device), tt_type.to(device), tt.to(device)
        nums, y = nums.to(device), y.to(device).float()
        logit = model(si, ti, st, tt_type, tt, nums)
        loss = criterion(logit, y)
        total_loss += loss.item() * si.size(0)
        pred = (torch.sigmoid(logit) >= 0.5).float()
        correct += (pred == y).sum().item()
        n += si.size(0)
        yc, pc = y.cpu(), pred.cpu()
        tp += ((pc == 1) & (yc == 1)).sum().item()
        fp += ((pc == 1) & (yc == 0)).sum().item()
        tn += ((pc == 0) & (yc == 0)).sum().item()
        fn += ((pc == 0) & (yc == 1)).sum().item()
    acc = correct / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return total_loss / n, acc, prec, rec, f1


class LabelSmoothBCE(nn.Module):
    """BCEWithLogitsLoss with label smoothing."""
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.s = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logit, target):
        target = target * (1 - self.s) + 0.5 * self.s
        return self.bce(logit, target)


def main():
    if not TORCH_AVAILABLE:
        raise SystemExit("pip install torch")

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = load_records(args.data)
    random.shuffle(records)
    n_val = max(1, len(records) // 5)
    train_recs, val_recs = records[n_val:], records[:n_val]

    n_pos = sum(1 for r in train_recs if r.get("label") == 1)
    n_neg = len(train_recs) - n_pos
    print(f"Train: {len(train_recs)} (pos={n_pos}, neg={n_neg})  Val: {len(val_recs)}")

    vocab = Vocab()
    for r in train_recs:
        s, t, _ = extract_features(r)
        vocab.add(s)
        vocab.add(t)
    vocab.build(min_count=2)
    tt_vocab = build_tt_vocab(train_recs)
    print(f"Token vocab: {len(vocab)}  Transform types: {len(tt_vocab)}")

    train_ds = TxDataset(train_recs, vocab, tt_vocab)
    val_ds = TxDataset(val_recs, vocab, tt_vocab)

    weights = [1.0 / n_pos if r.get("label") == 1 else 1.0 / n_neg for r in train_recs]
    sampler = WeightedRandomSampler(weights, len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    criterion = LabelSmoothBCE(smoothing=0.05)
    val_criterion = nn.BCEWithLogitsLoss()

    model = TransformClassifier(
        vocab_size=len(vocab), num_tt=len(tt_vocab), hidden=args.hidden, dropout=0.3,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=len(train_loader), pct_start=0.1, anneal_strategy="cos",
    )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    best_epoch = 0
    patience_left = args.patience

    for ep in range(1, args.epochs + 1):
        t_loss = train_one_epoch(model, train_loader, opt, criterion, device, sched)
        v_loss, v_acc, v_prec, v_rec, v_f1 = evaluate(model, val_loader, device, val_criterion)
        lr = opt.param_groups[0]["lr"]
        print(
            f"Epoch {ep:3d}  lr={lr:.2e}  t_loss={t_loss:.4f}  v_loss={v_loss:.4f}  "
            f"acc={v_acc:.4f}  prec={v_prec:.4f}  rec={v_rec:.4f}  f1={v_f1:.4f}"
        )
        if v_f1 > best_f1:
            best_f1 = v_f1
            best_epoch = ep
            patience_left = args.patience
            torch.save({
                "epoch": ep, "model_state": model.state_dict(),
                "vocab_tokens": vocab.ordered(), "tt_vocab": tt_vocab,
                "hidden": args.hidden, "num_tt": len(tt_vocab),
                "val_acc": v_acc, "val_prec": v_prec, "val_rec": v_rec, "val_f1": v_f1,
            }, args.save_dir / "best.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stop at epoch {ep}")
                break

    print(f"\nBest epoch {best_epoch}: f1={best_f1:.4f}")
    print(f"Checkpoints: {args.save_dir}")


if __name__ == "__main__":
    main()
