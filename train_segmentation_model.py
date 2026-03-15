"""
Segmentation Model  -  Training Script
========================================
Trains a per-character boundary detection model (SegModel) on the data
produced by  generate_seg_training_data.py .

Architecture
  Embedding -> Conv1d -> TransformerEncoder -> BiLSTM -> Linear(2)

Features
  - AdamW optimiser with cosine-annealing LR scheduler
  - Class-weighted CrossEntropyLoss  (boundary class upweighted)
  - Early stopping on validation F1  (patience configurable)
  - Saves drop-in checkpoint  segmentation_model_final.pt
  - Post-training validation on ~50 curated test cases
  - --resume  flag to continue from latest checkpoint

Run
  python train_segmentation_model.py
  python train_segmentation_model.py --epochs 40 --batch 128 --lr 3e-4
  python train_segmentation_model.py --resume
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# =====================================================================
# CONFIG
# =====================================================================
DEFAULT_DATA_DIR = "seg_training_data"
DEFAULT_OUTPUT = "segmentation_model_final.pt"
CHECKPOINT_DIR = "seg_checkpoints"
MAX_LEN = 40

# =====================================================================
# DATASET
# =====================================================================

class SegDataset(Dataset):
    def __init__(self, path: str, c2i: Optional[Dict[str, int]] = None):
        self.examples: List[Tuple[str, List[int]]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                text = obj["text"][:MAX_LEN]
                labels = obj["labels"][:MAX_LEN]
                self.examples.append((text, labels))

        # Build or accept char vocab
        if c2i is None:
            chars = set()
            for text, _ in self.examples:
                chars.update(text)
            self.c2i: Dict[str, int] = {"<pad>": 0, "<unk>": 1}
            for ch in sorted(chars):
                if ch not in self.c2i:
                    self.c2i[ch] = len(self.c2i)
        else:
            self.c2i = c2i

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text, labels = self.examples[idx]
        ids = [self.c2i.get(c, 1) for c in text]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long), len(ids)


def collate_fn(batch):
    """Pad sequences to max length in batch."""
    seqs, labels, lengths = zip(*batch)
    max_len = max(lengths)
    padded_seqs = torch.zeros(len(seqs), max_len, dtype=torch.long)
    padded_labels = torch.full((len(seqs), max_len), -1, dtype=torch.long)  # -1 = ignore
    for i, (s, l, ln) in enumerate(zip(seqs, labels, lengths)):
        padded_seqs[i, :ln] = s
        padded_labels[i, :ln] = l
    return padded_seqs, padded_labels, torch.tensor(lengths, dtype=torch.long)


# =====================================================================
# MODEL  (identical architecture to seg_classf_abbrev_test.py)
# =====================================================================

class SegModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128,
                 n_heads: int = 4, ff_dim: int = 256,
                 n_layers: int = 3, dropout: float = 0.15):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, embed_dim, 3, padding=1)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True,
        )
        self.tr = nn.TransformerEncoder(enc_layer, n_layers)
        self.lstm = nn.LSTM(embed_dim, embed_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(embed_dim * 2, 2)

    def forward(self, x):
        e = self.emb(x).permute(0, 2, 1)       # (B, D, L)
        h = F.relu(self.conv(e)).permute(0, 2, 1)  # (B, L, D)
        h = self.tr(h)
        h, _ = self.lstm(h)
        return self.fc(h)                        # (B, L, 2)


# =====================================================================
# METRICS
# =====================================================================

def compute_metrics(preds: np.ndarray, targets: np.ndarray):
    """Compute precision, recall, F1 for boundary class (1)."""
    mask = targets >= 0  # ignore padding
    p = preds[mask]
    t = targets[mask]

    tp = int(((p == 1) & (t == 1)).sum())
    fp = int(((p == 1) & (t == 0)).sum())
    fn = int(((p == 0) & (t == 1)).sum())
    tn = int(((p == 0) & (t == 0)).sum())

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# =====================================================================
# TRAINING
# =====================================================================

def train(
    data_dir: str = DEFAULT_DATA_DIR,
    output_path: str = DEFAULT_OUTPUT,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 3e-4,
    embed_dim: int = 128,
    n_layers: int = 3,
    dropout: float = 0.15,
    patience: int = 5,
    boundary_weight: float = 2.0,
    seed: int = 42,
    resume: bool = False,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Segmentation Model Training")
    print(f"{'='*60}")
    print(f"  Device:          {device}")
    print(f"  Data dir:        {data_dir}")
    print(f"  Output:          {output_path}")
    print(f"  Epochs:          {epochs}")
    print(f"  Batch size:      {batch_size}")
    print(f"  Learning rate:   {lr}")
    print(f"  Embed dim:       {embed_dim}")
    print(f"  Transformer layers: {n_layers}")
    print(f"  Dropout:         {dropout}")
    print(f"  Boundary weight: {boundary_weight}")
    print(f"  Patience:        {patience}")
    print(f"  Resume:          {resume}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    train_path = os.path.join(data_dir, "seg_train.jsonl")
    val_path = os.path.join(data_dir, "seg_val.jsonl")

    if not os.path.exists(train_path):
        print(f"[ERROR] Training data not found: {train_path}")
        print(f"  Run:  python generate_seg_training_data.py")
        return

    print("Loading training data...")
    train_ds = SegDataset(train_path)
    c2i = train_ds.c2i
    val_ds = SegDataset(val_path, c2i=c2i)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0, pin_memory=True)

    vocab_size = len(c2i)
    print(f"  Train examples: {len(train_ds)}")
    print(f"  Val examples:   {len(val_ds)}")
    print(f"  Vocab size:     {vocab_size}")

    # ------------------------------------------------------------------
    # 2. Model
    # ------------------------------------------------------------------
    model = SegModel(
        vocab_size=vocab_size, embed_dim=embed_dim,
        n_layers=n_layers, dropout=dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params:   {total_params:,} (trainable: {train_params:,})")

    # ------------------------------------------------------------------
    # 3. Loss, optimizer, scheduler
    # ------------------------------------------------------------------
    class_weights = torch.tensor([1.0, boundary_weight], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # ------------------------------------------------------------------
    # 4. Resume from checkpoint
    # ------------------------------------------------------------------
    start_epoch = 0
    best_f1 = 0.0
    no_improve = 0

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if resume:
        import glob
        ckpts = sorted(
            glob.glob(os.path.join(CHECKPOINT_DIR, "seg_ckpt_epoch_*.pt")),
            key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0]),
        )
        if ckpts:
            ckpt_path = ckpts[-1]
            print(f"\nResuming from: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            best_f1 = ckpt.get("best_f1", 0.0)
            no_improve = ckpt.get("no_improve", 0)
            print(f"  Resuming from epoch {start_epoch}, best F1={best_f1:.4f}")
        else:
            print(f"\n[WARN] --resume set but no checkpoints found. Training from scratch.")

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    print(f"\nStarting training...\n")
    t_start = time.time()

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for seqs, labels, lengths in train_loader:
            seqs = seqs.to(device)
            labels = labels.to(device)

            logits = model(seqs)  # (B, L, 2)
            loss = criterion(logits.view(-1, 2), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            n_batches += 1

        avg_train_loss = train_loss / max(1, n_batches)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for seqs, labels, lengths in val_loader:
                seqs = seqs.to(device)
                labels = labels.to(device)

                logits = model(seqs)
                loss = criterion(logits.view(-1, 2), labels.view(-1))
                val_loss += loss.item()
                val_batches += 1

                preds = logits.argmax(dim=-1).cpu().numpy()
                tgts = labels.cpu().numpy()
                all_preds.append(preds)
                all_targets.append(tgts)

        avg_val_loss = val_loss / max(1, val_batches)
        all_preds = np.concatenate([p.flatten() for p in all_preds])
        all_targets = np.concatenate([t.flatten() for t in all_targets])
        metrics = compute_metrics(all_preds, all_targets)

        current_lr = scheduler.get_last_lr()[0]

        print(f"  Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Acc: {metrics['accuracy']:.4f} | "
              f"F1: {metrics['f1']:.4f} | "
              f"P: {metrics['precision']:.4f} | "
              f"R: {metrics['recall']:.4f} | "
              f"LR: {current_lr:.2e}")

        # ---- Save checkpoint ----
        ckpt_data = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "c2i": c2i,
            "max_len": MAX_LEN,
            "epoch": epoch,
            "best_f1": max(best_f1, metrics["f1"]),
            "no_improve": no_improve,
            "metrics": metrics,
        }
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"seg_ckpt_epoch_{epoch+1:03d}.pt")
        torch.save(ckpt_data, ckpt_path)

        # ---- Early stopping on F1 ----
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            no_improve = 0
            # Save best model as the final output
            best_ckpt = {
                "model": model.state_dict(),
                "c2i": c2i,
                "max_len": MAX_LEN,
            }
            torch.save(best_ckpt, output_path)
            print(f"    >> New best F1={best_f1:.4f} - saved to {output_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

        # Cleanup old checkpoints (keep last 3)
        import glob
        all_ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "seg_ckpt_epoch_*.pt")))
        if len(all_ckpts) > 3:
            for old in all_ckpts[:-3]:
                os.remove(old)

    elapsed = time.time() - t_start
    print(f"\nTraining complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Best val F1: {best_f1:.4f}")
    print(f"  Model saved: {output_path}")

    # ------------------------------------------------------------------
    # 6. Post-training validation (curated test cases)
    # ------------------------------------------------------------------
    run_curated_validation(output_path, c2i, device)

    print(f"\n{'='*60}")
    print(f"  All Done!")
    print(f"{'='*60}")


# =====================================================================
# CURATED VALIDATION
# =====================================================================

CURATED_TESTS = [
    # (input, expected_parts)
    # --- Full words: must NOT split ---
    ("customer", ["customer"]),
    ("first", ["first"]),
    ("last", ["last"]),
    ("address", ["address"]),
    ("balance", ["balance"]),
    ("name", ["name"]),
    ("date", ["date"]),
    ("price", ["price"]),
    ("order", ["order"]),
    ("amount", ["amount"]),
    ("employee", ["employee"]),
    ("department", ["department"]),
    ("transaction", ["transaction"]),
    ("description", ["description"]),
    ("configuration", ["configuration"]),
    ("identifier", ["identifier"]),
    ("status", ["status"]),
    ("payment", ["payment"]),
    ("invoice", ["invoice"]),
    ("shipment", ["shipment"]),

    # --- 2-part abbreviation splits ---
    ("custid", ["cust", "id"]),
    ("acctbal", ["acct", "bal"]),
    ("empname", ["emp", "name"]),
    ("deptcode", ["dept", "code"]),
    ("txnamt", ["txn", "amt"]),
    ("shipaddr", ["ship", "addr"]),
    ("ordstat", ["ord", "stat"]),
    ("payamt", ["pay", "amt"]),
    ("usrid", ["usr", "id"]),
    ("mgrname", ["mgr", "name"]),

    # --- 3-part concatenations ---
    ("custacctbal", ["cust", "acct", "bal"]),
    ("empfirstname", ["emp", "first", "name"]),
    ("usrgrpmap", ["usr", "grp", "map"]),
    ("txnamtbal", ["txn", "amt", "bal"]),
    ("deptorgcode", ["dept", "org", "code"]),
    ("cfgmgrstat", ["cfg", "mgr", "stat"]),
    ("shipaddrdest", ["ship", "addr", "dest"]),

    # --- Mixed abbreviation + full-word ---
    ("custname", ["cust", "name"]),
    ("acctbalance", ["acct", "balance"]),
    ("empaddress", ["emp", "address"]),
    ("orgtitle", ["org", "title"]),
    ("deptdescription", ["dept", "description"]),
    ("paymentamt", ["payment", "amt"]),
    ("orderstatus", ["order", "status"]),

    # --- Edge cases ---
    ("id", ["id"]),
    ("a", ["a"]),
    ("cfg", ["cfg"]),
]


def run_curated_validation(model_path: str, c2i: Dict[str, int], device: torch.device):
    """Run model on curated test cases and report pass/fail."""
    print(f"\n{'='*60}")
    print(f"  Post-Training Curated Validation")
    print(f"{'='*60}")

    # Load best model
    ckpt = torch.load(model_path, map_location=device)
    vocab_size = len(ckpt["c2i"])

    model = SegModel(vocab_size=vocab_size).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    c2i_loaded = ckpt["c2i"]
    max_len = ckpt.get("max_len", MAX_LEN)

    passed = 0
    failed = 0
    results = []

    for text, expected in CURATED_TESTS:
        # Encode
        s = text.lower()[:max_len]
        ids = [c2i_loaded.get(c, 1) for c in s]
        x = torch.tensor(ids).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(x)[0]
            probs = torch.softmax(out, dim=1)[:, 1].cpu().tolist()

        # Apply same logic as segment_token with 0.65 threshold
        cuts = [i for i, p in enumerate(probs) if p > 0.65]

        parts = []
        start = 0
        for c in cuts:
            parts.append(s[start:c+1])
            start = c + 1
        if start < len(s):
            parts.append(s[start:])

        # Min sub-token length guard
        if len(parts) > 1 and any(len(p) < 2 for p in parts):
            parts = [s]

        if len(parts) <= 1:
            parts = [s]

        expected_lower = [e.lower() for e in expected]
        ok = parts == expected_lower

        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        results.append((status, text, expected_lower, parts))

    # Print results
    print(f"\n  {'Status':<6} {'Input':<25} {'Expected':<35} {'Got'}")
    print(f"  {'-'*6} {'-'*25} {'-'*35} {'-'*30}")

    for status, text, expected, got in results:
        exp_str = " | ".join(expected)
        got_str = " | ".join(got)
        marker = "[OK]" if status == "PASS" else "[XX]"
        print(f"  {marker:<6} {text:<25} {exp_str:<35} {got_str}")

    total = passed + failed
    pct = passed / total * 100 if total > 0 else 0
    print(f"\n  Results: {passed}/{total} passed ({pct:.1f}%)")

    if pct >= 95:
        print(f"  EXCELLENT - Model is production-ready!")
    elif pct >= 85:
        print(f"  GOOD - Minor issues remain, whitelist provides safety net.")
    elif pct >= 70:
        print(f"  FAIR - Consider more training data or epochs.")
    else:
        print(f"  NEEDS WORK - Significant improvements needed.")

    return pct


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Segmentation Model")
    parser.add_argument("--data", default=DEFAULT_DATA_DIR,
                        help=f"Training data directory (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"Output model path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Max training epochs (default: 30)")
    parser.add_argument("--batch", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--embed-dim", type=int, default=128,
                        help="Embedding dimension (default: 128)")
    parser.add_argument("--layers", type=int, default=3,
                        help="Transformer encoder layers (default: 3)")
    parser.add_argument("--dropout", type=float, default=0.15,
                        help="Dropout rate (default: 0.15)")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience (default: 5)")
    parser.add_argument("--boundary-weight", type=float, default=2.0,
                        help="Class weight for boundary label (default: 2.0)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")

    args = parser.parse_args()

    train(
        data_dir=args.data,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        embed_dim=args.embed_dim,
        n_layers=args.layers,
        dropout=args.dropout,
        patience=args.patience,
        boundary_weight=args.boundary_weight,
        seed=args.seed,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
