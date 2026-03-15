#!/usr/bin/env python3
"""
Train the expansion classifier (UniversalGatekeeper).

Architecture: Embedding + dual Conv1d (kernel 2,3) + BiGRU + MLP head
Task: Binary classification -- is this token an abbreviation (1) or full word (0)?

This script exactly matches the architecture in seg_classf_abbrev_test.py
so the saved model can be loaded directly.

Usage:
  python train_expansion_classifier.py [--epochs 50] [--batch 64] [--lr 0.001]
  python train_expansion_classifier.py --resume  # resume from latest checkpoint
"""

import argparse
import json
import os
import math
import random
import time
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)


# ═══════════════════════════════════════════════════════════════
# VOCAB (must match gatekeeper_vocab.json exactly)
# ═══════════════════════════════════════════════════════════════

VOCAB_PATH = "gatekeeper_vocab.json"

def load_or_create_vocab():
    if os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH) as f:
            c2i = json.load(f)
        print(f"[INFO] Loaded existing vocab from {VOCAB_PATH} ({len(c2i)} chars)")
        return c2i

    # Create fresh vocab (should not normally happen)
    chars = ["<pad>", "-"] + [str(i) for i in range(10)] + ["_"]
    chars += [chr(c) for c in range(ord('a'), ord('z') + 1)]
    c2i = {ch: i for i, ch in enumerate(chars)}
    with open(VOCAB_PATH, "w") as f:
        json.dump(c2i, f)
    print(f"[INFO] Created new vocab ({len(c2i)} chars)")
    return c2i


# ═══════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════

class ClfDataset(Dataset):
    def __init__(self, path, c2i):
        self.examples = []
        self.c2i = c2i
        self.pad = c2i.get("<pad>", 0)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.examples.append((obj["token"], obj["label"]))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        token, label = self.examples[idx]
        ids = [self.c2i.get(c.lower(), self.pad) for c in token]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.float32)


def collate_fn(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    max_len = max(lengths)
    padded = torch.zeros(len(seqs), max_len, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, :len(s)] = s
    return padded, lengths, torch.stack(labels)


# ═══════════════════════════════════════════════════════════════
# MODEL -- EXACT MATCH to UniversalGatekeeper in seg_classf_abbrev_test.py
# ═══════════════════════════════════════════════════════════════

class UniversalGatekeeper(nn.Module):

    def __init__(self, vocab, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden, padding_idx=0)
        self.conv2 = nn.Conv1d(hidden, hidden, 2, padding=1)
        self.conv3 = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.gru = nn.GRU(hidden * 3, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(64, 1)
        )

    def forward(self, x, lengths):
        e = self.emb(x)
        c = e.permute(0, 2, 1)
        f2 = F.relu(self.conv2(c))[:, :, :x.size(1)]
        f3 = F.relu(self.conv3(c))[:, :, :x.size(1)]
        feat = torch.cat([e, f2.permute(0, 2, 1), f3.permute(0, 2, 1)], dim=2)
        packed = nn.utils.rnn.pack_padded_sequence(
            feat, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.head(h)


# ═══════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════

def compute_metrics(model, loader, threshold=0.4):
    """Compute accuracy, precision, recall, F1 on a data loader."""
    model.eval()
    tp = fp = tn = fn = 0
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for padded, lengths, labels in loader:
            padded, lengths, labels = padded.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)
            logits = model(padded, lengths).squeeze(-1)
            total_loss += criterion(logits, labels).item() * len(labels)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    avg_loss = total_loss / total if total else 0

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


# ═══════════════════════════════════════════════════════════════
# CURATED TEST SUITE
# ═══════════════════════════════════════════════════════════════

CURATED_TESTS = [
    # (token, expected_is_abbreviation)
    # Full words that MUST NOT be classified as abbreviations
    ("first", False), ("last", False), ("name", False), ("date", False),
    ("time", False), ("type", False), ("code", False), ("flag", False),
    ("age", False), ("key", False), ("cost", False), ("rate", False),
    ("role", False), ("user", False), ("file", False), ("node", False),
    ("host", False), ("port", False), ("path", False), ("mode", False),
    ("step", False), ("lock", False), ("tier", False), ("slot", False),
    ("zone", False), ("unit", False), ("area", False), ("line", False),
    ("page", False), ("view", False), ("rule", False), ("test", False),
    ("plan", False), ("task", False), ("note", False), ("list", False),
    ("item", False), ("link", False), ("send", False), ("read", False),
    ("load", False), ("sync", False), ("push", False), ("pull", False),
    ("null", False), ("true", False), ("text", False), ("mask", False),
    ("salt", False), ("hash", False), ("seed", False), ("root", False),
    ("email", False), ("phone", False), ("price", False), ("order", False),
    ("value", False), ("count", False), ("total", False), ("index", False),
    ("table", False), ("field", False), ("store", False), ("group", False),
    ("level", False), ("state", False), ("event", False), ("error", False),
    ("alert", False), ("debug", False), ("trace", False), ("token", False),
    ("image", False), ("video", False), ("audio", False), ("label", False),
    ("title", False), ("query", False), ("limit", False), ("batch", False),
    ("queue", False), ("cache", False), ("proxy", False), ("model", False),
    ("class", False), ("scope", False), ("trade", False), ("share", False),
    ("grant", False), ("shift", False), ("leave", False), ("skill", False),
    ("score", False), ("grade", False), ("badge", False), ("point", False),
    ("bonus", False), ("extra", False), ("gross", False), ("debit", False),
    ("credit", False), ("weight", False), ("height", False), ("length", False),
    ("amount", False), ("number", False), ("gender", False), ("salary", False),
    ("status", False), ("active", False), ("street", False), ("region", False),
    ("source", False), ("target", False), ("origin", False), ("parent", False),
    ("master", False), ("backup", False), ("mobile", False), ("office", False),
    ("branch", False), ("vendor", False), ("client", False), ("member", False),
    ("worker", False), ("doctor", False), ("period", False), ("domain", False),
    ("object", False), ("record", False), ("schema", False), ("format", False),
    ("output", False), ("report", False), ("metric", False), ("signal", False),
    ("thread", False), ("cursor", False), ("socket", False), ("stream", False),
    ("buffer", False), ("filter", False), ("layout", False),
    ("customer", False), ("database", False), ("employee", False),
    ("shipping", False), ("tracking", False), ("discount", False),
    ("description", False), ("transaction", False),

    # Abbreviations that MUST be classified as abbreviations
    ("cust", True), ("acct", True), ("addr", True), ("nm", True),
    ("amt", True), ("bal", True), ("qty", True), ("txn", True),
    ("num", True), ("dt", True), ("dob", True), ("dept", True),
    ("mgr", True), ("emp", True), ("pmt", True), ("inv", True),
    ("desc", True), ("flg", True), ("sts", True), ("ctry", True),
    ("phn", True), ("sal", True), ("grp", True), ("usr", True),
    ("src", True), ("tgt", True), ("typ", True), ("idx", True),
    ("cd", True), ("shp", True), ("trk", True), ("wt", True),
    ("ht", True), ("cr", True), ("dr", True), ("dsc", True),
    ("cfg", True), ("svc", True), ("ctrl", True), ("proc", True),
    ("attr", True), ("seq", True), ("tbl", True), ("col", True),
    ("ts", True), ("tm", True), ("yr", True), ("fk", True),
    ("pk", True), ("ref", True), ("id", True), ("org", True),
]


def run_curated_tests(model, c2i, threshold=0.4):
    """Run curated test suite and report results."""
    model.eval()
    pad = c2i.get("<pad>", 0)
    passed = 0
    failed_cases = []

    for token, expected_expand in CURATED_TESTS:
        ids = torch.tensor(
            [c2i.get(c.lower(), pad) for c in token],
            dtype=torch.long
        ).unsqueeze(0).to(DEVICE)
        lengths = torch.tensor([len(token)])

        with torch.no_grad():
            logit = model(ids, lengths)
            prob = torch.sigmoid(logit)[0].item()

        predicted = prob > threshold
        if predicted == expected_expand:
            passed += 1
        else:
            failed_cases.append((token, expected_expand, predicted, prob))

    total = len(CURATED_TESTS)
    print(f"\n  Curated test: {passed}/{total} = {passed/total:.1%}")
    if failed_cases:
        print(f"  Failed cases:")
        for tok, exp, pred, prob in failed_cases[:15]:
            exp_str = "ABBREV" if exp else "FULL"
            pred_str = "ABBREV" if pred else "FULL"
            print(f"    '{tok}': expected={exp_str}, got={pred_str}, prob={prob:.3f}")
    return passed / total if total else 0


# ═══════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════

def train(
    train_path="clf_training_data/clf_train.jsonl",
    val_path="clf_training_data/clf_val.jsonl",
    epochs=60,
    batch_size=64,
    lr=0.001,
    hidden=128,
    patience=12,
    pos_weight_factor=1.5,
    seed=SEED,
    resume=False,
):
    random.seed(seed)
    torch.manual_seed(seed)

    c2i = load_or_create_vocab()
    vocab_size = len(c2i)

    # Datasets
    train_ds = ClfDataset(train_path, c2i)
    val_ds = ClfDataset(val_path, c2i)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, drop_last=False
    )

    # Class distribution
    train_pos = sum(1 for _, l in train_ds.examples if l == 1)
    train_neg = len(train_ds) - train_pos
    print(f"\nTrain: {len(train_ds)} examples ({train_pos} pos / {train_neg} neg)")
    print(f"Val:   {len(val_ds)} examples")
    print(f"Device: {DEVICE}")

    # Model
    model = UniversalGatekeeper(vocab_size, hidden).to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model params: {param_count:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Loss with positive class weight to handle any remaining imbalance
    pw = torch.tensor([pos_weight_factor]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    # Checkpoint setup
    ckpt_dir = Path("clf_checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    best_model_path = "gatekeeper_universal_enhanced.pth"

    # Resume from checkpoint if requested
    start_epoch = 0
    best_f1 = 0.0
    no_improve = 0

    if resume:
        import glob
        ckpts = sorted(
            glob.glob(str(ckpt_dir / "clf_epoch_*.pt")),
            key=lambda p: int(Path(p).stem.split("_")[-1])
        )
        if ckpts:
            latest = ckpts[-1]
            ckpt = torch.load(latest, map_location=DEVICE)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"] + 1
            best_f1 = ckpt.get("best_f1", 0.0)
            print(f"\n[RESUME] from {latest}, epoch {start_epoch}, best_f1={best_f1:.4f}")
        else:
            print(f"\n[WARN] --resume set but no checkpoints found. Training from scratch.")

    print(f"\nTraining for {epochs - start_epoch} epochs (starting at epoch {start_epoch})...")
    print(f"{'='*80}")

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for padded, lengths, labels in train_loader:
            padded, lengths, labels = padded.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(padded, lengths).squeeze(-1)
            loss = criterion(logits, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = total_loss / n_batches

        # Validation
        val_metrics = compute_metrics(model, val_loader)

        # Curated test (every 5 epochs or last epoch)
        curated_acc = 0
        if epoch % 5 == 0 or epoch == epochs - 1:
            curated_acc = run_curated_tests(model, c2i)

        print(f"  Epoch {epoch+1:3d}/{epochs} | "
              f"train_loss={avg_train_loss:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} | "
              f"acc={val_metrics['accuracy']:.3f} | "
              f"P={val_metrics['precision']:.3f} R={val_metrics['recall']:.3f} "
              f"F1={val_metrics['f1']:.3f} | "
              f"FP={val_metrics['fp']} FN={val_metrics['fn']} | "
              f"lr={scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            no_improve = 0

            # Save best model (compatible with seg_classf_abbrev_test.py)
            torch.save(model.state_dict(), best_model_path)
            print(f"    >> NEW BEST F1={best_f1:.4f} -- saved to {best_model_path}")
        else:
            no_improve += 1

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt_path = ckpt_dir / f"clf_epoch_{epoch+1}.pt"
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_f1": best_f1,
            }, ckpt_path)
            # Keep only last 3
            import glob
            old_ckpts = sorted(
                glob.glob(str(ckpt_dir / "clf_epoch_*.pt")),
                key=lambda p: int(Path(p).stem.split("_")[-1])
            )
            while len(old_ckpts) > 3:
                os.remove(old_ckpts.pop(0))

        # Early stopping
        if no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    print(f"\n{'='*80}")
    print(f"Training complete. Best F1: {best_f1:.4f}")
    print(f"Best model saved to: {best_model_path}")

    # Final curated test
    # Reload best model
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    print(f"\n--- Final Curated Test Suite (best model) ---")
    run_curated_tests(model, c2i)

    # Detailed per-token results on the hardest cases
    print(f"\n--- Detailed Hard Case Results ---")
    hard_cases = [
        "first", "last", "time", "type", "code", "flag", "age", "key",
        "cost", "rate", "role", "user", "date", "name", "tax", "end",
        "cust", "acct", "addr", "nm", "amt", "bal", "qty", "txn",
        "num", "dt", "dob", "dept", "mgr", "emp", "desc", "flg",
    ]
    pad = c2i.get("<pad>", 0)
    model.eval()
    for token in hard_cases:
        ids = torch.tensor(
            [c2i.get(c.lower(), pad) for c in token],
            dtype=torch.long
        ).unsqueeze(0).to(DEVICE)
        lengths = torch.tensor([len(token)])
        with torch.no_grad():
            logit = model(ids, lengths)
            prob = torch.sigmoid(logit)[0].item()
        label = "ABBREV" if prob > 0.4 else "FULL"
        print(f"  {token:15} -> prob={prob:.4f} ({label})")

    return best_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train expansion classifier")
    parser.add_argument("--train", default="clf_training_data/clf_train.jsonl")
    parser.add_argument("--val", default="clf_training_data/clf_val.jsonl")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--pos-weight", type=float, default=1.5,
                        help="Positive class weight for BCELoss")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    args = parser.parse_args()

    train(
        train_path=args.train,
        val_path=args.val,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        hidden=args.hidden,
        patience=args.patience,
        pos_weight_factor=args.pos_weight,
        resume=args.resume,
    )
