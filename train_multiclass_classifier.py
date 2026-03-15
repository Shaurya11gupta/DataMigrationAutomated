#!/usr/bin/env python3
"""
Train multi-class transformation type predictor — v2 with semantic generalization.

Key improvements over v1:
  1. Character trigram tokenization — handles ANY column name, even novel ones
     ("punctuation" → "pun","unc","nct"... → meaningful embedding even if unseen)
  2. Semantic keyword features — 28 keyword groups (cleaning, hashing, temporal, etc.)
     detect domain-relevant terms in source/target names
  3. Dual path: word-level tokens (for known words) + char trigrams (for novel words)
  4. Source/target separate paths with interaction

Architecture:
  Source path:
    word tokens → Embedding → mean pool → src_word (H/4)
    char trigrams → Embedding → mean pool → src_char (H/4)
    → concat → src_rep (H/2)
  Target path: (same) → tgt_rep (H/2)
  Interaction: src_rep * tgt_rep (H/2)
  Keywords: src_kw(32) + tgt_kw(32)
  Categorical: src_types(64) + tgt_type(16)
  Numeric: entropy etc. (48)
  Fusion → MLP → num_classes

Usage:
  python train_multiclass_classifier.py [--data path] [--epochs N]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

BASE = Path(__file__).resolve().parent
DEFAULT_DATA = BASE / "unified_transformation_training_data.jsonl"
DEFAULT_SAVE = BASE / "multiclass_classifier_checkpoints"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TYPE_VOCAB = {"<unk>": 0, "string": 1, "int": 2, "decimal": 3, "date": 4, "boolean": 5}
NUM_TYPES = len(TYPE_VOCAB)
MAX_WORD_TOKS = 24      # max word tokens per column side
MAX_CHAR_TOKS = 80      # max character trigrams per column side
NUM_NUMERIC = 7

# ---------------------------------------------------------------------------
# Semantic keyword groups — fires when source/target names contain these terms
# ---------------------------------------------------------------------------
KEYWORD_GROUPS = {
    "cleaning": {"clean", "cleaned", "sanitize", "sanitized", "strip", "stripped",
                 "remove", "removed", "filter", "filtered", "purge", "purged", "wash"},
    "hashing": {"hash", "hashed", "digest", "checksum", "fingerprint", "md5", "sha"},
    "encryption": {"encrypt", "encrypted", "decrypt", "cipher", "secret", "protected", "secure"},
    "masking": {"mask", "masked", "redact", "redacted", "anon", "anonymize",
                "anonymized", "hidden", "obscure", "obscured"},
    "aggregation": {"sum", "total", "count", "avg", "average", "min", "max",
                    "aggregate", "aggregated"},
    "formatting": {"format", "formatted", "display", "pretty", "human"},
    "parsing": {"parse", "parsed", "extract", "extracted"},
    "normalization": {"normalize", "normalized", "canonical", "standard",
                      "standardized", "harmonize", "harmonized"},
    "case_change": {"lower", "upper", "initcap", "capitalize", "capitalized",
                    "lowercase", "uppercase"},
    "trimming": {"trim", "trimmed", "pad", "padded", "lpad", "rpad"},
    "splitting": {"split", "token", "segment", "part", "delimit", "delimiter"},
    "concatenation": {"concat", "concatenate", "full", "combined", "merged",
                      "complete", "join", "joined"},
    "identity": {"copy", "copied", "duplicate", "duplicated", "mirror", "same",
                 "renamed", "alias"},
    "temporal_extract": {"year", "month", "day", "hour", "minute", "quarter",
                         "week", "fiscal"},
    "temporal_general": {"date", "time", "datetime", "timestamp", "ts", "period",
                         "interval", "duration"},
    "boolean_flag": {"is", "has", "flag", "indicator", "check", "valid", "active",
                     "enabled", "disabled"},
    "bucketing": {"bucket", "bucketed", "bin", "binned", "band", "tier", "range",
                  "quantile"},
    "ranking": {"rank", "ranked", "dense", "row", "number", "position", "sequence"},
    "window_ops": {"running", "cumulative", "moving", "rolling", "lag", "lead",
                   "prev", "next", "window"},
    "lookup": {"lookup", "enrich", "enriched", "dimension", "reference", "resolved",
               "mapped"},
    "imputation": {"impute", "imputed", "fill", "filled", "default", "coalesce",
                   "fallback", "missing"},
    "json_xml": {"json", "xml", "payload", "config", "metadata", "array", "nested",
                 "struct"},
    "punctuation": {"punctuation", "punct", "symbol", "special", "character", "char"},
    "raw_marker": {"raw", "original", "input", "source", "dirty", "unprocessed",
                   "included", "with", "before"},
    "output_marker": {"output", "result", "final", "processed", "converted",
                      "transformed", "without", "after"},
    "dedup": {"dedup", "deduplicate", "deduplicated", "canonical", "unique", "distinct"},
    "slug": {"slug", "slugify", "slugified", "seo", "permalink", "url"},
    "type_conversion": {"cast", "convert", "converted", "text", "str", "num",
                        "float", "bool", "to"},
}
NUM_KW_GROUPS = len(KEYWORD_GROUPS)
KW_GROUP_KEYS = sorted(KEYWORD_GROUPS.keys())  # deterministic order


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def type_id(t: str) -> int:
    return TYPE_VOCAB.get((t or "").lower().strip(), 0)


def _name_tokens(name: str) -> List[str]:
    """Split column name into word tokens."""
    return [w for w in re.split(r"\W+", (name or "").lower().replace("_", " ").replace("-", " ")) if w]


def _char_trigrams(word: str) -> List[str]:
    """Extract character trigrams with boundary markers."""
    w = "#" + word.lower() + "#"
    return [w[i:i+3] for i in range(len(w) - 2)]


def name_to_trigrams(name: str) -> List[str]:
    """Convert a column name to character trigrams."""
    tris = []
    for w in _name_tokens(name):
        tris.extend(_char_trigrams(w))
    return tris


def keyword_features(name: str) -> List[float]:
    """28 binary features: does this name contain keywords from each group?"""
    tokens = set(_name_tokens(name))
    return [1.0 if tokens & KEYWORD_GROUPS[k] else 0.0 for k in KW_GROUP_KEYS]


def extract_features(record: dict):
    """Returns (src_words, tgt_words, src_trigrams, tgt_trigrams, src_kw, tgt_kw, numeric)."""
    src_words, src_tris, src_kw_all = [], [], []
    for col in record.get("source_columns", []):
        name = col.get("name", "")
        src_words.extend(_name_tokens(name))
        src_tris.extend(name_to_trigrams(name))
        kw = keyword_features(name)
        if not src_kw_all:
            src_kw_all = kw
        else:
            src_kw_all = [max(a, b) for a, b in zip(src_kw_all, kw)]  # OR across cols
    if not src_kw_all:
        src_kw_all = [0.0] * NUM_KW_GROUPS

    tcol = record.get("target_column", {})
    tgt_name = tcol.get("name", "")
    tgt_words = _name_tokens(tgt_name)
    tgt_tris = name_to_trigrams(tgt_name)
    tgt_kw = keyword_features(tgt_name)

    # Numeric features
    src_cols = record.get("source_columns", [])
    src_ents = [c.get("entropy", 0.0) for c in src_cols]
    tgt_ent = tcol.get("entropy", 0.0)
    mean_s = sum(src_ents) / len(src_ents) if src_ents else 0.0
    max_s = max(src_ents) if src_ents else 0.0
    n_src = float(len(src_cols))
    pk = 1.0 if any(c.get("is_pk", False) for c in src_cols) else 0.0
    delta = tgt_ent - mean_s
    overlap = float(len(set(src_words) & set(tgt_words)))
    nums = [mean_s, max_s, tgt_ent, n_src, pk, delta, overlap]

    return src_words, tgt_words, src_tris, tgt_tris, src_kw_all, tgt_kw, nums


# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

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


def build_class_vocab(records):
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

if TORCH_AVAILABLE:

    class MultiClassDataset(Dataset):
        def __init__(self, records, word_vocab, tri_vocab, class_vocab):
            self.records = records
            self.word_vocab = word_vocab
            self.tri_vocab = tri_vocab
            self.class_vocab = class_vocab

        def __len__(self):
            return len(self.records)

        def __getitem__(self, i):
            r = self.records[i]
            sw, tw, st, tt, skw, tkw, nums = extract_features(r)
            src_word_ids = self.word_vocab.encode(sw, MAX_WORD_TOKS)
            tgt_word_ids = self.word_vocab.encode(tw, MAX_WORD_TOKS)
            src_tri_ids = self.tri_vocab.encode(st, MAX_CHAR_TOKS)
            tgt_tri_ids = self.tri_vocab.encode(tt, MAX_CHAR_TOKS)
            src_types = [type_id(c.get("type")) for c in r.get("source_columns", [])][:4]
            src_types += [0] * (4 - len(src_types))
            tgt_type = type_id((r.get("target_column") or {}).get("type"))
            class_id = self.class_vocab.get((r.get("transform_type") or "").strip(), 0)
            return (
                torch.tensor(src_word_ids, dtype=torch.long),
                torch.tensor(tgt_word_ids, dtype=torch.long),
                torch.tensor(src_tri_ids, dtype=torch.long),
                torch.tensor(tgt_tri_ids, dtype=torch.long),
                torch.tensor(src_types, dtype=torch.long),
                torch.tensor(tgt_type, dtype=torch.long),
                torch.tensor(skw, dtype=torch.float32),
                torch.tensor(tkw, dtype=torch.float32),
                torch.tensor(nums, dtype=torch.float32),
                class_id,
            )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    class MultiClassTransformPredictor(nn.Module):
        def __init__(self, word_vocab_size, tri_vocab_size, num_classes,
                     embed_dim=64, hidden=320, dropout=0.3):
            super().__init__()
            H4 = hidden // 4

            # Word-level path (for known words)
            self.word_embed = nn.Embedding(word_vocab_size, embed_dim, padding_idx=0)
            self.src_word_proj = nn.Linear(embed_dim, H4)
            self.tgt_word_proj = nn.Linear(embed_dim, H4)

            # Character trigram path (for novel words)
            self.tri_embed = nn.Embedding(tri_vocab_size, embed_dim // 2, padding_idx=0)
            self.src_tri_proj = nn.Linear(embed_dim // 2, H4)
            self.tgt_tri_proj = nn.Linear(embed_dim // 2, H4)

            # Categorical
            self.src_type_embed = nn.Embedding(NUM_TYPES, 16)
            self.tgt_type_embed = nn.Embedding(NUM_TYPES, 16)

            # Keyword features
            self.src_kw_proj = nn.Linear(NUM_KW_GROUPS, 32)
            self.tgt_kw_proj = nn.Linear(NUM_KW_GROUPS, 32)

            # Numeric
            self.num_proj = nn.Linear(NUM_NUMERIC, 48)

            # Fusion:
            # src_rep(H/2) + tgt_rep(H/2) + interact(H/2) + src_types(64) + tgt_type(16)
            # + src_kw(32) + tgt_kw(32) + nums(48)
            H2 = hidden // 2
            fusion_dim = H2 + H2 + H2 + 64 + 16 + 32 + 32 + 48

            self.classifier = nn.Sequential(
                nn.BatchNorm1d(fusion_dim),
                nn.Linear(fusion_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_classes),
            )

        def _pool(self, embed_layer, ids):
            mask = (ids != 0).float().unsqueeze(-1)
            e = embed_layer(ids) * mask
            return e.sum(1) / mask.sum(1).clamp(min=1e-9)

        def forward(self, src_wids, tgt_wids, src_tids, tgt_tids,
                    src_types, tgt_type, src_kw, tgt_kw, nums):
            # Word-level representations
            sw = torch.relu(self.src_word_proj(self._pool(self.word_embed, src_wids)))
            tw = torch.relu(self.tgt_word_proj(self._pool(self.word_embed, tgt_wids)))

            # Character trigram representations
            sc = torch.relu(self.src_tri_proj(self._pool(self.tri_embed, src_tids)))
            tc = torch.relu(self.tgt_tri_proj(self._pool(self.tri_embed, tgt_tids)))

            # Combined source/target representations
            src_rep = torch.cat([sw, sc], dim=1)  # (B, H/2)
            tgt_rep = torch.cat([tw, tc], dim=1)  # (B, H/2)

            # Interaction
            interact = src_rep * tgt_rep  # (B, H/2)

            # Categorical
            src_emb = self.src_type_embed(src_types).view(src_types.size(0), -1)
            tgt_emb = self.tgt_type_embed(tgt_type)

            # Keywords
            skw = torch.relu(self.src_kw_proj(src_kw))
            tkw = torch.relu(self.tgt_kw_proj(tgt_kw))

            # Numeric
            num_rep = torch.relu(self.num_proj(nums))

            fused = torch.cat([src_rep, tgt_rep, interact,
                               src_emb, tgt_emb, skw, tkw, num_rep], dim=1)
            return self.classifier(fused)


# ---------------------------------------------------------------------------
# Label smoothing CrossEntropy
# ---------------------------------------------------------------------------

class LabelSmoothCE(nn.Module):
    def __init__(self, num_classes, smoothing=0.02):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, targets):
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        smooth = -log_probs.mean(dim=-1)
        return (self.confidence * nll + self.smoothing * smooth).mean()


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

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
    for swi, twi, sti, tti, st, tt, skw, tkw, nums, y in loader:
        swi, twi = swi.to(device), twi.to(device)
        sti, tti = sti.to(device), tti.to(device)
        st, tt = st.to(device), tt.to(device)
        skw, tkw = skw.to(device), tkw.to(device)
        nums, y = nums.to(device), y.to(device)
        opt.zero_grad()
        logits = model(swi, twi, sti, tti, st, tt, skw, tkw, nums)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if sched:
            sched.step()
        total_loss += loss.item() * swi.size(0)
        n += swi.size(0)
    return total_loss / n if n else 0.0


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    total_loss = n = correct_top1 = correct_top3 = 0
    criterion = nn.CrossEntropyLoss()
    class_tp = [0] * num_classes
    class_fp = [0] * num_classes
    class_fn = [0] * num_classes

    for swi, twi, sti, tti, st, tt, skw, tkw, nums, y in loader:
        swi, twi = swi.to(device), twi.to(device)
        sti, tti = sti.to(device), tti.to(device)
        st, tt = st.to(device), tt.to(device)
        skw, tkw = skw.to(device), tkw.to(device)
        nums, y = nums.to(device), y.to(device)
        logits = model(swi, twi, sti, tti, st, tt, skw, tkw, nums)
        loss = criterion(logits, y)
        total_loss += loss.item() * swi.size(0)
        n += swi.size(0)
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
    model.eval()
    all_logits, all_targets = [], []
    for swi, twi, sti, tti, st, tt, skw, tkw, nums, y in val_loader:
        swi, twi = swi.to(device), twi.to(device)
        sti, tti = sti.to(device), tti.to(device)
        st, tt = st.to(device), tt.to(device)
        skw, tkw = skw.to(device), tkw.to(device)
        nums = nums.to(device)
        logits = model(swi, twi, sti, tti, st, tt, skw, tkw, nums)
        all_logits.append(logits.cpu())
        all_targets.append(y)
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

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
# Main
# ---------------------------------------------------------------------------

def main():
    if not TORCH_AVAILABLE:
        raise SystemExit("pip install torch")

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=320)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label-smoothing", type=float, default=0.02)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    print(f"Samples/class: min={min(counts.values())}, max={max(counts.values())}, avg={len(train_recs)//num_classes}")

    # Build word vocab
    word_vocab = Vocab()
    for r in train_recs:
        sw, tw, _, _, _, _, _ = extract_features(r)
        word_vocab.add(sw)
        word_vocab.add(tw)
    word_vocab.build(min_count=2)

    # Build trigram vocab
    tri_vocab = Vocab()
    for r in train_recs:
        _, _, st, tt, _, _, _ = extract_features(r)
        tri_vocab.add(st)
        tri_vocab.add(tt)
    tri_vocab.build(min_count=3)

    print(f"Word vocab: {len(word_vocab)}  Trigram vocab: {len(tri_vocab)}")

    train_ds = MultiClassDataset(train_recs, word_vocab, tri_vocab, class_vocab)
    val_ds = MultiClassDataset(val_recs, word_vocab, tri_vocab, class_vocab)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    criterion = LabelSmoothCE(num_classes, smoothing=args.label_smoothing)

    model = MultiClassTransformPredictor(
        word_vocab_size=len(word_vocab),
        tri_vocab_size=len(tri_vocab),
        num_classes=num_classes,
        embed_dim=args.embed_dim,
        hidden=args.hidden,
        dropout=0.3,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=len(train_loader), pct_start=0.1, anneal_strategy="cos",
    )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_top1 = -1.0
    best_epoch = 0
    patience_left = args.patience

    print(f"\n{'Ep':>3} {'LR':>9} {'TLoss':>7} {'VLoss':>7} "
          f"{'Top1':>6} {'Top3':>6} {'MPr':>6} {'MRe':>6} {'MF1':>6}")
    print("-" * 78)

    for ep in range(1, args.epochs + 1):
        t_loss = train_one_epoch(model, train_loader, opt, criterion, device, sched)
        v_loss, top1, top3, mp, mr, mf, c_tp, c_fp, c_fn = evaluate(
            model, val_loader, device, num_classes)
        lr = opt.param_groups[0]["lr"]
        print(f"{ep:3d} {lr:9.2e} {t_loss:7.4f} {v_loss:7.4f} "
              f"{top1:6.4f} {top3:6.4f} {mp:6.4f} {mr:6.4f} {mf:6.4f}")

        if top1 > best_top1:
            best_top1 = top1
            best_epoch = ep
            patience_left = args.patience
            torch.save({
                "epoch": ep, "model_state": model.state_dict(),
                "word_vocab_tokens": word_vocab.ordered(),
                "tri_vocab_tokens": tri_vocab.ordered(),
                "class_names": class_names, "class_vocab": class_vocab,
                "num_classes": num_classes, "embed_dim": args.embed_dim,
                "hidden": args.hidden,
                "val_top1": top1, "val_top3": top3,
                "val_macro_f1": mf, "val_macro_prec": mp, "val_macro_rec": mr,
            }, args.save_dir / "best.pt")
            best_tp, best_fp, best_fn = c_tp, c_fp, c_fn
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"\nEarly stop at epoch {ep}")
                break

    print(f"\nBest epoch {best_epoch}: top1={best_top1:.4f}")

    print("\n" + "=" * 90)
    print("PER-CLASS REPORT (worst first, from best epoch):")
    print_per_class_report(class_names, best_tp, best_fp, best_fn, top_n=15)

    # Temperature calibration
    print("\n" + "=" * 90)
    print("TEMPERATURE CALIBRATION:")
    ckpt = torch.load(args.save_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    opt_temp = calibrate_temperature(model, val_loader, device)
    ckpt["temperature"] = opt_temp
    torch.save(ckpt, args.save_dir / "best.pt")
    print(f"Optimal temperature: {opt_temp:.2f}")
    print(f"Checkpoints: {args.save_dir}")


if __name__ == "__main__":
    main()
