#!/usr/bin/env python3
"""
Inference script for the multi-class transformation predictor (v2).

Given source columns and a target column, predicts the top-K most likely
transformation types with confidence scores.

Uses character trigrams + semantic keyword features for robust generalization
to novel column names.

Usage:
  python predict_transform.py --interactive
  python predict_transform.py --json '{"source_columns": [...], "target_column": {...}}'
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Tuple

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Feature extraction (must match training exactly)
# ---------------------------------------------------------------------------

TYPE_VOCAB = {"<unk>": 0, "string": 1, "int": 2, "decimal": 3, "date": 4, "boolean": 5}
NUM_TYPES = len(TYPE_VOCAB)
MAX_WORD_TOKS = 24
MAX_CHAR_TOKS = 80
NUM_NUMERIC = 7

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
KW_GROUP_KEYS = sorted(KEYWORD_GROUPS.keys())


def type_id(t: str) -> int:
    return TYPE_VOCAB.get((t or "").lower().strip(), 0)


def _name_tokens(name: str) -> List[str]:
    return [w for w in re.split(r"\W+", (name or "").lower().replace("_", " ").replace("-", " ")) if w]


def _char_trigrams(word: str) -> List[str]:
    w = "#" + word.lower() + "#"
    return [w[i:i+3] for i in range(len(w) - 2)]


def name_to_trigrams(name: str) -> List[str]:
    tris = []
    for w in _name_tokens(name):
        tris.extend(_char_trigrams(w))
    return tris


def keyword_features(name: str) -> List[float]:
    tokens = set(_name_tokens(name))
    return [1.0 if tokens & KEYWORD_GROUPS[k] else 0.0 for k in KW_GROUP_KEYS]


def extract_features(record: dict):
    src_words, src_tris, src_kw_all = [], [], []
    for col in record.get("source_columns", []):
        name = col.get("name", "")
        src_words.extend(_name_tokens(name))
        src_tris.extend(name_to_trigrams(name))
        kw = keyword_features(name)
        if not src_kw_all:
            src_kw_all = kw
        else:
            src_kw_all = [max(a, b) for a, b in zip(src_kw_all, kw)]
    if not src_kw_all:
        src_kw_all = [0.0] * NUM_KW_GROUPS

    tcol = record.get("target_column", {})
    tgt_name = tcol.get("name", "")
    tgt_words = _name_tokens(tgt_name)
    tgt_tris = name_to_trigrams(tgt_name)
    tgt_kw = keyword_features(tgt_name)

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


class Vocab:
    PAD_ID = 0
    UNK_ID = 1

    def __init__(self, tokens: List[str]):
        self._idx = {}
        for i, t in enumerate(tokens):
            self._idx[t] = i

    def encode(self, tokens, max_len):
        ids = [self._idx.get(t, self.UNK_ID) for t in tokens[:max_len]]
        ids += [self.PAD_ID] * (max_len - len(ids))
        return ids


# ---------------------------------------------------------------------------
# Model (must match training architecture exactly)
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class MultiClassTransformPredictor(nn.Module):
        def __init__(self, word_vocab_size, tri_vocab_size, num_classes,
                     embed_dim=64, hidden=320, dropout=0.0):
            super().__init__()
            H4 = hidden // 4
            self.word_embed = nn.Embedding(word_vocab_size, embed_dim, padding_idx=0)
            self.src_word_proj = nn.Linear(embed_dim, H4)
            self.tgt_word_proj = nn.Linear(embed_dim, H4)
            self.tri_embed = nn.Embedding(tri_vocab_size, embed_dim // 2, padding_idx=0)
            self.src_tri_proj = nn.Linear(embed_dim // 2, H4)
            self.tgt_tri_proj = nn.Linear(embed_dim // 2, H4)
            self.src_type_embed = nn.Embedding(NUM_TYPES, 16)
            self.tgt_type_embed = nn.Embedding(NUM_TYPES, 16)
            self.src_kw_proj = nn.Linear(NUM_KW_GROUPS, 32)
            self.tgt_kw_proj = nn.Linear(NUM_KW_GROUPS, 32)
            self.num_proj = nn.Linear(NUM_NUMERIC, 48)
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
            sw = torch.relu(self.src_word_proj(self._pool(self.word_embed, src_wids)))
            tw = torch.relu(self.tgt_word_proj(self._pool(self.word_embed, tgt_wids)))
            sc = torch.relu(self.src_tri_proj(self._pool(self.tri_embed, src_tids)))
            tc = torch.relu(self.tgt_tri_proj(self._pool(self.tri_embed, tgt_tids)))
            src_rep = torch.cat([sw, sc], dim=1)
            tgt_rep = torch.cat([tw, tc], dim=1)
            interact = src_rep * tgt_rep
            src_emb = self.src_type_embed(src_types).view(src_types.size(0), -1)
            tgt_emb = self.tgt_type_embed(tgt_type)
            skw = torch.relu(self.src_kw_proj(src_kw))
            tkw = torch.relu(self.tgt_kw_proj(tgt_kw))
            num_rep = torch.relu(self.num_proj(nums))
            fused = torch.cat([src_rep, tgt_rep, interact,
                               src_emb, tgt_emb, skw, tkw, num_rep], dim=1)
            return self.classifier(fused)


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class TransformPredictor:
    def __init__(self, checkpoint_path: str):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not installed")

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.class_names = ckpt["class_names"]
        self.num_classes = ckpt["num_classes"]
        self.word_vocab = Vocab(ckpt["word_vocab_tokens"])
        self.tri_vocab = Vocab(ckpt["tri_vocab_tokens"])
        self.temperature = ckpt.get("temperature", 1.0)

        self.model = MultiClassTransformPredictor(
            word_vocab_size=len(ckpt["word_vocab_tokens"]),
            tri_vocab_size=len(ckpt["tri_vocab_tokens"]),
            num_classes=self.num_classes,
            embed_dim=ckpt.get("embed_dim", 64),
            hidden=ckpt.get("hidden", 320),
            dropout=0.0,
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        print(f"Loaded model: {self.num_classes} transform types, "
              f"top1={ckpt.get('val_top1', 0):.4f}, "
              f"macro_f1={ckpt.get('val_macro_f1', 0):.4f}, "
              f"temp={self.temperature:.2f}")

    @torch.no_grad()
    def predict(self, record: dict, top_k: int = 5) -> List[Tuple[str, float]]:
        sw, tw, st, tt, skw, tkw, nums = extract_features(record)
        swi = torch.tensor([self.word_vocab.encode(sw, MAX_WORD_TOKS)], dtype=torch.long)
        twi = torch.tensor([self.word_vocab.encode(tw, MAX_WORD_TOKS)], dtype=torch.long)
        sti = torch.tensor([self.tri_vocab.encode(st, MAX_CHAR_TOKS)], dtype=torch.long)
        tti = torch.tensor([self.tri_vocab.encode(tt, MAX_CHAR_TOKS)], dtype=torch.long)
        src_types = [type_id(c.get("type")) for c in record.get("source_columns", [])][:4]
        src_types += [0] * (4 - len(src_types))
        st_t = torch.tensor([src_types], dtype=torch.long)
        tt_t = torch.tensor([type_id((record.get("target_column") or {}).get("type"))], dtype=torch.long)
        skw_t = torch.tensor([skw], dtype=torch.float32)
        tkw_t = torch.tensor([tkw], dtype=torch.float32)
        nums_t = torch.tensor([nums], dtype=torch.float32)

        logits = self.model(swi, twi, sti, tti, st_t, tt_t, skw_t, tkw_t, nums_t)
        probs = torch.softmax(logits / self.temperature, dim=-1).squeeze(0)

        topk = probs.topk(min(top_k, self.num_classes))
        return [(self.class_names[idx], round(prob, 4))
                for idx, prob in zip(topk.indices.tolist(), topk.values.tolist())]


def interactive_mode(predictor):
    print("\n" + "=" * 60)
    print("INTERACTIVE TRANSFORM PREDICTOR (v2 — semantic)")
    print("Enter column info, or 'quit' to exit.")
    print("=" * 60)

    while True:
        print("\n--- New Prediction ---")
        try:
            n_src = input("Number of source columns (default 1): ").strip()
            n_src = int(n_src) if n_src else 1
        except (ValueError, EOFError):
            n_src = 1

        source_columns = []
        for i in range(n_src):
            print(f"\nSource column {i+1}:")
            name = input("  Name: ").strip()
            if name.lower() == "quit":
                return
            col_type = input("  Type (string/int/decimal/date/boolean) [string]: ").strip() or "string"
            entropy = input("  Entropy [3.0]: ").strip()
            entropy = float(entropy) if entropy else 3.0
            is_pk = input("  Is PK? (y/n) [n]: ").strip().lower() == "y"
            source_columns.append({"name": name, "type": col_type, "is_pk": is_pk, "entropy": entropy})

        print("\nTarget column:")
        tgt_name = input("  Name: ").strip()
        if tgt_name.lower() == "quit":
            return
        tgt_type = input("  Type (string/int/decimal/date/boolean) [string]: ").strip() or "string"
        tgt_entropy = input("  Entropy [3.0]: ").strip()
        tgt_entropy = float(tgt_entropy) if tgt_entropy else 3.0

        record = {
            "source_columns": source_columns,
            "target_column": {"name": tgt_name, "type": tgt_type, "entropy": tgt_entropy},
        }

        print(f"\nInput: {json.dumps(record, indent=2)}")
        results = predictor.predict(record, top_k=5)
        print(f"\nTop-5 predicted transformations:")
        for i, (tt, conf) in enumerate(results, 1):
            bar = "█" * int(conf * 30)
            print(f"  {i}. {tt:<35} {conf:6.2%}  {bar}")


def main():
    if not TORCH_AVAILABLE:
        raise SystemExit("pip install torch")

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=Path(__file__).resolve().parent / "multiclass_classifier_checkpoints" / "best.pt")
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--json-file", type=Path, default=None)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()

    predictor = TransformPredictor(str(args.checkpoint))

    if args.interactive:
        interactive_mode(predictor)
    elif args.json:
        record = json.loads(args.json)
        results = predictor.predict(record, args.top_k)
        print(f"\nPredicted transformations:")
        for i, (tt, conf) in enumerate(results, 1):
            print(f"  {i}. {tt:<35} {conf:6.2%}")
    elif args.json_file:
        for idx, line in enumerate(args.json_file.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            record = json.loads(line)
            results = predictor.predict(record, args.top_k)
            src_names = [c.get("name", "?") for c in record.get("source_columns", [])]
            tgt_name = (record.get("target_column") or {}).get("name", "?")
            print(f"\n[{idx+1}] {src_names} -> {tgt_name}")
            for i, (tt, conf) in enumerate(results, 1):
                print(f"     {i}. {tt:<35} {conf:6.2%}")
    else:
        print("Use --json, --json-file, or --interactive. See --help.")


if __name__ == "__main__":
    main()
