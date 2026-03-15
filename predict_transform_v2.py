#!/usr/bin/env python3
"""
Inference script for the Transformer-based multi-class transformation predictor.

Supports three modes:
  1. --json '{"source_columns": [...], "target_column": {...}}'
  2. --json-file path/to/input.json
  3. --interactive (guided prompt)

Usage:
  python predict_transform_v2.py --interactive
  python predict_transform_v2.py --json '{"source_columns":[{"name":"first_name","type":"string"}],"target_column":{"name":"full_name","type":"string"}}'
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
CKPT_DIR = BASE / "transformer_classifier_checkpoints"
NUM_NUMERIC = 7


# ---------------------------------------------------------------------------
# Text serialization (must match train_transformer_classifier.py exactly)
# ---------------------------------------------------------------------------

def _col_text(col: dict) -> str:
    name = col.get("name", "unknown")
    ctype = col.get("type", "string")
    return f"{name} [{ctype}]"


def record_to_text(record: dict) -> str:
    src_parts = [_col_text(c) for c in record.get("source_columns", [])]
    src_text = " , ".join(src_parts) if src_parts else "unknown [string]"
    tgt_text = _col_text(record.get("target_column", {}))
    return f"{src_text} -> {tgt_text}"


def extract_numeric(record: dict) -> List[float]:
    src_cols = record.get("source_columns", [])
    tcol = record.get("target_column", {})
    src_ents = [c.get("entropy", 0.0) for c in src_cols]
    tgt_ent = tcol.get("entropy", 0.0)
    mean_s = sum(src_ents) / len(src_ents) if src_ents else 0.0
    max_s = max(src_ents) if src_ents else 0.0
    n_src = float(len(src_cols))
    pk = 1.0 if any(c.get("is_pk", False) for c in src_cols) else 0.0
    delta = tgt_ent - mean_s
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
# Model (must match train_transformer_classifier.py architecture)
# ---------------------------------------------------------------------------

class TransformerClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, num_numeric: int = NUM_NUMERIC,
                 dropout: float = 0.2, freeze_layers: int = 0):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        bert_dim = self.bert.config.hidden_size

        self.num_proj = nn.Sequential(
            nn.Linear(num_numeric, 64),
            nn.GELU(),
            nn.Dropout(dropout),
        )

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
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = outputs.last_hidden_state[:, 0]
        num_rep = self.num_proj(numeric)
        combined = torch.cat([cls_rep, num_rep], dim=1)
        return self.classifier(combined)


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class TransformPredictor:
    def __init__(self, ckpt_dir: Path = CKPT_DIR):
        ckpt_path = ckpt_dir / "best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.class_names = ckpt["class_names"]
        self.num_classes = ckpt["num_classes"]
        self.max_len = ckpt.get("max_len", 96)
        self.temperature = ckpt.get("temperature", 1.0)
        model_name = ckpt.get("model_name", "distilbert-base-uncased")

        # Load tokenizer from saved copy or download
        tok_dir = ckpt_dir / "tokenizer"
        if tok_dir.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Build and load model
        self.model = TransformerClassifier(
            model_name=model_name,
            num_classes=self.num_classes,
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Print checkpoint info
        print(f"Model: {model_name} | Classes: {self.num_classes} | "
              f"Temperature: {self.temperature:.2f}")
        top1 = ckpt.get("val_top1", 0)
        mf1 = ckpt.get("val_macro_f1", 0)
        print(f"Checkpoint metrics: top1={top1:.4f}  macro_f1={mf1:.4f}")

    @torch.no_grad()
    def predict(self, record: dict, top_k: int = 5) -> List[Dict]:
        text = record_to_text(record)
        enc = self.tokenizer(
            text, max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        nums = torch.tensor([extract_numeric(record)], dtype=torch.float32)

        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        nums = nums.to(self.device)

        logits = self.model(ids, mask, nums)
        probs = torch.softmax(logits / self.temperature, dim=-1)[0]

        topk = probs.topk(min(top_k, self.num_classes))
        results = []
        for idx, prob in zip(topk.indices.cpu().tolist(), topk.values.cpu().tolist()):
            results.append({
                "transform_type": self.class_names[idx],
                "confidence": round(prob * 100, 2),
            })
        return results


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def interactive(predictor: TransformPredictor):
    print("\n=== Interactive Transformation Predictor (Transformer) ===")
    print("Describe source and target columns. Type 'quit' to exit.\n")

    while True:
        try:
            n_src = input("Number of source columns (default 1): ").strip() or "1"
            if n_src.lower() == "quit":
                break
            n_src = int(n_src)
        except ValueError:
            print("Invalid number, using 1")
            n_src = 1

        src_cols = []
        for i in range(n_src):
            print(f"\nSource column {i + 1}:")
            name = input("  Name: ").strip()
            if name.lower() == "quit":
                return
            ctype = input("  Type (string/int/decimal/date/boolean) [string]: ").strip() or "string"
            entropy = float(input("  Entropy [3.0]: ").strip() or "3.0")
            is_pk = input("  Is PK? (y/n) [n]: ").strip().lower() == "y"
            src_cols.append({"name": name, "type": ctype, "is_pk": is_pk, "entropy": entropy})

        print("\nTarget column:")
        tgt_name = input("  Name: ").strip()
        if tgt_name.lower() == "quit":
            return
        tgt_type = input("  Type (string/int/decimal/date/boolean) [string]: ").strip() or "string"
        tgt_ent = float(input("  Entropy [3.0]: ").strip() or "3.0")

        record = {
            "source_columns": src_cols,
            "target_column": {"name": tgt_name, "type": tgt_type, "entropy": tgt_ent},
        }

        print(f"\nInput: {json.dumps(record, indent=2)}")
        results = predictor.predict(record)
        print("\nTop-5 predicted transformations:")
        for i, r in enumerate(results, 1):
            bar = "\u2588" * int(r["confidence"] / 3.5)
            print(f"  {i}. {r['transform_type']:<35} {r['confidence']:>6.2f}%  {bar}")

        print("\n--- New Prediction ---")


def main():
    ap = argparse.ArgumentParser(description="Predict transformation type (Transformer model)")
    ap.add_argument("--ckpt-dir", type=Path, default=CKPT_DIR)
    ap.add_argument("--json", type=str, help="JSON record string")
    ap.add_argument("--json-file", type=Path, help="Path to JSON file with record")
    ap.add_argument("--interactive", action="store_true", help="Interactive mode")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    predictor = TransformPredictor(args.ckpt_dir)

    if args.json:
        record = json.loads(args.json)
        results = predictor.predict(record, args.top_k)
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['transform_type']:<35} {r['confidence']:>6.2f}%")

    elif args.json_file:
        record = json.loads(args.json_file.read_text(encoding="utf-8"))
        results = predictor.predict(record, args.top_k)
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['transform_type']:<35} {r['confidence']:>6.2f}%")

    elif args.interactive:
        interactive(predictor)

    else:
        print("Use --interactive, --json, or --json-file")
        ap.print_help()


if __name__ == "__main__":
    main()
