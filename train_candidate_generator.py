#!/usr/bin/env python3
"""
Train Bi-Encoder + Cross-Encoder for Candidate Generation V2
=============================================================
Fine-tunes both models on the generated schema mapping training data.

Usage (GPU recommended):
  python train_candidate_generator.py --mode bi      # Train bi-encoder only
  python train_candidate_generator.py --mode cross   # Train cross-encoder only
  python train_candidate_generator.py --mode both    # Train both sequentially (recommended)

Requirements:
  pip install sentence-transformers torch datasets accelerate

GPU Training Tips:
  - With NVIDIA GPU (8GB+ VRAM): default settings work well
  - With NVIDIA GPU (24GB+ VRAM): increase batch size to 128/64
  - Training time on GPU: ~5-10 min bi-encoder, ~3-5 min cross-encoder
  - Training time on CPU: ~2-3 hours total (not recommended)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation,
)
from sentence_transformers.cross_encoder import CrossEncoder

# Handle API change across sentence-transformers versions
try:
    from sentence_transformers.cross_encoder.evaluation import CrossEncoderClassificationEvaluator as _CEEval
    _CE_EVAL_NEW_API = True
except ImportError:
    from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator as _CEEval
    _CE_EVAL_NEW_API = False


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  GPU detected: {name} ({mem:.1f} GB)")
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("  Apple MPS detected")
        return "mps"
    else:
        print("  WARNING: No GPU detected, using CPU (training will be slow)")
        return "cpu"


# ───────────────────────────────────────────────────────
# Data loading
# ───────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def make_bi_encoder_examples(records: List[Dict]) -> List[InputExample]:
    """Convert bi-encoder training records to InputExamples.
    
    For MultipleNegativesRankingLoss:
    - 2-element: (query, positive) → in-batch negatives
    - 3-element: (query, positive, hard_negative) → explicit hard negative
    """
    examples = []
    for r in records:
        query = r["query"]
        positive = r["positive"]
        negatives = r.get("negatives", [])

        # Always include (query, positive) for in-batch negatives
        examples.append(InputExample(texts=[query, positive]))

        # Add up to 3 hard negatives as triplets
        for neg in negatives[:3]:
            examples.append(InputExample(texts=[query, positive, neg]))

    return examples


def make_cross_encoder_examples(records: List[Dict]) -> List[InputExample]:
    """Convert cross-encoder records to (text_pair, label) format."""
    examples = []
    for r in records:
        examples.append(InputExample(
            texts=[r["query"], r["candidate"]],
            label=float(r["label"]),
        ))
    return examples


# ───────────────────────────────────────────────────────
# Bi-encoder training
# ───────────────────────────────────────────────────────

def train_bi_encoder(
    data_dir: str = "candidate_training_data",
    output_dir: str = "candidate_v2_models/bi_encoder",
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 2e-5,
    warmup_ratio: float = 0.1,
    seed: int = 42,
):
    set_seed(seed)
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f" BI-ENCODER TRAINING")
    print(f"{'='*60}")
    print(f"  Base model:  {base_model}")
    print(f"  Epochs:      {epochs}")
    print(f"  Batch size:  {batch_size}")
    print(f"  LR:          {lr}")
    device = detect_device()

    # Load data
    train_data = load_jsonl(os.path.join(data_dir, "bi_encoder_train.jsonl"))
    val_data = load_jsonl(os.path.join(data_dir, "bi_encoder_val.jsonl"))
    print(f"  Train:       {len(train_data)} records")
    print(f"  Val:         {len(val_data)} records")

    train_examples = make_bi_encoder_examples(train_data)
    print(f"  Examples:    {len(train_examples)} (with hard negatives)")

    # Information Retrieval evaluator
    queries_dict = {}
    corpus_dict = {}
    relevant_docs = {}
    for i, r in enumerate(val_data):
        qid = f"q_{i}"
        pid = f"p_{i}"
        queries_dict[qid] = r["query"]
        corpus_dict[pid] = r["positive"]
        relevant_docs[qid] = {pid}
        for j, neg in enumerate(r.get("negatives", [])[:5]):
            nid = f"n_{i}_{j}"
            corpus_dict[nid] = neg

    ir_evaluator = evaluation.InformationRetrievalEvaluator(
        queries=queries_dict,
        corpus=corpus_dict,
        relevant_docs=relevant_docs,
        name="val_ir",
        show_progress_bar=False,
    )

    # Load model
    model = SentenceTransformer(base_model)

    # Training
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    total_steps = len(train_dataloader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    eval_steps = max(50, len(train_dataloader) // 2)

    print(f"  Steps/epoch: {len(train_dataloader)}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup:      {warmup_steps}")
    print(f"  Eval every:  {eval_steps} steps")
    print(f"{'='*60}")
    print(f"  Training started...")

    os.makedirs(output_dir, exist_ok=True)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=ir_evaluator,
        epochs=epochs,
        evaluation_steps=eval_steps,
        warmup_steps=warmup_steps,
        output_path=output_dir,
        optimizer_params={"lr": lr},
        save_best_model=True,
        show_progress_bar=True,
    )

    elapsed = time.time() - t_start
    print(f"\n  Bi-encoder training complete in {elapsed/60:.1f} min")
    print(f"  Model saved to: {output_dir}")
    return output_dir


# ───────────────────────────────────────────────────────
# Cross-encoder training
# ───────────────────────────────────────────────────────

def train_cross_encoder(
    data_dir: str = "candidate_training_data",
    output_dir: str = "candidate_v2_models/cross_encoder",
    base_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 2e-5,
    warmup_ratio: float = 0.1,
    seed: int = 42,
):
    set_seed(seed)
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f" CROSS-ENCODER TRAINING")
    print(f"{'='*60}")
    print(f"  Base model:  {base_model}")
    print(f"  Epochs:      {epochs}")
    print(f"  Batch size:  {batch_size}")
    print(f"  LR:          {lr}")
    device = detect_device()

    # Load data
    train_data = load_jsonl(os.path.join(data_dir, "cross_encoder_train.jsonl"))
    val_data = load_jsonl(os.path.join(data_dir, "cross_encoder_val.jsonl"))
    print(f"  Train:       {len(train_data)} records")
    print(f"  Val:         {len(val_data)} records")

    train_examples = make_cross_encoder_examples(train_data)
    val_examples = make_cross_encoder_examples(val_data)

    pos = sum(1 for e in train_examples if e.label > 0.5)
    neg = len(train_examples) - pos
    print(f"  Positives:   {pos}")
    print(f"  Negatives:   {neg}")
    print(f"  Ratio:       1:{neg/max(pos,1):.1f}")

    # Evaluator (handle API change across versions)
    val_labels = [int(e.label) for e in val_examples]
    if _CE_EVAL_NEW_API:
        val_pairs = [[e.texts[0], e.texts[1]] for e in val_examples]
        evaluator = _CEEval(
            sentence_pairs=val_pairs,
            labels=val_labels,
            name="val",
        )
    else:
        val_sentences1 = [e.texts[0] for e in val_examples]
        val_sentences2 = [e.texts[1] for e in val_examples]
        evaluator = _CEEval(
            val_sentences1, val_sentences2, val_labels, name="val",
        )

    # Load model
    model = CrossEncoder(base_model, num_labels=1)

    total_steps = (len(train_examples) // batch_size) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    eval_steps = max(50, len(train_examples) // batch_size // 2)

    print(f"  Total steps: {total_steps}")
    print(f"  Warmup:      {warmup_steps}")
    print(f"  Eval every:  {eval_steps} steps")
    print(f"{'='*60}")
    print(f"  Training started...")

    os.makedirs(output_dir, exist_ok=True)

    model.fit(
        train_dataloader=DataLoader(train_examples, shuffle=True, batch_size=batch_size),
        evaluator=evaluator,
        epochs=epochs,
        evaluation_steps=eval_steps,
        warmup_steps=warmup_steps,
        output_path=output_dir,
        optimizer_params={"lr": lr},
        save_best_model=True,
        show_progress_bar=True,
    )

    elapsed = time.time() - t_start
    print(f"\n  Cross-encoder training complete in {elapsed/60:.1f} min")
    print(f"  Model saved to: {output_dir}")
    return output_dir


# ───────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train bi-encoder + cross-encoder for candidate generation V2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train both models (recommended)
  python train_candidate_generator.py --mode both

  # GPU with 24GB VRAM - use larger batches for speed
  python train_candidate_generator.py --mode both --bi-batch 128 --cross-batch 64

  # Just re-train cross-encoder
  python train_candidate_generator.py --mode cross --cross-epochs 8

  # Custom data directory
  python train_candidate_generator.py --mode both --data-dir my_data/
""",
    )
    parser.add_argument("--mode", choices=["bi", "cross", "both"], default="both",
                        help="Which model(s) to train (default: both)")
    parser.add_argument("--data-dir", default="candidate_training_data",
                        help="Directory with training JSONL files")
    parser.add_argument("--output-dir", default="candidate_v2_models",
                        help="Root output directory for models")
    parser.add_argument("--bi-base", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Base bi-encoder model from HuggingFace")
    parser.add_argument("--cross-base", default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                        help="Base cross-encoder model from HuggingFace")
    parser.add_argument("--bi-epochs", type=int, default=10,
                        help="Bi-encoder training epochs (default: 10)")
    parser.add_argument("--cross-epochs", type=int, default=5,
                        help="Cross-encoder training epochs (default: 5)")
    parser.add_argument("--bi-batch", type=int, default=64,
                        help="Bi-encoder batch size (default: 64)")
    parser.add_argument("--cross-batch", type=int, default=32,
                        help="Cross-encoder batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate (default: 2e-5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"  CANDIDATE GENERATION V2 - MODEL TRAINING")
    print(f"{'#'*60}")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU:             {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:            {torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB")
    print(f"  Mode:            {args.mode}")
    print(f"  Data dir:        {args.data_dir}")
    print(f"  Output dir:      {args.output_dir}")

    t_total = time.time()

    if args.mode in ("bi", "both"):
        train_bi_encoder(
            data_dir=args.data_dir,
            output_dir=os.path.join(args.output_dir, "bi_encoder"),
            base_model=args.bi_base,
            epochs=args.bi_epochs,
            batch_size=args.bi_batch,
            lr=args.lr,
            seed=args.seed,
        )

    if args.mode in ("cross", "both"):
        train_cross_encoder(
            data_dir=args.data_dir,
            output_dir=os.path.join(args.output_dir, "cross_encoder"),
            base_model=args.cross_base,
            epochs=args.cross_epochs,
            batch_size=args.cross_batch,
            lr=args.lr,
            seed=args.seed,
        )

    elapsed = time.time() - t_total
    print(f"\n{'#'*60}")
    print(f"  ALL TRAINING COMPLETE")
    print(f"  Total time: {elapsed/60:.1f} min")
    print(f"  Models saved to: {args.output_dir}/")
    print(f"{'#'*60}")
    print(f"\nNext step: Copy the '{args.output_dir}/' folder back to the project")
    print(f"           and the V2 engine will auto-detect the fine-tuned models.")


if __name__ == "__main__":
    main()
