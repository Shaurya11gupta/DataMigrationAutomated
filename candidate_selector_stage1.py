#!/usr/bin/env python3
"""
Stage-1 Candidate Selector
==========================

Pipeline:
  1) Bi-encoder retrieval model
  2) Cross-encoder reranker

Primary metric:
  - Recall@K (query-level)

Input data format:
  JSONL where each row is one (target_query, candidate_set, label) instance.
  See `stage1_training_input_sample.jsonl`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from sentence_transformers.cross_encoder import CrossEncoder
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Missing dependencies. Install with: "
        "pip install sentence-transformers torch numpy"
    ) from exc


# ---------------------------
# Reproducibility
# ---------------------------


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------
# Data model
# ---------------------------


@dataclass
class TargetColumn:
    table: str
    column: str
    type: str
    description: str = ""


@dataclass
class SourceColumn:
    table: str
    column: str
    type: str = "string"
    description: str = ""


@dataclass
class CandidateSet:
    id: str
    columns: List[SourceColumn]
    join_path: List[Dict[str, Any]]
    transform_hint: str = ""


@dataclass
class TrainingRecord:
    query_id: str
    split: str
    target: TargetColumn
    candidate_set: CandidateSet
    label: int
    weight: float = 1.0
    domain: str = "unknown"


# ---------------------------
# Serialization helpers
# ---------------------------


def _safe_get(d: Dict[str, Any], key: str, default: Any = "") -> Any:
    v = d.get(key, default)
    return default if v is None else v


def _tokenize_column_name(name: str) -> str:
    """Convert column_name to 'column name' for better semantic matching."""
    import re as _re
    # Split camelCase
    name = _re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    # Replace underscores, hyphens, dots with spaces
    name = _re.sub(r"[_\-\.]+", " ", name)
    return name.lower().strip()


def _infer_semantic_description(name: str, col_type: str = "") -> str:
    """
    Infer a brief semantic description of a column from its name + type.
    Uses a lightweight approach that complements the SentenceTransformer's
    own semantic understanding.

    The key idea: instead of hardcoded token-to-concept mappings we generate
    natural-language context phrases that the embedding model can leverage.
    """
    import re as _re
    tokens = _tokenize_column_name(name).split()
    if not tokens:
        return ""

    # Build context from the name's natural language reading
    readable = " ".join(tokens)

    # Use type information to add context
    type_hints = {
        "int": "numeric integer value",
        "integer": "numeric integer value",
        "bigint": "numeric large integer",
        "float": "numeric decimal value",
        "double": "numeric decimal value",
        "decimal": "numeric decimal value",
        "number": "numeric value",
        "numeric": "numeric value",
        "string": "text string value",
        "varchar": "text string value",
        "text": "text string value",
        "char": "text character value",
        "date": "calendar date temporal",
        "datetime": "date and time temporal",
        "timestamp": "date time temporal",
        "time": "time temporal",
        "boolean": "true false flag indicator",
        "bool": "true false flag indicator",
    }
    type_context = type_hints.get(col_type.lower(), col_type) if col_type else ""

    return f"{readable} {type_context}".strip()


def target_to_text(t: TargetColumn) -> str:
    """Serialize target column into rich text for embedding models.

    Includes:
      1. Table-qualified column name
      2. Human-readable tokenized name
      3. Inferred semantic description (context for the model)
      4. Type information
      5. Any explicit description
    """
    col_tokens = _tokenize_column_name(t.column)
    sem_desc = _infer_semantic_description(t.column, t.type)
    parts = [
        f"target: {t.table}.{t.column}",
        f"meaning: {col_tokens}",
        f"context: {sem_desc}",
        f"type: {t.type}",
    ]
    if t.description:
        parts.append(f"desc: {t.description}")
    return " | ".join(parts)


def candidate_to_text(c: CandidateSet) -> str:
    """Serialize candidate set into rich text for embedding models.

    Includes:
      1. Table-qualified column names with types
      2. Human-readable tokenized names
      3. Inferred semantic descriptions
      4. Join path information
      5. Transform hint from Stage A
    """
    col_text = []
    for sc in c.columns:
        col_tokens = _tokenize_column_name(sc.column)
        sem_desc = _infer_semantic_description(sc.column, sc.type)
        piece = f"{sc.table}.{sc.column}<{sc.type}> [{col_tokens}] ({sem_desc})"
        if sc.description:
            piece += f" desc:{sc.description}"
        col_text.append(piece)

    join_txt = []
    for step in c.join_path:
        frm = _safe_get(step, "from", "")
        to = _safe_get(step, "to", "")
        lk = ",".join(_safe_get(step, "left_cols", []))
        rk = ",".join(_safe_get(step, "right_cols", []))
        join_txt.append(f"{frm}->{to}:{lk}={rk}")

    parts = [f"columns: {' ; '.join(col_text)}"]
    if join_txt:
        parts.append(f"join: {' ; '.join(join_txt)}")
    if c.transform_hint:
        parts.append(f"transform: {c.transform_hint}")
    return " | ".join(parts)


def record_to_pair_text(rec: TrainingRecord) -> Tuple[str, str]:
    return target_to_text(rec.target), candidate_to_text(rec.candidate_set)


# ---------------------------
# Data loading
# ---------------------------


def _parse_target(obj: Dict[str, Any]) -> TargetColumn:
    return TargetColumn(
        table=str(_safe_get(obj, "table", "")),
        column=str(_safe_get(obj, "column", "")),
        type=str(_safe_get(obj, "type", "string")),
        description=str(_safe_get(obj, "description", "")),
    )


def _parse_source_columns(cols: Sequence[Dict[str, Any]]) -> List[SourceColumn]:
    out = []
    for c in cols:
        out.append(
            SourceColumn(
                table=str(_safe_get(c, "table", "")),
                column=str(_safe_get(c, "column", "")),
                type=str(_safe_get(c, "type", "string")),
                description=str(_safe_get(c, "description", "")),
            )
        )
    return out


def _parse_candidate(obj: Dict[str, Any]) -> CandidateSet:
    return CandidateSet(
        id=str(_safe_get(obj, "id", "")),
        columns=_parse_source_columns(obj.get("columns", [])),
        join_path=list(obj.get("join_path", [])),
        transform_hint=str(_safe_get(obj, "transform_hint", "")),
    )


def load_training_records(path: str) -> List[TrainingRecord]:
    rows = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)

        rec = TrainingRecord(
            query_id=str(_safe_get(obj, "query_id", "")),
            split=str(_safe_get(obj, "split", "train")).lower(),
            target=_parse_target(obj.get("target", {})),
            candidate_set=_parse_candidate(obj.get("candidate_set", {})),
            label=int(_safe_get(obj, "label", 0)),
            weight=float(_safe_get(obj, "weight", 1.0)),
            domain=str(_safe_get(obj, "domain", "unknown")),
        )
        if not rec.query_id:
            raise ValueError(f"Missing query_id at line {i}")
        if rec.candidate_set.id == "":
            raise ValueError(f"Missing candidate_set.id at line {i}")
        if rec.split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split '{rec.split}' at line {i}")
        if rec.label not in {0, 1}:
            raise ValueError(f"Invalid label '{rec.label}' at line {i}")
        rows.append(rec)

    return rows


def split_records(records: Sequence[TrainingRecord]) -> Dict[str, List[TrainingRecord]]:
    buckets: Dict[str, List[TrainingRecord]] = defaultdict(list)
    for r in records:
        buckets[r.split].append(r)
    return buckets


def group_by_query(records: Sequence[TrainingRecord]) -> Dict[str, List[TrainingRecord]]:
    g: Dict[str, List[TrainingRecord]] = defaultdict(list)
    for r in records:
        g[r.query_id].append(r)
    return g


# ---------------------------
# Metrics
# ---------------------------


def recall_at_k(
    query_to_ranked_ids: Dict[str, List[str]],
    query_to_positive_ids: Dict[str, set],
    k: int,
) -> float:
    if not query_to_positive_ids:
        return 0.0
    hits = 0
    total = 0
    for qid, pos_ids in query_to_positive_ids.items():
        if not pos_ids:
            continue
        total += 1
        ranked = query_to_ranked_ids.get(qid, [])[:k]
        if any(cid in pos_ids for cid in ranked):
            hits += 1
    return hits / max(1, total)


# ---------------------------
# Bi-encoder
# ---------------------------


def build_biencoder_examples(records: Sequence[TrainingRecord]) -> List[InputExample]:
    examples = []
    for r in records:
        q, c = record_to_pair_text(r)
        examples.append(InputExample(texts=[q, c], label=float(r.label)))
    return examples


def build_mnrl_examples(records: Sequence[TrainingRecord]) -> List[InputExample]:
    """Build examples for MultipleNegativesRankingLoss.

    Only includes positive pairs: (query, positive_candidate).
    In-batch negatives provide implicit hard negatives.
    """
    examples = []
    for r in records:
        if r.label == 1:
            q, c = record_to_pair_text(r)
            examples.append(InputExample(texts=[q, c]))
    return examples


def train_biencoder(
    train_records: Sequence[TrainingRecord],
    model_name: str,
    output_dir: str,
    epochs: int = 2,
    batch_size: int = 32,
    lr: float = 2e-5,
    warmup_ratio: float = 0.1,
) -> SentenceTransformer:
    model = SentenceTransformer(model_name)

    # Use MultipleNegativesRankingLoss for better retrieval performance.
    # Falls back to CosineSimilarityLoss if not enough positives.
    mnrl_examples = build_mnrl_examples(train_records)
    if len(mnrl_examples) >= 8:
        train_examples = mnrl_examples
        train_loader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        train_loss = losses.MultipleNegativesRankingLoss(model)
    else:
        train_examples = build_biencoder_examples(train_records)
        train_loader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        train_loss = losses.CosineSimilarityLoss(model)

    warmup_steps = int(len(train_loader) * epochs * warmup_ratio)
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        show_progress_bar=True,
    )

    os.makedirs(output_dir, exist_ok=True)
    model.save(output_dir)
    return model


def _ensure_metric(metrics: Dict[str, float], metric_name: str) -> float:
    if metric_name in metrics:
        return float(metrics[metric_name])
    # fallback for accidental naming mismatch
    if metric_name.startswith("pipeline_"):
        alt = metric_name.replace("pipeline_", "")
        if alt in metrics:
            return float(metrics[alt])
    if not metrics:
        return 0.0
    # deterministic fallback
    key = sorted(metrics.keys())[-1]
    return float(metrics[key])


def _save_best_model_checkpoint(model: SentenceTransformer, best_dir: Path) -> None:
    if best_dir.exists():
        shutil.rmtree(best_dir)
    best_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(best_dir))


def train_biencoder_with_early_stopping(
    train_records: Sequence[TrainingRecord],
    val_records: Sequence[TrainingRecord],
    model_name: str,
    output_dir: str,
    monitor_metric: str = "recall@10",
    epochs: int = 8,
    batch_size: int = 32,
    lr: float = 2e-5,
    warmup_ratio: float = 0.1,
    early_stopping: bool = True,
    patience: int = 3,
    min_delta: float = 5e-4,
    min_epochs: int = 2,
    eval_ks: Sequence[int] = (1, 3, 5, 10),
) -> Tuple[SentenceTransformer, List[Dict[str, Any]], Dict[str, Any]]:
    model = SentenceTransformer(model_name)

    # Use MNRL for better retrieval; fall back to cosine if not enough positives
    mnrl_examples = build_mnrl_examples(train_records)
    if len(mnrl_examples) >= 8:
        train_examples = mnrl_examples
        train_loader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        train_loss = losses.MultipleNegativesRankingLoss(model)
    else:
        train_examples = build_biencoder_examples(train_records)
        train_loader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        train_loss = losses.CosineSimilarityLoss(model)

    total_warmup_steps = int(len(train_loader) * max(1, epochs) * warmup_ratio)
    best_dir = Path(output_dir) / "_best_checkpoint"
    history: List[Dict[str, Any]] = []
    best_score = -1.0
    best_epoch = 0
    best_metrics: Dict[str, float] = {}
    stale_epochs = 0

    for epoch_idx in range(1, epochs + 1):
        print(f"\n[Bi-encoder] Epoch {epoch_idx}/{epochs}")
        model.fit(
            train_objectives=[(train_loader, train_loss)],
            epochs=1,
            warmup_steps=total_warmup_steps if epoch_idx == 1 else 0,
            optimizer_params={"lr": lr},
            show_progress_bar=True,
        )

        val_metrics = evaluate_biencoder_recall(model, val_records, ks=eval_ks)
        score = _ensure_metric(val_metrics, monitor_metric)
        improved = score > (best_score + min_delta)

        if improved:
            best_score = score
            best_epoch = epoch_idx
            best_metrics = dict(val_metrics)
            stale_epochs = 0
            _save_best_model_checkpoint(model, best_dir)
        else:
            stale_epochs += 1

        hist_row = {
            "epoch": epoch_idx,
            "val_metrics": val_metrics,
            "monitor_metric": monitor_metric,
            "monitor_value": round(score, 6),
            "improved": improved,
            "stale_epochs": stale_epochs,
        }
        history.append(hist_row)
        print(f"[Bi-encoder] val metrics: {val_metrics}")
        print(
            f"[Bi-encoder] monitor {monitor_metric}={score:.6f} | "
            f"{'improved' if improved else 'no_improve'} | stale={stale_epochs}"
        )

        if early_stopping and epoch_idx >= min_epochs and stale_epochs >= patience:
            print(
                f"[Bi-encoder] Early stopping triggered at epoch {epoch_idx} "
                f"(patience={patience}, min_delta={min_delta})"
            )
            break

    if best_epoch > 0 and best_dir.exists():
        model = SentenceTransformer(str(best_dir))
        print(f"[Bi-encoder] Restored best checkpoint from epoch {best_epoch}")

    os.makedirs(output_dir, exist_ok=True)
    model.save(output_dir)

    summary = {
        "best_epoch": best_epoch,
        "best_monitor_metric": monitor_metric,
        "best_monitor_value": round(best_score, 6) if best_score >= 0 else 0.0,
        "best_val_metrics": best_metrics,
        "epochs_ran": len(history),
        "early_stopping_enabled": early_stopping,
        "patience": patience,
        "min_delta": min_delta,
        "min_epochs": min_epochs,
    }
    return model, history, summary


def _embed_biencoder(
    model: SentenceTransformer,
    records: Sequence[TrainingRecord],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
    """
    Returns:
      query_embeddings: query_id -> vector
      cand_embeddings: candidate_id -> vector
      candidate_meta: candidate_id -> {"query_id":..., "record":...}
    """
    query_text_by_id: Dict[str, str] = {}
    candidate_text_by_id: Dict[str, str] = {}
    candidate_meta: Dict[str, Dict[str, Any]] = {}

    for r in records:
        q_text, c_text = record_to_pair_text(r)
        query_text_by_id[r.query_id] = q_text
        candidate_text_by_id[r.candidate_set.id] = c_text
        candidate_meta[r.candidate_set.id] = {"query_id": r.query_id, "record": r}

    qids = sorted(query_text_by_id.keys())
    cids = sorted(candidate_text_by_id.keys())
    q_texts = [query_text_by_id[q] for q in qids]
    c_texts = [candidate_text_by_id[c] for c in cids]

    q_emb = model.encode(q_texts, normalize_embeddings=True, show_progress_bar=False)
    c_emb = model.encode(c_texts, normalize_embeddings=True, show_progress_bar=False)

    query_embeddings = {qids[i]: q_emb[i] for i in range(len(qids))}
    cand_embeddings = {cids[i]: c_emb[i] for i in range(len(cids))}
    return query_embeddings, cand_embeddings, candidate_meta


def _rank_candidates_for_query(
    query_vec: np.ndarray,
    candidate_ids: Sequence[str],
    cand_embeddings: Dict[str, np.ndarray],
    top_k: int,
) -> List[str]:
    if not candidate_ids:
        return []
    mat = np.vstack([cand_embeddings[cid] for cid in candidate_ids])
    scores = mat @ query_vec
    idx = np.argsort(-scores)[:top_k]
    return [candidate_ids[i] for i in idx.tolist()]


def evaluate_biencoder_recall(
    model: SentenceTransformer,
    records: Sequence[TrainingRecord],
    ks: Sequence[int] = (1, 3, 5, 10),
) -> Dict[str, float]:
    query_embeddings, cand_embeddings, _ = _embed_biencoder(model, records)
    by_query = group_by_query(records)

    query_to_ranked: Dict[str, List[str]] = {}
    query_to_pos: Dict[str, set] = {}

    for qid, recs in by_query.items():
        candidate_ids = [r.candidate_set.id for r in recs]
        ranked = _rank_candidates_for_query(
            query_vec=query_embeddings[qid],
            candidate_ids=candidate_ids,
            cand_embeddings=cand_embeddings,
            top_k=max(ks),
        )
        query_to_ranked[qid] = ranked
        query_to_pos[qid] = {r.candidate_set.id for r in recs if r.label == 1}

    return {f"recall@{k}": round(recall_at_k(query_to_ranked, query_to_pos, k), 4) for k in ks}


# ---------------------------
# Hard-negative mining
# ---------------------------


def build_reranker_training_rows(
    biencoder: SentenceTransformer,
    records: Sequence[TrainingRecord],
    top_n: int = 20,
    max_neg_per_query: int = 8,
) -> List[Tuple[str, str, int]]:
    """
    Build cross-encoder rows from:
      - all positives
      - hard negatives: top bi-encoder retrieved non-positives
    """
    query_embeddings, cand_embeddings, _ = _embed_biencoder(biencoder, records)
    by_query = group_by_query(records)
    rows: List[Tuple[str, str, int]] = []

    for qid, recs in by_query.items():
        query_text = target_to_text(recs[0].target)
        pos_ids = {r.candidate_set.id for r in recs if r.label == 1}
        all_ids = [r.candidate_set.id for r in recs]
        ranked = _rank_candidates_for_query(query_embeddings[qid], all_ids, cand_embeddings, top_n)

        # Add all positives
        by_id = {r.candidate_set.id: r for r in recs}
        for pid in pos_ids:
            rows.append((query_text, candidate_to_text(by_id[pid].candidate_set), 1))

        # Add hard negatives
        neg_count = 0
        for cid in ranked:
            if cid in pos_ids:
                continue
            rows.append((query_text, candidate_to_text(by_id[cid].candidate_set), 0))
            neg_count += 1
            if neg_count >= max_neg_per_query:
                break

    return rows


# ---------------------------
# Cross-encoder reranker
# ---------------------------


def train_cross_encoder(
    train_rows: Sequence[Tuple[str, str, int]],
    model_name: str,
    output_dir: str,
    epochs: int = 2,
    batch_size: int = 16,
    lr: float = 1e-5,
    warmup_ratio: float = 0.1,
) -> CrossEncoder:
    model = CrossEncoder(model_name, num_labels=1)
    examples = [InputExample(texts=[q, c], label=float(y)) for q, c, y in train_rows]
    train_loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    warmup_steps = int(len(train_loader) * epochs * warmup_ratio)

    model.fit(
        train_dataloader=train_loader,
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        show_progress_bar=True,
    )

    os.makedirs(output_dir, exist_ok=True)
    model.save(output_dir)
    return model


def _save_best_cross_checkpoint(model: CrossEncoder, best_dir: Path) -> None:
    if best_dir.exists():
        shutil.rmtree(best_dir)
    best_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(best_dir))


def train_cross_encoder_with_early_stopping(
    train_rows: Sequence[Tuple[str, str, int]],
    model_name: str,
    output_dir: str,
    eval_fn: Callable[[CrossEncoder], Dict[str, float]],
    monitor_metric: str = "pipeline_recall@10",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-5,
    warmup_ratio: float = 0.1,
    early_stopping: bool = True,
    patience: int = 2,
    min_delta: float = 5e-4,
    min_epochs: int = 2,
) -> Tuple[CrossEncoder, List[Dict[str, Any]], Dict[str, Any]]:
    model = CrossEncoder(model_name, num_labels=1)
    examples = [InputExample(texts=[q, c], label=float(y)) for q, c, y in train_rows]
    train_loader = DataLoader(examples, shuffle=True, batch_size=batch_size)

    total_warmup_steps = int(len(train_loader) * max(1, epochs) * warmup_ratio)
    best_dir = Path(output_dir) / "_best_checkpoint"
    history: List[Dict[str, Any]] = []
    best_score = -1.0
    best_epoch = 0
    best_metrics: Dict[str, float] = {}
    stale_epochs = 0

    for epoch_idx in range(1, epochs + 1):
        print(f"\n[Cross-encoder] Epoch {epoch_idx}/{epochs}")
        model.fit(
            train_dataloader=train_loader,
            epochs=1,
            warmup_steps=total_warmup_steps if epoch_idx == 1 else 0,
            optimizer_params={"lr": lr},
            show_progress_bar=True,
        )

        val_metrics = eval_fn(model)
        score = _ensure_metric(val_metrics, monitor_metric)
        improved = score > (best_score + min_delta)

        if improved:
            best_score = score
            best_epoch = epoch_idx
            best_metrics = dict(val_metrics)
            stale_epochs = 0
            _save_best_cross_checkpoint(model, best_dir)
        else:
            stale_epochs += 1

        hist_row = {
            "epoch": epoch_idx,
            "val_metrics": val_metrics,
            "monitor_metric": monitor_metric,
            "monitor_value": round(score, 6),
            "improved": improved,
            "stale_epochs": stale_epochs,
        }
        history.append(hist_row)
        print(f"[Cross-encoder] val metrics: {val_metrics}")
        print(
            f"[Cross-encoder] monitor {monitor_metric}={score:.6f} | "
            f"{'improved' if improved else 'no_improve'} | stale={stale_epochs}"
        )

        if early_stopping and epoch_idx >= min_epochs and stale_epochs >= patience:
            print(
                f"[Cross-encoder] Early stopping triggered at epoch {epoch_idx} "
                f"(patience={patience}, min_delta={min_delta})"
            )
            break

    if best_epoch > 0 and best_dir.exists():
        model = CrossEncoder(str(best_dir), num_labels=1)
        print(f"[Cross-encoder] Restored best checkpoint from epoch {best_epoch}")

    os.makedirs(output_dir, exist_ok=True)
    model.save(output_dir)

    summary = {
        "best_epoch": best_epoch,
        "best_monitor_metric": monitor_metric,
        "best_monitor_value": round(best_score, 6) if best_score >= 0 else 0.0,
        "best_val_metrics": best_metrics,
        "epochs_ran": len(history),
        "early_stopping_enabled": early_stopping,
        "patience": patience,
        "min_delta": min_delta,
        "min_epochs": min_epochs,
    }
    return model, history, summary


def evaluate_pipeline_recall(
    biencoder: SentenceTransformer,
    cross_encoder: CrossEncoder,
    records: Sequence[TrainingRecord],
    retrieval_k: int = 30,
    rerank_k: int = 10,
) -> Dict[str, float]:
    query_embeddings, cand_embeddings, _ = _embed_biencoder(biencoder, records)
    by_query = group_by_query(records)

    query_to_ranked: Dict[str, List[str]] = {}
    query_to_pos: Dict[str, set] = {}

    for qid, recs in by_query.items():
        query_text = target_to_text(recs[0].target)
        by_id = {r.candidate_set.id: r for r in recs}
        all_ids = [r.candidate_set.id for r in recs]

        stage1_ids = _rank_candidates_for_query(
            query_embeddings[qid], all_ids, cand_embeddings, retrieval_k
        )
        pairs = [(query_text, candidate_to_text(by_id[cid].candidate_set)) for cid in stage1_ids]
        if not pairs:
            query_to_ranked[qid] = []
            query_to_pos[qid] = {r.candidate_set.id for r in recs if r.label == 1}
            continue

        scores = cross_encoder.predict(pairs, batch_size=64)
        order = np.argsort(-np.array(scores))[:rerank_k]
        ranked = [stage1_ids[i] for i in order.tolist()]

        query_to_ranked[qid] = ranked
        query_to_pos[qid] = {r.candidate_set.id for r in recs if r.label == 1}

    return {
        "pipeline_recall@1": round(recall_at_k(query_to_ranked, query_to_pos, 1), 4),
        "pipeline_recall@3": round(recall_at_k(query_to_ranked, query_to_pos, 3), 4),
        "pipeline_recall@5": round(recall_at_k(query_to_ranked, query_to_pos, 5), 4),
        "pipeline_recall@10": round(recall_at_k(query_to_ranked, query_to_pos, 10), 4),
    }


# ---------------------------
# Inference API
# ---------------------------


class CandidateSelectorStage1:
    def __init__(self, biencoder_path: str, cross_encoder_path: str,
                 bi_weight: float = 0.25, cross_weight: float = 0.75):
        self.biencoder = SentenceTransformer(biencoder_path)
        self.cross_encoder = CrossEncoder(cross_encoder_path, num_labels=1)
        self.bi_weight = bi_weight
        self.cross_weight = cross_weight

    def rank(
        self,
        target: TargetColumn,
        candidate_sets: Sequence[CandidateSet],
        retrieval_k: int = 50,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        if not candidate_sets:
            return []

        q_text = target_to_text(target)
        q_emb = self.biencoder.encode([q_text], normalize_embeddings=True, show_progress_bar=False)[0]

        cids = [c.id for c in candidate_sets]
        ctexts = [candidate_to_text(c) for c in candidate_sets]
        cemb = self.biencoder.encode(ctexts, normalize_embeddings=True, show_progress_bar=False)
        sims = np.array(cemb) @ q_emb

        stage1_order = np.argsort(-sims)[: min(retrieval_k, len(cids))]
        stage1_ids = [cids[i] for i in stage1_order.tolist()]
        stage1_texts = [ctexts[i] for i in stage1_order.tolist()]
        stage1_sims = [float(sims[i]) for i in stage1_order.tolist()]

        pairs = [(q_text, txt) for txt in stage1_texts]
        ce_scores = self.cross_encoder.predict(pairs, batch_size=64)

        # Combine bi-encoder and cross-encoder scores for final ranking.
        # Normalize cross-encoder scores to [0, 1] range for combination.
        ce_arr = np.array(ce_scores, dtype=float)
        ce_min, ce_max = float(ce_arr.min()), float(ce_arr.max())
        ce_range = max(ce_max - ce_min, 1e-8)
        ce_norm = (ce_arr - ce_min) / ce_range

        bi_arr = np.array(stage1_sims, dtype=float)
        bi_min, bi_max = float(bi_arr.min()), float(bi_arr.max())
        bi_range = max(bi_max - bi_min, 1e-8)
        bi_norm = (bi_arr - bi_min) / bi_range

        combined = self.cross_weight * ce_norm + self.bi_weight * bi_norm

        rerank = np.argsort(-combined)[: min(top_k, len(stage1_ids))]
        out = []
        for rank_idx, i in enumerate(rerank.tolist(), start=1):
            out.append(
                {
                    "rank": rank_idx,
                    "candidate_id": stage1_ids[i],
                    "cross_encoder_score": float(ce_scores[i]),
                    "bi_encoder_similarity": float(stage1_sims[i]),
                    "combined_score": float(combined[i]),
                }
            )
        return out


# ---------------------------
# CLI
# ---------------------------


def run_train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    records = load_training_records(args.input_jsonl)
    splits = split_records(records)
    train_records = splits.get("train", [])
    val_records = splits.get("val", [])
    test_records = splits.get("test", [])

    if not train_records:
        raise ValueError("No train split found in input data.")
    if not val_records:
        print("WARN: no val split found; using train split for quick validation metrics.")
        val_records = train_records

    out_dir = Path(args.output_dir)
    biencoder_dir = out_dir / "biencoder"
    cross_dir = out_dir / "cross_encoder"
    os.makedirs(out_dir, exist_ok=True)

    bi_monitor_metric = f"recall@{args.bi_monitor_k}"
    cross_monitor_metric = f"pipeline_recall@{args.cross_monitor_k}"
    early_stopping_enabled = not args.disable_early_stopping

    print("Training bi-encoder (epoch-wise with validation)...")
    biencoder, bi_epoch_history, bi_train_summary = train_biencoder_with_early_stopping(
        train_records=train_records,
        val_records=val_records,
        model_name=args.bi_model_name,
        output_dir=str(biencoder_dir),
        monitor_metric=bi_monitor_metric,
        epochs=args.bi_epochs,
        batch_size=args.bi_batch_size,
        lr=args.bi_lr,
        early_stopping=early_stopping_enabled,
        patience=args.bi_patience,
        min_delta=args.bi_min_delta,
        min_epochs=args.bi_min_epochs,
        eval_ks=(1, 3, 5, max(10, args.bi_monitor_k)),
    )

    print("Evaluating bi-encoder...")
    bi_val = evaluate_biencoder_recall(biencoder, val_records)
    bi_test = evaluate_biencoder_recall(biencoder, test_records) if test_records else {}
    print("Bi-encoder val:", bi_val)
    if bi_test:
        print("Bi-encoder test:", bi_test)

    print("Mining hard negatives for cross-encoder...")
    rerank_train_rows = build_reranker_training_rows(
        biencoder=biencoder,
        records=train_records,
        top_n=args.hard_negative_top_n,
        max_neg_per_query=args.max_neg_per_query,
    )

    print("Training cross-encoder (epoch-wise with validation)...")

    def _cross_val_eval(model: CrossEncoder) -> Dict[str, float]:
        return evaluate_pipeline_recall(
            biencoder=biencoder,
            cross_encoder=model,
            records=val_records,
            retrieval_k=args.retrieval_k,
            rerank_k=args.rerank_k,
        )

    cross, cross_epoch_history, cross_train_summary = train_cross_encoder_with_early_stopping(
        train_rows=rerank_train_rows,
        model_name=args.cross_model_name,
        output_dir=str(cross_dir),
        eval_fn=_cross_val_eval,
        monitor_metric=cross_monitor_metric,
        epochs=args.cross_epochs,
        batch_size=args.cross_batch_size,
        lr=args.cross_lr,
        early_stopping=early_stopping_enabled,
        patience=args.cross_patience,
        min_delta=args.cross_min_delta,
        min_epochs=args.cross_min_epochs,
    )

    print("Evaluating full retrieval+rereank pipeline...")
    pipeline_val = evaluate_pipeline_recall(
        biencoder=biencoder,
        cross_encoder=cross,
        records=val_records,
        retrieval_k=args.retrieval_k,
        rerank_k=args.rerank_k,
    )
    pipeline_test = (
        evaluate_pipeline_recall(
            biencoder=biencoder,
            cross_encoder=cross,
            records=test_records,
            retrieval_k=args.retrieval_k,
            rerank_k=args.rerank_k,
        )
        if test_records
        else {}
    )
    print("Pipeline val:", pipeline_val)
    if pipeline_test:
        print("Pipeline test:", pipeline_test)

    meta = {
        "input_jsonl": args.input_jsonl,
        "bi_model_name": args.bi_model_name,
        "cross_model_name": args.cross_model_name,
        "bi_epochs": args.bi_epochs,
        "cross_epochs": args.cross_epochs,
        "early_stopping_enabled": early_stopping_enabled,
        "bi_monitor_metric": bi_monitor_metric,
        "cross_monitor_metric": cross_monitor_metric,
        "bi_train_summary": bi_train_summary,
        "cross_train_summary": cross_train_summary,
        "bi_epoch_history": bi_epoch_history,
        "cross_epoch_history": cross_epoch_history,
        "hard_negative_top_n": args.hard_negative_top_n,
        "max_neg_per_query": args.max_neg_per_query,
        "retrieval_k": args.retrieval_k,
        "rerank_k": args.rerank_k,
        "metrics": {
            "bi_val": bi_val,
            "bi_test": bi_test,
            "pipeline_val": pipeline_val,
            "pipeline_test": pipeline_test,
        },
    }
    (out_dir / "training_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved models and metadata under: {out_dir}")


def run_predict(args: argparse.Namespace) -> None:
    selector = CandidateSelectorStage1(
        biencoder_path=args.biencoder_path,
        cross_encoder_path=args.cross_encoder_path,
    )
    payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))

    target = _parse_target(payload["target"])
    cands = [_parse_candidate(x) for x in payload.get("candidate_sets", [])]
    ranked = selector.rank(
        target=target,
        candidate_sets=cands,
        retrieval_k=args.retrieval_k,
        top_k=args.top_k,
    )

    out = {"target": payload["target"], "ranked_candidates": ranked}
    print(json.dumps(out, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage-1 candidate selector training/inference")
    sub = p.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="Train bi-encoder + cross-encoder")
    tr.add_argument("--input-jsonl", required=True)
    tr.add_argument("--output-dir", default="artifacts/stage1_candidate_selector")
    tr.add_argument("--seed", type=int, default=42)

    tr.add_argument("--bi-model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    tr.add_argument("--bi-epochs", type=int, default=8)
    tr.add_argument("--bi-batch-size", type=int, default=32)
    tr.add_argument("--bi-lr", type=float, default=2e-5)
    tr.add_argument("--bi-monitor-k", type=int, default=10)
    tr.add_argument("--bi-patience", type=int, default=3)
    tr.add_argument("--bi-min-delta", type=float, default=0.0005)
    tr.add_argument("--bi-min-epochs", type=int, default=2)

    tr.add_argument("--cross-model-name", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    tr.add_argument("--cross-epochs", type=int, default=5)
    tr.add_argument("--cross-batch-size", type=int, default=16)
    tr.add_argument("--cross-lr", type=float, default=1e-5)
    tr.add_argument("--cross-monitor-k", type=int, default=10)
    tr.add_argument("--cross-patience", type=int, default=2)
    tr.add_argument("--cross-min-delta", type=float, default=0.0005)
    tr.add_argument("--cross-min-epochs", type=int, default=2)
    tr.add_argument("--disable-early-stopping", action="store_true")

    tr.add_argument("--hard-negative-top-n", type=int, default=20)
    tr.add_argument("--max-neg-per-query", type=int, default=8)
    tr.add_argument("--retrieval-k", type=int, default=30)
    tr.add_argument("--rerank-k", type=int, default=10)
    tr.set_defaults(func=run_train)

    pr = sub.add_parser("predict", help="Inference ranking with trained models")
    pr.add_argument("--biencoder-path", required=True)
    pr.add_argument("--cross-encoder-path", required=True)
    pr.add_argument("--input-json", required=True)
    pr.add_argument("--retrieval-k", type=int, default=30)
    pr.add_argument("--top-k", type=int, default=10)
    pr.set_defaults(func=run_predict)

    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
