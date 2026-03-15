# Stage-1 Candidate Selector (Bi-Encoder + Cross-Encoder)

This module trains a two-stage ranking system:

1. **Bi-encoder retrieval** (fast, scalable)
2. **Cross-encoder reranker** (high precision)

Primary quality metric is **Recall@K**.

---

## 1) Training input structure

Training file is JSONL: one line = one `(target query, candidate set, label)` pair.

### Required fields

```json
{
  "query_id": "q_unique_target_id",
  "split": "train|val|test",
  "domain": "optional_domain_name",
  "target": {
    "table": "target_table_name",
    "column": "target_column_name",
    "type": "string|int|date|...",
    "description": "optional natural language description"
  },
  "candidate_set": {
    "id": "unique_candidate_set_id",
    "columns": [
      {
        "table": "source_table",
        "column": "source_column",
        "type": "string|int|...",
        "description": "optional"
      }
    ],
    "join_path": [
      {
        "from": "table_a",
        "to": "table_b",
        "left_cols": ["a_id"],
        "right_cols": ["b_id"]
      }
    ],
    "transform_hint": "optional hint such as concat_space"
  },
  "label": 1
}
```

### Labeling rule
- `label=1`: candidate set can produce target (correct source set)
- `label=0`: candidate set is incorrect

### Dataset quality requirements
- Each `query_id` should have:
  - at least 1 positive
  - multiple negatives (hard negatives preferred)
- Keep split by query (avoid same query in train and val/test)

---

## 2) Files

- `candidate_selector_stage1.py` -> training + inference CLI
- `stage1_training_input_sample.jsonl` -> sample input rows

---

## 3) Install dependencies

```bash
pip install sentence-transformers torch numpy
```

---

## 4) Train

```bash
python3 candidate_selector_stage1.py train \
  --input-jsonl stage1_training_input_sample.jsonl \
  --output-dir artifacts/stage1_candidate_selector \
  --bi-model-name sentence-transformers/all-MiniLM-L6-v2 \
  --cross-model-name cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --bi-epochs 12 \
  --cross-epochs 8 \
  --bi-monitor-k 10 \
  --cross-monitor-k 10 \
  --bi-patience 3 \
  --cross-patience 2 \
  --bi-min-delta 0.0005 \
  --cross-min-delta 0.0005
```

### Early stopping and per-epoch validation

The trainer now performs validation after every epoch and restores the best checkpoint.

- **Bi-encoder monitor:** `recall@K` via `--bi-monitor-k`
- **Cross-encoder monitor:** `pipeline_recall@K` via `--cross-monitor-k`

Useful flags:

- `--bi-patience`, `--cross-patience`
- `--bi-min-delta`, `--cross-min-delta`
- `--bi-min-epochs`, `--cross-min-epochs`
- `--disable-early-stopping` (if you want fixed full-epoch training)

Outputs:
- `artifacts/stage1_candidate_selector/biencoder/`
- `artifacts/stage1_candidate_selector/cross_encoder/`
- `artifacts/stage1_candidate_selector/training_metadata.json`

---

## 5) Predict / rank candidates

Prepare a JSON payload:

```json
{
  "target": {
    "table": "dim_customer",
    "column": "full_name",
    "type": "string",
    "description": "customer full name"
  },
  "candidate_sets": [
    {
      "id": "cand_1",
      "columns": [
        {"table":"src_customers","column":"first_name","type":"string"},
        {"table":"src_customers","column":"last_name","type":"string"}
      ],
      "join_path": [],
      "transform_hint": "concat_space"
    }
  ]
}
```

Run:

```bash
python3 candidate_selector_stage1.py predict \
  --biencoder-path artifacts/stage1_candidate_selector/biencoder \
  --cross-encoder-path artifacts/stage1_candidate_selector/cross_encoder \
  --input-json your_payload.json \
  --retrieval-k 30 \
  --top-k 10
```

---

## 6) What metric to optimize first

Prioritize:
1. `recall@10` on validation
2. then `recall@5`
3. then reranker precision improvements

High recall in stage-1 is more important than strict precision because stage-2 transformation identification can refine candidates further.
