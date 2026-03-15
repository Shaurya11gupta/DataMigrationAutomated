#!/usr/bin/env python3
"""
Merge transformation JSONL files into a single unified dataset for the classifier.

Usage:
  python build_unified_training_data.py

Reads (if present):
  - string_transformations_training_data.jsonl
  - numeric_date_boolean_transformations_training_data.jsonl
  - full_transformation_training_data.jsonl  (comprehensive A–L categories)

Writes:
  - unified_transformation_training_data.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path


SEED = 20260211
BASE = Path(__file__).resolve().parent
STRING_PATH = BASE / "string_transformations_training_data.jsonl"
NUMERIC_DATE_BOOL_PATH = BASE / "numeric_date_boolean_transformations_training_data.jsonl"
FULL_PATH = BASE / "full_transformation_training_data.jsonl"
OUTPUT_PATH = BASE / "unified_transformation_training_data.jsonl"

# Normalize legacy type names to the canonical names from the full generator
TYPE_ALIAS = {
    "format_date": "format_datetime",
    "parse_date": "parse_datetime",
    "date_diff_days": "date_difference",
    "add_days": "add_interval",
    "date_trunc_year": "truncate_to_period",
    "date_trunc_week": "truncate_to_period",
    "date_trunc_month": "truncate_to_period",
    "date_trunc_day": "truncate_to_period",
    "extract_month": "extract_quarter",   # similar temporal extraction
    "cast_to_int": "type_cast",
    "cast_to_decimal": "type_cast",
    "in_list": "in_list_check",
    "between": "range_check",
    "is_null": "null_presence_flag",
    "is_not_null": "null_presence_flag",
    "eq": "equality_check",
    "gt": "threshold_flag",
    "gte": "threshold_flag",
    "lt": "threshold_flag",
    "lte": "threshold_flag",
    "and_op": "case_when_multi",
    "or_op": "case_when_multi",
    "not_op": "case_when_multi",
}


def load_jsonl(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def main() -> None:
    all_records = []

    if STRING_PATH.exists():
        string_records = load_jsonl(STRING_PATH)
        all_records.extend(string_records)
        print(f"Loaded {len(string_records)} string transformation records")
    else:
        print(f"Warning: {STRING_PATH} not found, skipping")

    if NUMERIC_DATE_BOOL_PATH.exists():
        other_records = load_jsonl(NUMERIC_DATE_BOOL_PATH)
        all_records.extend(other_records)
        print(f"Loaded {len(other_records)} numeric/date/boolean records")
    else:
        print(f"Warning: {NUMERIC_DATE_BOOL_PATH} not found, skipping")

    if FULL_PATH.exists():
        full_records = load_jsonl(FULL_PATH)
        all_records.extend(full_records)
        print(f"Loaded {len(full_records)} full (A–L) transformation records")
    else:
        print(f"Warning: {FULL_PATH} not found, skipping")

    if not all_records:
        raise SystemExit("No records to write. Ensure at least one source JSONL exists.")

    # Normalize legacy type names to canonical names
    normalized = 0
    for r in all_records:
        tt = r.get("transform_type", "")
        if tt in TYPE_ALIAS:
            r["transform_type"] = TYPE_ALIAS[tt]
            normalized += 1
    if normalized:
        print(f"Normalized {normalized} legacy type names to canonical types")

    rng = random.Random(SEED)
    rng.shuffle(all_records)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")

    labels = sum(1 for r in all_records if r.get("label") == 1), sum(1 for r in all_records if r.get("label") == 0)
    print(f"Wrote {len(all_records)} records to {OUTPUT_PATH}")
    print(f"Labels: positive={labels[0]}, negative={labels[1]}")


if __name__ == "__main__":
    main()
