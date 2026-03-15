#!/usr/bin/env python3
"""
Generate training data for numeric, date, and boolean transformation types.

Output schema matches string_transformations_training_data.jsonl:
  - source_columns: [{name, type, is_pk, entropy}, ...]
  - target_column: {name, type, entropy}
  - transform_name, transform_type, label (1=positive, 0=negative)

Produces rich, diverse examples for model generalization while preserving
the same structure and quality as the string transformation dataset.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


SEED = 20260211
OUTPUT_PATH = Path(__file__).resolve().parent / "numeric_date_boolean_transformations_training_data.jsonl"

# Per-type positive/negative counts for balance and generalization
NUMERIC_POSITIVE = 320
NUMERIC_NEGATIVE = 120
DATE_POSITIVE = 280
DATE_NEGATIVE = 100
BOOLEAN_POSITIVE = 280
BOOLEAN_NEGATIVE = 100

DOMAIN_DB_PAIRS = [
    ("it", "adventureworks"),
    ("retail", "northwind"),
    ("film", "sakila"),
    ("media", "chinook"),
    ("analytics", "tpch"),
    ("logistics", "wideworldimporters"),
    ("healthcare", "mimic_iv"),
    ("finance", "adventureworks"),
    ("telecom", "tpch"),
    ("hr", "northwind"),
]

DOMAIN_ENTITIES = {
    "it": ["user", "server", "incident", "deployment", "ticket"],
    "retail": ["customer", "order", "product", "supplier", "invoice"],
    "film": ["actor", "film", "rental", "store", "staff"],
    "media": ["artist", "track", "playlist", "invoice"],
    "analytics": ["customer", "orders", "lineitem", "supplier", "region"],
    "logistics": ["warehouse", "parcel", "container", "manifest", "driver"],
    "healthcare": ["patient", "admission", "diagnosis", "procedure", "claim"],
    "finance": ["account", "transaction", "ledger", "card", "payment"],
    "telecom": ["subscriber", "msisdn", "session", "plan", "billing"],
    "hr": ["employee", "department", "payroll", "benefit", "attendance"],
}

# Unrelated targets for negative examples (semantic mismatch)
NUMERIC_UNRELATED = ["full_name", "email_domain", "country_code", "status_text", "invoice_id_text"]
DATE_UNRELATED = ["amount_local", "quantity", "risk_score", "is_active", "postal_code"]
BOOLEAN_UNRELATED = ["first_name", "amount_local", "signup_date", "age", "department_code"]


def rand_entropy(rng: random.Random, low: float, high: float) -> float:
    return round(rng.uniform(low, high), 2)


def infer_pk(rng: random.Random, name: str) -> bool:
    if name.endswith("_id") or name.endswith("_key"):
        return rng.random() < 0.6
    return rng.random() < 0.08


def make_column(rng: random.Random, name: str, col_type: str, low: float, high: float) -> dict:
    return {
        "name": name,
        "type": col_type,
        "is_pk": infer_pk(rng, name),
        "entropy": rand_entropy(rng, low, high),
    }


def pick_entity(rng: random.Random, domain: str) -> str:
    return rng.choice(DOMAIN_ENTITIES.get(domain, ["entity"]))


def make_transform_name(db_ref: str, domain: str, base_name: str, label: int, serial: int) -> str:
    polarity = "pos" if label == 1 else "neg"
    return f"{db_ref}_{domain}_{base_name}_{polarity}_{serial:04d}"


# ---------- Numeric (int, decimal) ----------

NUMERIC_POSITIVE_PATTERNS = [
    # (source_col_specs, target_name, target_type, base_name)
    (["amount_local", "tax_amount"], "total_amount", "decimal", "add_amount_plus_tax"),
    (["gross_amount", "discount_amount"], "net_amount", "decimal", "subtract_discount_from_gross"),
    (["unit_price", "quantity"], "line_total", "decimal", "multiply_price_quantity"),
    (["amount_local", "fx_rate"], "amount_usd", "decimal", "multiply_amount_fx"),
    (["revenue_amount", "cost_amount"], "margin_amount", "decimal", "subtract_cost_from_revenue"),
    (["balance_local", "credit_limit"], "utilization_ratio", "decimal", "divide_balance_by_limit"),
    (["risk_score"], "risk_score_rounded", "decimal", "round_risk_score"),
    (["risk_score"], "risk_score_floor", "int", "floor_risk_score"),
    (["risk_score"], "risk_score_ceil", "int", "ceil_risk_score"),
    (["amount_local"], "amount_abs", "decimal", "abs_amount"),
    (["amount_local", "tax_amount"], "amount_plus_tax", "decimal", "add_amount_tax"),
    (["quantity", "used_units"], "remaining_units", "int", "subtract_used_from_quantity"),
    (["total_units", "quantity"], "delta_units", "int", "subtract_quantity_from_total"),
    (["age"], "age_years_int", "int", "cast_age_to_int"),
    (["amount_text"], "amount_decimal", "decimal", "cast_to_decimal"),
    (["balance_local", "credit_limit"], "headroom", "decimal", "subtract_balance_from_limit"),
    (["gross_amount"], "gross_rounded_2", "decimal", "round_gross_2dp"),
    (["fx_rate"], "fx_rate_rounded_4", "decimal", "round_fx_4dp"),
    (["amount_local"], "amount_floor", "int", "floor_amount"),
    (["amount_local"], "amount_ceil", "int", "ceil_amount"),
]

NUMERIC_NEGATIVE_PATTERNS = [
    (["first_name", "last_name"], "full_name", "string", "numeric_wrong_concat"),
    (["signup_date"], "signup_year", "int", "numeric_from_date_without_extract"),
    (["amount_local"], "country_code", "string", "numeric_wrong_target_type"),
    (["status_code"], "status_numeric", "int", "numeric_from_categorical"),
    (["email_text"], "email_hash", "int", "numeric_from_string_without_parse"),
]


def _norm_numeric_source_type(name: str) -> str:
    if "amount" in name or "score" in name or "rate" in name or "price" in name:
        return "decimal"
    return "int"


def build_numeric_positive_record(rng: random.Random, domain: str, db_ref: str, entity: str, serial: int) -> dict:
    spec = rng.choice(NUMERIC_POSITIVE_PATTERNS)
    source_specs, target_name, target_type, base_name = spec
    source_columns = []
    for name in source_specs:
        col_type = _norm_numeric_source_type(name)
        source_columns.append(make_column(rng, f"{entity}_{name}", col_type, 1.8, 4.9))
    target_entropy = min(5.5, max(c["entropy"] for c in source_columns) + rand_entropy(rng, 0.1, 1.0))
    transform_type_map = {
        "add_amount_plus_tax": "add", "add_amount_tax": "add",
        "subtract_discount_from_gross": "subtract", "subtract_cost_from_revenue": "subtract", "subtract_used_from_quantity": "subtract", "subtract_quantity_from_total": "subtract", "subtract_balance_from_limit": "subtract",
        "multiply_price_quantity": "multiply", "multiply_amount_fx": "multiply",
        "divide_balance_by_limit": "divide",
        "round_risk_score": "round", "round_gross_2dp": "round", "round_fx_4dp": "round",
        "floor_risk_score": "floor", "floor_amount": "floor",
        "ceil_risk_score": "ceil", "ceil_amount": "ceil",
        "abs_amount": "abs",
        "cast_age_to_int": "cast_to_int", "cast_to_decimal": "cast_to_decimal",
    }
    transform_type = transform_type_map.get(base_name, "add")
    return {
        "source_columns": source_columns,
        "target_column": {"name": f"{entity}_{target_name}", "type": target_type, "entropy": round(target_entropy, 2)},
        "transform_name": make_transform_name(db_ref, domain, base_name, 1, serial),
        "transform_type": transform_type,
        "label": 1,
    }


def build_numeric_negative_record(rng: random.Random, domain: str, db_ref: str, entity: str, serial: int) -> dict:
    spec = rng.choice(NUMERIC_NEGATIVE_PATTERNS)
    source_specs, _, wrong_target_name, base_name = spec
    source_columns = [
        make_column(rng, f"{entity}_{name}", "string" if name in ("first_name", "last_name", "email_text", "status_code") else "decimal" if "amount" in name else "date" if "date" in name else "int", 1.5, 4.5)
        for name in source_specs
    ]
    target_name = rng.choice(NUMERIC_UNRELATED)
    return {
        "source_columns": source_columns,
        "target_column": {"name": target_name, "type": "string", "entropy": rand_entropy(rng, 1.5, 4.0)},
        "transform_name": make_transform_name(db_ref, domain, base_name, 0, serial),
        "transform_type": rng.choice(["add", "subtract", "multiply", "round", "cast_to_int"]),
        "label": 0,
    }


# ---------- Date ----------

DATE_POSITIVE_PATTERNS = [
    (["signup_date"], "signup_year", "date_trunc_year"),
    (["signup_date"], "signup_month", "date_trunc_month"),
    (["event_date"], "event_week", "date_trunc_week"),
    (["txn_date"], "txn_day", "date_trunc_day"),
    (["signup_date"], "signup_year_extract", "extract_year"),
    (["signup_date"], "signup_month_extract", "extract_month"),
    (["birth_date"], "birth_day_of_month", "extract_day"),
    (["event_date", "created_at"], "days_between", "date_diff_days"),
    (["due_date", "txn_date"], "days_until_due", "date_diff_days"),
    (["start_date"], "start_plus_30_days", "add_days"),
    (["event_date"], "event_plus_7_days", "add_days"),
    (["signup_date"], "signup_yyyy_mm", "format_date"),
    (["txn_date"], "txn_iso_date", "format_date"),
    (["date_text"], "parsed_date", "parse_date"),
    (["created_at_text"], "created_at_parsed", "parse_date"),
]
# For parse_date, source is string type
DATE_PARSE_SOURCE_NAMES = ["date_text", "created_at_text", "timestamp_text"]

DATE_NEGATIVE_PATTERNS = [
    (["amount_local"], "txn_year", "date_from_numeric"),
    (["first_name"], "signup_date", "date_from_string"),
    (["event_date"], "amount_local", "date_wrong_target"),
    (["status_code"], "event_month", "date_from_categorical"),
]


def build_date_positive_record(rng: random.Random, domain: str, db_ref: str, entity: str, serial: int) -> dict:
    spec = rng.choice(DATE_POSITIVE_PATTERNS)
    source_specs, target_suffix, base_name = spec[0], spec[1], spec[2]
    source_columns = []
    for src in source_specs:
        col_type = "string" if src in DATE_PARSE_SOURCE_NAMES or "text" in src else "date"
        source_columns.append(make_column(rng, f"{entity}_{src}", col_type, 2.0, 4.5))
    target_entropy = max(1.2, source_columns[0]["entropy"] - rand_entropy(rng, 0.3, 1.5))
    return {
        "source_columns": source_columns,
        "target_column": {"name": f"{entity}_{target_suffix}", "type": "date", "entropy": round(target_entropy, 2)},
        "transform_name": make_transform_name(db_ref, domain, base_name, 1, serial),
        "transform_type": base_name,
        "label": 1,
    }


def build_date_negative_record(rng: random.Random, domain: str, db_ref: str, entity: str, serial: int) -> dict:
    source_name, _, base_name = rng.choice(DATE_NEGATIVE_PATTERNS)
    col_type = "decimal" if "amount" in source_name else "string" if "name" in source_name or "status" in source_name else "date"
    source_columns = [make_column(rng, f"{entity}_{source_name}", col_type, 1.5, 4.5)]
    target_name = rng.choice(DATE_UNRELATED)
    return {
        "source_columns": source_columns,
        "target_column": {"name": target_name, "type": "decimal" if "amount" in target_name or "quantity" in target_name else "int" if target_name == "quantity" else "string", "entropy": rand_entropy(rng, 1.5, 4.0)},
        "transform_name": make_transform_name(db_ref, domain, base_name, 0, serial),
        "transform_type": rng.choice(["date_trunc_year", "extract_year", "date_diff_days", "format_date"]),
        "label": 0,
    }


# ---------- Boolean ----------

BOOLEAN_POSITIVE_PATTERNS = [
    (["is_active", "is_verified"], "is_active_and_verified", "and_op"),
    (["is_primary", "is_default"], "is_primary_or_default", "or_op"),
    (["is_cancelled"], "is_not_cancelled", "not_op"),
    (["amount_local"], "amount_is_null", "is_null"),
    (["email_text"], "email_is_not_null", "is_not_null"),
    (["status_code", "active_status"], "status_equals_active", "eq"),
    (["quantity", "reorder_level"], "quantity_below_reorder", "lt"),
    (["amount_local", "credit_limit"], "amount_over_limit", "gt"),
    (["age", "min_age"], "age_gte_min", "gte"),
    (["risk_score", "threshold"], "risk_lte_threshold", "lte"),
    (["status_code"], "status_in_list", "in_list"),
    (["amount_local", "min_val", "max_val"], "amount_between", "between"),
]

BOOLEAN_NEGATIVE_PATTERNS = [
    (["first_name"], "is_valid", "boolean_from_string"),
    (["amount_local"], "full_name", "boolean_wrong_target"),
    (["signup_date"], "is_active", "boolean_from_date"),
    (["status_code"], "amount_gt_zero", "boolean_semantic_mismatch"),
]


def build_boolean_positive_record(rng: random.Random, domain: str, db_ref: str, entity: str, serial: int) -> dict:
    spec = rng.choice(BOOLEAN_POSITIVE_PATTERNS)
    source_specs, target_suffix, base_name = spec
    source_columns = []
    for name in source_specs:
        t = "boolean" if name.startswith("is_") or "active" in name or "primary" in name or "default" in name or "cancelled" in name else "decimal" if "amount" in name or "limit" in name or "score" in name or "val" in name else "int" if name in ("quantity", "age", "reorder_level", "min_age") else "string"
        source_columns.append(make_column(rng, f"{entity}_{name}", t, 1.2, 4.2))
    target_entropy = rand_entropy(rng, 0.8, 2.5)
    return {
        "source_columns": source_columns,
        "target_column": {"name": f"{entity}_{target_suffix}", "type": "boolean", "entropy": round(target_entropy, 2)},
        "transform_name": make_transform_name(db_ref, domain, base_name, 1, serial),
        "transform_type": base_name,
        "label": 1,
    }


def build_boolean_negative_record(rng: random.Random, domain: str, db_ref: str, entity: str, serial: int) -> dict:
    source_name, _, base_name = rng.choice(BOOLEAN_NEGATIVE_PATTERNS)
    col_type = "string" if "name" in source_name or "status" in source_name else "decimal" if "amount" in source_name else "date" if "date" in source_name else "string"
    source_columns = [make_column(rng, f"{entity}_{source_name}", col_type, 1.5, 4.5)]
    target_name = rng.choice(BOOLEAN_UNRELATED)
    return {
        "source_columns": source_columns,
        "target_column": {"name": target_name, "type": "string" if target_name != "amount_local" else "decimal", "entropy": rand_entropy(rng, 1.5, 4.0)},
        "transform_name": make_transform_name(db_ref, domain, base_name, 0, serial),
        "transform_type": rng.choice(["and_op", "or_op", "eq", "gt", "lt", "is_null"]),
        "label": 0,
    }


def generate_all() -> List[dict]:
    rng = random.Random(SEED)
    records: List[dict] = []
    serial = 1

    for _ in range(NUMERIC_POSITIVE):
        domain, db_ref = rng.choice(DOMAIN_DB_PAIRS)
        entity = pick_entity(rng, domain)
        records.append(build_numeric_positive_record(rng, domain, db_ref, entity, serial))
        serial += 1
    for _ in range(NUMERIC_NEGATIVE):
        domain, db_ref = rng.choice(DOMAIN_DB_PAIRS)
        entity = pick_entity(rng, domain)
        records.append(build_numeric_negative_record(rng, domain, db_ref, entity, serial))
        serial += 1

    for _ in range(DATE_POSITIVE):
        domain, db_ref = rng.choice(DOMAIN_DB_PAIRS)
        entity = pick_entity(rng, domain)
        records.append(build_date_positive_record(rng, domain, db_ref, entity, serial))
        serial += 1
    for _ in range(DATE_NEGATIVE):
        domain, db_ref = rng.choice(DOMAIN_DB_PAIRS)
        entity = pick_entity(rng, domain)
        records.append(build_date_negative_record(rng, domain, db_ref, entity, serial))
        serial += 1

    for _ in range(BOOLEAN_POSITIVE):
        domain, db_ref = rng.choice(DOMAIN_DB_PAIRS)
        entity = pick_entity(rng, domain)
        records.append(build_boolean_positive_record(rng, domain, db_ref, entity, serial))
        serial += 1
    for _ in range(BOOLEAN_NEGATIVE):
        domain, db_ref = rng.choice(DOMAIN_DB_PAIRS)
        entity = pick_entity(rng, domain)
        records.append(build_boolean_negative_record(rng, domain, db_ref, entity, serial))
        serial += 1

    rng.shuffle(records)
    return records


def main() -> None:
    records = generate_all()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    labels = Counter(r["label"] for r in records)
    types_count = Counter()
    for r in records:
        types_count[r["target_column"]["type"]] += 1
    transform_types = Counter(r["transform_type"] for r in records)
    print(f"Wrote {len(records)} records to {OUTPUT_PATH}")
    print(f"Label distribution: {dict(labels)}")
    print(f"Target type distribution: {dict(types_count)}")
    print(f"Transform type distribution: {dict(transform_types)}")


if __name__ == "__main__":
    main()
