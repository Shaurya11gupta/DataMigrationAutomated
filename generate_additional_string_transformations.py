#!/usr/bin/env python3
"""
Append 900 additional string-transformation training examples to JSONL dataset.

This generator creates both positive (label=1) and negative (label=0) examples
across concat, substring, regex_extract, split, lower, upper, and trim.
It also injects domain + sample database references into transform_name to
increase contextual richness for model training.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path


DATASET_PATH = Path("/workspace/string_transformations_training_data.jsonl")
SEED = 20260210

POSITIVE_COUNTS = {
    "concat": 100,
    "substring": 90,
    "regex_extract": 90,
    "split": 90,
    "lower": 95,
    "upper": 95,
    "trim": 90,
}

NEGATIVE_COUNTS = {
    "concat": 40,
    "substring": 35,
    "regex_extract": 35,
    "split": 35,
    "lower": 35,
    "upper": 35,
    "trim": 35,
}

DOMAIN_DB_PAIRS = [
    ("it", "adventureworks"),
    ("retail", "northwind"),
    ("film", "sakila"),
    ("media", "chinook"),
    ("analytics", "tpch"),
    ("logistics", "wideworldimporters"),
    ("healthcare", "mimic_iv"),
    ("it", "pagila"),
    ("ecommerce", "dvdrental"),
    ("finance", "adventureworks"),
    ("telecom", "tpch"),
    ("education", "northwind"),
    ("public_sector", "wideworldimporters"),
    ("manufacturing", "adventureworks"),
    ("hr", "northwind"),
]

DOMAIN_ENTITIES = {
    "it": ["user", "server", "repository", "incident", "deployment", "ticket"],
    "retail": ["customer", "order", "product", "supplier", "shipment", "invoice"],
    "film": ["actor", "film", "category", "rental", "store", "staff"],
    "media": ["artist", "album", "track", "playlist", "listener", "invoice"],
    "analytics": ["customer", "orders", "lineitem", "supplier", "region", "nation"],
    "logistics": ["warehouse", "route", "parcel", "container", "manifest", "driver"],
    "healthcare": ["patient", "admission", "diagnosis", "procedure", "provider", "claim"],
    "ecommerce": ["cart", "checkout", "catalog", "merchant", "fulfillment", "coupon"],
    "finance": ["account", "transaction", "ledger", "card", "branch", "payment"],
    "telecom": ["subscriber", "msisdn", "tower", "session", "plan", "billing"],
    "education": ["student", "course", "campus", "program", "faculty", "enrollment"],
    "public_sector": ["citizen", "permit", "district", "registry", "service", "case"],
    "manufacturing": ["plant", "machine", "batch", "workorder", "component", "inspection"],
    "hr": ["employee", "department", "position", "payroll", "benefit", "attendance"],
}

DB_ENTITIES = {
    "northwind": ["customers", "employees", "orders", "products", "suppliers", "shippers"],
    "sakila": ["actor", "film", "customer", "address", "city", "staff"],
    "adventureworks": ["person", "salesorder", "product", "vendor", "territory", "employee"],
    "chinook": ["artist", "album", "track", "customer", "invoice", "genre"],
    "tpch": ["customer", "orders", "lineitem", "partsupp", "supplier", "nation"],
    "wideworldimporters": ["stockitem", "orderline", "delivery", "invoice", "warehouse", "customer"],
    "mimic_iv": ["patient", "admission", "diagnosis", "procedure", "icu", "prescription"],
    "pagila": ["film", "actor", "rental", "payment", "inventory", "staff"],
    "dvdrental": ["film", "customer", "rental", "payment", "staff", "store"],
}

UNRELATED_TARGETS = [
    "country",
    "blood_type",
    "os_distribution",
    "vaccine_brand",
    "invoice_total",
    "continent_name",
    "species_name",
    "weather_condition",
    "preferred_language",
    "emergency_contact",
    "currency_symbol",
    "diagnosis_text",
    "citizenship_status",
    "department_budget",
]


def rand_entropy(rng: random.Random, low: float, high: float) -> float:
    return round(rng.uniform(low, high), 2)


def infer_pk(rng: random.Random, name: str) -> bool:
    if name.endswith("_id"):
        return True
    if "identifier" in name or name.endswith("_key"):
        return rng.random() < 0.65
    return rng.random() < 0.08


def make_column(rng: random.Random, name: str, low: float, high: float) -> dict:
    return {
        "name": name,
        "type": "string",
        "is_pk": infer_pk(rng, name),
        "entropy": rand_entropy(rng, low, high),
    }


def pick_entity(rng: random.Random, domain: str, db_ref: str) -> str:
    pool = DOMAIN_ENTITIES.get(domain, []) + DB_ENTITIES.get(db_ref, [])
    return rng.choice(pool)


def make_transform_name(
    db_ref: str,
    domain: str,
    base_name: str,
    label: int,
    serial: int,
) -> str:
    polarity = "pos" if label == 1 else "neg"
    return f"{db_ref}_{domain}_{base_name}_{polarity}_{serial:04d}"


def build_positive_record(
    rng: random.Random,
    transform_type: str,
    domain: str,
    db_ref: str,
    entity: str,
    serial: int,
) -> dict:
    if transform_type == "concat":
        patterns = [
            ([f"{entity}_first_name", f"{entity}_last_name"], f"{entity}_full_name", "name_concat_space"),
            ([f"{entity}_city", f"{entity}_state_code"], f"{entity}_city_state_display", "city_state_concat_comma"),
            ([f"{entity}_country_code", f"{entity}_phone_number"], f"{entity}_e164_phone", "country_phone_concat_plus"),
            ([f"{entity}_schema_name", f"{entity}_table_name"], f"{entity}_qualified_table", "schema_table_concat_dot"),
            ([f"{entity}_year_text", f"{entity}_month_text"], f"{entity}_period_yyyymm", "period_concat_no_sep"),
            ([f"{entity}_code", f"{entity}_id"], f"{entity}_composite_key", "code_id_concat_dash"),
            ([f"{entity}_brand_name", f"{entity}_model_name"], f"{entity}_display_title", "brand_model_concat_space"),
            ([f"{entity}_route_code", f"{entity}_stop_code"], f"{entity}_route_stop_ref", "route_stop_concat_colon"),
        ]
        source_names, target_name, base_name = rng.choice(patterns)
        source_columns = [make_column(rng, name, 1.8, 4.9) for name in source_names]
        target_entropy = min(5.8, max(col["entropy"] for col in source_columns) + rand_entropy(rng, 0.25, 1.15))

    elif transform_type == "substring":
        patterns = [
            (f"{entity}_identifier", f"{entity}_prefix4", "identifier_prefix_substring"),
            (f"{entity}_identifier", f"{entity}_suffix4", "identifier_suffix_substring"),
            (f"{entity}_date_text", f"{entity}_year", "date_year_substring"),
            (f"{entity}_date_text", f"{entity}_month", "date_month_substring"),
            (f"{entity}_code", f"{entity}_family3", "code_family_substring"),
            (f"{entity}_serial_number", f"{entity}_tail6", "serial_tail_substring"),
            (f"{entity}_timestamp_text", f"{entity}_hour_token", "timestamp_hour_substring"),
        ]
        source_name, target_name, base_name = rng.choice(patterns)
        source_columns = [make_column(rng, source_name, 2.8, 5.3)]
        src_entropy = source_columns[0]["entropy"]
        target_entropy = max(1.0, src_entropy - rand_entropy(rng, 0.5, 2.0))

    elif transform_type == "regex_extract":
        patterns = [
            (f"{entity}_email", f"{entity}_email_username", "regex_email_username_extract"),
            (f"{entity}_email", f"{entity}_email_domain", "regex_email_domain_extract"),
            (f"{entity}_log_message", "ipv4_address", "regex_ipv4_extract"),
            (f"{entity}_url", f"{entity}_domain_name", "regex_domain_extract"),
            (f"{entity}_note_text", f"{entity}_ticket_id", "regex_ticket_id_extract"),
            (f"{entity}_address_blob", "zip5", "regex_zip5_extract"),
            (f"{entity}_payment_text", "amount_numeric", "regex_amount_extract"),
            (f"{entity}_clinical_note", "icd10_code", "regex_icd10_extract"),
        ]
        source_name, target_name, base_name = rng.choice(patterns)
        source_columns = [make_column(rng, source_name, 3.0, 5.5)]
        src_entropy = source_columns[0]["entropy"]
        target_entropy = max(1.1, min(5.3, src_entropy - rand_entropy(rng, 0.3, 1.9)))

    elif transform_type == "split":
        patterns = [
            (f"{entity}_full_name", f"{entity}_first_name_token", "split_full_name_first_token"),
            (f"{entity}_full_name", f"{entity}_last_name_token", "split_full_name_last_token"),
            (f"{entity}_email", f"{entity}_email_username_split", "split_email_username"),
            (f"{entity}_email", f"{entity}_email_domain_split", "split_email_domain"),
            (f"{entity}_path", f"{entity}_path_first_segment", "split_path_first_segment"),
            (f"{entity}_version", f"{entity}_version_major", "split_version_major"),
            (f"{entity}_version", f"{entity}_version_minor", "split_version_minor"),
            (f"{entity}_compound_id", f"{entity}_compound_right_part", "split_compound_id_right"),
        ]
        source_name, target_name, base_name = rng.choice(patterns)
        source_columns = [make_column(rng, source_name, 2.5, 5.2)]
        src_entropy = source_columns[0]["entropy"]
        target_entropy = max(1.0, src_entropy - rand_entropy(rng, 0.4, 2.2))

    elif transform_type == "lower":
        patterns = [
            (f"{entity}_email_raw", f"{entity}_email_lower", "lowercase_email"),
            (f"{entity}_username_raw", f"{entity}_username_lower", "lowercase_username"),
            (f"{entity}_city_raw", f"{entity}_city_lower", "lowercase_city"),
            (f"{entity}_slug_raw", f"{entity}_slug_lower", "lowercase_slug"),
            (f"{entity}_env_raw", f"{entity}_env_lower", "lowercase_environment"),
            (f"{entity}_token_hex_raw", f"{entity}_token_hex_lower", "lowercase_hex_token"),
        ]
        source_name, target_name, base_name = rng.choice(patterns)
        source_columns = [make_column(rng, source_name, 2.0, 5.2)]
        src_entropy = source_columns[0]["entropy"]
        target_entropy = max(1.0, src_entropy - rand_entropy(rng, 0.0, 0.25))

    elif transform_type == "upper":
        patterns = [
            (f"{entity}_state_code_raw", f"{entity}_state_code_upper", "uppercase_state_code"),
            (f"{entity}_country_code_raw", f"{entity}_country_code_upper", "uppercase_country_code"),
            (f"{entity}_currency_code_raw", f"{entity}_currency_code_upper", "uppercase_currency_code"),
            (f"{entity}_status_raw", f"{entity}_status_upper", "uppercase_status"),
            (f"{entity}_tier_code_raw", f"{entity}_tier_code_upper", "uppercase_tier_code"),
            (f"{entity}_plate_raw", f"{entity}_plate_upper", "uppercase_plate"),
        ]
        source_name, target_name, base_name = rng.choice(patterns)
        source_columns = [make_column(rng, source_name, 1.3, 4.8)]
        src_entropy = source_columns[0]["entropy"]
        target_entropy = max(1.0, src_entropy - rand_entropy(rng, 0.0, 0.25))

    elif transform_type == "trim":
        patterns = [
            (f"{entity}_name_raw_padded", f"{entity}_name_trimmed", "trim_name_spaces"),
            (f"{entity}_email_raw_space", f"{entity}_email_trimmed", "trim_email_spaces"),
            (f"{entity}_code_raw_space", f"{entity}_code_trimmed", "trim_code_edges"),
            (f"{entity}_comment_raw_tabs", f"{entity}_comment_trimmed", "trim_comment_tabs"),
            (f"{entity}_title_raw_quotes", f"{entity}_title_trimmed", "trim_title_quotes"),
            (f"{entity}_message_raw_newline", f"{entity}_message_trimmed", "trim_message_newline"),
            (f"{entity}_account_raw_zero", f"{entity}_account_ltrim_zero", "trim_account_leading_zero"),
        ]
        source_name, target_name, base_name = rng.choice(patterns)
        source_columns = [make_column(rng, source_name, 2.0, 5.3)]
        src_entropy = source_columns[0]["entropy"]
        target_entropy = max(1.0, src_entropy - rand_entropy(rng, 0.0, 0.35))

    else:
        raise ValueError(f"Unsupported positive transform_type: {transform_type}")

    return {
        "source_columns": source_columns,
        "target_column": {
            "name": target_name,
            "type": "string",
            "entropy": round(target_entropy, 2),
        },
        "transform_name": make_transform_name(db_ref, domain, base_name, 1, serial),
        "transform_type": transform_type,
        "label": 1,
    }


def build_negative_record(
    rng: random.Random,
    transform_type: str,
    domain: str,
    db_ref: str,
    entity: str,
    serial: int,
) -> dict:
    if transform_type == "concat":
        source_variants = [
            [f"{entity}_first_name", f"{entity}_last_name"],
            [f"{entity}_city", f"{entity}_zip_code"],
            [f"{entity}_brand_name", f"{entity}_model_name"],
            [f"{entity}_country_code", f"{entity}_phone_number"],
        ]
        base_name = rng.choice(["wrong_concat", "semantic_mismatch", "invalid_concat_assumption"])
        source_names = rng.choice(source_variants)
        target_name = rng.choice(UNRELATED_TARGETS)
        source_columns = [make_column(rng, name, 1.6, 4.9) for name in source_names]
        target_entropy = rand_entropy(rng, 1.2, 4.3)

    elif transform_type == "substring":
        source_name = rng.choice(
            [
                f"{entity}_date_text",
                f"{entity}_identifier",
                f"{entity}_serial_number",
                f"{entity}_url",
            ]
        )
        base_name = rng.choice(["wrong_substring_target", "substring_semantic_mismatch", "substring_unrelated_output"])
        target_name = rng.choice(UNRELATED_TARGETS)
        source_columns = [make_column(rng, source_name, 2.3, 5.4)]
        target_entropy = rand_entropy(rng, 1.1, 4.1)

    elif transform_type == "regex_extract":
        source_name = rng.choice(
            [
                f"{entity}_city_name",
                f"{entity}_country_name",
                f"{entity}_department_name",
                f"{entity}_status_label",
            ]
        )
        base_name = rng.choice(["regex_wrong_pattern_assumption", "regex_semantic_mismatch", "regex_extract_unrelated"])
        target_name = rng.choice(["email_username", "ipv4_address", "icd10_code", "zip5", "domain_name"])
        source_columns = [make_column(rng, source_name, 1.5, 4.4)]
        target_entropy = rand_entropy(rng, 1.2, 3.8)

    elif transform_type == "split":
        source_name = rng.choice(
            [
                f"{entity}_country_code",
                f"{entity}_single_word_label",
                f"{entity}_numeric_id",
                f"{entity}_status",
            ]
        )
        base_name = rng.choice(["split_without_delimiter", "split_semantic_mismatch", "split_invalid_tokenization"])
        target_name = rng.choice(["first_name", "email_domain", "path_segment", "version_major", "hour_token"])
        source_columns = [make_column(rng, source_name, 1.4, 4.1)]
        target_entropy = rand_entropy(rng, 1.0, 3.2)

    elif transform_type == "lower":
        source_name = rng.choice(
            [
                f"{entity}_numeric_id",
                f"{entity}_amount_value",
                f"{entity}_postal_code",
                f"{entity}_blood_type",
            ]
        )
        base_name = rng.choice(["lowercase_unrelated_target", "lower_semantic_mismatch", "wrong_lower_projection"])
        target_name = rng.choice(UNRELATED_TARGETS)
        source_columns = [make_column(rng, source_name, 1.1, 4.6)]
        target_entropy = rand_entropy(rng, 1.0, 4.0)

    elif transform_type == "upper":
        source_name = rng.choice(
            [
                f"{entity}_date_text",
                f"{entity}_age_band",
                f"{entity}_salary_range",
                f"{entity}_latitude",
            ]
        )
        base_name = rng.choice(["uppercase_unrelated_target", "upper_semantic_mismatch", "wrong_upper_projection"])
        target_name = rng.choice(UNRELATED_TARGETS)
        source_columns = [make_column(rng, source_name, 1.1, 4.8)]
        target_entropy = rand_entropy(rng, 1.0, 4.2)

    elif transform_type == "trim":
        source_name = rng.choice(
            [
                f"{entity}_country_code",
                f"{entity}_status_flag",
                f"{entity}_gender_code",
                f"{entity}_risk_level",
            ]
        )
        base_name = rng.choice(["trim_semantic_mismatch", "trim_unrelated_target", "wrong_trim_assumption"])
        target_name = rng.choice(UNRELATED_TARGETS)
        source_columns = [make_column(rng, source_name, 1.0, 4.2)]
        target_entropy = rand_entropy(rng, 1.0, 3.9)

    else:
        raise ValueError(f"Unsupported negative transform_type: {transform_type}")

    return {
        "source_columns": source_columns,
        "target_column": {
            "name": target_name,
            "type": "string",
            "entropy": round(target_entropy, 2),
        },
        "transform_name": make_transform_name(db_ref, domain, base_name, 0, serial),
        "transform_type": transform_type,
        "label": 0,
    }


def generate_records() -> list[dict]:
    rng = random.Random(SEED)
    records: list[dict] = []
    serial = 1

    for transform_type, count in POSITIVE_COUNTS.items():
        for _ in range(count):
            domain, db_ref = rng.choice(DOMAIN_DB_PAIRS)
            entity = pick_entity(rng, domain, db_ref)
            records.append(
                build_positive_record(
                    rng=rng,
                    transform_type=transform_type,
                    domain=domain,
                    db_ref=db_ref,
                    entity=entity,
                    serial=serial,
                )
            )
            serial += 1

    for transform_type, count in NEGATIVE_COUNTS.items():
        for _ in range(count):
            domain, db_ref = rng.choice(DOMAIN_DB_PAIRS)
            entity = pick_entity(rng, domain, db_ref)
            records.append(
                build_negative_record(
                    rng=rng,
                    transform_type=transform_type,
                    domain=domain,
                    db_ref=db_ref,
                    entity=entity,
                    serial=serial,
                )
            )
            serial += 1

    rng.shuffle(records)
    return records


def count_existing_rows(path: Path) -> int:
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def main() -> None:
    if not DATASET_PATH.exists():
        raise SystemExit(f"Dataset file not found: {DATASET_PATH}")

    existing_rows = count_existing_rows(DATASET_PATH)
    expected_start_rows = 100
    if existing_rows != expected_start_rows:
        raise SystemExit(
            f"Expected {expected_start_rows} existing rows, found {existing_rows}. "
            "Aborting to avoid duplicate appends."
        )

    new_records = generate_records()
    if len(new_records) != 900:
        raise SystemExit(f"Expected 900 generated records, got {len(new_records)}")

    with DATASET_PATH.open("a", encoding="utf-8") as f:
        for record in new_records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")

    labels = Counter(r["label"] for r in new_records)
    transforms = Counter(r["transform_type"] for r in new_records)
    print(f"Appended rows: {len(new_records)}")
    print(f"Label distribution: {dict(labels)}")
    print(f"Transform distribution: {dict(transforms)}")
    print(f"Final row count: {count_existing_rows(DATASET_PATH)}")


if __name__ == "__main__":
    main()
