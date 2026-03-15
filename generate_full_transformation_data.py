#!/usr/bin/env python3
"""
Generate comprehensive transformation training data for categories A–L (v3).

Key improvements (v3 – optimized for transformer fine-tuning):
  - 500 pos/neg per type for robust generalization
  - Naming convention variations: snake_case, camelCase, PascalCase, dot.notation
  - Massively expanded synonym pool (100+ concepts)
  - Richer templates per transform type (15-20+ per type)
  - Domain-specific and entity-specific column name generation
  - Diverse negatives (type mismatch, semantic mismatch, hard, cross-type)

Target: best-in-class generalization for DistilBERT-based transformer.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

SEED = 20260212
BASE = Path(__file__).resolve().parent
OUTPUT_PATH = BASE / "full_transformation_training_data.jsonl"

# 50/50 pos/neg per type for balanced training — 500 each for transformer
DEFAULT_POS = 500
DEFAULT_NEG = 500

DOMAINS = ["retail", "finance", "healthcare", "logistics", "hr", "analytics",
           "it", "media", "telecom", "public_sector", "insurance", "education",
           "real_estate", "manufacturing", "travel", "legal"]
ENTITIES = ["customer", "order", "patient", "invoice", "employee", "product",
            "transaction", "event", "account", "claim", "user", "vendor",
            "shipment", "contract", "policy", "ticket", "campaign", "asset"]

# ---------------------------------------------------------------------------
# Naming convention variations — key for transformer generalization
# ---------------------------------------------------------------------------

def to_camel(name: str) -> str:
    """snake_case → camelCase"""
    parts = name.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])

def to_pascal(name: str) -> str:
    """snake_case → PascalCase"""
    return "".join(p.capitalize() for p in name.split("_"))

def to_dot(name: str) -> str:
    """snake_case → dot.notation"""
    return name.replace("_", ".")

def to_hyphen(name: str) -> str:
    """snake_case → hyphen-case"""
    return name.replace("_", "-")

CONVENTION_FNS = [
    lambda x: x,            # snake_case (60% weight)
    lambda x: x,
    lambda x: x,
    to_camel,                # camelCase (20% weight)
    to_pascal,               # PascalCase (10% weight)
    to_dot,                  # dot.notation (5% weight)
    to_hyphen,               # hyphen-case (5% weight)
]

def vary_convention(rng: random.Random, name: str) -> str:
    """Randomly apply a naming convention variation."""
    fn = rng.choice(CONVENTION_FNS)
    return fn(name)


# ---------------------------------------------------------------------------
# Name augmentation system — generates diverse column name variations
# ---------------------------------------------------------------------------

# Source column modifiers (suggest raw/input data)
SRC_PREFIXES = ["raw_", "original_", "input_", "source_", "old_", "legacy_", "imported_",
                "dirty_", "staging_", "src_", "incoming_", "ext_", "unprocessed_", "temp_"]
SRC_SUFFIXES = ["_raw", "_original", "_input", "_source", "_old", "_dirty", "_v1",
                "_staging", "_src", "_incoming", "_unprocessed", "_orig"]

# Target column modifiers (suggest processed/output data)
TGT_PREFIXES = ["processed_", "clean_", "final_", "output_", "new_", "target_", "derived_",
                "transformed_", "tgt_", "computed_", "resolved_", "enriched_", "standardized_"]
TGT_SUFFIXES = ["_processed", "_clean", "_cleaned", "_final", "_output", "_new", "_v2",
                "_result", "_transformed", "_tgt", "_computed", "_resolved", "_enriched", "_std"]

# Massively expanded synonym pools for common concepts
NAME_SYNONYMS = {
    "first_name": ["given_name", "fname", "forename", "name_first", "givenname", "first", "person_first"],
    "last_name": ["surname", "family_name", "lname", "name_last", "familyname", "last", "person_last"],
    "full_name": ["complete_name", "display_name", "whole_name", "person_name", "fullname", "name_full", "name_display"],
    "email_text": ["email_address", "email_raw", "user_email", "contact_email", "email_str", "email_addr", "mail_address"],
    "phone_number": ["phone", "telephone", "mobile", "contact_number", "tel_number", "mobile_number", "cell_phone", "phone_num"],
    "customer_name": ["client_name", "buyer_name", "cust_name", "shopper_name", "patron_name", "consumer_name", "account_holder"],
    "amount_local": ["amount", "value", "price_amount", "cost_value", "payment_amount", "txn_amount", "monetary_value"],
    "total_amount": ["grand_total", "sum_amount", "total_value", "total_cost", "amount_total", "aggregate_amount", "total_sum"],
    "address_raw": ["street_address", "mailing_address", "location_text", "addr_line", "postal_address", "residence_addr", "home_address"],
    "status_code": ["status", "state_code", "status_val", "current_status", "status_text", "state", "status_indicator"],
    "description_text": ["description", "desc", "summary_text", "details", "remarks", "narrative", "body_text", "content"],
    "comment_text": ["comment", "note", "feedback_text", "review_text", "remark_text", "annotation", "observation"],
    "product_title": ["product_name", "item_name", "prod_title", "product_desc", "item_title", "goods_name", "merchandise_name"],
    "risk_score": ["risk_rating", "risk_value", "risk_index", "risk_metric", "riskiness", "risk_level", "risk_grade"],
    "created_at": ["creation_date", "date_created", "create_timestamp", "created_date", "insert_date", "creation_ts", "created_on"],
    "order_date": ["purchase_date", "order_timestamp", "ordered_at", "buy_date", "sale_date", "order_dt"],
    "birth_date": ["date_of_birth", "dob", "birthday", "born_date", "birth_dt", "date_birth"],
    "hire_date": ["employment_date", "start_date", "joining_date", "onboard_date", "hire_dt"],
    "unit_price": ["price", "rate", "cost_per_unit", "item_price", "selling_price", "unit_cost"],
    "quantity": ["qty", "count", "num_items", "units", "pieces", "item_qty"],
    "tax_amount": ["tax", "tax_value", "vat_amount", "tax_total", "gst_amount"],
    "discount_amount": ["discount", "rebate", "markdown", "price_reduction", "savings"],
    "city_name": ["city", "town", "municipality", "metro_area", "locality"],
    "country_code": ["country", "nation_code", "country_iso", "cc", "country_id"],
    "postal_code": ["zip_code", "zipcode", "postcode", "pin_code", "zip"],
    "ssn_text": ["social_security", "ssn", "national_id", "tax_id", "sin_number"],
    "category_raw": ["category", "cat_code", "product_category", "item_category", "classification"],
    "department_name": ["department", "dept", "division", "business_unit", "org_unit"],
    "company_name": ["organization", "business_name", "firm_name", "corp_name", "employer_name"],
    "username": ["user_name", "login", "login_id", "user_id_str", "handle"],
    "file_path": ["filepath", "path", "file_location", "document_path", "resource_path"],
    "url_text": ["url", "web_address", "link", "hyperlink", "web_url", "page_url"],
    "region_name": ["region", "area", "territory", "zone", "district"],
    "weight_kg": ["weight", "mass", "weight_value", "body_weight", "net_weight"],
    "height_cm": ["height", "stature", "body_height", "tallness"],
    "temperature": ["temp", "temperature_value", "temp_reading", "degrees"],
    "serial_number": ["serial", "serial_num", "serial_no", "sn", "part_number"],
    "gender_code": ["gender", "sex", "sex_code", "gender_val"],
    "age_years": ["age", "years_old", "patient_age", "person_age"],
    "balance_local": ["balance", "account_balance", "remaining", "outstanding"],
    "credit_limit": ["credit_max", "max_credit", "credit_ceiling", "credit_allowance"],
}


def augment_name(rng: random.Random, name: str, is_source: bool = True) -> str:
    """Randomly augment a column name with synonyms, prefixes, suffixes, and convention changes."""
    # 40% chance: use a synonym if available (increased from 30%)
    if rng.random() < 0.4 and name in NAME_SYNONYMS:
        name = rng.choice(NAME_SYNONYMS[name])

    # 25% chance: add a prefix or suffix (increased from 20%)
    if rng.random() < 0.25:
        if is_source:
            if rng.random() < 0.5 and not any(name.startswith(p) for p in SRC_PREFIXES):
                name = rng.choice(SRC_PREFIXES) + name
            elif not any(name.endswith(s) for s in SRC_SUFFIXES):
                name = name + rng.choice(SRC_SUFFIXES)
        else:
            if rng.random() < 0.5 and not any(name.startswith(p) for p in TGT_PREFIXES):
                name = rng.choice(TGT_PREFIXES) + name
            elif not any(name.endswith(s) for s in TGT_SUFFIXES):
                name = name + rng.choice(TGT_SUFFIXES)

    # 20% chance: vary naming convention (camelCase, PascalCase, etc.)
    if rng.random() < 0.2:
        name = vary_convention(rng, name)

    return name

# Diverse column name pools for each data type (massively expanded)
STRING_NAMES = [
    "first_name", "last_name", "full_name", "email_text", "phone_number",
    "city_name", "country_code", "postal_code", "address_raw", "street_line",
    "product_title", "description_text", "comment_text", "status_code",
    "department_name", "category_raw", "tag_name", "url_text", "file_path",
    "diagnosis_text", "ssn_text", "serial_number", "username", "gender_code",
    "middle_name", "suffix_name", "company_name", "brand_name", "region_name",
    "note_text", "title_raw", "code_raw", "email_domain", "slug_text",
    "password_hash", "api_key", "token_value", "reference_code", "tracking_number",
    "license_plate", "iban_number", "swift_code", "routing_number", "account_number",
    "policy_number", "claim_number", "ticket_number", "invoice_number", "receipt_code",
    "contract_id_text", "coupon_code", "promo_code", "barcode", "sku",
    "icd_code", "medication_name", "allergy_text", "symptom_desc", "treatment_plan",
    "job_title", "skill_name", "certification", "language_code", "currency_code",
    "mime_type", "user_agent", "ip_address", "mac_address", "hostname",
]
INT_NAMES = [
    "order_id", "customer_id", "product_id", "employee_id", "quantity",
    "line_number", "department_id", "region_id", "rank_value", "age_years",
    "tenure_months", "item_count", "sequence_num", "batch_id", "version_num",
    "store_id", "warehouse_id", "campaign_id", "session_id", "visit_count",
    "retry_count", "error_count", "page_views", "click_count", "num_dependents",
    "floor_number", "bed_count", "seat_number", "priority_level", "severity_level",
]
DECIMAL_NAMES = [
    "amount_local", "tax_amount", "total_amount", "net_amount", "discount_amount",
    "unit_price", "fx_rate", "risk_score", "credit_limit", "balance_local",
    "weight_kg", "height_cm", "temperature", "latitude", "longitude",
    "score_raw", "pct_value", "margin_pct", "commission", "hourly_rate",
    "interest_rate", "exchange_rate", "conversion_rate", "utilization_pct",
    "click_through_rate", "bounce_rate", "open_rate", "churn_rate", "retention_rate",
    "blood_pressure", "bmi_value", "glucose_level", "heart_rate_bpm",
    "revenue", "profit_margin", "operating_cost", "shipping_cost", "handling_fee",
]
DATE_NAMES = [
    "created_at", "updated_at", "signup_date", "birth_date", "order_date",
    "ship_date", "event_date", "start_date", "end_date", "due_date",
    "invoice_date", "hire_date", "termination_date", "txn_date", "modified_at",
    "expiry_date", "renewal_date", "last_login", "first_purchase", "last_activity",
    "discharge_date", "admission_date", "appointment_date", "delivery_date",
    "payment_date", "settlement_date", "maturity_date", "effective_date",
]
BOOL_NAMES = [
    "is_active", "is_primary", "is_valid", "is_deleted", "has_email",
    "is_verified", "is_premium", "has_shipped", "is_refunded", "is_flagged",
    "is_archived", "is_locked", "is_public", "has_consent", "is_subscribed",
    "is_eligible", "is_compliant", "has_dependencies", "is_billable", "is_taxable",
]

TYPE_POOLS = {
    "string": STRING_NAMES,
    "int": INT_NAMES,
    "decimal": DECIMAL_NAMES,
    "date": DATE_NAMES,
    "boolean": BOOL_NAMES,
}


def rand_entropy(rng: random.Random, low: float = 1.5, high: float = 4.8) -> float:
    return round(rng.uniform(low, high), 2)


def infer_pk(rng: random.Random, name: str) -> bool:
    if name.endswith("_id") or name.endswith("_key"):
        return rng.random() < 0.5
    return rng.random() < 0.06


def make_col(rng: random.Random, name: str, col_type: str) -> dict:
    return {"name": name, "type": col_type, "is_pk": infer_pk(rng, name), "entropy": rand_entropy(rng)}


def make_target(name: str, col_type: str, entropy: float) -> dict:
    return {"name": name, "type": col_type, "entropy": round(entropy, 2)}


def aug_src(rng, entity, name):
    """Create augmented source column name with entity prefix."""
    return nwe(entity, augment_name(rng, name, is_source=True))


def aug_tgt(rng, entity, name):
    """Create augmented target column name with entity prefix."""
    return nwe(entity, augment_name(rng, name, is_source=False))


def make_record(srcs, tgt, tt, tname, label):
    return {"source_columns": srcs, "target_column": tgt, "transform_name": tname, "transform_type": tt, "label": label}


def pick(rng, lst):
    return rng.choice(lst)


def ent(rng):
    return pick(rng, ENTITIES)


def nwe(entity, name):
    """Name with entity prefix."""
    if name.startswith(entity + "_"):
        return name
    return f"{entity}_{name}"


def rand_col(rng, entity, col_type):
    """Random column of given type with entity prefix."""
    name = pick(rng, TYPE_POOLS.get(col_type, STRING_NAMES))
    return make_col(rng, nwe(entity, name), col_type)


def rand_wrong_col(rng, entity, wrong_type, avoid_type):
    """Column of wrong type for a negative example."""
    pool = [t for t in ["string", "int", "decimal", "date", "boolean"] if t != avoid_type]
    actual = pick(rng, pool) if wrong_type is None else wrong_type
    return rand_col(rng, entity, actual)


# ---------- Generic negative generators (diverse per category) ----------

def neg_type_mismatch(rng, entity, serial, tt, expected_src_type, expected_tgt_type):
    """Negative: wrong source type for the transform."""
    wrong_src = rand_wrong_col(rng, entity, None, expected_src_type)
    tgt = rand_col(rng, entity, expected_tgt_type)
    return make_record([wrong_src], make_target(tgt["name"], tgt["type"], rand_entropy(rng)), tt, f"{entity}_{tt}_{serial:06d}", 0)


def neg_semantic_mismatch(rng, entity, serial, tt):
    """Negative: random source/target names that don't semantically match the transform."""
    src_type = pick(rng, ["string", "int", "decimal", "date", "boolean"])
    tgt_type = pick(rng, ["string", "int", "decimal", "date", "boolean"])
    src = rand_col(rng, entity, src_type)
    tgt_name = nwe(pick(rng, ENTITIES), pick(rng, TYPE_POOLS[tgt_type]))
    return make_record([src], make_target(tgt_name, tgt_type, rand_entropy(rng)), tt, f"{entity}_{tt}_{serial:06d}", 0)


# ---------- A. Identity / structural ----------

A_TEMPLATES = {
    "direct_copy": [
        ("customer_name", "customer_name_copy", "string"),
        ("amount_local", "amount_local_copy", "decimal"),
        ("order_id", "order_id_copy", "int"),
        ("event_date", "event_date_copy", "date"),
        ("is_active", "is_active_copy", "boolean"),
        ("email_text", "email_text_copy", "string"),
        ("quantity", "quantity_copy", "int"),
        ("balance_local", "balance_local_copy", "decimal"),
        ("phone_number", "phone_number_copy", "string"),
        ("risk_score", "risk_score_copy", "decimal"),
        ("hire_date", "hire_date_copy", "date"),
        ("is_verified", "is_verified_copy", "boolean"),
        ("product_title", "product_title_copy", "string"),
        ("serial_number", "serial_number_dup", "string"),
    ],
    "rename_only": [
        ("first_name", "given_name", "string"),
        ("created_at", "created_ts", "date"),
        ("amount_local", "amount_value", "decimal"),
        ("customer_id", "cust_id", "int"),
        ("is_active", "active_flag", "boolean"),
        ("postal_code", "zip_code", "string"),
        ("phone_number", "contact_phone", "string"),
        ("hire_date", "employment_date", "date"),
        ("last_name", "surname", "string"),
        ("email_text", "email_address", "string"),
        ("city_name", "city", "string"),
        ("company_name", "organization", "string"),
        ("birth_date", "dob", "date"),
        ("department_name", "dept", "string"),
        ("ssn_text", "social_security", "string"),
        ("unit_price", "price_per_unit", "decimal"),
    ],
    "type_cast": [
        ("quantity_text", "quantity_int", "int"),
        ("amount_text", "amount_decimal", "decimal"),
        ("date_text", "signup_date", "date"),
        ("status_flag", "status_boolean", "boolean"),
        ("order_id", "order_id_text", "string"),
        ("price_text", "price_num", "decimal"),
        ("age_text", "age_int", "int"),
        ("rate_text", "rate_decimal", "decimal"),
        ("boolean_text", "is_active", "boolean"),
        ("timestamp_text", "event_timestamp", "date"),
        ("count_text", "count_num", "int"),
        ("score_string", "score_decimal", "decimal"),
    ],
    "coalesce": [
        ("middle_name", "middle_name_filled", "string"),
        ("phone_number", "phone_filled", "string"),
        ("email_text", "email_filled", "string"),
        ("amount_local", "amount_filled", "decimal"),
        ("birth_date", "birth_date_filled", "date"),
        ("address_raw", "address_filled", "string"),
        ("emergency_contact", "emergency_filled", "string"),
        ("secondary_phone", "contact_filled", "string"),
        ("alternate_email", "email_resolved", "string"),
    ],
    "default_replacement": [
        ("status_code", "status_code_default", "string"),
        ("category_raw", "category_default", "string"),
        ("risk_score", "risk_score_default", "decimal"),
        ("department_name", "dept_default", "string"),
        ("region_name", "region_default", "string"),
        ("currency_code", "currency_default", "string"),
        ("language_code", "language_default", "string"),
        ("priority_level", "priority_default", "int"),
        ("country_code", "country_default", "string"),
    ],
}


def gen_a_identity(rng, entity, serial, tt, label):
    if label == 1:
        templates = A_TEMPLATES.get(tt, A_TEMPLATES["direct_copy"])
        src_name, tgt_name, col_type = pick(rng, templates)
        src = make_col(rng, aug_src(rng, entity, src_name), col_type if tt != "type_cast" else "string")
        tgt_ent = rand_entropy(rng)
        return make_record([src], make_target(aug_tgt(rng, entity, tgt_name), col_type, tgt_ent), tt, f"{entity}_{tt}_{serial:06d}", 1)
    neg_kind = rng.randint(0, 2)
    if neg_kind == 0:
        return neg_type_mismatch(rng, entity, serial, tt, "string", "decimal")
    elif neg_kind == 1:
        return neg_semantic_mismatch(rng, entity, serial, tt)
    else:
        src = rand_col(rng, entity, pick(rng, ["string", "int", "decimal"]))
        tgt_name = nwe(pick(rng, ENTITIES), pick(rng, TYPE_POOLS[pick(rng, ["date", "boolean"])]))
        return make_record([src], make_target(tgt_name, "date", rand_entropy(rng)), tt, f"{entity}_{tt}_{serial:06d}", 0)


# ---------- B. String ----------

B_TEMPLATES = {
    "concat": [
        (["first_name", "last_name"], "full_name"),
        (["street_line", "city_name"], "full_address"),
        (["brand_name", "product_title"], "product_full"),
        (["first_name", "middle_name", "last_name"], "complete_name"),
        (["city_name", "region_name", "country_code"], "location_full"),
        (["title_raw", "first_name", "last_name"], "formal_name"),
        (["street_line", "city_name", "postal_code", "country_code"], "mailing_address"),
        (["area_code", "phone_number"], "full_phone"),
        (["department_name", "job_title"], "role_description"),
        (["first_name", "last_name", "email_text"], "contact_info"),
        (["year_part", "month_part", "day_part"], "date_string"),
        (["currency_code", "amount_local"], "formatted_amount"),
    ],
    "substring": [
        ("phone_number", "area_code"), ("postal_code", "zip_prefix"),
        ("serial_number", "serial_prefix"), ("ssn_text", "ssn_last4"),
        ("email_text", "email_local_part"), ("code_raw", "code_prefix"),
        ("account_number", "account_prefix"), ("iban_number", "country_prefix"),
        ("tracking_number", "carrier_code"), ("credit_card", "card_last4"),
        ("ip_address", "subnet_part"), ("license_plate", "state_code"),
        ("barcode", "barcode_prefix"), ("sku", "sku_category"),
    ],
    "split": [
        ("full_name", "first_token"), ("full_name", "last_token"),
        ("address_raw", "street_part"), ("file_path", "filename"),
        ("url_text", "host_part"), ("email_text", "email_user"),
        ("full_address", "city_part"), ("full_name", "middle_part"),
        ("reference_code", "ref_prefix"), ("mime_type", "type_part"),
        ("ip_address", "first_octet"), ("user_agent", "browser_name"),
        ("hostname", "subdomain"), ("tag_name", "tag_namespace"),
    ],
    "regex_extract": [
        ("email_text", "email_domain"), ("url_text", "url_host"),
        ("phone_number", "country_code_part"), ("address_raw", "zip_extracted"),
        ("description_text", "first_number"), ("comment_text", "hashtag_extracted"),
        ("user_agent", "version_number"), ("ip_address", "ip_class"),
        ("log_line", "timestamp_part"), ("html_content", "text_content"),
        ("json_text", "key_value"), ("csv_line", "field_extracted"),
        ("sql_query", "table_name_extracted"), ("error_message", "error_code_extracted"),
    ],
    "regex_replace": [
        ("comment_text", "comment_cleaned"), ("description_text", "description_sanitized"),
        ("note_text", "note_cleaned"), ("title_raw", "title_sanitized"),
        ("address_raw", "address_cleaned"), ("html_content", "plain_text"),
        ("phone_raw", "phone_formatted"), ("ssn_raw", "ssn_formatted"),
        ("log_entry", "log_sanitized"), ("user_input", "input_sanitized"),
        ("markdown_text", "text_stripped"), ("xml_content", "content_cleaned"),
        ("script_text", "script_sanitized"), ("query_string", "query_cleaned"),
    ],
    "lower": [
        ("email_raw", "email_lower"), ("status_code", "status_lower"),
        ("category_raw", "category_lower"), ("tag_name", "tag_lower"),
        ("username", "username_lower"), ("country_code", "country_lower"),
        ("brand_name", "brand_lower"), ("product_title", "title_lowercase"),
        ("department_name", "dept_lower"), ("hostname", "hostname_lower"),
        ("search_term", "search_normalized"), ("keyword", "keyword_lower"),
    ],
    "upper": [
        ("country_code", "country_upper"), ("status_code", "status_upper"),
        ("code_raw", "code_upper"), ("region_name", "region_upper"),
        ("currency_code", "currency_upper"), ("language_code", "lang_upper"),
        ("license_plate", "plate_upper"), ("iata_code", "iata_normalized"),
        ("swift_code", "swift_upper"), ("ticker_symbol", "ticker_upper"),
    ],
    "initcap": [
        ("first_name", "first_name_cap"), ("city_name", "city_cap"),
        ("company_name", "company_cap"), ("last_name", "last_name_cap"),
        ("street_line", "street_proper"), ("title_raw", "title_proper"),
        ("department_name", "dept_proper"), ("region_name", "region_proper"),
        ("brand_name", "brand_proper"), ("job_title", "job_title_proper"),
    ],
    "trim": [
        ("name_padded", "name_trimmed"), ("address_raw", "address_trimmed"),
        ("code_raw", "code_trimmed"), ("note_text", "note_trimmed"),
        ("description_text", "desc_trimmed"), ("comment_text", "comment_trimmed"),
        ("input_value", "input_cleaned"), ("csv_field", "field_trimmed"),
        ("user_input", "input_trimmed"), ("whitespace_text", "clean_text"),
        ("padded_string", "unpadded_string"), ("spaced_value", "compact_value"),
    ],
    "normalize_whitespace": [
        ("address_raw", "address_normalized"), ("description_text", "desc_normalized"),
        ("comment_text", "comment_normalized"), ("note_text", "note_normalized"),
        ("multiline_text", "single_line_text"), ("messy_input", "clean_input"),
        ("raw_paragraph", "normalized_paragraph"), ("body_text", "body_clean"),
        ("free_text", "formatted_text"), ("user_comment", "comment_formatted"),
    ],
    "replace_punctuation": [
        ("title_raw", "title_clean"), ("description_text", "desc_clean"),
        ("note_text", "note_clean"), ("comment_text", "comment_nopunct"),
        ("text_with_punctuation", "text_without_punctuation"),
        ("punctuated_text", "clean_text"), ("input_string", "sanitized_string"),
        ("raw_content", "filtered_content"), ("dirty_text", "stripped_text"),
        ("messy_text", "cleaned_text"), ("special_chars_text", "alphanumeric_text"),
        ("symbol_text", "plain_text"), ("formatted_text", "unformatted_text"),
        ("marked_up_text", "unmarked_text"), ("annotated_text", "bare_text"),
        ("decorated_text", "simple_text"), ("punctuation_field", "cleaned_field"),
        ("noisy_string", "clean_string"), ("raw_name", "normalized_name"),
        ("user_text_raw", "user_text_sanitized"),
    ],
    "lpad": [
        ("code_raw", "code_padded"), ("serial_number", "serial_padded"),
        ("postal_code", "postal_padded"), ("account_number", "account_padded"),
        ("employee_id_text", "emp_id_padded"), ("invoice_number", "invoice_padded"),
        ("batch_number", "batch_padded"), ("sequence_text", "sequence_padded"),
    ],
    "rpad": [
        ("code_raw", "code_rpadded"), ("serial_number", "serial_rpadded"),
        ("field_value", "field_fixed_width"), ("record_id_text", "record_padded"),
        ("name_text", "name_fixed_len"), ("description_text", "desc_fixed_width"),
    ],
    "slugify": [
        ("product_title", "product_slug"), ("title_raw", "title_slug"),
        ("company_name", "company_slug"), ("article_title", "article_slug"),
        ("page_title", "page_slug"), ("category_raw", "category_slug"),
        ("brand_name", "brand_slug"), ("tag_name", "tag_slug"),
        ("event_name", "event_slug"), ("blog_title", "blog_slug"),
    ],
    "email_parse": [
        ("email_text", "username"), ("email_text", "email_domain"),
        ("email_address", "email_user"), ("user_email", "domain_part"),
        ("contact_email", "email_local"), ("mail_address", "mail_domain"),
    ],
    "url_parse": [
        ("url_text", "url_host"), ("url_text", "url_path"),
        ("web_address", "protocol"), ("page_url", "query_string"),
        ("api_endpoint", "endpoint_path"), ("download_link", "filename_part"),
    ],
    "path_parse": [
        ("file_path", "file_extension"), ("file_path", "directory_name"),
        ("document_path", "document_name"), ("resource_path", "resource_type"),
        ("image_path", "image_filename"), ("log_path", "log_filename"),
    ],
    "anonymization": [
        ("ssn_text", "ssn_masked"), ("email_text", "email_anon"),
        ("phone_number", "phone_masked"), ("first_name", "name_anon"),
        ("credit_card", "card_anonymized"), ("address_raw", "address_anon"),
        ("account_number", "account_masked"), ("ip_address", "ip_anonymized"),
        ("last_name", "surname_masked"), ("birth_date", "dob_masked"),
    ],
}


def gen_b_string(rng, entity, serial, tt, label):
    if label == 1:
        templates = B_TEMPLATES.get(tt, [("text_field", "text_out")])
        if tt == "concat":
            src_names, tgt_name = pick(rng, templates)
            srcs = [make_col(rng, aug_src(rng, entity, s), "string") for s in src_names]
        else:
            if isinstance(templates[0], tuple) and len(templates[0]) == 2:
                src_name, tgt_name = pick(rng, templates)
            else:
                src_name, tgt_name = templates[0], templates[1] if len(templates) > 1 else "out"
            srcs = [make_col(rng, aug_src(rng, entity, src_name), "string")]
        tgt_ent = rand_entropy(rng)
        return make_record(srcs, make_target(aug_tgt(rng, entity, tgt_name), "string", tgt_ent), tt, f"{entity}_{tt}_{serial:06d}", 1)
    # Negatives: wrong type source, or semantically unrelated columns
    neg_kind = rng.randint(0, 3)
    if neg_kind == 0:
        return neg_type_mismatch(rng, entity, serial, tt, "string", "string")
    elif neg_kind == 1:
        return neg_semantic_mismatch(rng, entity, serial, tt)
    elif neg_kind == 2:
        # String source → wrong type target
        src = rand_col(rng, entity, "string")
        tgt_type = pick(rng, ["int", "decimal", "date", "boolean"])
        tgt_name = nwe(entity, pick(rng, TYPE_POOLS[tgt_type]))
        return make_record([src], make_target(tgt_name, tgt_type, rand_entropy(rng)), tt, f"{entity}_{tt}_{serial:06d}", 0)
    else:
        # Numeric source for a string transform
        src = rand_col(rng, entity, pick(rng, ["int", "decimal"]))
        tgt = rand_col(rng, entity, "string")
        return make_record([src], make_target(tgt["name"], tgt["type"], rand_entropy(rng)), tt, f"{entity}_{tt}_{serial:06d}", 0)


# ---------- C. Numeric ----------

C_TEMPLATES = {
    "add": [
        (["amount_local", "tax_amount"], "total_amount", ["decimal", "decimal"], "decimal"),
        (["unit_price", "commission"], "total_price", ["decimal", "decimal"], "decimal"),
        (["quantity", "batch_id"], "combined_qty", ["int", "int"], "int"),
        (["balance_local", "discount_amount"], "adjusted_balance", ["decimal", "decimal"], "decimal"),
        (["revenue", "interest_rate"], "total_revenue", ["decimal", "decimal"], "decimal"),
        (["shipping_cost", "handling_fee"], "delivery_cost", ["decimal", "decimal"], "decimal"),
        (["base_salary", "bonus_amount"], "total_compensation", ["decimal", "decimal"], "decimal"),
        (["principal", "interest"], "total_due", ["decimal", "decimal"], "decimal"),
    ],
    "subtract": [
        (["total_amount", "tax_amount"], "net_amount", ["decimal", "decimal"], "decimal"),
        (["balance_local", "discount_amount"], "final_balance", ["decimal", "decimal"], "decimal"),
        (["unit_price", "commission"], "net_price", ["decimal", "decimal"], "decimal"),
        (["revenue", "operating_cost"], "profit", ["decimal", "decimal"], "decimal"),
        (["gross_salary", "deductions"], "net_salary", ["decimal", "decimal"], "decimal"),
        (["total_amount", "refund_amount"], "remaining_balance", ["decimal", "decimal"], "decimal"),
    ],
    "multiply": [
        (["unit_price", "quantity"], "line_total", ["decimal", "int"], "decimal"),
        (["amount_local", "fx_rate"], "amount_converted", ["decimal", "decimal"], "decimal"),
        (["hourly_rate", "quantity"], "total_pay", ["decimal", "int"], "decimal"),
        (["base_rate", "multiplier"], "adjusted_rate", ["decimal", "decimal"], "decimal"),
        (["principal", "interest_rate"], "interest_amount", ["decimal", "decimal"], "decimal"),
        (["area_sqft", "price_per_sqft"], "property_value", ["decimal", "decimal"], "decimal"),
    ],
    "divide": [
        (["total_amount", "quantity"], "unit_price_calc", ["decimal", "int"], "decimal"),
        (["net_amount", "item_count"], "avg_item_price", ["decimal", "int"], "decimal"),
        (["total_cost", "num_items"], "cost_per_item", ["decimal", "int"], "decimal"),
        (["annual_salary", "months_worked"], "monthly_rate", ["decimal", "int"], "decimal"),
        (["total_distance", "total_time"], "avg_speed", ["decimal", "decimal"], "decimal"),
    ],
    "ratio_percentage": [
        (["net_amount", "total_amount"], "net_pct", ["decimal", "decimal"], "decimal"),
        (["discount_amount", "unit_price"], "discount_pct", ["decimal", "decimal"], "decimal"),
        (["profit", "revenue"], "profit_margin_pct", ["decimal", "decimal"], "decimal"),
        (["returns_count", "orders_count"], "return_rate", ["int", "int"], "decimal"),
        (["defect_count", "total_produced"], "defect_rate", ["int", "int"], "decimal"),
    ],
    "scaling_unit_conversion": [
        ("amount_local", "amount_usd", "decimal"), ("weight_kg", "weight_lb", "decimal"),
        ("temperature", "temperature_f", "decimal"), ("height_cm", "height_in", "decimal"),
        ("distance_km", "distance_miles", "decimal"), ("speed_kmh", "speed_mph", "decimal"),
        ("amount_cents", "amount_dollars", "decimal"), ("volume_liters", "volume_gallons", "decimal"),
        ("pressure_psi", "pressure_bar", "decimal"), ("energy_kwh", "energy_joules", "decimal"),
    ],
    "round": [
        ("risk_score", "risk_rounded", "decimal"), ("amount_local", "amount_rounded", "decimal"),
        ("pct_value", "pct_rounded", "decimal"), ("gpa_value", "gpa_rounded", "decimal"),
        ("exchange_rate", "rate_rounded", "decimal"), ("temperature", "temp_rounded", "decimal"),
        ("latitude", "lat_rounded", "decimal"), ("longitude", "lon_rounded", "decimal"),
    ],
    "floor": [
        ("risk_score", "risk_floor", "int"), ("amount_local", "amount_floor", "int"),
        ("price_value", "price_floor", "int"), ("rating", "rating_floor", "int"),
        ("temperature", "temp_floor", "int"), ("bmi_value", "bmi_floor", "int"),
    ],
    "ceil": [
        ("risk_score", "risk_ceil", "int"), ("amount_local", "amount_ceil", "int"),
        ("shipping_weight", "weight_ceil", "int"), ("time_hours", "hours_ceil", "int"),
        ("utilization_pct", "utilization_ceil", "int"),
    ],
    "trunc": [
        ("risk_score", "risk_trunc", "int"), ("pct_value", "pct_trunc", "int"),
        ("decimal_value", "integer_part", "int"), ("price_exact", "price_trunc", "int"),
    ],
    "clipping_winsorization": [
        ("amount_local", "amount_clipped", "decimal"), ("risk_score", "risk_clipped", "decimal"),
        ("outlier_value", "clipped_value", "decimal"), ("raw_score", "winsorized_score", "decimal"),
        ("salary_amount", "salary_capped", "decimal"), ("temperature", "temp_bounded", "decimal"),
    ],
    "abs": [
        ("amount_local", "amount_abs", "decimal"), ("score_raw", "score_abs", "decimal"),
        ("delta_value", "delta_magnitude", "decimal"), ("profit_loss", "amount_absolute", "decimal"),
        ("variance", "variance_abs", "decimal"), ("deviation", "abs_deviation", "decimal"),
    ],
    "sign": [
        ("amount_local", "amount_sign", "int"), ("score_raw", "score_sign", "int"),
        ("profit_loss", "is_profit", "int"), ("balance_change", "direction", "int"),
        ("temperature_delta", "warming_cooling", "int"),
    ],
    "log": [
        ("amount_local", "amount_log", "decimal"), ("score_raw", "score_log", "decimal"),
        ("population", "log_population", "decimal"), ("revenue", "log_revenue", "decimal"),
        ("page_views", "log_views", "decimal"), ("frequency", "log_frequency", "decimal"),
    ],
    "power": [
        ("score_raw", "score_sq", "decimal"), ("risk_score", "risk_sq", "decimal"),
        ("distance", "distance_squared", "decimal"), ("error", "error_squared", "decimal"),
        ("deviation", "variance", "decimal"),
    ],
    "min_max_normalize": [
        ("score_raw", "score_normalized", "decimal"), ("risk_score", "risk_normalized", "decimal"),
        ("amount_local", "amount_scaled", "decimal"), ("feature_value", "feature_norm", "decimal"),
        ("rating_value", "rating_normalized", "decimal"), ("price_raw", "price_normalized", "decimal"),
    ],
    "z_score_normalize": [
        ("score_raw", "score_zscore", "decimal"), ("amount_local", "amount_zscore", "decimal"),
        ("height_cm", "height_zscore", "decimal"), ("weight_kg", "weight_zscore", "decimal"),
        ("test_score", "standardized_score", "decimal"),
    ],
    "bucketing_binning": [
        ("amount_local", "amount_bucket", "string"), ("risk_score", "risk_tier", "string"),
        ("age_years", "age_band", "string"), ("income_level", "income_bracket", "string"),
        ("credit_score", "credit_tier", "string"), ("temperature", "temp_range", "string"),
        ("bmi_value", "bmi_category", "string"), ("revenue", "revenue_segment", "string"),
    ],
}


def gen_c_numeric(rng, entity, serial, tt, label):
    if label == 1:
        templates = C_TEMPLATES.get(tt, [("score_raw", "score_out", "decimal")])
        tmpl = pick(rng, templates)
        if isinstance(tmpl[0], list):
            src_names, tgt_name, src_types, tgt_type = tmpl
            srcs = [make_col(rng, aug_src(rng, entity, n), t) for n, t in zip(src_names, src_types)]
        else:
            src_name, tgt_name, tgt_type = tmpl
            srcs = [make_col(rng, aug_src(rng, entity, src_name), "decimal")]
        tgt_ent = rand_entropy(rng)
        return make_record(srcs, make_target(aug_tgt(rng, entity, tgt_name), tgt_type, tgt_ent), tt, f"{entity}_{tt}_{serial:06d}", 1)
    # Diverse negatives
    neg_kind = rng.randint(0, 3)
    if neg_kind == 0:
        # String source for numeric transform
        src = rand_col(rng, entity, "string")
        tgt = rand_col(rng, entity, "decimal")
        return make_record([src], make_target(tgt["name"], tgt["type"], rand_entropy(rng)), tt, f"{entity}_{tt}_{serial:06d}", 0)
    elif neg_kind == 1:
        # Date source for numeric transform
        src = rand_col(rng, entity, "date")
        tgt = rand_col(rng, entity, "decimal")
        return make_record([src], make_target(tgt["name"], tgt["type"], rand_entropy(rng)), tt, f"{entity}_{tt}_{serial:06d}", 0)
    elif neg_kind == 2:
        return neg_semantic_mismatch(rng, entity, serial, tt)
    else:
        # Numeric source but boolean/string target (wrong target type)
        src = rand_col(rng, entity, "decimal")
        tgt_type = pick(rng, ["boolean", "string", "date"])
        tgt_name = nwe(entity, pick(rng, TYPE_POOLS[tgt_type]))
        return make_record([src], make_target(tgt_name, tgt_type, rand_entropy(rng)), tt, f"{entity}_{tt}_{serial:06d}", 0)


# ---------- D. Date/time ----------

D_TEMPLATES = {
    "parse_datetime": [
        ("date_text", "parsed_date"), ("timestamp_text", "parsed_ts"), ("dob_text", "birth_date"),
        ("date_string", "date_value"), ("datetime_str", "datetime_val"), ("iso_date_text", "parsed_iso"),
        ("schedule_text", "schedule_date"), ("expiry_text", "expiry_date"), ("effective_text", "effective_date"),
    ],
    "format_datetime": [
        ("signup_date", "signup_yyyy_mm"), ("event_date", "event_formatted"), ("birth_date", "dob_text"),
        ("created_at", "created_display"), ("order_date", "order_date_str"), ("invoice_date", "invoice_display"),
        ("hire_date", "hire_date_formatted"), ("payment_date", "payment_display"),
    ],
    "extract_year": [
        ("event_date", "event_year"), ("birth_date", "birth_year"), ("hire_date", "hire_year"),
        ("order_date", "order_year"), ("created_at", "created_year"), ("txn_date", "txn_year"),
        ("signup_date", "signup_year"), ("invoice_date", "invoice_year"),
    ],
    "extract_quarter": [
        ("txn_date", "txn_quarter"), ("order_date", "order_quarter"),
        ("invoice_date", "invoice_quarter"), ("payment_date", "payment_quarter"),
        ("created_at", "created_quarter"), ("signup_date", "signup_quarter"),
    ],
    "extract_week": [
        ("event_date", "event_week"), ("order_date", "order_week"),
        ("txn_date", "txn_week"), ("ship_date", "ship_week"),
        ("created_at", "created_week"),
    ],
    "extract_day": [
        ("event_date", "event_day"), ("birth_date", "birth_day"),
        ("order_date", "order_day_of_month"), ("hire_date", "hire_day"),
        ("txn_date", "txn_day"), ("created_at", "created_day_num"),
    ],
    "extract_hour": [
        ("created_at", "created_hour"), ("event_date", "event_hour"),
        ("last_login", "login_hour"), ("txn_date", "txn_hour"),
        ("appointment_date", "appointment_hour"),
    ],
    "add_interval": [
        ("start_date", "end_date"), ("order_date", "expected_delivery"),
        ("hire_date", "probation_end"), ("subscription_start", "renewal_date"),
        ("payment_date", "next_payment"), ("effective_date", "expiry_date"),
        ("admission_date", "expected_discharge"), ("signup_date", "trial_end"),
    ],
    "subtract_interval": [
        ("end_date", "start_date_calc"), ("due_date", "reminder_date"),
        ("expiry_date", "warning_date"), ("renewal_date", "notice_date"),
        ("delivery_date", "cutoff_date"), ("payment_date", "grace_start"),
    ],
    "date_difference": [
        (["end_date", "start_date"], "days_between"),
        (["termination_date", "hire_date"], "tenure_days"),
        (["ship_date", "order_date"], "fulfillment_days"),
        (["discharge_date", "admission_date"], "length_of_stay"),
        (["birth_date", "created_at"], "age_at_signup"),
        (["due_date", "payment_date"], "days_overdue"),
        (["expiry_date", "effective_date"], "coverage_days"),
    ],
    "truncate_to_period": [
        ("txn_date", "month_start"), ("event_date", "week_start"),
        ("order_date", "quarter_start"), ("created_at", "year_start"),
        ("payment_date", "period_start"), ("last_activity", "activity_month"),
    ],
    "timezone_conversion": [
        ("created_at", "created_local"), ("event_date", "event_local"),
        ("txn_date", "txn_utc"), ("last_login", "login_local"),
        ("appointment_date", "appointment_local"), ("meeting_time", "local_time"),
    ],
    "fiscal_calendar": [
        ("txn_date", "fiscal_quarter"), ("order_date", "fiscal_year"),
        ("invoice_date", "fiscal_period"), ("payment_date", "fiscal_month"),
        ("created_at", "fiscal_week"), ("revenue_date", "fiscal_year_quarter"),
    ],
}


def gen_d_datetime(rng, entity, serial, tt, label):
    if label == 1:
        templates = D_TEMPLATES.get(tt, [("event_date", "date_out")])
        tmpl = pick(rng, templates)
        if tt == "parse_datetime":
            src_name, tgt_name = tmpl
            srcs = [make_col(rng, aug_src(rng, entity, src_name), "string")]
            tgt_type = "date"
        elif tt == "format_datetime":
            src_name, tgt_name = tmpl
            srcs = [make_col(rng, aug_src(rng, entity, src_name), "date")]
            tgt_type = "string"
        elif tt.startswith("extract_"):
            src_name, tgt_name = tmpl
            srcs = [make_col(rng, aug_src(rng, entity, src_name), "date")]
            tgt_type = "int"
        elif tt == "date_difference":
            src_names, tgt_name = tmpl
            srcs = [make_col(rng, aug_src(rng, entity, n), "date") for n in src_names]
            tgt_type = "int"
        elif tt == "fiscal_calendar":
            src_name, tgt_name = tmpl
            srcs = [make_col(rng, aug_src(rng, entity, src_name), "date")]
            tgt_type = "string"
        else:
            src_name, tgt_name = tmpl
            srcs = [make_col(rng, aug_src(rng, entity, src_name), "date")]
            tgt_type = "date"
        return make_record(srcs, make_target(aug_tgt(rng, entity, tgt_name), tgt_type, rand_entropy(rng)), tt, f"{entity}_{tt}_{serial:06d}", 1)
    # Negatives
    neg_kind = rng.randint(0, 3)
    if neg_kind == 0:
        # Numeric source for date transform
        src = rand_col(rng, entity, "decimal")
        tgt = rand_col(rng, entity, "date")
        return make_record([src], make_target(tgt["name"], tgt["type"], rand_entropy(rng)), tt, f"{entity}_{tt}_{serial:06d}", 0)
    elif neg_kind == 1:
        # String source for non-parse date transform
        src = rand_col(rng, entity, "string")
        tgt_type = pick(rng, ["date", "int", "string"])
        tgt_name = nwe(entity, pick(rng, TYPE_POOLS[tgt_type]))
        return make_record([src], make_target(tgt_name, tgt_type, rand_entropy(rng)), tt, f"{entity}_{tt}_{serial:06d}", 0)
    elif neg_kind == 2:
        return neg_semantic_mismatch(rng, entity, serial, tt)
    else:
        # Boolean source for date transform
        src = rand_col(rng, entity, "boolean")
        tgt_name = nwe(entity, pick(rng, DATE_NAMES))
        return make_record([src], make_target(tgt_name, "date", rand_entropy(rng)), tt, f"{entity}_{tt}_{serial:06d}", 0)


# ---------- E. Boolean / conditional ----------

E_TEMPLATES = {
    "threshold_flag": [
        ("amount_local", "is_high_value", "decimal"),
        ("risk_score", "is_risky", "decimal"),
        ("age_years", "is_adult", "int"),
        ("credit_limit", "is_high_credit", "decimal"),
        ("balance_local", "is_overdrawn", "decimal"),
        ("temperature", "is_fever", "decimal"),
        ("blood_pressure", "is_hypertensive", "decimal"),
        ("bmi_value", "is_obese", "decimal"),
        ("utilization_pct", "is_overloaded", "decimal"),
        ("tenure_months", "is_veteran", "int"),
    ],
    "equality_check": [
        ("status_code", "is_active", "string"),
        ("country_code", "is_domestic", "string"),
        ("gender_code", "is_male", "string"),
        ("currency_code", "is_usd", "string"),
        ("language_code", "is_english", "string"),
        ("priority_level", "is_urgent", "int"),
        ("payment_method", "is_credit_card", "string"),
    ],
    "range_check": [
        ("amount_local", "in_range_flag", "decimal"),
        ("age_years", "is_eligible_age", "int"),
        ("risk_score", "is_normal_risk", "decimal"),
        ("temperature", "is_normal_temp", "decimal"),
        ("credit_score", "is_good_credit", "int"),
        ("glucose_level", "is_normal_glucose", "decimal"),
        ("heart_rate_bpm", "is_normal_heart", "decimal"),
    ],
    "in_list_check": [
        ("status_code", "is_valid_status", "string"),
        ("country_code", "is_eu_country", "string"),
        ("category_raw", "is_core_category", "string"),
        ("department_name", "is_engineering", "string"),
        ("currency_code", "is_major_currency", "string"),
        ("state_code", "is_west_coast", "string"),
    ],
    "case_when_multi": [
        (["status_code", "amount_local"], "tier_flag"),
        (["risk_score", "is_verified"], "approval_flag"),
        (["category_raw", "quantity"], "priority_flag"),
        (["age_years", "income_level"], "eligibility_flag"),
        (["credit_score", "employment_status"], "loan_eligible"),
        (["severity_level", "priority_level"], "escalation_flag"),
    ],
    "null_presence_flag": [
        ("middle_name", "has_middle_name"),
        ("email_text", "has_email"),
        ("phone_number", "has_phone"),
        ("birth_date", "has_dob"),
        ("address_raw", "has_address"),
        ("ssn_text", "has_ssn"),
        ("emergency_contact", "has_emergency"),
        ("profile_photo", "has_photo"),
    ],
    "regex_match_flag": [
        ("email_text", "is_valid_email"),
        ("phone_number", "is_valid_phone"),
        ("postal_code", "is_valid_zip"),
        ("url_text", "is_valid_url"),
        ("ip_address", "is_valid_ip"),
        ("ssn_text", "is_valid_ssn"),
        ("iban_number", "is_valid_iban"),
        ("credit_card", "is_valid_card"),
    ],
}


def gen_e_boolean(rng, entity, serial, tt, label):
    if label == 1:
        templates = E_TEMPLATES.get(tt, [("status_code", "flag", "string")])
        tmpl = pick(rng, templates)
        if tt == "case_when_multi":
            src_names, tgt_name = tmpl
            src_type_map = {"status_code": "string", "amount_local": "decimal", "risk_score": "decimal",
                            "is_verified": "boolean", "category_raw": "string", "quantity": "int"}
            srcs = [make_col(rng, aug_src(rng, entity, n), src_type_map.get(n, "string")) for n in src_names]
        elif tt in ("null_presence_flag", "regex_match_flag"):
            src_name, tgt_name = tmpl
            srcs = [make_col(rng, aug_src(rng, entity, src_name), "string" if src_name not in ("birth_date",) else "date")]
        else:
            src_name, tgt_name, src_type = tmpl
            srcs = [make_col(rng, aug_src(rng, entity, src_name), src_type)]
        return make_record(srcs, make_target(aug_tgt(rng, entity, tgt_name), "boolean", rand_entropy(rng, 0.5, 1.5)), tt, f"{entity}_{tt}_{serial:06d}", 1)
    neg_kind = rng.randint(0, 3)
    if neg_kind == 0:
        return neg_type_mismatch(rng, entity, serial, tt, "string", "boolean")
    elif neg_kind == 1:
        # Boolean transform with non-boolean target
        src = rand_col(rng, entity, pick(rng, ["string", "decimal", "int"]))
        tgt_type = pick(rng, ["string", "decimal", "int", "date"])
        tgt_name = nwe(entity, pick(rng, TYPE_POOLS[tgt_type]))
        return make_record([src], make_target(tgt_name, tgt_type, rand_entropy(rng)), tt, f"{entity}_{tt}_{serial:06d}", 0)
    elif neg_kind == 2:
        return neg_semantic_mismatch(rng, entity, serial, tt)
    else:
        src = rand_col(rng, entity, "date")
        return make_record([src], make_target(nwe(entity, pick(rng, BOOL_NAMES)), "boolean", rand_entropy(rng, 0.5, 1.5)), tt, f"{entity}_{tt}_{serial:06d}", 0)


# ---------- F. Categorical / semantic ----------

F_TEMPLATES = {
    "code_to_label": [
        ("gender_code", "gender_label"), ("country_code", "country_name"),
        ("status_code", "status_label"), ("category_raw", "category_label"),
        ("region_name", "region_full_name"), ("department_name", "dept_label"),
        ("currency_code", "currency_name"), ("language_code", "language_name"),
        ("icd_code", "diagnosis_name"), ("state_code", "state_name"),
        ("severity_level", "severity_label"), ("priority_level", "priority_label"),
    ],
    "category_harmonization": [
        ("category_raw", "category_standard"), ("status_code", "status_harmonized"),
        ("brand_name", "brand_standard"), ("region_name", "region_standard"),
        ("department_name", "dept_harmonized"), ("job_title", "job_family"),
        ("product_title", "product_group"), ("diagnosis_text", "diagnosis_standard"),
        ("medication_name", "drug_class"), ("industry_code", "industry_standard"),
    ],
    "synonym_normalization": [
        ("product_title", "product_canonical"), ("company_name", "company_canonical"),
        ("city_name", "city_canonical"), ("department_name", "dept_canonical"),
        ("skill_name", "skill_canonical"), ("medication_name", "drug_canonical"),
        ("address_raw", "address_canonical"), ("certification", "cert_canonical"),
    ],
    "hierarchy_rollup": [
        ("city_name", "region_band"), ("department_name", "division_name"),
        ("product_title", "product_category"), ("region_name", "country_group"),
        ("sku", "product_line"), ("zip_code", "metro_area"),
        ("job_title", "job_level"), ("store_id_text", "district"),
        ("symptom_desc", "condition_group"), ("account_type", "account_class"),
    ],
}


def gen_f_categorical(rng, entity, serial, tt, label):
    if label == 1:
        templates = F_TEMPLATES.get(tt, [("code_raw", "label_out")])
        src_name, tgt_name = pick(rng, templates)
        src = make_col(rng, aug_src(rng, entity, src_name), "string")
        return make_record([src], make_target(aug_tgt(rng, entity, tgt_name), "string", rand_entropy(rng)), tt, f"{entity}_{tt}_{serial:06d}", 1)
    neg_kind = rng.randint(0, 3)
    if neg_kind == 0:
        return neg_type_mismatch(rng, entity, serial, tt, "string", "string")
    elif neg_kind == 1:
        # Numeric source for categorical transform
        src = rand_col(rng, entity, pick(rng, ["decimal", "int"]))
        tgt = rand_col(rng, entity, "string")
        return make_record([src], make_target(tgt["name"], tgt["type"], rand_entropy(rng)), tt, f"{entity}_{tt}_{serial:06d}", 0)
    elif neg_kind == 2:
        return neg_semantic_mismatch(rng, entity, serial, tt)
    else:
        src = rand_col(rng, entity, "date")
        tgt = rand_col(rng, entity, "string")
        return make_record([src], make_target(tgt["name"], tgt["type"], rand_entropy(rng)), tt, f"{entity}_{tt}_{serial:06d}", 0)


# ---------- G. Lookup/join ----------

G_TEMPLATES = {
    "fk_dimension_enrichment": [
        ("country_id", "country_name", "int", "string"),
        ("customer_id", "customer_name", "int", "string"),
        ("product_id", "product_title", "int", "string"),
        ("department_id", "department_name", "int", "string"),
        ("store_id", "store_name", "int", "string"),
        ("employee_id", "employee_name", "int", "string"),
        ("vendor_id", "vendor_name", "int", "string"),
        ("campaign_id", "campaign_name", "int", "string"),
        ("category_id", "category_name", "int", "string"),
        ("warehouse_id", "warehouse_location", "int", "string"),
    ],
    "multi_hop_lookup": [
        ("region_id", "country_name", "int", "string"),
        ("order_id", "customer_email", "int", "string"),
        ("product_id", "brand_name", "int", "string"),
        ("ticket_id", "assignee_email", "int", "string"),
        ("shipment_id", "destination_city", "int", "string"),
        ("invoice_id", "vendor_contact", "int", "string"),
        ("policy_id", "underwriter_name", "int", "string"),
    ],
    "fallback_lookup": [
        ("primary_code", "resolved_label", "string", "string"),
        ("email_text", "customer_name", "string", "string"),
        ("status_code", "status_label", "string", "string"),
        ("alternate_id", "resolved_name", "string", "string"),
        ("legacy_code", "current_code", "string", "string"),
        ("old_sku", "new_sku", "string", "string"),
    ],
}


def gen_g_lookup(rng, entity, serial, tt, label):
    if label == 1:
        templates = G_TEMPLATES.get(tt, [("lookup_id", "lookup_name", "int", "string")])
        src_name, tgt_name, src_type, tgt_type = pick(rng, templates)
        src = make_col(rng, aug_src(rng, entity, src_name), src_type)
        return make_record([src], make_target(aug_tgt(rng, entity, tgt_name), tgt_type, rand_entropy(rng)), tt, f"{entity}_{tt}_{serial:06d}", 1)
    neg_kind = rng.randint(0, 2)
    if neg_kind == 0:
        return neg_semantic_mismatch(rng, entity, serial, tt)
    elif neg_kind == 1:
        # String source for FK enrichment (should be int)
        src = rand_col(rng, entity, "string")
        tgt = rand_col(rng, entity, pick(rng, ["decimal", "date", "boolean"]))
        return make_record([src], make_target(tgt["name"], tgt["type"], rand_entropy(rng)), tt, f"{entity}_{tt}_{serial:06d}", 0)
    else:
        return neg_type_mismatch(rng, entity, serial, tt, "int", "string")


# ---------- H. Aggregation ----------

H_TEMPLATES = {
    "sum_agg": [
        ("amount_local", "total_amount"), ("unit_price", "revenue_total"),
        ("discount_amount", "total_discount"), ("quantity", "total_qty"),
        ("shipping_cost", "total_shipping"), ("tax_amount", "total_tax"),
        ("hours_worked", "total_hours"), ("commission", "total_commission"),
    ],
    "avg_agg": [
        ("amount_local", "avg_amount"), ("risk_score", "avg_risk"),
        ("unit_price", "avg_price"), ("rating_value", "avg_rating"),
        ("response_time", "avg_response"), ("age_years", "avg_age"),
        ("tenure_months", "avg_tenure"), ("salary", "avg_salary"),
    ],
    "count_agg": [
        ("order_id", "order_count"), ("customer_id", "customer_count"),
        ("product_id", "product_count"), ("ticket_id", "ticket_count"),
        ("claim_id", "claim_count"), ("session_id", "session_count"),
        ("event_id", "event_count"), ("error_id", "error_count"),
    ],
    "min_agg": [
        ("amount_local", "min_amount"), ("risk_score", "min_risk"),
        ("unit_price", "min_price"), ("temperature", "min_temp"),
        ("created_at", "earliest_date"), ("balance_local", "min_balance"),
    ],
    "max_agg": [
        ("amount_local", "max_amount"), ("risk_score", "max_risk"),
        ("unit_price", "max_price"), ("temperature", "max_temp"),
        ("updated_at", "latest_date"), ("balance_local", "max_balance"),
    ],
    "distinct_count": [
        ("customer_id", "unique_customers"), ("product_id", "unique_products"),
        ("status_code", "unique_statuses"), ("city_name", "unique_cities"),
        ("department_name", "unique_departments"), ("country_code", "unique_countries"),
        ("category_raw", "unique_categories"), ("vendor_id", "unique_vendors"),
    ],
    "string_agg": [
        ("tag_name", "tags_list"), ("status_code", "status_history"),
        ("category_raw", "categories_concat"), ("skill_name", "skills_list"),
        ("department_name", "departments_list"), ("product_title", "products_concat"),
        ("comment_text", "comments_combined"), ("note_text", "notes_combined"),
    ],
}


def gen_h_aggregation(rng, entity, serial, tt, label):
    if label == 1:
        templates = H_TEMPLATES.get(tt, [("value", "agg_value")])
        tmpl = pick(rng, templates)
        src_name, tgt_name = tmpl
        src_type = "int" if src_name.endswith("_id") else "string" if tt == "string_agg" else "decimal"
        tgt_type = "int" if tt in ("count_agg", "distinct_count") else "string" if tt == "string_agg" else "decimal"
        src = make_col(rng, aug_src(rng, entity, src_name), src_type)
        return make_record([src], make_target(aug_tgt(rng, entity, tgt_name), tgt_type, rand_entropy(rng)), tt, f"{entity}_{tt}_{serial:06d}", 1)
    neg_kind = rng.randint(0, 3)
    if neg_kind == 0:
        # String source for numeric aggregation
        src = rand_col(rng, entity, "string")
        tgt = rand_col(rng, entity, "decimal")
        return make_record([src], make_target(tgt["name"], tgt["type"], rand_entropy(rng)), tt, f"{entity}_{tt}_{serial:06d}", 0)
    elif neg_kind == 1:
        return neg_semantic_mismatch(rng, entity, serial, tt)
    elif neg_kind == 2:
        # Date source for sum (doesn't make sense)
        src = rand_col(rng, entity, "date")
        tgt = rand_col(rng, entity, pick(rng, ["decimal", "int"]))
        return make_record([src], make_target(tgt["name"], tgt["type"], rand_entropy(rng)), tt, f"{entity}_{tt}_{serial:06d}", 0)
    else:
        return neg_type_mismatch(rng, entity, serial, tt, "decimal", "decimal")


# ---------- I. Window ----------

I_TEMPLATES = {
    "row_number": [
        ("order_id", "row_num", "int", "int"), ("customer_id", "rank_num", "int", "int"),
        ("txn_date", "sequence_num", "date", "int"), ("created_at", "record_num", "date", "int"),
        ("event_id", "event_seq", "int", "int"), ("session_id", "visit_seq", "int", "int"),
    ],
    "rank": [
        ("amount_local", "amount_rank", "decimal", "int"), ("risk_score", "risk_rank", "decimal", "int"),
        ("score_raw", "score_rank", "decimal", "int"), ("revenue", "revenue_rank", "decimal", "int"),
        ("rating_value", "rating_rank", "decimal", "int"), ("performance_score", "perf_rank", "decimal", "int"),
    ],
    "dense_rank": [
        ("score_raw", "score_drank", "decimal", "int"), ("amount_local", "amount_drank", "decimal", "int"),
        ("salary", "salary_drank", "decimal", "int"), ("gpa_value", "gpa_drank", "decimal", "int"),
    ],
    "lag": [
        ("amount_local", "prev_amount", "decimal", "decimal"), ("unit_price", "prev_price", "decimal", "decimal"),
        ("balance_local", "prev_balance", "decimal", "decimal"), ("status_code", "prev_status", "string", "string"),
        ("risk_score", "prev_risk", "decimal", "decimal"), ("quantity", "prev_qty", "int", "int"),
    ],
    "lead": [
        ("amount_local", "next_amount", "decimal", "decimal"), ("risk_score", "next_risk", "decimal", "decimal"),
        ("balance_local", "next_balance", "decimal", "decimal"), ("status_code", "next_status", "string", "string"),
        ("unit_price", "next_price", "decimal", "decimal"), ("event_date", "next_event", "date", "date"),
    ],
    "cumulative_sum": [
        ("amount_local", "running_total", "decimal", "decimal"), ("quantity", "running_qty", "int", "int"),
        ("revenue", "cumulative_revenue", "decimal", "decimal"), ("cost", "cumulative_cost", "decimal", "decimal"),
        ("hours_worked", "running_hours", "decimal", "decimal"),
    ],
    "cumulative_count": [
        ("order_id", "running_count", "int", "int"), ("customer_id", "cum_customers", "int", "int"),
        ("event_id", "cum_events", "int", "int"), ("ticket_id", "cum_tickets", "int", "int"),
    ],
    "moving_avg": [
        ("amount_local", "moving_avg_7", "decimal", "decimal"), ("risk_score", "moving_avg_30", "decimal", "decimal"),
        ("temperature", "temp_moving_avg", "decimal", "decimal"), ("unit_price", "price_sma", "decimal", "decimal"),
        ("revenue", "revenue_ma_7d", "decimal", "decimal"), ("click_count", "clicks_moving_avg", "int", "decimal"),
    ],
    "window_stats": [
        ("amount_local", "window_std", "decimal", "decimal"), ("score_raw", "window_var", "decimal", "decimal"),
        ("unit_price", "price_stddev", "decimal", "decimal"), ("revenue", "revenue_variance", "decimal", "decimal"),
    ],
}


def gen_i_window(rng, entity, serial, tt, label):
    if label == 1:
        templates = I_TEMPLATES.get(tt, [("value", "window_value", "decimal", "decimal")])
        src_name, tgt_name, src_type, tgt_type = pick(rng, templates)
        src = make_col(rng, aug_src(rng, entity, src_name), src_type)
        return make_record([src], make_target(aug_tgt(rng, entity, tgt_name), tgt_type, rand_entropy(rng)), tt, f"{entity}_{tt}_{serial:06d}", 1)
    neg_kind = rng.randint(0, 2)
    if neg_kind == 0:
        src = rand_col(rng, entity, "string")
        tgt = rand_col(rng, entity, "int")
        return make_record([src], make_target(tgt["name"], tgt["type"], rand_entropy(rng)), tt, f"{entity}_{tt}_{serial:06d}", 0)
    elif neg_kind == 1:
        return neg_semantic_mismatch(rng, entity, serial, tt)
    else:
        src = rand_col(rng, entity, "date")
        tgt = rand_col(rng, entity, "decimal")
        return make_record([src], make_target(tgt["name"], tgt["type"], rand_entropy(rng)), tt, f"{entity}_{tt}_{serial:06d}", 0)


# ---------- J. Semi-structured ----------

J_TEMPLATES = {
    "json_extract": [
        ("payload_json", "extracted_field", "string"), ("config_json", "config_value", "string"),
        ("metadata_json", "meta_field", "string"), ("api_response", "response_data", "string"),
        ("event_payload", "event_type", "string"), ("user_preferences", "preference_value", "string"),
        ("audit_data", "changed_field", "string"), ("settings_json", "setting_value", "string"),
        ("profile_json", "bio_text", "string"), ("geojson_data", "coordinates", "string"),
    ],
    "xml_extract": [
        ("payload_xml", "xml_field", "string"), ("config_xml", "xml_value", "string"),
        ("soap_response", "result_value", "string"), ("feed_xml", "item_title", "string"),
        ("report_xml", "summary_text", "string"), ("message_xml", "body_content", "string"),
    ],
    "array_index": [
        ("tags_array", "first_tag", "string"), ("items_json", "first_item", "string"),
        ("categories_array", "primary_category", "string"), ("scores_array", "highest_score", "decimal"),
        ("addresses_array", "primary_address", "string"), ("phones_array", "primary_phone", "string"),
    ],
    "explode_flatten_aggregate": [
        ("items_json", "item_count", "int"), ("tags_array", "tag_count", "int"),
        ("line_items", "line_count", "int"), ("attachments", "attachment_count", "int"),
        ("participants", "participant_count", "int"), ("nested_records", "flat_count", "int"),
    ],
}


def gen_j_semistructured(rng, entity, serial, tt, label):
    if label == 1:
        templates = J_TEMPLATES.get(tt, [("payload", "field_out", "string")])
        src_name, tgt_name, tgt_type = pick(rng, templates)
        src = make_col(rng, aug_src(rng, entity, src_name), "string")
        return make_record([src], make_target(aug_tgt(rng, entity, tgt_name), tgt_type, rand_entropy(rng)), tt, f"{entity}_{tt}_{serial:06d}", 1)
    neg_kind = rng.randint(0, 2)
    if neg_kind == 0:
        src = rand_col(rng, entity, pick(rng, ["decimal", "int", "date"]))
        tgt = rand_col(rng, entity, "string")
        return make_record([src], make_target(tgt["name"], tgt["type"], rand_entropy(rng)), tt, f"{entity}_{tt}_{serial:06d}", 0)
    elif neg_kind == 1:
        return neg_semantic_mismatch(rng, entity, serial, tt)
    else:
        return neg_type_mismatch(rng, entity, serial, tt, "string", "string")


# ---------- K. Data quality ----------

K_TEMPLATES = {
    "dedup_canonical": [
        ("customer_name", "canonical_name"), ("company_name", "company_canonical"),
        ("product_title", "product_canonical"), ("address_raw", "address_canonical"),
        ("city_name", "city_canonical"), ("brand_name", "brand_canonical"),
        ("vendor_name", "vendor_canonical"), ("hospital_name", "hospital_canonical"),
    ],
    "missing_imputation": [
        ("middle_name", "middle_imputed"), ("email_text", "email_imputed"),
        ("phone_number", "phone_imputed"), ("birth_date", "dob_imputed"),
        ("address_raw", "address_imputed"), ("income_level", "income_imputed"),
        ("gender_code", "gender_imputed"), ("zip_code", "zip_imputed"),
        ("emergency_contact", "emergency_filled"), ("blood_type", "blood_type_imputed"),
    ],
    "invalid_standardization": [
        ("status_raw", "status_standard"), ("category_raw", "category_clean"),
        ("gender_code", "gender_standard"), ("country_code", "country_standard"),
        ("phone_raw", "phone_standard"), ("date_text", "date_standard"),
        ("currency_code", "currency_standard"), ("state_code", "state_standard"),
    ],
}


def gen_k_data_quality(rng, entity, serial, tt, label):
    if label == 1:
        templates = K_TEMPLATES.get(tt, [("value", "value_clean")])
        src_name, tgt_name = pick(rng, templates)
        col_type = "date" if "date" in src_name or "dob" in src_name else "string"
        src = make_col(rng, aug_src(rng, entity, src_name), col_type)
        return make_record([src], make_target(aug_tgt(rng, entity, tgt_name), col_type, rand_entropy(rng)), tt, f"{entity}_{tt}_{serial:06d}", 1)
    neg_kind = rng.randint(0, 2)
    if neg_kind == 0:
        src = rand_col(rng, entity, pick(rng, ["int", "decimal"]))
        tgt = rand_col(rng, entity, "string")
        return make_record([src], make_target(tgt["name"], tgt["type"], rand_entropy(rng)), tt, f"{entity}_{tt}_{serial:06d}", 0)
    elif neg_kind == 1:
        return neg_semantic_mismatch(rng, entity, serial, tt)
    else:
        return neg_type_mismatch(rng, entity, serial, tt, "string", "string")


# ---------- L. Security/privacy ----------

L_TEMPLATES = {
    "hash": [
        ("email_text", "email_hash"), ("ssn_text", "ssn_hash"), ("phone_number", "phone_hash"),
        ("customer_id", "customer_hash"), ("password_text", "password_hashed"),
        ("api_key", "key_hash"), ("account_number", "account_hash"),
        ("credit_card", "card_hash"), ("national_id", "id_hash"),
        ("ip_address", "ip_hash"), ("device_id", "device_hash"),
    ],
    "tokenization": [
        ("ssn_text", "ssn_token"), ("email_text", "email_token"), ("phone_number", "phone_token"),
        ("credit_card", "card_token"), ("account_number", "account_token"),
        ("patient_id", "patient_token"), ("policy_number", "policy_token"),
    ],
    "masking": [
        ("ssn_text", "ssn_masked"), ("phone_number", "phone_masked"), ("email_text", "email_masked"),
        ("credit_card", "card_masked"), ("account_number", "account_masked"),
        ("name_text", "name_masked"), ("address_raw", "address_masked"),
        ("ip_address", "ip_masked"), ("license_plate", "plate_masked"),
    ],
    "redaction": [
        ("ssn_text", "ssn_redacted"), ("comment_text", "comment_redacted"),
        ("address_raw", "address_redacted"), ("medical_notes", "notes_redacted"),
        ("legal_text", "legal_redacted"), ("diagnosis_text", "diagnosis_redacted"),
        ("financial_details", "details_redacted"), ("personal_info", "info_redacted"),
    ],
    "deterministic_encryption": [
        ("pii_field", "pii_encrypted"), ("ssn_text", "ssn_encrypted"),
        ("email_text", "email_encrypted"), ("phone_number", "phone_encrypted"),
        ("credit_card", "card_encrypted"), ("medical_record", "record_encrypted"),
        ("salary", "salary_encrypted"), ("account_number", "account_encrypted"),
    ],
}


def gen_l_security(rng, entity, serial, tt, label):
    if label == 1:
        templates = L_TEMPLATES.get(tt, [("sensitive", "sensitive_protected")])
        src_name, tgt_name = pick(rng, templates)
        src_type = "int" if src_name.endswith("_id") else "string"
        src = make_col(rng, aug_src(rng, entity, src_name), src_type)
        return make_record([src], make_target(aug_tgt(rng, entity, tgt_name), "string", rand_entropy(rng)), tt, f"{entity}_{tt}_{serial:06d}", 1)
    neg_kind = rng.randint(0, 2)
    if neg_kind == 0:
        src = rand_col(rng, entity, pick(rng, ["decimal", "date"]))
        tgt = rand_col(rng, entity, "string")
        return make_record([src], make_target(tgt["name"], tgt["type"], rand_entropy(rng)), tt, f"{entity}_{tt}_{serial:06d}", 0)
    elif neg_kind == 1:
        return neg_semantic_mismatch(rng, entity, serial, tt)
    else:
        return neg_type_mismatch(rng, entity, serial, tt, "string", "string")


# ---------- Config ----------
TRANSFORM_CONFIG: List[Tuple[str, str, int, int]] = [
    # A. Identity/structural
    ("a", "direct_copy", DEFAULT_POS, DEFAULT_NEG),
    ("a", "rename_only", DEFAULT_POS, DEFAULT_NEG),
    ("a", "type_cast", DEFAULT_POS, DEFAULT_NEG),
    ("a", "coalesce", DEFAULT_POS, DEFAULT_NEG),
    ("a", "default_replacement", DEFAULT_POS, DEFAULT_NEG),
    # B. String
    ("b", "concat", DEFAULT_POS, DEFAULT_NEG),
    ("b", "substring", DEFAULT_POS, DEFAULT_NEG),
    ("b", "split", DEFAULT_POS, DEFAULT_NEG),
    ("b", "regex_extract", DEFAULT_POS, DEFAULT_NEG),
    ("b", "regex_replace", DEFAULT_POS, DEFAULT_NEG),
    ("b", "lower", DEFAULT_POS, DEFAULT_NEG),
    ("b", "upper", DEFAULT_POS, DEFAULT_NEG),
    ("b", "initcap", DEFAULT_POS, DEFAULT_NEG),
    ("b", "trim", DEFAULT_POS, DEFAULT_NEG),
    ("b", "normalize_whitespace", DEFAULT_POS, DEFAULT_NEG),
    ("b", "replace_punctuation", DEFAULT_POS, DEFAULT_NEG),
    ("b", "lpad", DEFAULT_POS, DEFAULT_NEG),
    ("b", "rpad", DEFAULT_POS, DEFAULT_NEG),
    ("b", "slugify", DEFAULT_POS, DEFAULT_NEG),
    ("b", "email_parse", DEFAULT_POS, DEFAULT_NEG),
    ("b", "url_parse", DEFAULT_POS, DEFAULT_NEG),
    ("b", "path_parse", DEFAULT_POS, DEFAULT_NEG),
    ("b", "anonymization", DEFAULT_POS, DEFAULT_NEG),
    # C. Numeric
    ("c", "add", DEFAULT_POS, DEFAULT_NEG),
    ("c", "subtract", DEFAULT_POS, DEFAULT_NEG),
    ("c", "multiply", DEFAULT_POS, DEFAULT_NEG),
    ("c", "divide", DEFAULT_POS, DEFAULT_NEG),
    ("c", "ratio_percentage", DEFAULT_POS, DEFAULT_NEG),
    ("c", "scaling_unit_conversion", DEFAULT_POS, DEFAULT_NEG),
    ("c", "round", DEFAULT_POS, DEFAULT_NEG),
    ("c", "floor", DEFAULT_POS, DEFAULT_NEG),
    ("c", "ceil", DEFAULT_POS, DEFAULT_NEG),
    ("c", "trunc", DEFAULT_POS, DEFAULT_NEG),
    ("c", "clipping_winsorization", DEFAULT_POS, DEFAULT_NEG),
    ("c", "abs", DEFAULT_POS, DEFAULT_NEG),
    ("c", "sign", DEFAULT_POS, DEFAULT_NEG),
    ("c", "log", DEFAULT_POS, DEFAULT_NEG),
    ("c", "power", DEFAULT_POS, DEFAULT_NEG),
    ("c", "min_max_normalize", DEFAULT_POS, DEFAULT_NEG),
    ("c", "z_score_normalize", DEFAULT_POS, DEFAULT_NEG),
    ("c", "bucketing_binning", DEFAULT_POS, DEFAULT_NEG),
    # D. Date/time
    ("d", "parse_datetime", DEFAULT_POS, DEFAULT_NEG),
    ("d", "format_datetime", DEFAULT_POS, DEFAULT_NEG),
    ("d", "extract_year", DEFAULT_POS, DEFAULT_NEG),
    ("d", "extract_quarter", DEFAULT_POS, DEFAULT_NEG),
    ("d", "extract_week", DEFAULT_POS, DEFAULT_NEG),
    ("d", "extract_day", DEFAULT_POS, DEFAULT_NEG),
    ("d", "extract_hour", DEFAULT_POS, DEFAULT_NEG),
    ("d", "add_interval", DEFAULT_POS, DEFAULT_NEG),
    ("d", "subtract_interval", DEFAULT_POS, DEFAULT_NEG),
    ("d", "date_difference", DEFAULT_POS, DEFAULT_NEG),
    ("d", "truncate_to_period", DEFAULT_POS, DEFAULT_NEG),
    ("d", "timezone_conversion", DEFAULT_POS, DEFAULT_NEG),
    ("d", "fiscal_calendar", DEFAULT_POS, DEFAULT_NEG),
    # E. Boolean
    ("e", "threshold_flag", DEFAULT_POS, DEFAULT_NEG),
    ("e", "equality_check", DEFAULT_POS, DEFAULT_NEG),
    ("e", "range_check", DEFAULT_POS, DEFAULT_NEG),
    ("e", "in_list_check", DEFAULT_POS, DEFAULT_NEG),
    ("e", "case_when_multi", DEFAULT_POS, DEFAULT_NEG),
    ("e", "null_presence_flag", DEFAULT_POS, DEFAULT_NEG),
    ("e", "regex_match_flag", DEFAULT_POS, DEFAULT_NEG),
    # F. Categorical
    ("f", "code_to_label", DEFAULT_POS, DEFAULT_NEG),
    ("f", "category_harmonization", DEFAULT_POS, DEFAULT_NEG),
    ("f", "synonym_normalization", DEFAULT_POS, DEFAULT_NEG),
    ("f", "hierarchy_rollup", DEFAULT_POS, DEFAULT_NEG),
    # G. Lookup
    ("g", "fk_dimension_enrichment", DEFAULT_POS, DEFAULT_NEG),
    ("g", "multi_hop_lookup", DEFAULT_POS, DEFAULT_NEG),
    ("g", "fallback_lookup", DEFAULT_POS, DEFAULT_NEG),
    # H. Aggregation
    ("h", "sum_agg", DEFAULT_POS, DEFAULT_NEG),
    ("h", "avg_agg", DEFAULT_POS, DEFAULT_NEG),
    ("h", "count_agg", DEFAULT_POS, DEFAULT_NEG),
    ("h", "min_agg", DEFAULT_POS, DEFAULT_NEG),
    ("h", "max_agg", DEFAULT_POS, DEFAULT_NEG),
    ("h", "distinct_count", DEFAULT_POS, DEFAULT_NEG),
    ("h", "string_agg", DEFAULT_POS, DEFAULT_NEG),
    # I. Window
    ("i", "row_number", DEFAULT_POS, DEFAULT_NEG),
    ("i", "rank", DEFAULT_POS, DEFAULT_NEG),
    ("i", "dense_rank", DEFAULT_POS, DEFAULT_NEG),
    ("i", "lag", DEFAULT_POS, DEFAULT_NEG),
    ("i", "lead", DEFAULT_POS, DEFAULT_NEG),
    ("i", "cumulative_sum", DEFAULT_POS, DEFAULT_NEG),
    ("i", "cumulative_count", DEFAULT_POS, DEFAULT_NEG),
    ("i", "moving_avg", DEFAULT_POS, DEFAULT_NEG),
    ("i", "window_stats", DEFAULT_POS, DEFAULT_NEG),
    # J. Semi-structured
    ("j", "json_extract", DEFAULT_POS, DEFAULT_NEG),
    ("j", "xml_extract", DEFAULT_POS, DEFAULT_NEG),
    ("j", "array_index", DEFAULT_POS, DEFAULT_NEG),
    ("j", "explode_flatten_aggregate", DEFAULT_POS, DEFAULT_NEG),
    # K. Data quality
    ("k", "dedup_canonical", DEFAULT_POS, DEFAULT_NEG),
    ("k", "missing_imputation", DEFAULT_POS, DEFAULT_NEG),
    ("k", "invalid_standardization", DEFAULT_POS, DEFAULT_NEG),
    # L. Security
    ("l", "hash", DEFAULT_POS, DEFAULT_NEG),
    ("l", "tokenization", DEFAULT_POS, DEFAULT_NEG),
    ("l", "masking", DEFAULT_POS, DEFAULT_NEG),
    ("l", "redaction", DEFAULT_POS, DEFAULT_NEG),
    ("l", "deterministic_encryption", DEFAULT_POS, DEFAULT_NEG),
]


def dispatch(cat, rng, entity, serial, tt, label):
    fn = {"a": gen_a_identity, "b": gen_b_string, "c": gen_c_numeric,
          "d": gen_d_datetime, "e": gen_e_boolean, "f": gen_f_categorical,
          "g": gen_g_lookup, "h": gen_h_aggregation, "i": gen_i_window,
          "j": gen_j_semistructured, "k": gen_k_data_quality, "l": gen_l_security}[cat]
    return fn(rng, entity, serial, tt, label)


# ---------- Hard negatives ----------

HARD_NEG_SWAPS = [
    ("customer_name", "string", "product_code", "string"),
    ("order_date", "date", "invoice_date", "date"),
    ("amount_local", "decimal", "fx_rate", "decimal"),
    ("employee_id", "int", "department_id", "int"),
    ("first_name", "string", "last_name", "string"),
    ("email_text", "string", "phone_number", "string"),
    ("city_name", "string", "country_code", "string"),
    ("quantity", "int", "line_number", "int"),
    ("risk_score", "decimal", "credit_limit", "decimal"),
    ("signup_date", "date", "birth_date", "date"),
    ("status_code", "string", "department_code", "string"),
    ("is_active", "boolean", "is_primary", "boolean"),
    ("balance_local", "decimal", "discount_amount", "decimal"),
    ("postal_code", "string", "serial_number", "string"),
    ("product_title", "string", "diagnosis_text", "string"),
    ("total_amount", "decimal", "weight_kg", "decimal"),
    ("hire_date", "date", "ship_date", "date"),
    ("customer_id", "int", "product_id", "int"),
    ("address_raw", "string", "url_text", "string"),
    ("company_name", "string", "tag_name", "string"),
    ("hourly_rate", "decimal", "temperature", "decimal"),
    ("unit_price", "decimal", "latitude", "decimal"),
    ("comment_text", "string", "ssn_text", "string"),
    ("brand_name", "string", "file_path", "string"),
    ("age_years", "int", "batch_id", "int"),
    ("order_id", "int", "version_num", "int"),
    ("score_raw", "decimal", "margin_pct", "decimal"),
    ("start_date", "date", "due_date", "date"),
    ("created_at", "date", "modified_at", "date"),
    ("is_verified", "boolean", "is_deleted", "boolean"),
]


def build_hard_negative(rng, entity, serial, tt):
    swap = pick(rng, HARD_NEG_SWAPS)
    src_name, src_type, tgt_name, tgt_type = swap
    src = make_col(rng, nwe(entity, src_name), src_type)
    tgt = make_target(nwe(pick(rng, ENTITIES), tgt_name), tgt_type, rand_entropy(rng))
    return make_record([src], tgt, tt, f"{entity}_{tt}_{serial:06d}", 0)


CROSS_SRC = [
    ("amount_local", "decimal"), ("first_name", "string"), ("event_date", "date"),
    ("status_code", "string"), ("is_active", "boolean"), ("customer_id", "int"),
    ("risk_score", "decimal"), ("phone_number", "string"), ("quantity", "int"),
    ("balance_local", "decimal"), ("email_text", "string"), ("hire_date", "date"),
    ("product_title", "string"), ("order_id", "int"), ("ssn_text", "string"),
]
CROSS_TGT = [
    ("total_amount", "decimal"), ("full_name", "string"), ("event_year", "int"),
    ("status_upper", "string"), ("is_valid", "boolean"), ("customer_name", "string"),
    ("risk_rounded", "decimal"), ("phone_masked", "string"), ("running_qty", "int"),
    ("net_amount", "decimal"), ("email_domain", "string"), ("tenure_days", "int"),
    ("product_slug", "string"), ("order_count", "int"), ("ssn_hash", "string"),
]


def build_cross_type_negative(rng, entity, serial, tt):
    all_types = [t for _, t, _, _ in TRANSFORM_CONFIG if t != tt]
    wrong_tt = pick(rng, all_types)  # noqa: F841 (just for documentation)
    src_name, src_type = pick(rng, CROSS_SRC)
    tgt_name, tgt_type = pick(rng, CROSS_TGT)
    src = make_col(rng, nwe(entity, src_name), src_type)
    tgt = make_target(nwe(entity, tgt_name), tgt_type, rand_entropy(rng))
    return make_record([src], tgt, tt, f"{entity}_{tt}_{serial:06d}", 0)


# ---------- Main generator ----------

def generate_all():
    rng = random.Random(SEED)
    records = []
    serial = 0
    for cat, tt, n_pos, n_neg in TRANSFORM_CONFIG:
        for _ in range(n_pos):
            serial += 1
            entity = pick(rng, ENTITIES)
            try:
                rec = dispatch(cat, rng, entity, serial, tt, 1)
                rec["transform_name"] = f"{entity}_{tt}_{serial:06d}"
                records.append(rec)
            except Exception:
                pass
        # Split negatives: 40% easy, 30% hard, 30% cross-type
        n_easy = max(1, int(n_neg * 0.4))
        n_hard = max(1, int(n_neg * 0.3))
        n_cross = n_neg - n_easy - n_hard
        for _ in range(n_easy):
            serial += 1
            entity = pick(rng, ENTITIES)
            try:
                rec = dispatch(cat, rng, entity, serial, tt, 0)
                rec["transform_name"] = f"{entity}_{tt}_{serial:06d}"
                records.append(rec)
            except Exception:
                pass
        for _ in range(n_hard):
            serial += 1
            entity = pick(rng, ENTITIES)
            try:
                records.append(build_hard_negative(rng, entity, serial, tt))
            except Exception:
                pass
        for _ in range(n_cross):
            serial += 1
            entity = pick(rng, ENTITIES)
            try:
                records.append(build_cross_type_negative(rng, entity, serial, tt))
            except Exception:
                pass
    rng.shuffle(records)
    return records


def main():
    records = generate_all()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    labels = Counter(r["label"] for r in records)
    types = Counter(r["transform_type"] for r in records)
    print(f"Wrote {len(records)} records to {OUTPUT_PATH}")
    print(f"Labels: {dict(labels)}")
    print(f"Transform types: {len(types)}")


if __name__ == "__main__":
    main()
