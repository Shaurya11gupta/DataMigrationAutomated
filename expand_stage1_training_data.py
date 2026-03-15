#!/usr/bin/env python3
"""
Generate rich Stage-1 candidate-selector training data with hard negatives.

Output format matches candidate_selector_stage1.py loader:
  - query_id
  - split
  - domain
  - target {...}
  - candidate_set {...}
  - label

Additional metadata fields are included for traceability:
  - db_reference
  - scenario
  - hard_negative_type
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


SEED = 20260210
QUERY_COUNT = 2000
NEGATIVES_PER_QUERY = 5
OUTPUT_PATH = Path("/workspace/stage1_training_input_sample.jsonl")


def pluralize(word: str) -> str:
    if word.endswith("y") and len(word) > 1:
        return word[:-1] + "ies"
    if word.endswith("s"):
        return word + "es"
    return word + "s"


def assign_split(query_id: str) -> str:
    h = int(hashlib.md5(query_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if h < 80:
        return "train"
    if h < 90:
        return "val"
    return "test"


def safe_choice(rng: random.Random, arr: Sequence[Any]) -> Any:
    return arr[rng.randrange(len(arr))]


def normalize_type(t: str) -> str:
    t = t.lower()
    if t in {"varchar", "text", "char"}:
        return "string"
    if t in {"int", "integer", "bigint", "smallint"}:
        return "int"
    if t in {"float", "double", "decimal", "numeric", "number"}:
        return "decimal"
    if t in {"datetime", "timestamp"}:
        return "date"
    if t in {"bool"}:
        return "boolean"
    return t


def incompatible_types(target_type: str) -> List[str]:
    t = normalize_type(target_type)
    if t == "string":
        return ["int", "decimal", "date", "boolean"]
    if t in {"int", "decimal"}:
        return ["string", "date", "boolean"]
    if t == "date":
        return ["int", "decimal", "boolean"]
    if t == "boolean":
        return ["string", "date", "decimal"]
    return ["string", "int", "decimal", "date", "boolean"]


@dataclass
class DomainProfile:
    domain: str
    db_reference: str
    entity: str
    event: str
    txn: str
    aux: str
    lookup: str
    tables: Dict[str, str]
    role_columns: Dict[str, List[Dict[str, str]]]
    role_edges: List[Tuple[str, str, str, str]]


def build_profile(domain: str, db_reference: str, entity: str, event: str, txn: str, aux: str, lookup: str) -> DomainProfile:
    tables = {
        "main": f"src_{pluralize(entity)}",
        "event": f"src_{pluralize(event)}",
        "txn": f"src_{pluralize(txn)}",
        "aux": f"src_{pluralize(aux)}",
        "lookup": f"dim_{lookup}",
    }

    ctx = {
        "entity": entity,
        "event": event,
        "txn": txn,
        "aux": aux,
        "lookup": lookup,
        "entity_id": f"{entity}_id",
        "event_id": f"{event}_id",
        "txn_id": f"{txn}_id",
        "aux_id": f"{aux}_id",
        "lookup_id": f"{lookup}_id",
    }

    role_columns = {
        "main": [
            {"name": ctx["entity_id"], "type": "int", "description": f"Primary identifier for {entity}"},
            {"name": ctx["lookup_id"], "type": "int", "description": f"{lookup} dimension foreign key"},
            {"name": "first_name", "type": "string", "description": f"{entity} first name"},
            {"name": "last_name", "type": "string", "description": f"{entity} last name"},
            {"name": "middle_name", "type": "string", "description": f"{entity} middle name"},
            {"name": "display_name", "type": "string", "description": f"display label for {entity}"},
            {"name": "email", "type": "string", "description": "primary email address"},
            {"name": "username", "type": "string", "description": "login username"},
            {"name": "phone_number", "type": "string", "description": "national phone number"},
            {"name": "country_code", "type": "string", "description": "ISO country code"},
            {"name": "city", "type": "string", "description": "city name"},
            {"name": "state_code", "type": "string", "description": "state or province code"},
            {"name": "postal_code", "type": "string", "description": "postal code"},
            {"name": "department_code", "type": "string", "description": "department code"},
            {"name": "status_code", "type": "string", "description": "status code"},
            {"name": "status_text", "type": "string", "description": "status text"},
            {"name": "signup_date", "type": "date", "description": "registration date"},
            {"name": "birth_date", "type": "date", "description": "date of birth"},
            {"name": "created_at_text", "type": "string", "description": "raw created timestamp text"},
            {"name": "risk_score", "type": "decimal", "description": "risk score"},
            {"name": "balance_local", "type": "decimal", "description": "balance in local currency"},
            {"name": "credit_limit", "type": "decimal", "description": "credit limit"},
            {"name": "age", "type": "int", "description": "age in years"},
            {"name": "is_active", "type": "boolean", "description": "active status flag"},
        ],
        "event": [
            {"name": ctx["event_id"], "type": "int", "description": f"Primary identifier for {event}"},
            {"name": ctx["entity_id"], "type": "int", "description": f"{entity} reference"},
            {"name": ctx["lookup_id"], "type": "int", "description": f"{lookup} reference"},
            {"name": ctx["txn_id"], "type": "int", "description": f"{txn} reference"},
            {"name": "event_date", "type": "date", "description": f"{event} date"},
            {"name": "created_at_text", "type": "string", "description": "raw event timestamp text"},
            {"name": "closed_at", "type": "date", "description": "event close date"},
            {"name": "status_text", "type": "string", "description": "event status text"},
            {"name": "status_code", "type": "string", "description": "event status code"},
            {"name": "amount_local", "type": "decimal", "description": "amount in local currency"},
            {"name": "gross_amount", "type": "decimal", "description": "gross amount"},
            {"name": "discount_amount", "type": "decimal", "description": "discount amount"},
            {"name": "revenue_amount", "type": "decimal", "description": "revenue amount"},
            {"name": "cost_amount", "type": "decimal", "description": "cost amount"},
            {"name": "quantity", "type": "int", "description": "quantity units"},
            {"name": "used_units", "type": "int", "description": "used units"},
            {"name": "total_units", "type": "int", "description": "total units"},
            {"name": "diag_code", "type": "string", "description": "diagnosis or classification code"},
            {"name": "service_code", "type": "string", "description": "service code"},
            {"name": "ticket_subject", "type": "string", "description": "subject text"},
            {"name": "country_code", "type": "string", "description": "event country code"},
            {"name": "is_primary", "type": "boolean", "description": "primary marker"},
        ],
        "txn": [
            {"name": ctx["txn_id"], "type": "int", "description": f"Primary identifier for {txn}"},
            {"name": ctx["entity_id"], "type": "int", "description": f"{entity} reference"},
            {"name": ctx["lookup_id"], "type": "int", "description": f"{lookup} reference"},
            {"name": "currency_code", "type": "string", "description": "currency code"},
            {"name": "amount_local", "type": "decimal", "description": "amount in local currency"},
            {"name": "net_amount", "type": "decimal", "description": "net amount"},
            {"name": "tax_amount", "type": "decimal", "description": "tax amount"},
            {"name": "gross_amount", "type": "decimal", "description": "gross amount"},
            {"name": "discount_amount", "type": "decimal", "description": "discount amount"},
            {"name": "quantity", "type": "int", "description": "quantity units"},
            {"name": "fx_rate", "type": "decimal", "description": "exchange rate"},
            {"name": "txn_date", "type": "date", "description": f"{txn} date"},
            {"name": "due_date", "type": "date", "description": "due date"},
            {"name": "reference_code", "type": "string", "description": "reference code"},
            {"name": "invoice_number", "type": "string", "description": "invoice number"},
            {"name": "status_code", "type": "string", "description": "transaction status"},
            {"name": "is_reversal", "type": "boolean", "description": "reversal flag"},
        ],
        "aux": [
            {"name": ctx["aux_id"], "type": "int", "description": f"Primary identifier for {aux}"},
            {"name": ctx["event_id"], "type": "int", "description": f"{event} reference"},
            {"name": ctx["txn_id"], "type": "int", "description": f"{txn} reference"},
            {"name": "item_id", "type": "int", "description": "item id"},
            {"name": "line_no", "type": "int", "description": "line number"},
            {"name": "file_path", "type": "string", "description": "full file path"},
            {"name": "file_name", "type": "string", "description": "file name"},
            {"name": "msisdn", "type": "string", "description": "mobile number"},
            {"name": "imei", "type": "string", "description": "device imei"},
            {"name": "plan_code", "type": "string", "description": "plan code"},
            {"name": "product_code", "type": "string", "description": "product code"},
            {"name": "warehouse_code", "type": "string", "description": "warehouse code"},
            {"name": "batch_code", "type": "string", "description": "batch code"},
            {"name": "quantity", "type": "int", "description": "item quantity"},
            {"name": "amount_local", "type": "decimal", "description": "line amount"},
            {"name": "created_at_text", "type": "string", "description": "raw aux created timestamp text"},
        ],
        "lookup": [
            {"name": ctx["lookup_id"], "type": "int", "description": f"Primary identifier for {lookup}"},
            {"name": "lookup_code", "type": "string", "description": "lookup code"},
            {"name": "lookup_name", "type": "string", "description": "lookup label"},
            {"name": "country_name", "type": "string", "description": "country full name"},
            {"name": "category_name", "type": "string", "description": "category name"},
            {"name": "group_name", "type": "string", "description": "group name"},
            {"name": "diagnosis_group", "type": "string", "description": "diagnosis grouping"},
            {"name": "department_code", "type": "string", "description": "department code"},
            {"name": "department_name", "type": "string", "description": "department name"},
            {"name": "service_category", "type": "string", "description": "service category"},
            {"name": "plan_name", "type": "string", "description": "plan name"},
            {"name": "risk_band", "type": "string", "description": "risk band"},
            {"name": "fx_rate", "type": "decimal", "description": "exchange rate"},
            {"name": "active_flag", "type": "boolean", "description": "active lookup flag"},
        ],
    }

    role_edges = [
        ("event", "main", ctx["entity_id"], ctx["entity_id"]),
        ("txn", "main", ctx["entity_id"], ctx["entity_id"]),
        ("aux", "event", ctx["event_id"], ctx["event_id"]),
        ("aux", "txn", ctx["txn_id"], ctx["txn_id"]),
        ("event", "txn", ctx["txn_id"], ctx["txn_id"]),
        ("main", "lookup", ctx["lookup_id"], ctx["lookup_id"]),
        ("event", "lookup", ctx["lookup_id"], ctx["lookup_id"]),
        ("txn", "lookup", ctx["lookup_id"], ctx["lookup_id"]),
    ]

    return DomainProfile(
        domain=domain,
        db_reference=db_reference,
        entity=entity,
        event=event,
        txn=txn,
        aux=aux,
        lookup=lookup,
        tables=tables,
        role_columns=role_columns,
        role_edges=role_edges,
    )


PROFILES = [
    build_profile("retail", "northwind", "customer", "order", "invoice", "order_item", "country"),
    build_profile("healthcare", "mimic_iv", "patient", "admission", "claim", "procedure", "diagnosis"),
    build_profile("finance", "tpch", "account", "payment", "transaction", "ledger_entry", "branch"),
    build_profile("it", "adventureworks", "user", "ticket", "session", "deployment", "department"),
    build_profile("telecom", "wideworldimporters", "subscriber", "usage", "billing", "cdr", "plan"),
    build_profile("manufacturing", "adventureworks", "worker", "workorder", "costing", "batch_item", "plant"),
    build_profile("logistics", "wideworldimporters", "shipper", "shipment", "freight", "manifest", "warehouse"),
    build_profile("education", "sakila", "student", "enrollment", "payment", "grade_item", "course"),
    build_profile("media", "chinook", "listener", "stream", "invoice", "playlist_item", "genre"),
    build_profile("public_sector", "open_data", "citizen", "service_case", "fee", "document", "district"),
]


SCENARIOS = [
    {
        "id": "full_name_concat",
        "target_kind": "dim_entity",
        "target_type": "string",
        "target_names": ["full_name", "{entity}_full_name", "display_full_name"],
        "transform_hint": "concat_space",
        "description": "Canonical full name for {entity} analytics in {domain} ({db_reference})",
        "positive": [("main", "first_name"), ("main", "last_name")],
    },
    {
        "id": "full_name_with_middle",
        "target_kind": "dim_entity",
        "target_type": "string",
        "target_names": ["full_name_with_middle", "{entity}_name_long"],
        "transform_hint": "format_name_with_middle",
        "description": "Extended name including middle token for {entity}",
        "positive": [("main", "first_name"), ("main", "middle_name"), ("main", "last_name")],
    },
    {
        "id": "email_username_split",
        "target_kind": "dim_entity",
        "target_type": "string",
        "target_names": ["email_username", "{entity}_email_user", "login_name_seed"],
        "transform_hint": "split_at",
        "description": "Extract local part from {entity} email",
        "positive": [("main", "email")],
    },
    {
        "id": "email_domain_split",
        "target_kind": "dim_entity",
        "target_type": "string",
        "target_names": ["email_domain", "{entity}_email_domain"],
        "transform_hint": "split_at_domain",
        "description": "Extract domain from primary email",
        "positive": [("main", "email")],
    },
    {
        "id": "city_state_display",
        "target_kind": "dim_entity",
        "target_type": "string",
        "target_names": ["city_state_display", "{entity}_city_state"],
        "transform_hint": "concat_comma",
        "description": "Create city and state display label",
        "positive": [("main", "city"), ("main", "state_code")],
    },
    {
        "id": "e164_phone",
        "target_kind": "dim_entity",
        "target_type": "string",
        "target_names": ["e164_phone", "phone_global_format"],
        "transform_hint": "concat_plus",
        "description": "Build international phone format",
        "positive": [("main", "country_code"), ("main", "phone_number")],
    },
    {
        "id": "normalized_status",
        "target_kind": "fct_event",
        "target_type": "string",
        "target_names": ["status_normalized", "{event}_status_norm"],
        "transform_hint": "lower_trim",
        "description": "Normalize free-text status for {event}",
        "positive": [("event", "status_text")],
    },
    {
        "id": "postal_prefix",
        "target_kind": "dim_entity",
        "target_type": "string",
        "target_names": ["postal_prefix", "zip_prefix"],
        "transform_hint": "substring_prefix3",
        "description": "Extract postal region prefix",
        "positive": [("main", "postal_code")],
    },
    {
        "id": "file_extension",
        "target_kind": "fct_aux",
        "target_type": "string",
        "target_names": ["file_extension", "document_ext"],
        "transform_hint": "regex_extract_extension",
        "description": "Extract file extension from path",
        "positive": [("aux", "file_path")],
    },
    {
        "id": "department_code_upper",
        "target_kind": "dim_lookup",
        "target_type": "string",
        "target_names": ["department_code_upper", "dept_code_norm"],
        "transform_hint": "upper",
        "description": "Standardize department code to uppercase",
        "positive": [("lookup", "department_code")],
    },
    {
        "id": "net_amount_calc",
        "target_kind": "fct_txn",
        "target_type": "decimal",
        "target_names": ["net_amount_calculated", "amount_after_discount"],
        "transform_hint": "arithmetic_subtract",
        "description": "Compute net amount from gross and discount",
        "positive": [("txn", "gross_amount"), ("txn", "discount_amount")],
    },
    {
        "id": "total_amount_with_tax",
        "target_kind": "fct_txn",
        "target_type": "decimal",
        "target_names": ["gross_with_tax", "total_amount_tax_inclusive"],
        "transform_hint": "arithmetic_add",
        "description": "Compute total amount including tax",
        "positive": [("txn", "net_amount"), ("txn", "tax_amount")],
    },
    {
        "id": "utilization_ratio",
        "target_kind": "fct_event",
        "target_type": "decimal",
        "target_names": ["utilization_ratio", "{event}_usage_ratio"],
        "transform_hint": "ratio",
        "description": "Calculate usage ratio from used and total units",
        "positive": [("event", "used_units"), ("event", "total_units")],
    },
    {
        "id": "unit_price",
        "target_kind": "fct_txn",
        "target_type": "decimal",
        "target_names": ["unit_price", "avg_unit_cost"],
        "transform_hint": "ratio",
        "description": "Estimate unit price from amount and quantity",
        "positive": [("txn", "amount_local"), ("txn", "quantity")],
    },
    {
        "id": "amount_usd",
        "target_kind": "fct_txn",
        "target_type": "decimal",
        "target_names": ["amount_usd", "amount_base_currency"],
        "transform_hint": "lookup_fx_rate",
        "description": "Convert local amount using lookup exchange rate",
        "positive": [("txn", "amount_local"), ("lookup", "fx_rate")],
    },
    {
        "id": "event_year",
        "target_kind": "fct_event",
        "target_type": "int",
        "target_names": ["event_year", "{event}_year"],
        "transform_hint": "date_part_year",
        "description": "Extract year from event date",
        "positive": [("event", "event_date")],
    },
    {
        "id": "signup_month",
        "target_kind": "dim_entity",
        "target_type": "int",
        "target_names": ["signup_month", "registration_month"],
        "transform_hint": "date_part_month",
        "description": "Extract month from signup date",
        "positive": [("main", "signup_date")],
    },
    {
        "id": "parsed_created_date",
        "target_kind": "fct_event",
        "target_type": "date",
        "target_names": ["created_date_parsed", "event_created_date"],
        "transform_hint": "parse_date",
        "description": "Parse raw created timestamp text to date",
        "positive": [("event", "created_at_text")],
    },
    {
        "id": "is_high_risk",
        "target_kind": "dim_entity",
        "target_type": "boolean",
        "target_names": ["is_high_risk", "high_risk_flag"],
        "transform_hint": "conditional_threshold",
        "description": "Flag records with high risk score",
        "positive": [("main", "risk_score")],
    },
    {
        "id": "is_active_standardized",
        "target_kind": "dim_entity",
        "target_type": "boolean",
        "target_names": ["is_active_standardized", "active_flag_standard"],
        "transform_hint": "conditional_status",
        "description": "Infer active flag from status code",
        "positive": [("main", "status_code")],
    },
    {
        "id": "diagnosis_group_lookup",
        "target_kind": "fct_event",
        "target_type": "string",
        "target_names": ["diagnosis_group", "classification_group"],
        "transform_hint": "lookup_join",
        "description": "Map diagnosis code to diagnosis group from dimension",
        "positive": [("event", "diag_code"), ("lookup", "diagnosis_group")],
    },
    {
        "id": "country_name_lookup",
        "target_kind": "dim_entity",
        "target_type": "string",
        "target_names": ["country_name", "country_label"],
        "transform_hint": "lookup_join",
        "description": "Enrich country code to full country name",
        "positive": [("main", "country_code"), ("lookup", "country_name")],
    },
    {
        "id": "lookup_name_enrichment",
        "target_kind": "dim_entity",
        "target_type": "string",
        "target_names": ["lookup_name", "{lookup}_name_enriched"],
        "transform_hint": "lookup_join",
        "description": "Enrich main records with lookup label",
        "positive": [("main", "{lookup_id}"), ("lookup", "lookup_name")],
    },
    {
        "id": "department_name_lookup",
        "target_kind": "dim_entity",
        "target_type": "string",
        "target_names": ["department_name", "dept_name_enriched"],
        "transform_hint": "lookup_join",
        "description": "Resolve department name using lookup dimension",
        "positive": [("main", "{lookup_id}"), ("lookup", "department_name")],
    },
    {
        "id": "event_category_lookup",
        "target_kind": "fct_event",
        "target_type": "string",
        "target_names": ["event_category_name", "category_from_lookup"],
        "transform_hint": "lookup_join",
        "description": "Map event records to category via lookup key",
        "positive": [("event", "{lookup_id}"), ("lookup", "category_name")],
    },
    {
        "id": "plan_name_lookup",
        "target_kind": "fct_event",
        "target_type": "string",
        "target_names": ["plan_name", "plan_label_enriched"],
        "transform_hint": "lookup_join",
        "description": "Resolve plan label from lookup table",
        "positive": [("event", "{lookup_id}"), ("lookup", "plan_name")],
    },
    {
        "id": "service_lookup_label",
        "target_kind": "fct_event",
        "target_type": "string",
        "target_names": ["service_lookup_label", "service_dimension_label"],
        "transform_hint": "lookup_join",
        "description": "Attach lookup-driven service label",
        "positive": [("event", "{lookup_id}"), ("lookup", "service_category")],
    },
    {
        "id": "service_category_lookup",
        "target_kind": "fct_event",
        "target_type": "string",
        "target_names": ["service_category", "service_group_label"],
        "transform_hint": "lookup_join",
        "description": "Map service code to service category",
        "positive": [("event", "service_code"), ("lookup", "service_category")],
    },
    {
        "id": "total_event_amount_agg",
        "target_kind": "fct_event",
        "target_type": "decimal",
        "target_names": ["total_event_amount", "sum_amount_per_{entity}"],
        "transform_hint": "aggregate_sum",
        "description": "Aggregate event amount by {entity}",
        "positive": [("event", "amount_local"), ("event", "{entity_id}")],
    },
    {
        "id": "item_count_agg",
        "target_kind": "fct_event",
        "target_type": "int",
        "target_names": ["item_count", "line_item_count"],
        "transform_hint": "aggregate_count",
        "description": "Count item rows per event",
        "positive": [("aux", "item_id"), ("aux", "{event_id}")],
    },
    {
        "id": "business_key_concat",
        "target_kind": "dim_entity",
        "target_type": "string",
        "target_names": ["business_key", "{entity}_business_key"],
        "transform_hint": "concat_dash",
        "description": "Build business key from department and id",
        "positive": [("main", "department_code"), ("main", "{entity_id}")],
    },
    {
        "id": "service_ticket_label",
        "target_kind": "fct_event",
        "target_type": "string",
        "target_names": ["service_ticket_label", "service_subject_label"],
        "transform_hint": "format",
        "description": "Compose service and subject label for event",
        "positive": [("event", "service_code"), ("event", "ticket_subject")],
    },
    {
        "id": "age_band",
        "target_kind": "dim_entity",
        "target_type": "string",
        "target_names": ["age_band", "age_group"],
        "transform_hint": "conditional_bucket",
        "description": "Bucket age into standard age bands",
        "positive": [("main", "age")],
    },
    {
        "id": "fiscal_quarter_label",
        "target_kind": "fct_txn",
        "target_type": "string",
        "target_names": ["fiscal_quarter_label", "quarter_label"],
        "transform_hint": "date_to_quarter_format",
        "description": "Generate quarter label from transaction date",
        "positive": [("txn", "txn_date")],
    },
    {
        "id": "recent_activity_flag",
        "target_kind": "fct_event",
        "target_type": "boolean",
        "target_names": ["recent_activity_flag", "is_recent_activity"],
        "transform_hint": "conditional_recent",
        "description": "Flag recently closed events",
        "positive": [("event", "closed_at")],
    },
    {
        "id": "normalized_reference",
        "target_kind": "fct_txn",
        "target_type": "string",
        "target_names": ["reference_normalized", "reference_clean"],
        "transform_hint": "trim_upper",
        "description": "Normalize reference code format",
        "positive": [("txn", "reference_code")],
    },
    {
        "id": "username_lower_norm",
        "target_kind": "dim_entity",
        "target_type": "string",
        "target_names": ["username_normalized", "username_lower"],
        "transform_hint": "lower",
        "description": "Normalize username casing",
        "positive": [("main", "username")],
    },
    {
        "id": "display_name_trimmed",
        "target_kind": "dim_entity",
        "target_type": "string",
        "target_names": ["display_name_clean", "display_name_trimmed"],
        "transform_hint": "trim",
        "description": "Trim display label edge whitespace",
        "positive": [("main", "display_name")],
    },
    {
        "id": "main_created_date_parsed",
        "target_kind": "dim_entity",
        "target_type": "date",
        "target_names": ["entity_created_date", "created_date_standard"],
        "transform_hint": "parse_date",
        "description": "Parse main created timestamp text into date",
        "positive": [("main", "created_at_text")],
    },
    {
        "id": "birth_year",
        "target_kind": "dim_entity",
        "target_type": "int",
        "target_names": ["birth_year", "{entity}_birth_year"],
        "transform_hint": "date_part_year",
        "description": "Extract birth year from birth date",
        "positive": [("main", "birth_date")],
    },
    {
        "id": "tenure_days",
        "target_kind": "fct_event",
        "target_type": "int",
        "target_names": ["tenure_days", "{entity}_tenure_days"],
        "transform_hint": "date_diff",
        "description": "Difference between event date and signup date",
        "positive": [("main", "signup_date"), ("event", "event_date")],
    },
    {
        "id": "risk_band_lookup",
        "target_kind": "dim_entity",
        "target_type": "string",
        "target_names": ["risk_band_name", "risk_band_lookup"],
        "transform_hint": "lookup_join",
        "description": "Fetch risk band from lookup using lookup key",
        "positive": [("main", "{lookup_id}"), ("lookup", "risk_band")],
    },
    {
        "id": "lookup_active_flag",
        "target_kind": "dim_entity",
        "target_type": "boolean",
        "target_names": ["lookup_active_flag", "is_lookup_active"],
        "transform_hint": "lookup_join",
        "description": "Bring active flag from lookup dimension",
        "positive": [("main", "{lookup_id}"), ("lookup", "active_flag")],
    },
    {
        "id": "event_country_name_lookup",
        "target_kind": "fct_event",
        "target_type": "string",
        "target_names": ["event_country_name", "country_name_enriched"],
        "transform_hint": "lookup_join",
        "description": "Enrich event records with country name",
        "positive": [("event", "{lookup_id}"), ("lookup", "country_name")],
    },
    {
        "id": "event_margin_amount",
        "target_kind": "fct_event",
        "target_type": "decimal",
        "target_names": ["margin_amount", "revenue_minus_cost"],
        "transform_hint": "arithmetic_subtract",
        "description": "Compute event margin from revenue and cost",
        "positive": [("event", "revenue_amount"), ("event", "cost_amount")],
    },
    {
        "id": "event_discount_ratio",
        "target_kind": "fct_event",
        "target_type": "decimal",
        "target_names": ["event_discount_ratio", "discount_share"],
        "transform_hint": "ratio",
        "description": "Discount ratio using discount and gross amount",
        "positive": [("event", "discount_amount"), ("event", "gross_amount")],
    },
    {
        "id": "event_amount_per_unit",
        "target_kind": "fct_event",
        "target_type": "decimal",
        "target_names": ["amount_per_unit", "avg_amount_per_unit"],
        "transform_hint": "ratio",
        "description": "Average amount per unit for event",
        "positive": [("event", "amount_local"), ("event", "quantity")],
    },
    {
        "id": "invoice_number_clean",
        "target_kind": "fct_txn",
        "target_type": "string",
        "target_names": ["invoice_number_clean", "invoice_ref_clean"],
        "transform_hint": "trim_upper",
        "description": "Normalize invoice number format",
        "positive": [("txn", "invoice_number")],
    },
    {
        "id": "due_days",
        "target_kind": "fct_txn",
        "target_type": "int",
        "target_names": ["due_days", "days_to_due"],
        "transform_hint": "date_diff",
        "description": "Difference between due date and transaction date",
        "positive": [("txn", "due_date"), ("txn", "txn_date")],
    },
    {
        "id": "txn_month",
        "target_kind": "fct_txn",
        "target_type": "int",
        "target_names": ["txn_month", "transaction_month"],
        "transform_hint": "date_part_month",
        "description": "Extract month from transaction date",
        "positive": [("txn", "txn_date")],
    },
    {
        "id": "reversal_flag_standard",
        "target_kind": "fct_txn",
        "target_type": "boolean",
        "target_names": ["reversal_flag_standard", "is_reversal_standard"],
        "transform_hint": "identity",
        "description": "Standardized reversal boolean flag",
        "positive": [("txn", "is_reversal")],
    },
    {
        "id": "line_amount_usd",
        "target_kind": "fct_aux",
        "target_type": "decimal",
        "target_names": ["line_amount_usd", "aux_amount_base_currency"],
        "transform_hint": "lookup_fx_rate",
        "description": "Convert auxiliary amount using lookup exchange rate",
        "positive": [("aux", "amount_local"), ("lookup", "fx_rate")],
    },
    {
        "id": "file_basename",
        "target_kind": "fct_aux",
        "target_type": "string",
        "target_names": ["file_basename", "document_name_no_ext"],
        "transform_hint": "split",
        "description": "Extract base file name from full path",
        "positive": [("aux", "file_path")],
    },
    {
        "id": "imei_prefix",
        "target_kind": "fct_aux",
        "target_type": "string",
        "target_names": ["imei_prefix", "device_tac"],
        "transform_hint": "substring_prefix8",
        "description": "Extract imei prefix/tac segment",
        "positive": [("aux", "imei")],
    },
    {
        "id": "msisdn_normalized",
        "target_kind": "fct_aux",
        "target_type": "string",
        "target_names": ["msisdn_normalized", "phone_normalized"],
        "transform_hint": "trim",
        "description": "Normalize mobile number text",
        "positive": [("aux", "msisdn")],
    },
    {
        "id": "event_group_lookup",
        "target_kind": "fct_event",
        "target_type": "string",
        "target_names": ["event_group_name", "group_lookup_name"],
        "transform_hint": "lookup_join",
        "description": "Map event records to group name from lookup",
        "positive": [("event", "{lookup_id}"), ("lookup", "group_name")],
    },
    {
        "id": "quantity_bucket",
        "target_kind": "fct_event",
        "target_type": "string",
        "target_names": ["quantity_bucket", "usage_bucket"],
        "transform_hint": "conditional_bucket",
        "description": "Bucket event quantity into usage classes",
        "positive": [("event", "quantity")],
    },
    {
        "id": "dept_service_business_key",
        "target_kind": "fct_event",
        "target_type": "string",
        "target_names": ["dept_service_key", "department_service_business_key"],
        "transform_hint": "concat_dash",
        "description": "Compose business key from department and service code",
        "positive": [("main", "department_code"), ("event", "service_code")],
    },
    {
        "id": "country_from_lookup_main",
        "target_kind": "dim_entity",
        "target_type": "string",
        "target_names": ["country_from_lookup", "country_label_lookup"],
        "transform_hint": "lookup_join",
        "description": "Use lookup to enrich country label for main records",
        "positive": [("main", "{lookup_id}"), ("lookup", "country_name")],
    },
]


JOIN_HEAVY_SCENARIO_IDS = {
    s["id"]
    for s in SCENARIOS
    if len({role for role, _ in s["positive"]}) > 1
}


def make_context(profile: DomainProfile) -> Dict[str, str]:
    return {
        "domain": profile.domain,
        "db_reference": profile.db_reference,
        "entity": profile.entity,
        "event": profile.event,
        "txn": profile.txn,
        "aux": profile.aux,
        "lookup": profile.lookup,
        "entity_id": f"{profile.entity}_id",
        "event_id": f"{profile.event}_id",
        "txn_id": f"{profile.txn}_id",
        "aux_id": f"{profile.aux}_id",
        "lookup_id": f"{profile.lookup}_id",
    }


def resolve_text(template: str, ctx: Dict[str, str]) -> str:
    return template.format(**ctx)


def role_column_type(profile: DomainProfile, role: str, column_name: str) -> str:
    for c in profile.role_columns[role]:
        if c["name"] == column_name:
            return c["type"]
    # fallback heuristic
    if column_name.endswith("_id") or column_name in {"quantity", "line_no", "age"}:
        return "int"
    if "date" in column_name or "at" in column_name:
        return "date"
    if "amount" in column_name or "rate" in column_name or "score" in column_name:
        return "decimal"
    if column_name.startswith("is_") or column_name.endswith("_flag"):
        return "boolean"
    return "string"


def role_column_desc(profile: DomainProfile, role: str, column_name: str) -> str:
    for c in profile.role_columns[role]:
        if c["name"] == column_name:
            return c["description"]
    return f"{column_name} from {profile.tables[role]}"


def role_graph(profile: DomainProfile) -> Dict[str, List[Tuple[str, str, str]]]:
    g: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    for a, b, a_col, b_col in profile.role_edges:
        g[a].append((b, a_col, b_col))
        g[b].append((a, b_col, a_col))
    return g


def find_role_path(profile: DomainProfile, start_role: str, end_role: str) -> List[Tuple[str, str, str, str]]:
    if start_role == end_role:
        return []
    g = role_graph(profile)
    q = [(start_role, [])]
    seen = {start_role}
    while q:
        node, path = q.pop(0)
        for nxt, lcol, rcol in g.get(node, []):
            if nxt in seen:
                continue
            new_path = path + [(node, nxt, lcol, rcol)]
            if nxt == end_role:
                return new_path
            seen.add(nxt)
            q.append((nxt, new_path))
    return []


def build_join_path(profile: DomainProfile, roles: List[str]) -> List[Dict[str, Any]]:
    roles = list(dict.fromkeys(roles))
    if len(roles) <= 1:
        return []
    anchor_priority = {"main": 0, "event": 1, "txn": 2, "aux": 3, "lookup": 4}
    anchor = sorted(roles, key=lambda x: anchor_priority.get(x, 100))[0]
    steps = []
    used = set()
    for role in roles:
        if role == anchor:
            continue
        path = find_role_path(profile, anchor, role)
        for a, b, a_col, b_col in path:
            key = (a, b, a_col, b_col)
            if key in used:
                continue
            used.add(key)
            steps.append(
                {
                    "from": profile.tables[a],
                    "to": profile.tables[b],
                    "left_cols": [a_col],
                    "right_cols": [b_col],
                }
            )
    return steps


def table_kind_to_name(profile: DomainProfile, kind: str) -> str:
    if kind == "dim_entity":
        return f"dim_{profile.entity}"
    if kind == "fct_event":
        return f"fct_{profile.event}"
    if kind == "fct_txn":
        return f"fct_{profile.txn}"
    if kind == "fct_aux":
        return f"fct_{profile.aux}"
    if kind == "dim_lookup":
        return f"dim_{profile.lookup}"
    return f"dim_{profile.entity}"


def expand_positive_columns(profile: DomainProfile, scenario: Dict[str, Any], ctx: Dict[str, str]) -> List[Dict[str, str]]:
    cols = []
    for role, col_t in scenario["positive"]:
        col_name = resolve_text(col_t, ctx)
        cols.append(
            {
                "role": role,
                "table": profile.tables[role],
                "column": col_name,
                "type": role_column_type(profile, role, col_name),
                "description": role_column_desc(profile, role, col_name),
            }
        )
    return cols


def all_columns_catalog(profile: DomainProfile) -> List[Dict[str, str]]:
    rows = []
    for role, cols in profile.role_columns.items():
        table = profile.tables[role]
        for c in cols:
            rows.append(
                {
                    "role": role,
                    "table": table,
                    "column": c["name"],
                    "type": c["type"],
                    "description": c["description"],
                }
            )
    return rows


def pick_by_type(
    rng: random.Random,
    catalog: List[Dict[str, str]],
    desired_types: Sequence[str],
    exclude: Set[Tuple[str, str]],
    preferred_roles: Optional[Sequence[str]] = None,
) -> Optional[Dict[str, str]]:
    desired = {normalize_type(t) for t in desired_types}
    pool = []
    for c in catalog:
        if (c["table"], c["column"]) in exclude:
            continue
        if normalize_type(c["type"]) not in desired:
            continue
        if preferred_roles and c["role"] not in set(preferred_roles):
            continue
        pool.append(c)
    if not pool:
        return None
    return safe_choice(rng, pool)


def canonical_candidate(columns: List[Dict[str, str]], join_path: List[Dict[str, Any]]) -> Tuple[Any, ...]:
    col_key = tuple(sorted((c["table"], c["column"]) for c in columns))
    path_key = tuple(
        sorted(
            (
                step["from"],
                step["to"],
                tuple(step.get("left_cols", [])),
                tuple(step.get("right_cols", [])),
            )
            for step in join_path
        )
    )
    return col_key, path_key


def candidate_id_from_signature(
    query_id: str,
    columns: List[Dict[str, str]],
    join_path: List[Dict[str, Any]],
    salt: str,
) -> str:
    sig = {
        "query_id": query_id,
        "salt": salt,
        "columns": sorted((c["table"], c["column"], c["type"]) for c in columns),
        "join_path": sorted(
            (
                step["from"],
                step["to"],
                tuple(step.get("left_cols", [])),
                tuple(step.get("right_cols", [])),
            )
            for step in join_path
        ),
    }
    digest = hashlib.md5(json.dumps(sig, sort_keys=True).encode("utf-8")).hexdigest()[:14]
    return f"cand_{digest}"


def make_wrong_join_path(rng: random.Random, columns: List[Dict[str, str]], profile: DomainProfile, catalog: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    tables = sorted({c["table"] for c in columns})
    cols = list(columns)
    if len(tables) < 2:
        base_table = tables[0] if tables else ""
        pool = [c for c in catalog if c["table"] != base_table]
        if pool:
            cols.append(dict(safe_choice(rng, pool)))
        tables = sorted({c["table"] for c in cols})
    if len(tables) < 2:
        return cols, []
    t1 = tables[0]
    t2 = tables[-1]
    wrong_pairs = [
        ("status_code", "lookup_code"),
        ("country_code", "department_code"),
        ("created_at_text", "lookup_name"),
        ("line_no", "lookup_id"),
    ]
    lcol, rcol = safe_choice(rng, wrong_pairs)
    return cols, [{"from": t1, "to": t2, "left_cols": [lcol], "right_cols": [rcol]}]


def build_negative_candidates(
    rng: random.Random,
    profile: DomainProfile,
    scenario: Dict[str, Any],
    ctx: Dict[str, str],
    positive_cols: List[Dict[str, str]],
    target_type: str,
    used_signatures: Set[Tuple[Any, ...]],
) -> List[Tuple[List[Dict[str, str]], List[Dict[str, Any]], str]]:
    catalog = all_columns_catalog(profile)
    pos_set = {(c["table"], c["column"]) for c in positive_cols}
    pos_types = [normalize_type(c["type"]) for c in positive_cols]
    strategies = ["partial_projection", "semantic_mismatch", "type_mismatch", "wrong_join_path", "lookalike_confusion"]
    negatives: List[Tuple[List[Dict[str, str]], List[Dict[str, Any]], str]] = []

    for strat in strategies:
        cols: List[Dict[str, str]] = []
        join_path: List[Dict[str, Any]] = []
        attempts = 0
        while attempts < 30:
            attempts += 1
            cols = []
            if strat == "partial_projection":
                if len(positive_cols) > 1:
                    keep = rng.sample(positive_cols, k=len(positive_cols) - 1)
                    cols = [dict(x) for x in keep]
                else:
                    # same type different semantic
                    picked = pick_by_type(
                        rng,
                        catalog,
                        [positive_cols[0]["type"]],
                        exclude=pos_set,
                        preferred_roles=[positive_cols[0]["role"]],
                    ) or pick_by_type(rng, catalog, [positive_cols[0]["type"]], exclude=pos_set)
                    if picked:
                        cols = [dict(picked)]
            elif strat == "semantic_mismatch":
                for pc in positive_cols:
                    picked = pick_by_type(
                        rng,
                        catalog,
                        [pc["type"]],
                        exclude=pos_set | {(c["table"], c["column"]) for c in cols},
                        preferred_roles=[pc["role"]],
                    )
                    if picked is None:
                        picked = pick_by_type(
                            rng,
                            catalog,
                            [pc["type"]],
                            exclude=pos_set | {(c["table"], c["column"]) for c in cols},
                        )
                    if picked:
                        cols.append(dict(picked))
            elif strat == "type_mismatch":
                wrong_types = incompatible_types(target_type)
                arity = max(1, len(positive_cols))
                for _ in range(arity):
                    picked = pick_by_type(
                        rng,
                        catalog,
                        wrong_types,
                        exclude=pos_set | {(c["table"], c["column"]) for c in cols},
                    )
                    if picked:
                        cols.append(dict(picked))
            elif strat == "wrong_join_path":
                seed = safe_choice(rng, positive_cols)
                cols.append(dict(seed))
                picked = pick_by_type(
                    rng,
                    catalog,
                    ["string", "int", "decimal", "date", "boolean"],
                    exclude=pos_set | {(seed["table"], seed["column"])},
                )
                if picked:
                    cols.append(dict(picked))
                cols, join_path = make_wrong_join_path(rng, cols, profile, catalog)
            elif strat == "lookalike_confusion":
                lookalike_pool = [
                    c for c in catalog
                    if (c["column"].endswith("_id") or c["column"].endswith("_code") or "status" in c["column"])
                    and (c["table"], c["column"]) not in pos_set
                ]
                if lookalike_pool:
                    arity = max(1, min(2, len(positive_cols)))
                    picks = rng.sample(lookalike_pool, k=min(arity, len(lookalike_pool)))
                    cols = [dict(x) for x in picks]

            if not cols:
                continue
            roles = [c["role"] for c in cols]
            if strat != "wrong_join_path":
                join_path = build_join_path(profile, roles)

            sig = canonical_candidate(cols, join_path)
            if sig in used_signatures:
                continue
            if sig == canonical_candidate(positive_cols, build_join_path(profile, [c["role"] for c in positive_cols])):
                continue
            used_signatures.add(sig)
            negatives.append((cols, join_path, strat))
            break

    return negatives


def build_query_records(
    rng: random.Random,
    q_idx: int,
    profile: DomainProfile,
    scenario: Dict[str, Any],
) -> List[Dict[str, Any]]:
    ctx = make_context(profile)
    query_id = f"q_{profile.domain}_{scenario['id']}_{q_idx:05d}"
    split = assign_split(query_id)

    target = {
        "table": table_kind_to_name(profile, scenario["target_kind"]),
        "column": resolve_text(safe_choice(rng, scenario["target_names"]), ctx),
        "type": normalize_type(scenario["target_type"]),
        "description": resolve_text(scenario["description"], ctx),
    }

    positive_cols = expand_positive_columns(profile, scenario, ctx)
    positive_join = build_join_path(profile, [c["role"] for c in positive_cols])
    used_signatures = {canonical_candidate(positive_cols, positive_join)}

    records = []
    pos_record = {
        "query_id": query_id,
        "split": split,
        "domain": profile.domain,
        "db_reference": profile.db_reference,
        "scenario": scenario["id"],
        "target": target,
        "candidate_set": {
            "id": candidate_id_from_signature(
                query_id=query_id,
                columns=positive_cols,
                join_path=positive_join,
                salt="base",
            ),
            "columns": [
                {
                    "table": c["table"],
                    "column": c["column"],
                    "type": normalize_type(c["type"]),
                    "description": c["description"],
                }
                for c in positive_cols
            ],
            "join_path": positive_join,
            "transform_hint": scenario["transform_hint"],
        },
        "label": 1,
        "weight": 1.25,
    }
    records.append(pos_record)

    negatives = build_negative_candidates(
        rng=rng,
        profile=profile,
        scenario=scenario,
        ctx=ctx,
        positive_cols=positive_cols,
        target_type=target["type"],
        used_signatures=used_signatures,
    )

    # If a strategy failed, backfill with random mismatches.
    catalog = all_columns_catalog(profile)
    neg_needed = NEGATIVES_PER_QUERY - len(negatives)
    backfill_count = 0
    while neg_needed > 0 and backfill_count < 50:
        backfill_count += 1
        arity = rng.choice([1, 1, 2, 2, 3])
        cols = []
        excluded = {(c["table"], c["column"]) for c in positive_cols}
        for _ in range(arity):
            c = safe_choice(rng, catalog)
            if (c["table"], c["column"]) in excluded:
                continue
            if (c["table"], c["column"]) in {(x["table"], x["column"]) for x in cols}:
                continue
            cols.append(dict(c))
        if not cols:
            continue
        jpath = build_join_path(profile, [c["role"] for c in cols])
        sig = canonical_candidate(cols, jpath)
        if sig in used_signatures:
            continue
        used_signatures.add(sig)
        negatives.append((cols, jpath, "random_mismatch"))
        neg_needed -= 1

    negatives = negatives[:NEGATIVES_PER_QUERY]
    for i, (cols, jpath, ntype) in enumerate(negatives, 1):
        records.append(
            {
                "query_id": query_id,
                "split": split,
                "domain": profile.domain,
                "db_reference": profile.db_reference,
                "scenario": scenario["id"],
                "hard_negative_type": ntype,
                "target": target,
                "candidate_set": {
                    "id": candidate_id_from_signature(
                        query_id=query_id,
                        columns=cols,
                        join_path=jpath,
                        salt=f"neg{i}",
                    ),
                    "columns": [
                        {
                            "table": c["table"],
                            "column": c["column"],
                            "type": normalize_type(c["type"]),
                            "description": c["description"],
                        }
                        for c in cols
                    ],
                    "join_path": jpath,
                    "transform_hint": ntype,
                },
                "label": 0,
                "weight": 1.0,
            }
        )

    return records


def generate_dataset(query_count: int = QUERY_COUNT) -> List[Dict[str, Any]]:
    rng = random.Random(SEED)
    rows: List[Dict[str, Any]] = []

    # Ensure broad coverage: first pass cycles through domain x scenario.
    all_pairs = [(p, s) for p in PROFILES for s in SCENARIOS]
    rng.shuffle(all_pairs)

    q_idx = 1
    for profile, scenario in all_pairs:
        if q_idx > query_count:
            break
        rows.extend(build_query_records(rng, q_idx, profile, scenario))
        q_idx += 1

    while q_idx <= query_count:
        profile = safe_choice(rng, PROFILES)
        # Bias towards join-heavy scenarios for better cross-table generalization.
        weighted = []
        for s in SCENARIOS:
            w = 2.4 if s["id"] in JOIN_HEAVY_SCENARIO_IDS else 1.0
            weighted.append((s, w))
        total_w = sum(w for _, w in weighted)
        r = rng.random() * total_w
        upto = 0.0
        scenario = SCENARIOS[0]
        for s, w in weighted:
            upto += w
            if upto >= r:
                scenario = s
                break
        rows.extend(build_query_records(rng, q_idx, profile, scenario))
        q_idx += 1

    # Guarantee no exact duplicates.
    seen_lines = set()
    deduped = []
    for r in rows:
        line = json.dumps(r, sort_keys=True)
        if line in seen_lines:
            continue
        seen_lines.add(line)
        deduped.append(r)
    return deduped


def validate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    q_to_labels = defaultdict(list)
    cand_ids = set()
    split_counts = defaultdict(int)
    label_counts = defaultdict(int)
    domain_counts = defaultdict(int)
    scenario_counts = defaultdict(int)

    for r in rows:
        qid = r["query_id"]
        q_to_labels[qid].append(r["label"])
        split_counts[r["split"]] += 1
        label_counts[r["label"]] += 1
        domain_counts[r["domain"]] += 1
        scenario_counts[r["scenario"]] += 1

        cid = r["candidate_set"]["id"]
        if cid in cand_ids:
            raise ValueError(f"Duplicate candidate_set.id found: {cid}")
        cand_ids.add(cid)

        if r["label"] not in (0, 1):
            raise ValueError(f"Invalid label: {r['label']}")
        if r["split"] not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: {r['split']}")
        if not r["candidate_set"]["columns"]:
            raise ValueError(f"No columns in candidate set for query {qid}")

    # per-query checks
    for qid, labels in q_to_labels.items():
        if 1 not in labels:
            raise ValueError(f"Query has no positive candidate: {qid}")
        if labels.count(0) < 3:
            raise ValueError(f"Query has insufficient negatives (<3): {qid}")

    return {
        "row_count": len(rows),
        "query_count": len(q_to_labels),
        "split_counts": dict(split_counts),
        "label_counts": dict(label_counts),
        "domain_count": len(domain_counts),
        "scenario_count": len(scenario_counts),
        "domain_counts": dict(domain_counts),
    }


def main() -> None:
    rows = generate_dataset(query_count=QUERY_COUNT)
    stats = validate(rows)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")

    print("Generated dataset written to:", OUTPUT_PATH)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
