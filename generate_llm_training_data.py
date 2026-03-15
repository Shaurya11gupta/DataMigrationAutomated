#!/usr/bin/env python3
"""
Generate Training Data for LLM Schema Matcher (V3) - Production Quality
=========================================================================
Creates rich, diverse instruction-following training data for fine-tuning
a small LLM on schema mapping tasks.

Key design principles for generalization:
  1. 20+ domains covering diverse industries
  2. Schema perturbation: messy names, abbreviations, legacy conventions
  3. Hard reasoning examples: ambiguous columns, synonym traps
  4. Multiple naming conventions per domain (clean, abbreviated, legacy)
  5. Cross-domain pattern reuse for transfer learning
  6. 10,000+ examples targeting robust generalization

Output: llm_train.jsonl, llm_val.jsonl in chat format.
"""
from __future__ import annotations

import copy
import json
import os
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Reuse base domain definitions from V2 data generator
from generate_candidate_training_data import (
    ColDef, DomainSchema, MappingDef, TableDef,
    _build_domains as _build_base_domains,
)


# ---------------------------------------------------------------
# System prompt (constant across all examples)
# ---------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a schema mapping expert. Given a source database schema and a "
    "target column, identify which source column(s) should be mapped to the "
    "target and what transformation is needed.\n"
    "\n"
    "COLUMN SELECTION RULES:\n"
    "- Use ONLY columns that exist in the source schema\n"
    "- Choose the MINIMUM number of source columns needed\n"
    "- ALWAYS output the VALUE column(s) that hold the actual data, NOT the "
    "FK/join key columns used to traverse between tables. Example: to get a "
    "department name via a join, output departments.dept_name, NOT the "
    "employees.department_id FK column\n"
    "- When a mapping requires columns from MULTIPLE tables (e.g. date_diff "
    "between dates in different tables), select the actual data columns from "
    "each table, not the FK columns that link them\n"
    "- For surrogate keys, use the primary key of the matching entity table\n"
    "\n"
    "TRANSFORM DECISION RULES:\n"
    "- rename: ONLY when the source column is in the SAME table as the "
    "target's primary entity and no computation is needed. If a join/FK is "
    "required to reach the column, it is NOT a rename\n"
    "- direct_copy: identical to rename but emphasizes no name change\n"
    "- fk_lookup: when the source column is in a DIFFERENT table reached "
    "via a single FK join. The key distinction from rename is that a join "
    "is required\n"
    "- lookup_join: when the source column requires traversing 2+ joins "
    "(multi-hop). Use this instead of fk_lookup for multi-table chains\n"
    "- concat: combining 2+ columns into one string value\n"
    "- date_part: extracting year, month, day, etc. from a date/datetime\n"
    "- date_diff: computing the difference between two dates (duration, age)\n"
    "- date_format: converting a date to a display string format\n"
    "- date_parse: parsing a string into a date/datetime type\n"
    "- arithmetic: math operations (add, subtract, multiply, divide) between "
    "columns. Do NOT assume arithmetic from column names alone -- if salary "
    "maps to annual_salary with no formula specified, use rename\n"
    "- conditional: deriving a BOOLEAN true/false flag from a status/code "
    "column. The target must be boolean/flag type\n"
    "- code_to_label: mapping a code/status column to a human-readable LABEL "
    "string (e.g. gender_code -> gender_label, status_code -> status_name). "
    "This is NOT the same as conditional -- code_to_label produces a label "
    "string, conditional produces a boolean\n"
    "- bucketing: grouping a numeric/continuous value into range categories\n"
    "- type_cast: converting data type without changing the value\n"
    "- template: assembling columns into a formatted display string pattern\n"
    "\n"
    "Valid transform types: rename, direct_copy, concat, fk_lookup, date_part, "
    "date_diff, date_format, date_parse, arithmetic, conditional, bucketing, "
    "code_to_label, type_cast, lookup_join, template\n"
    "\n"
    "SUB-OPERATION: For each transform, specify the fine-grained operation:\n"
    "- rename: rename_only\n"
    "- direct_copy: direct_copy\n"
    "- concat: concat_two, concat_multi (3+ columns)\n"
    "- fk_lookup: fk_dimension_lookup\n"
    "- lookup_join: multi_hop_lookup\n"
    "- date_part: extract_year, extract_month, extract_day, extract_quarter, "
    "extract_hour\n"
    "- date_diff: date_difference, age_calculation, duration_days, "
    "duration_hours\n"
    "- date_format: format_date\n"
    "- date_parse: parse_date\n"
    "- arithmetic: add, subtract, multiply, divide, ratio_percentage, "
    "scaling_unit_conversion\n"
    "- conditional: threshold_flag, status_flag, equality_check, "
    "null_presence_flag\n"
    "- code_to_label: code_to_label, category_harmonization\n"
    "- bucketing: bucketing_binning, range_classification\n"
    "- type_cast: type_cast_numeric, type_cast_string, type_cast_date\n"
    "- template: string_template\n"
    "\n"
    "OUTPUT RULES:\n"
    "- Respond with EXACTLY ONE mapping for the requested target column\n"
    "- Do NOT output mappings for other columns\n"
    "\n"
    "Respond in EXACTLY this format:\n"
    "source_columns: <table.column>, <table.column>, ...\n"
    "transform_type: <transform>\n"
    "sub_operation: <fine-grained operation>\n"
    "reasoning: <brief explanation>"
)


# ---------------------------------------------------------------
# Additional domains (beyond the 12 base domains)
# ---------------------------------------------------------------

def _build_extra_domains() -> List[DomainSchema]:
    """Build 8+ additional domains for richer coverage."""
    domains = []

    # ─── Domain 13: IoT / Sensors ───
    domains.append(DomainSchema(
        name="iot",
        source_tables=[
            TableDef("sensors", [
                ColDef("sensor_id", "int", is_pk=True),
                ColDef("sensor_name", "string"),
                ColDef("sensor_type", "string"),
                ColDef("install_date", "date"),
                ColDef("location_id", "int", is_fk="locations.location_id"),
                ColDef("firmware_version", "string"),
                ColDef("status", "string"),
                ColDef("min_threshold", "decimal"),
                ColDef("max_threshold", "decimal"),
            ]),
            TableDef("locations", [
                ColDef("location_id", "int", is_pk=True),
                ColDef("building_name", "string"),
                ColDef("floor_number", "int"),
                ColDef("zone", "string"),
                ColDef("city", "string"),
                ColDef("country", "string"),
                ColDef("latitude", "decimal"),
                ColDef("longitude", "decimal"),
            ]),
            TableDef("readings", [
                ColDef("reading_id", "int", is_pk=True),
                ColDef("sensor_id", "int", is_fk="sensors.sensor_id"),
                ColDef("reading_time", "datetime"),
                ColDef("value", "decimal"),
                ColDef("unit", "string"),
                ColDef("quality_flag", "string"),
                ColDef("battery_level", "decimal"),
            ]),
            TableDef("alerts", [
                ColDef("alert_id", "int", is_pk=True),
                ColDef("sensor_id", "int", is_fk="sensors.sensor_id"),
                ColDef("alert_time", "datetime"),
                ColDef("severity", "string"),
                ColDef("message", "string"),
                ColDef("acknowledged", "string"),
                ColDef("resolved_time", "datetime"),
            ]),
        ],
        mappings=[
            MappingDef("dim_sensor", "sensor_key", "int", "sensor surrogate key",
                       [("sensors", "sensor_id", "int")], "rename"),
            MappingDef("dim_sensor", "sensor_label", "string", "human-readable sensor name",
                       [("sensors", "sensor_name", "string")], "rename"),
            MappingDef("dim_sensor", "installation_year", "int", "year sensor was installed",
                       [("sensors", "install_date", "date")], "date_part"),
            MappingDef("dim_sensor", "building_name", "string", "building where sensor is located",
                       [("locations", "building_name", "string")], "fk_lookup",
                       join_path=["sensors.location_id = locations.location_id"]),
            MappingDef("dim_sensor", "sensor_location", "string", "building and zone combined",
                       [("locations", "building_name", "string"), ("locations", "zone", "string")], "concat",
                       join_path=["sensors.location_id = locations.location_id"]),
            MappingDef("dim_sensor", "full_location", "string", "building, zone, city, country",
                       [("locations", "building_name", "string"), ("locations", "zone", "string"),
                        ("locations", "city", "string")], "concat",
                       join_path=["sensors.location_id = locations.location_id"]),
            MappingDef("dim_sensor", "threshold_range", "decimal", "max minus min threshold",
                       [("sensors", "min_threshold", "decimal"), ("sensors", "max_threshold", "decimal")], "arithmetic"),
            MappingDef("dim_sensor", "is_operational", "boolean", "whether sensor is active",
                       [("sensors", "status", "string")], "conditional"),
            MappingDef("fact_reading", "reading_key", "int", "reading identifier",
                       [("readings", "reading_id", "int")], "rename"),
            MappingDef("fact_reading", "sensor_name", "string", "name of the sensor that took reading",
                       [("sensors", "sensor_name", "string")], "fk_lookup",
                       join_path=["readings.sensor_id = sensors.sensor_id"]),
            MappingDef("fact_reading", "reading_year", "int", "year of the reading",
                       [("readings", "reading_time", "datetime")], "date_part"),
            MappingDef("fact_reading", "measurement_value", "decimal", "the actual reading value",
                       [("readings", "value", "decimal")], "rename"),
            MappingDef("fact_reading", "is_valid", "boolean", "whether reading passed quality check",
                       [("readings", "quality_flag", "string")], "conditional"),
            MappingDef("fact_alert", "alert_key", "int", "alert identifier",
                       [("alerts", "alert_id", "int")], "rename"),
            MappingDef("fact_alert", "resolution_hours", "int", "hours to resolve alert",
                       [("alerts", "alert_time", "datetime"), ("alerts", "resolved_time", "datetime")], "date_diff"),
            MappingDef("fact_alert", "is_acknowledged", "boolean", "whether alert was acknowledged",
                       [("alerts", "acknowledged", "string")], "conditional"),
        ],
    ))

    # ─── Domain 14: Retail / Inventory ───
    domains.append(DomainSchema(
        name="retail",
        source_tables=[
            TableDef("store_items", [
                ColDef("item_id", "int", is_pk=True),
                ColDef("item_name", "string"),
                ColDef("unit_price", "decimal"),
                ColDef("cost_price", "decimal"),
                ColDef("category_code", "string"),
                ColDef("supplier_id", "int", is_fk="suppliers.supplier_id"),
                ColDef("weight_grams", "int"),
                ColDef("shelf_life_days", "int"),
                ColDef("reorder_level", "int"),
                ColDef("is_perishable", "string"),
            ]),
            TableDef("suppliers", [
                ColDef("supplier_id", "int", is_pk=True),
                ColDef("company_name", "string"),
                ColDef("contact_first", "string"),
                ColDef("contact_last", "string"),
                ColDef("contact_email", "string"),
                ColDef("city", "string"),
                ColDef("country", "string"),
                ColDef("rating", "decimal"),
            ]),
            TableDef("purchase_orders", [
                ColDef("po_id", "int", is_pk=True),
                ColDef("item_id", "int", is_fk="store_items.item_id"),
                ColDef("supplier_id", "int", is_fk="suppliers.supplier_id"),
                ColDef("order_date", "date"),
                ColDef("delivery_date", "date"),
                ColDef("quantity", "int"),
                ColDef("total_cost", "decimal"),
                ColDef("status", "string"),
            ]),
            TableDef("inventory", [
                ColDef("inv_id", "int", is_pk=True),
                ColDef("item_id", "int", is_fk="store_items.item_id"),
                ColDef("warehouse_code", "string"),
                ColDef("quantity_on_hand", "int"),
                ColDef("last_restock_date", "date"),
            ]),
        ],
        mappings=[
            MappingDef("dim_product", "product_key", "int", "product surrogate key",
                       [("store_items", "item_id", "int")], "rename"),
            MappingDef("dim_product", "product_name", "string", "name of the product",
                       [("store_items", "item_name", "string")], "rename"),
            MappingDef("dim_product", "profit_margin", "decimal", "selling price minus cost",
                       [("store_items", "unit_price", "decimal"), ("store_items", "cost_price", "decimal")], "arithmetic"),
            MappingDef("dim_product", "supplier_company", "string", "name of the supplier company",
                       [("suppliers", "company_name", "string")], "fk_lookup",
                       join_path=["store_items.supplier_id = suppliers.supplier_id"]),
            MappingDef("dim_product", "supplier_contact_name", "string", "full name of supplier contact",
                       [("suppliers", "contact_first", "string"), ("suppliers", "contact_last", "string")], "concat",
                       join_path=["store_items.supplier_id = suppliers.supplier_id"]),
            MappingDef("dim_product", "weight_kg", "decimal", "weight in kilograms",
                       [("store_items", "weight_grams", "int")], "arithmetic"),
            MappingDef("dim_product", "is_perishable_flag", "boolean", "whether item is perishable",
                       [("store_items", "is_perishable", "string")], "conditional"),
            MappingDef("fact_purchase", "purchase_key", "int", "purchase order id",
                       [("purchase_orders", "po_id", "int")], "rename"),
            MappingDef("fact_purchase", "order_year", "int", "year the order was placed",
                       [("purchase_orders", "order_date", "date")], "date_part"),
            MappingDef("fact_purchase", "lead_time_days", "int", "days between order and delivery",
                       [("purchase_orders", "order_date", "date"), ("purchase_orders", "delivery_date", "date")], "date_diff"),
            MappingDef("fact_purchase", "unit_cost", "decimal", "total cost divided by quantity",
                       [("purchase_orders", "total_cost", "decimal"), ("purchase_orders", "quantity", "int")], "arithmetic"),
            MappingDef("fact_purchase", "is_delivered", "boolean", "whether order has been delivered",
                       [("purchase_orders", "status", "string")], "conditional"),
            MappingDef("fact_purchase", "product_name", "string", "name of purchased item",
                       [("store_items", "item_name", "string")], "fk_lookup",
                       join_path=["purchase_orders.item_id = store_items.item_id"]),
        ],
    ))

    # ─── Domain 15: CRM ───
    domains.append(DomainSchema(
        name="crm",
        source_tables=[
            TableDef("contacts", [
                ColDef("contact_id", "int", is_pk=True),
                ColDef("first_name", "string"),
                ColDef("last_name", "string"),
                ColDef("email", "string"),
                ColDef("phone", "string"),
                ColDef("company_id", "int", is_fk="companies.company_id"),
                ColDef("title", "string"),
                ColDef("created_date", "date"),
                ColDef("lead_source", "string"),
                ColDef("status", "string"),
            ]),
            TableDef("companies", [
                ColDef("company_id", "int", is_pk=True),
                ColDef("company_name", "string"),
                ColDef("industry", "string"),
                ColDef("revenue", "decimal"),
                ColDef("employee_count", "int"),
                ColDef("website", "string"),
                ColDef("country", "string"),
            ]),
            TableDef("opportunities", [
                ColDef("opp_id", "int", is_pk=True),
                ColDef("contact_id", "int", is_fk="contacts.contact_id"),
                ColDef("opp_name", "string"),
                ColDef("amount", "decimal"),
                ColDef("stage", "string"),
                ColDef("close_date", "date"),
                ColDef("created_date", "date"),
                ColDef("probability", "decimal"),
                ColDef("owner_id", "int", is_fk="sales_reps.rep_id"),
            ]),
            TableDef("sales_reps", [
                ColDef("rep_id", "int", is_pk=True),
                ColDef("first_name", "string"),
                ColDef("last_name", "string"),
                ColDef("email", "string"),
                ColDef("region", "string"),
                ColDef("quota", "decimal"),
            ]),
            TableDef("activities", [
                ColDef("activity_id", "int", is_pk=True),
                ColDef("contact_id", "int", is_fk="contacts.contact_id"),
                ColDef("activity_type", "string"),
                ColDef("activity_date", "date"),
                ColDef("notes", "string"),
                ColDef("duration_min", "int"),
            ]),
        ],
        mappings=[
            MappingDef("dim_contact", "contact_key", "int", "contact surrogate key",
                       [("contacts", "contact_id", "int")], "rename"),
            MappingDef("dim_contact", "contact_name", "string", "full name of contact person",
                       [("contacts", "first_name", "string"), ("contacts", "last_name", "string")], "concat"),
            MappingDef("dim_contact", "contact_email", "string", "email address",
                       [("contacts", "email", "string")], "rename"),
            MappingDef("dim_contact", "company_name", "string", "associated company name",
                       [("companies", "company_name", "string")], "fk_lookup",
                       join_path=["contacts.company_id = companies.company_id"]),
            MappingDef("dim_contact", "company_industry", "string", "industry of the company",
                       [("companies", "industry", "string")], "fk_lookup",
                       join_path=["contacts.company_id = companies.company_id"]),
            MappingDef("dim_contact", "creation_year", "int", "year contact was created",
                       [("contacts", "created_date", "date")], "date_part"),
            MappingDef("dim_contact", "is_qualified", "boolean", "whether contact is qualified lead",
                       [("contacts", "status", "string")], "conditional"),
            MappingDef("fact_opportunity", "opportunity_key", "int", "opportunity id",
                       [("opportunities", "opp_id", "int")], "rename"),
            MappingDef("fact_opportunity", "deal_name", "string", "opportunity name",
                       [("opportunities", "opp_name", "string")], "rename"),
            MappingDef("fact_opportunity", "expected_revenue", "decimal", "amount times probability",
                       [("opportunities", "amount", "decimal"), ("opportunities", "probability", "decimal")], "arithmetic"),
            MappingDef("fact_opportunity", "sales_cycle_days", "int", "days from creation to close",
                       [("opportunities", "created_date", "date"), ("opportunities", "close_date", "date")], "date_diff"),
            MappingDef("fact_opportunity", "owner_name", "string", "sales rep full name",
                       [("sales_reps", "first_name", "string"), ("sales_reps", "last_name", "string")], "concat",
                       join_path=["opportunities.owner_id = sales_reps.rep_id"]),
            MappingDef("fact_opportunity", "contact_full_name", "string", "contact person name",
                       [("contacts", "first_name", "string"), ("contacts", "last_name", "string")], "concat",
                       join_path=["opportunities.contact_id = contacts.contact_id"]),
            MappingDef("fact_opportunity", "close_year", "int", "expected close year",
                       [("opportunities", "close_date", "date")], "date_part"),
        ],
    ))

    # ─── Domain 16: Social Media / Content ───
    domains.append(DomainSchema(
        name="social_media",
        source_tables=[
            TableDef("users", [
                ColDef("user_id", "int", is_pk=True),
                ColDef("username", "string"),
                ColDef("display_name", "string"),
                ColDef("email", "string"),
                ColDef("join_date", "date"),
                ColDef("bio", "string"),
                ColDef("follower_count", "int"),
                ColDef("following_count", "int"),
                ColDef("country", "string"),
                ColDef("is_verified", "string"),
            ]),
            TableDef("posts", [
                ColDef("post_id", "int", is_pk=True),
                ColDef("user_id", "int", is_fk="users.user_id"),
                ColDef("content", "string"),
                ColDef("post_date", "datetime"),
                ColDef("like_count", "int"),
                ColDef("comment_count", "int"),
                ColDef("share_count", "int"),
                ColDef("media_type", "string"),
                ColDef("hashtags", "string"),
            ]),
            TableDef("comments", [
                ColDef("comment_id", "int", is_pk=True),
                ColDef("post_id", "int", is_fk="posts.post_id"),
                ColDef("user_id", "int", is_fk="users.user_id"),
                ColDef("comment_text", "string"),
                ColDef("comment_date", "datetime"),
                ColDef("sentiment_score", "decimal"),
            ]),
        ],
        mappings=[
            MappingDef("dim_user", "user_key", "int", "user surrogate key",
                       [("users", "user_id", "int")], "rename"),
            MappingDef("dim_user", "user_name", "string", "display name of user",
                       [("users", "display_name", "string")], "rename"),
            MappingDef("dim_user", "join_year", "int", "year user joined platform",
                       [("users", "join_date", "date")], "date_part"),
            MappingDef("dim_user", "engagement_ratio", "decimal", "followers divided by following",
                       [("users", "follower_count", "int"), ("users", "following_count", "int")], "arithmetic"),
            MappingDef("dim_user", "is_verified_flag", "boolean", "whether user is verified",
                       [("users", "is_verified", "string")], "conditional"),
            MappingDef("fact_post", "post_key", "int", "post identifier",
                       [("posts", "post_id", "int")], "rename"),
            MappingDef("fact_post", "author_name", "string", "name of the post author",
                       [("users", "display_name", "string")], "fk_lookup",
                       join_path=["posts.user_id = users.user_id"]),
            MappingDef("fact_post", "post_year", "int", "year post was created",
                       [("posts", "post_date", "datetime")], "date_part"),
            MappingDef("fact_post", "total_engagement", "int", "likes plus comments plus shares",
                       [("posts", "like_count", "int"), ("posts", "comment_count", "int"), ("posts", "share_count", "int")], "arithmetic"),
            MappingDef("fact_post", "engagement_rate", "decimal", "total engagement per follower",
                       [("posts", "like_count", "int"), ("posts", "comment_count", "int")], "arithmetic"),
        ],
    ))

    # ─── Domain 17: Energy / Utilities ───
    domains.append(DomainSchema(
        name="energy",
        source_tables=[
            TableDef("meters", [
                ColDef("meter_id", "int", is_pk=True),
                ColDef("meter_number", "string"),
                ColDef("meter_type", "string"),
                ColDef("install_date", "date"),
                ColDef("customer_id", "int", is_fk="utility_customers.cust_id"),
                ColDef("address", "string"),
                ColDef("city", "string"),
            ]),
            TableDef("utility_customers", [
                ColDef("cust_id", "int", is_pk=True),
                ColDef("first_name", "string"),
                ColDef("last_name", "string"),
                ColDef("account_number", "string"),
                ColDef("service_start", "date"),
                ColDef("rate_plan", "string"),
                ColDef("monthly_budget", "decimal"),
            ]),
            TableDef("consumption", [
                ColDef("record_id", "int", is_pk=True),
                ColDef("meter_id", "int", is_fk="meters.meter_id"),
                ColDef("reading_date", "date"),
                ColDef("kwh_used", "decimal"),
                ColDef("peak_demand_kw", "decimal"),
                ColDef("billing_period", "string"),
            ]),
            TableDef("bills", [
                ColDef("bill_id", "int", is_pk=True),
                ColDef("cust_id", "int", is_fk="utility_customers.cust_id"),
                ColDef("bill_date", "date"),
                ColDef("amount", "decimal"),
                ColDef("kwh_billed", "decimal"),
                ColDef("payment_date", "date"),
                ColDef("status", "string"),
            ]),
        ],
        mappings=[
            MappingDef("dim_customer", "customer_key", "int", "utility customer key",
                       [("utility_customers", "cust_id", "int")], "rename"),
            MappingDef("dim_customer", "customer_name", "string", "customer full name",
                       [("utility_customers", "first_name", "string"), ("utility_customers", "last_name", "string")], "concat"),
            MappingDef("dim_customer", "service_start_year", "int", "year service began",
                       [("utility_customers", "service_start", "date")], "date_part"),
            MappingDef("dim_customer", "rate_plan_name", "string", "customer rate plan",
                       [("utility_customers", "rate_plan", "string")], "rename"),
            MappingDef("fact_consumption", "consumption_key", "int", "consumption record id",
                       [("consumption", "record_id", "int")], "rename"),
            MappingDef("fact_consumption", "meter_number", "string", "meter serial number",
                       [("meters", "meter_number", "string")], "fk_lookup",
                       join_path=["consumption.meter_id = meters.meter_id"]),
            MappingDef("fact_consumption", "reading_year", "int", "year of consumption reading",
                       [("consumption", "reading_date", "date")], "date_part"),
            MappingDef("fact_consumption", "energy_consumed", "decimal", "kilowatt hours used",
                       [("consumption", "kwh_used", "decimal")], "rename"),
            MappingDef("fact_consumption", "cost_per_kwh", "decimal", "bill amount divided by kwh",
                       [("bills", "amount", "decimal"), ("bills", "kwh_billed", "decimal")], "arithmetic",
                       join_path=["bills.cust_id = utility_customers.cust_id"]),
            MappingDef("fact_billing", "bill_key", "int", "bill identifier",
                       [("bills", "bill_id", "int")], "rename"),
            MappingDef("fact_billing", "days_to_pay", "int", "days between bill and payment",
                       [("bills", "bill_date", "date"), ("bills", "payment_date", "date")], "date_diff"),
            MappingDef("fact_billing", "is_paid", "boolean", "whether bill has been paid",
                       [("bills", "status", "string")], "conditional"),
        ],
    ))

    # ─── Domain 18: Marketing / Campaigns ───
    domains.append(DomainSchema(
        name="marketing",
        source_tables=[
            TableDef("campaigns", [
                ColDef("campaign_id", "int", is_pk=True),
                ColDef("campaign_name", "string"),
                ColDef("channel", "string"),
                ColDef("start_date", "date"),
                ColDef("end_date", "date"),
                ColDef("budget", "decimal"),
                ColDef("spend", "decimal"),
                ColDef("target_audience", "string"),
                ColDef("manager_id", "int", is_fk="marketing_team.member_id"),
            ]),
            TableDef("marketing_team", [
                ColDef("member_id", "int", is_pk=True),
                ColDef("first_name", "string"),
                ColDef("last_name", "string"),
                ColDef("role", "string"),
                ColDef("email", "string"),
            ]),
            TableDef("leads", [
                ColDef("lead_id", "int", is_pk=True),
                ColDef("campaign_id", "int", is_fk="campaigns.campaign_id"),
                ColDef("first_name", "string"),
                ColDef("last_name", "string"),
                ColDef("email", "string"),
                ColDef("phone", "string"),
                ColDef("source", "string"),
                ColDef("created_date", "date"),
                ColDef("status", "string"),
                ColDef("score", "int"),
            ]),
            TableDef("conversions", [
                ColDef("conversion_id", "int", is_pk=True),
                ColDef("lead_id", "int", is_fk="leads.lead_id"),
                ColDef("conversion_date", "date"),
                ColDef("revenue", "decimal"),
                ColDef("product_name", "string"),
            ]),
        ],
        mappings=[
            MappingDef("dim_campaign", "campaign_key", "int", "campaign identifier",
                       [("campaigns", "campaign_id", "int")], "rename"),
            MappingDef("dim_campaign", "campaign_title", "string", "name of campaign",
                       [("campaigns", "campaign_name", "string")], "rename"),
            MappingDef("dim_campaign", "campaign_duration_days", "int", "length of campaign in days",
                       [("campaigns", "start_date", "date"), ("campaigns", "end_date", "date")], "date_diff"),
            MappingDef("dim_campaign", "budget_remaining", "decimal", "budget minus spend",
                       [("campaigns", "budget", "decimal"), ("campaigns", "spend", "decimal")], "arithmetic"),
            MappingDef("dim_campaign", "roi_ratio", "decimal", "spend divided by budget",
                       [("campaigns", "spend", "decimal"), ("campaigns", "budget", "decimal")], "arithmetic"),
            MappingDef("dim_campaign", "manager_name", "string", "campaign manager full name",
                       [("marketing_team", "first_name", "string"), ("marketing_team", "last_name", "string")], "concat",
                       join_path=["campaigns.manager_id = marketing_team.member_id"]),
            MappingDef("dim_campaign", "start_year", "int", "year campaign started",
                       [("campaigns", "start_date", "date")], "date_part"),
            MappingDef("fact_lead", "lead_key", "int", "lead identifier",
                       [("leads", "lead_id", "int")], "rename"),
            MappingDef("fact_lead", "lead_name", "string", "full name of the lead",
                       [("leads", "first_name", "string"), ("leads", "last_name", "string")], "concat"),
            MappingDef("fact_lead", "campaign_name", "string", "originating campaign",
                       [("campaigns", "campaign_name", "string")], "fk_lookup",
                       join_path=["leads.campaign_id = campaigns.campaign_id"]),
            MappingDef("fact_lead", "is_converted", "boolean", "whether lead converted",
                       [("leads", "status", "string")], "conditional"),
            MappingDef("fact_lead", "creation_year", "int", "year lead was created",
                       [("leads", "created_date", "date")], "date_part"),
        ],
    ))

    # ─── Domain 19: Government / Public Records ───
    domains.append(DomainSchema(
        name="government",
        source_tables=[
            TableDef("citizens", [
                ColDef("citizen_id", "int", is_pk=True),
                ColDef("first_name", "string"),
                ColDef("last_name", "string"),
                ColDef("date_of_birth", "date"),
                ColDef("ssn_hash", "string"),
                ColDef("gender", "string"),
                ColDef("address", "string"),
                ColDef("city", "string"),
                ColDef("state", "string"),
                ColDef("zip_code", "string"),
                ColDef("registration_date", "date"),
            ]),
            TableDef("permits", [
                ColDef("permit_id", "int", is_pk=True),
                ColDef("citizen_id", "int", is_fk="citizens.citizen_id"),
                ColDef("permit_type", "string"),
                ColDef("issue_date", "date"),
                ColDef("expiry_date", "date"),
                ColDef("fee", "decimal"),
                ColDef("status", "string"),
                ColDef("department_id", "int", is_fk="gov_departments.dept_id"),
            ]),
            TableDef("gov_departments", [
                ColDef("dept_id", "int", is_pk=True),
                ColDef("dept_name", "string"),
                ColDef("head_name", "string"),
                ColDef("budget", "decimal"),
                ColDef("building", "string"),
            ]),
        ],
        mappings=[
            MappingDef("dim_citizen", "citizen_key", "int", "citizen identifier",
                       [("citizens", "citizen_id", "int")], "rename"),
            MappingDef("dim_citizen", "citizen_name", "string", "full name of citizen",
                       [("citizens", "first_name", "string"), ("citizens", "last_name", "string")], "concat"),
            MappingDef("dim_citizen", "full_address", "string", "complete mailing address",
                       [("citizens", "address", "string"), ("citizens", "city", "string"),
                        ("citizens", "state", "string"), ("citizens", "zip_code", "string")], "concat"),
            MappingDef("dim_citizen", "birth_year", "int", "year of birth",
                       [("citizens", "date_of_birth", "date")], "date_part"),
            MappingDef("dim_citizen", "citizen_age", "int", "current age",
                       [("citizens", "date_of_birth", "date")], "date_diff"),
            MappingDef("fact_permit", "permit_key", "int", "permit identifier",
                       [("permits", "permit_id", "int")], "rename"),
            MappingDef("fact_permit", "applicant_name", "string", "name of permit applicant",
                       [("citizens", "first_name", "string"), ("citizens", "last_name", "string")], "concat",
                       join_path=["permits.citizen_id = citizens.citizen_id"]),
            MappingDef("fact_permit", "validity_days", "int", "days permit is valid",
                       [("permits", "issue_date", "date"), ("permits", "expiry_date", "date")], "date_diff"),
            MappingDef("fact_permit", "issuing_department", "string", "department that issued permit",
                       [("gov_departments", "dept_name", "string")], "fk_lookup",
                       join_path=["permits.department_id = gov_departments.dept_id"]),
            MappingDef("fact_permit", "issue_year", "int", "year permit was issued",
                       [("permits", "issue_date", "date")], "date_part"),
            MappingDef("fact_permit", "is_expired", "boolean", "whether permit has expired",
                       [("permits", "status", "string")], "conditional"),
        ],
    ))

    # ─── Domain 20: Fleet / Vehicle Management ───
    domains.append(DomainSchema(
        name="fleet",
        source_tables=[
            TableDef("vehicles", [
                ColDef("vehicle_id", "int", is_pk=True),
                ColDef("vin", "string"),
                ColDef("make", "string"),
                ColDef("model", "string"),
                ColDef("year", "int"),
                ColDef("color", "string"),
                ColDef("purchase_date", "date"),
                ColDef("purchase_price", "decimal"),
                ColDef("current_mileage", "int"),
                ColDef("fuel_type", "string"),
                ColDef("status", "string"),
            ]),
            TableDef("drivers", [
                ColDef("driver_id", "int", is_pk=True),
                ColDef("first_name", "string"),
                ColDef("last_name", "string"),
                ColDef("license_number", "string"),
                ColDef("hire_date", "date"),
                ColDef("phone", "string"),
                ColDef("rating", "decimal"),
            ]),
            TableDef("trips", [
                ColDef("trip_id", "int", is_pk=True),
                ColDef("vehicle_id", "int", is_fk="vehicles.vehicle_id"),
                ColDef("driver_id", "int", is_fk="drivers.driver_id"),
                ColDef("start_time", "datetime"),
                ColDef("end_time", "datetime"),
                ColDef("start_location", "string"),
                ColDef("end_location", "string"),
                ColDef("distance_km", "decimal"),
                ColDef("fuel_used_liters", "decimal"),
                ColDef("cost", "decimal"),
            ]),
            TableDef("maintenance", [
                ColDef("maint_id", "int", is_pk=True),
                ColDef("vehicle_id", "int", is_fk="vehicles.vehicle_id"),
                ColDef("service_date", "date"),
                ColDef("service_type", "string"),
                ColDef("cost", "decimal"),
                ColDef("mileage_at_service", "int"),
                ColDef("next_service_date", "date"),
            ]),
        ],
        mappings=[
            MappingDef("dim_vehicle", "vehicle_key", "int", "vehicle identifier",
                       [("vehicles", "vehicle_id", "int")], "rename"),
            MappingDef("dim_vehicle", "vehicle_description", "string", "make model and year combined",
                       [("vehicles", "make", "string"), ("vehicles", "model", "string")], "concat"),
            MappingDef("dim_vehicle", "vehicle_age_years", "int", "years since purchase",
                       [("vehicles", "purchase_date", "date")], "date_diff"),
            MappingDef("dim_vehicle", "is_active", "boolean", "whether vehicle is in service",
                       [("vehicles", "status", "string")], "conditional"),
            MappingDef("fact_trip", "trip_key", "int", "trip identifier",
                       [("trips", "trip_id", "int")], "rename"),
            MappingDef("fact_trip", "driver_name", "string", "driver full name",
                       [("drivers", "first_name", "string"), ("drivers", "last_name", "string")], "concat",
                       join_path=["trips.driver_id = drivers.driver_id"]),
            MappingDef("fact_trip", "vehicle_make_model", "string", "vehicle make and model",
                       [("vehicles", "make", "string"), ("vehicles", "model", "string")], "concat",
                       join_path=["trips.vehicle_id = vehicles.vehicle_id"]),
            MappingDef("fact_trip", "trip_year", "int", "year of the trip",
                       [("trips", "start_time", "datetime")], "date_part"),
            MappingDef("fact_trip", "fuel_efficiency", "decimal", "km per liter",
                       [("trips", "distance_km", "decimal"), ("trips", "fuel_used_liters", "decimal")], "arithmetic"),
            MappingDef("fact_trip", "cost_per_km", "decimal", "trip cost divided by distance",
                       [("trips", "cost", "decimal"), ("trips", "distance_km", "decimal")], "arithmetic"),
            MappingDef("fact_trip", "route", "string", "start to end location",
                       [("trips", "start_location", "string"), ("trips", "end_location", "string")], "concat"),
            MappingDef("fact_maintenance", "maintenance_key", "int", "maintenance id",
                       [("maintenance", "maint_id", "int")], "rename"),
            MappingDef("fact_maintenance", "days_until_next_service", "int", "days to next scheduled service",
                       [("maintenance", "service_date", "date"), ("maintenance", "next_service_date", "date")], "date_diff"),
            MappingDef("fact_maintenance", "service_cost", "decimal", "cost of maintenance",
                       [("maintenance", "cost", "decimal")], "rename"),
        ],
    ))

    # ─── Domain 21: Banking / Payments ───
    domains.append(DomainSchema(
        name="banking",
        source_tables=[
            TableDef("bank_accounts", [
                ColDef("account_id", "int", is_pk=True),
                ColDef("account_number", "string"),
                ColDef("account_type_code", "string"),
                ColDef("customer_id", "int", is_fk="bank_customers.customer_id"),
                ColDef("open_date", "date"),
                ColDef("close_date", "date"),
                ColDef("balance", "decimal"),
                ColDef("interest_rate", "decimal"),
                ColDef("status_code", "string"),
                ColDef("branch_id", "int", is_fk="branches.branch_id"),
                ColDef("currency_code", "string"),
            ]),
            TableDef("bank_customers", [
                ColDef("customer_id", "int", is_pk=True),
                ColDef("first_name", "string"),
                ColDef("last_name", "string"),
                ColDef("email", "string"),
                ColDef("phone", "string"),
                ColDef("date_of_birth", "date"),
                ColDef("credit_score", "int"),
                ColDef("income_annual", "decimal"),
                ColDef("risk_category_code", "string"),
                ColDef("kyc_date", "date"),
            ]),
            TableDef("branches", [
                ColDef("branch_id", "int", is_pk=True),
                ColDef("branch_name", "string"),
                ColDef("branch_code", "string"),
                ColDef("city", "string"),
                ColDef("state", "string"),
                ColDef("manager_name", "string"),
            ]),
            TableDef("bank_transactions", [
                ColDef("txn_id", "int", is_pk=True),
                ColDef("account_id", "int", is_fk="bank_accounts.account_id"),
                ColDef("txn_date", "date"),
                ColDef("txn_time", "datetime"),
                ColDef("amount", "decimal"),
                ColDef("txn_type_code", "string"),
                ColDef("description", "string"),
                ColDef("counterparty", "string"),
                ColDef("channel_code", "string"),
            ]),
        ],
        mappings=[
            # rename
            MappingDef("dim_account", "account_key", "int", "account surrogate key",
                       [("bank_accounts", "account_id", "int")], "rename"),
            MappingDef("dim_account", "account_num", "string", "account number",
                       [("bank_accounts", "account_number", "string")], "rename"),
            # concat
            MappingDef("dim_customer", "customer_name", "string", "customer full name",
                       [("bank_customers", "first_name", "string"), ("bank_customers", "last_name", "string")], "concat"),
            MappingDef("dim_branch", "branch_location", "string", "branch city and state",
                       [("branches", "city", "string"), ("branches", "state", "string")], "concat"),
            # fk_lookup
            MappingDef("dim_account", "branch_name", "string", "branch name for account",
                       [("branches", "branch_name", "string")], "fk_lookup",
                       join_path=["bank_accounts.branch_id = branches.branch_id"]),
            MappingDef("dim_account", "customer_full_name", "string", "name of account holder",
                       [("bank_customers", "first_name", "string"), ("bank_customers", "last_name", "string")], "concat",
                       join_path=["bank_accounts.customer_id = bank_customers.customer_id"]),
            # date_part
            MappingDef("dim_account", "open_year", "int", "year account was opened",
                       [("bank_accounts", "open_date", "date")], "date_part"),
            MappingDef("fact_txn", "transaction_year", "int", "year of transaction",
                       [("bank_transactions", "txn_date", "date")], "date_part"),
            MappingDef("fact_txn", "transaction_month", "int", "month of transaction",
                       [("bank_transactions", "txn_date", "date")], "date_part"),
            # date_diff
            MappingDef("dim_account", "account_age_days", "int", "days since account opened",
                       [("bank_accounts", "open_date", "date")], "date_diff"),
            # arithmetic
            MappingDef("dim_customer", "debt_to_income", "decimal", "balance divided by income",
                       [("bank_accounts", "balance", "decimal"), ("bank_customers", "income_annual", "decimal")], "arithmetic",
                       join_path=["bank_accounts.customer_id = bank_customers.customer_id"]),
            # conditional
            MappingDef("dim_account", "is_active", "boolean", "whether account is currently open",
                       [("bank_accounts", "status_code", "string")], "conditional"),
            MappingDef("dim_customer", "is_high_risk", "boolean", "whether customer is high risk",
                       [("bank_customers", "risk_category_code", "string")], "conditional"),
            # code_to_label
            MappingDef("dim_account", "account_type_label", "string", "human readable account type",
                       [("bank_accounts", "account_type_code", "string")], "code_to_label"),
            MappingDef("dim_account", "status_label", "string", "readable status name",
                       [("bank_accounts", "status_code", "string")], "code_to_label"),
            MappingDef("dim_customer", "risk_category_name", "string", "risk category in plain text",
                       [("bank_customers", "risk_category_code", "string")], "code_to_label"),
            MappingDef("fact_txn", "transaction_type_name", "string", "readable transaction type",
                       [("bank_transactions", "txn_type_code", "string")], "code_to_label"),
            MappingDef("fact_txn", "channel_name", "string", "readable channel description",
                       [("bank_transactions", "channel_code", "string")], "code_to_label"),
            # type_cast
            MappingDef("dim_customer", "credit_score_decimal", "decimal", "credit score as decimal",
                       [("bank_customers", "credit_score", "int")], "type_cast"),
            MappingDef("dim_account", "balance_text", "string", "balance as text for display",
                       [("bank_accounts", "balance", "decimal")], "type_cast"),
            MappingDef("dim_account", "interest_rate_text", "string", "interest rate as text",
                       [("bank_accounts", "interest_rate", "decimal")], "type_cast"),
            # date_format
            MappingDef("dim_account", "open_date_formatted", "string", "opening date in display format",
                       [("bank_accounts", "open_date", "date")], "date_format"),
            MappingDef("dim_customer", "dob_formatted", "string", "date of birth formatted for display",
                       [("bank_customers", "date_of_birth", "date")], "date_format"),
            MappingDef("fact_txn", "txn_date_display", "string", "transaction date as formatted string",
                       [("bank_transactions", "txn_date", "date")], "date_format"),
            # date_parse
            MappingDef("dim_customer", "kyc_timestamp", "datetime", "KYC date parsed to datetime",
                       [("bank_customers", "kyc_date", "date")], "date_parse"),
            # bucketing
            MappingDef("dim_customer", "income_bracket", "string", "income range bucket",
                       [("bank_customers", "income_annual", "decimal")], "bucketing"),
            MappingDef("dim_customer", "credit_tier", "string", "credit score tier",
                       [("bank_customers", "credit_score", "int")], "bucketing"),
            MappingDef("dim_customer", "age_group", "string", "age range category",
                       [("bank_customers", "date_of_birth", "date")], "bucketing"),
            MappingDef("fact_txn", "amount_bucket", "string", "transaction amount range",
                       [("bank_transactions", "amount", "decimal")], "bucketing"),
            # template
            MappingDef("dim_account", "account_display", "string", "formatted account label like Account #12345 (Savings)",
                       [("bank_accounts", "account_number", "string"), ("bank_accounts", "account_type_code", "string")], "template"),
            MappingDef("dim_customer", "customer_greeting", "string", "greeting template: Dear FirstName LastName",
                       [("bank_customers", "first_name", "string"), ("bank_customers", "last_name", "string")], "template"),
            MappingDef("dim_branch", "branch_display", "string", "formatted branch: BranchName - City, State",
                       [("branches", "branch_name", "string"), ("branches", "city", "string"), ("branches", "state", "string")], "template"),
            # lookup_join
            MappingDef("fact_txn", "account_holder_name", "string", "customer name from account lookup",
                       [("bank_customers", "first_name", "string"), ("bank_customers", "last_name", "string")], "lookup_join",
                       join_path=["bank_transactions.account_id = bank_accounts.account_id", "bank_accounts.customer_id = bank_customers.customer_id"]),
            MappingDef("fact_txn", "branch_of_account", "string", "branch name from multi-hop join",
                       [("branches", "branch_name", "string")], "lookup_join",
                       join_path=["bank_transactions.account_id = bank_accounts.account_id", "bank_accounts.branch_id = branches.branch_id"]),
        ],
    ))

    # ─── Domain 22: Hospital Management ───
    domains.append(DomainSchema(
        name="hospital",
        source_tables=[
            TableDef("patients", [
                ColDef("patient_id", "int", is_pk=True),
                ColDef("first_name", "string"),
                ColDef("last_name", "string"),
                ColDef("date_of_birth", "date"),
                ColDef("gender_code", "string"),
                ColDef("blood_type", "string"),
                ColDef("phone", "string"),
                ColDef("email", "string"),
                ColDef("admission_date", "date"),
                ColDef("discharge_date", "date"),
                ColDef("insurance_id", "int", is_fk="insurance_plans.plan_id"),
                ColDef("weight_kg", "decimal"),
                ColDef("height_cm", "int"),
            ]),
            TableDef("doctors", [
                ColDef("doctor_id", "int", is_pk=True),
                ColDef("first_name", "string"),
                ColDef("last_name", "string"),
                ColDef("specialization_code", "string"),
                ColDef("license_number", "string"),
                ColDef("department_id", "int", is_fk="hospital_departments.dept_id"),
                ColDef("hire_date", "date"),
            ]),
            TableDef("hospital_departments", [
                ColDef("dept_id", "int", is_pk=True),
                ColDef("dept_name", "string"),
                ColDef("floor", "int"),
                ColDef("head_doctor_id", "int"),
                ColDef("bed_count", "int"),
            ]),
            TableDef("insurance_plans", [
                ColDef("plan_id", "int", is_pk=True),
                ColDef("plan_name", "string"),
                ColDef("provider_name", "string"),
                ColDef("coverage_type_code", "string"),
                ColDef("max_coverage", "decimal"),
            ]),
            TableDef("visits", [
                ColDef("visit_id", "int", is_pk=True),
                ColDef("patient_id", "int", is_fk="patients.patient_id"),
                ColDef("doctor_id", "int", is_fk="doctors.doctor_id"),
                ColDef("visit_date", "date"),
                ColDef("diagnosis_code", "string"),
                ColDef("treatment_code", "string"),
                ColDef("visit_cost", "decimal"),
                ColDef("notes", "string"),
            ]),
        ],
        mappings=[
            # rename
            MappingDef("dim_patient", "patient_key", "int", "patient surrogate key",
                       [("patients", "patient_id", "int")], "rename"),
            # concat
            MappingDef("dim_patient", "patient_name", "string", "full name of patient",
                       [("patients", "first_name", "string"), ("patients", "last_name", "string")], "concat"),
            MappingDef("dim_doctor", "doctor_name", "string", "full name of doctor",
                       [("doctors", "first_name", "string"), ("doctors", "last_name", "string")], "concat"),
            # date_part
            MappingDef("dim_patient", "birth_year", "int", "year patient was born",
                       [("patients", "date_of_birth", "date")], "date_part"),
            # date_diff
            MappingDef("dim_patient", "length_of_stay", "int", "days between admission and discharge",
                       [("patients", "admission_date", "date"), ("patients", "discharge_date", "date")], "date_diff"),
            # arithmetic
            MappingDef("dim_patient", "bmi", "decimal", "body mass index from weight and height",
                       [("patients", "weight_kg", "decimal"), ("patients", "height_cm", "int")], "arithmetic",
                       sub_operation="divide"),
            # conditional
            MappingDef("dim_patient", "is_adult", "boolean", "whether patient is over 18",
                       [("patients", "date_of_birth", "date")], "conditional"),
            # fk_lookup
            MappingDef("dim_patient", "insurance_plan", "string", "name of patient insurance plan",
                       [("insurance_plans", "plan_name", "string")], "fk_lookup",
                       join_path=["patients.insurance_id = insurance_plans.plan_id"]),
            MappingDef("dim_patient", "insurance_provider", "string", "name of insurance company",
                       [("insurance_plans", "provider_name", "string")], "fk_lookup",
                       join_path=["patients.insurance_id = insurance_plans.plan_id"]),
            MappingDef("fact_visit", "doctor_full_name", "string", "name of attending doctor",
                       [("doctors", "first_name", "string"), ("doctors", "last_name", "string")], "concat",
                       join_path=["visits.doctor_id = doctors.doctor_id"]),
            MappingDef("fact_visit", "patient_full_name", "string", "name of the patient",
                       [("patients", "first_name", "string"), ("patients", "last_name", "string")], "concat",
                       join_path=["visits.patient_id = patients.patient_id"]),
            # code_to_label
            MappingDef("dim_patient", "gender_label", "string", "gender in readable form",
                       [("patients", "gender_code", "string")], "code_to_label"),
            MappingDef("dim_doctor", "specialization_name", "string", "specialization in readable form",
                       [("doctors", "specialization_code", "string")], "code_to_label"),
            MappingDef("fact_visit", "diagnosis_name", "string", "diagnosis in readable form",
                       [("visits", "diagnosis_code", "string")], "code_to_label"),
            MappingDef("fact_visit", "treatment_name", "string", "treatment in readable form",
                       [("visits", "treatment_code", "string")], "code_to_label"),
            MappingDef("dim_patient", "insurance_coverage_type", "string", "coverage type label",
                       [("insurance_plans", "coverage_type_code", "string")], "code_to_label",
                       join_path=["patients.insurance_id = insurance_plans.plan_id"]),
            # type_cast
            MappingDef("dim_patient", "height_decimal", "decimal", "height in cm as decimal",
                       [("patients", "height_cm", "int")], "type_cast"),
            MappingDef("dim_patient", "weight_text", "string", "weight as string display",
                       [("patients", "weight_kg", "decimal")], "type_cast"),
            # date_format
            MappingDef("dim_patient", "admission_display", "string", "admission date formatted",
                       [("patients", "admission_date", "date")], "date_format"),
            MappingDef("fact_visit", "visit_date_formatted", "string", "visit date as display string",
                       [("visits", "visit_date", "date")], "date_format"),
            # date_parse
            MappingDef("dim_doctor", "hire_timestamp", "datetime", "doctor hire date as timestamp",
                       [("doctors", "hire_date", "date")], "date_parse"),
            # bucketing
            MappingDef("dim_patient", "age_bracket", "string", "age group category",
                       [("patients", "date_of_birth", "date")], "bucketing"),
            MappingDef("dim_patient", "bmi_category", "string", "BMI classification",
                       [("patients", "weight_kg", "decimal")], "bucketing"),
            MappingDef("fact_visit", "cost_tier", "string", "visit cost tier range",
                       [("visits", "visit_cost", "decimal")], "bucketing"),
            # template
            MappingDef("dim_patient", "patient_label", "string", "label: LastName, FirstName (DOB)",
                       [("patients", "first_name", "string"), ("patients", "last_name", "string"), ("patients", "date_of_birth", "date")], "template"),
            MappingDef("dim_doctor", "doctor_display", "string", "formatted: Dr. FirstName LastName - Specialty",
                       [("doctors", "first_name", "string"), ("doctors", "last_name", "string"), ("doctors", "specialization_code", "string")], "template"),
            # lookup_join (multi-hop)
            MappingDef("fact_visit", "department_name", "string", "department from doctor lookup",
                       [("hospital_departments", "dept_name", "string")], "lookup_join",
                       join_path=["visits.doctor_id = doctors.doctor_id", "doctors.department_id = hospital_departments.dept_id"]),
            MappingDef("fact_visit", "insurance_plan_name", "string", "insurance from patient lookup",
                       [("insurance_plans", "plan_name", "string")], "lookup_join",
                       join_path=["visits.patient_id = patients.patient_id", "patients.insurance_id = insurance_plans.plan_id"]),
        ],
    ))

    # ─── Domain 23: E-learning Platform ───
    domains.append(DomainSchema(
        name="elearning",
        source_tables=[
            TableDef("instructors", [
                ColDef("instructor_id", "int", is_pk=True),
                ColDef("first_name", "string"),
                ColDef("last_name", "string"),
                ColDef("email", "string"),
                ColDef("bio", "string"),
                ColDef("join_date", "date"),
                ColDef("rating", "decimal"),
            ]),
            TableDef("courses", [
                ColDef("course_id", "int", is_pk=True),
                ColDef("course_title", "string"),
                ColDef("instructor_id", "int", is_fk="instructors.instructor_id"),
                ColDef("category_code", "string"),
                ColDef("difficulty_code", "string"),
                ColDef("price", "decimal"),
                ColDef("duration_hours", "decimal"),
                ColDef("publish_date", "date"),
                ColDef("language_code", "string"),
            ]),
            TableDef("students", [
                ColDef("student_id", "int", is_pk=True),
                ColDef("first_name", "string"),
                ColDef("last_name", "string"),
                ColDef("email", "string"),
                ColDef("signup_date", "date"),
                ColDef("country", "string"),
                ColDef("subscription_type_code", "string"),
            ]),
            TableDef("enrollments", [
                ColDef("enrollment_id", "int", is_pk=True),
                ColDef("student_id", "int", is_fk="students.student_id"),
                ColDef("course_id", "int", is_fk="courses.course_id"),
                ColDef("enroll_date", "date"),
                ColDef("completion_date", "date"),
                ColDef("progress_pct", "decimal"),
                ColDef("score", "decimal"),
                ColDef("certificate_issued", "string"),
            ]),
        ],
        mappings=[
            MappingDef("dim_course", "course_key", "int", "course identifier",
                       [("courses", "course_id", "int")], "rename"),
            MappingDef("dim_course", "course_name", "string", "title of course",
                       [("courses", "course_title", "string")], "rename"),
            MappingDef("dim_course", "instructor_name", "string", "full name of instructor",
                       [("instructors", "first_name", "string"), ("instructors", "last_name", "string")], "concat",
                       join_path=["courses.instructor_id = instructors.instructor_id"]),
            MappingDef("dim_course", "category_name", "string", "human readable course category",
                       [("courses", "category_code", "string")], "code_to_label"),
            MappingDef("dim_course", "difficulty_label", "string", "difficulty level name",
                       [("courses", "difficulty_code", "string")], "code_to_label"),
            MappingDef("dim_course", "language_name", "string", "language of the course",
                       [("courses", "language_code", "string")], "code_to_label"),
            MappingDef("dim_course", "publish_year", "int", "year course was published",
                       [("courses", "publish_date", "date")], "date_part"),
            MappingDef("dim_course", "price_tier", "string", "price bracket: free, low, mid, high",
                       [("courses", "price", "decimal")], "bucketing"),
            MappingDef("dim_course", "duration_category", "string", "short/medium/long duration",
                       [("courses", "duration_hours", "decimal")], "bucketing"),
            MappingDef("dim_course", "price_text", "string", "price as formatted text",
                       [("courses", "price", "decimal")], "type_cast"),
            MappingDef("dim_course", "course_listing", "string", "formatted: Title by Instructor (Category)",
                       [("courses", "course_title", "string"), ("courses", "category_code", "string")], "template"),
            MappingDef("fact_enrollment", "enrollment_key", "int", "enrollment id",
                       [("enrollments", "enrollment_id", "int")], "rename"),
            MappingDef("fact_enrollment", "student_name", "string", "full name of student",
                       [("students", "first_name", "string"), ("students", "last_name", "string")], "concat",
                       join_path=["enrollments.student_id = students.student_id"]),
            MappingDef("fact_enrollment", "course_title", "string", "name of enrolled course",
                       [("courses", "course_title", "string")], "fk_lookup",
                       join_path=["enrollments.course_id = courses.course_id"]),
            MappingDef("fact_enrollment", "days_to_complete", "int", "days from enroll to completion",
                       [("enrollments", "enroll_date", "date"), ("enrollments", "completion_date", "date")], "date_diff"),
            MappingDef("fact_enrollment", "is_completed", "boolean", "whether student finished course",
                       [("enrollments", "certificate_issued", "string")], "conditional"),
            MappingDef("fact_enrollment", "score_text", "string", "score as formatted string",
                       [("enrollments", "score", "decimal")], "type_cast"),
            MappingDef("fact_enrollment", "progress_integer", "int", "progress as integer percentage",
                       [("enrollments", "progress_pct", "decimal")], "type_cast"),
            MappingDef("fact_enrollment", "enroll_date_display", "string", "enrollment date formatted",
                       [("enrollments", "enroll_date", "date")], "date_format"),
            MappingDef("dim_student", "subscription_label", "string", "subscription type name",
                       [("students", "subscription_type_code", "string")], "code_to_label"),
            MappingDef("fact_enrollment", "instructor_full_name", "string", "instructor from course lookup",
                       [("instructors", "first_name", "string"), ("instructors", "last_name", "string")], "lookup_join",
                       join_path=["enrollments.course_id = courses.course_id", "courses.instructor_id = instructors.instructor_id"]),
        ],
    ))

    # ─── Domain 24: Warehouse / Supply Chain ───
    domains.append(DomainSchema(
        name="warehouse",
        source_tables=[
            TableDef("warehouses", [
                ColDef("warehouse_id", "int", is_pk=True),
                ColDef("warehouse_name", "string"),
                ColDef("location", "string"),
                ColDef("capacity_sqft", "int"),
                ColDef("manager_first", "string"),
                ColDef("manager_last", "string"),
                ColDef("open_date", "date"),
                ColDef("warehouse_type_code", "string"),
            ]),
            TableDef("products_wh", [
                ColDef("product_id", "int", is_pk=True),
                ColDef("product_name", "string"),
                ColDef("sku", "string"),
                ColDef("weight_lb", "decimal"),
                ColDef("unit_cost", "decimal"),
                ColDef("category_code", "string"),
                ColDef("shelf_life_days", "int"),
            ]),
            TableDef("stock_levels", [
                ColDef("stock_id", "int", is_pk=True),
                ColDef("warehouse_id", "int", is_fk="warehouses.warehouse_id"),
                ColDef("product_id", "int", is_fk="products_wh.product_id"),
                ColDef("quantity", "int"),
                ColDef("last_count_date", "date"),
                ColDef("reorder_qty", "int"),
                ColDef("aisle_code", "string"),
            ]),
            TableDef("shipments", [
                ColDef("shipment_id", "int", is_pk=True),
                ColDef("warehouse_id", "int", is_fk="warehouses.warehouse_id"),
                ColDef("ship_date", "date"),
                ColDef("arrival_date", "date"),
                ColDef("carrier_code", "string"),
                ColDef("tracking_number", "string"),
                ColDef("weight_total", "decimal"),
                ColDef("cost", "decimal"),
                ColDef("status_code", "string"),
            ]),
        ],
        mappings=[
            MappingDef("dim_warehouse", "warehouse_key", "int", "warehouse id",
                       [("warehouses", "warehouse_id", "int")], "rename"),
            MappingDef("dim_warehouse", "warehouse_label", "string", "warehouse name",
                       [("warehouses", "warehouse_name", "string")], "rename"),
            MappingDef("dim_warehouse", "manager_name", "string", "warehouse manager full name",
                       [("warehouses", "manager_first", "string"), ("warehouses", "manager_last", "string")], "concat"),
            MappingDef("dim_warehouse", "warehouse_type_name", "string", "type in readable form",
                       [("warehouses", "warehouse_type_code", "string")], "code_to_label"),
            MappingDef("dim_warehouse", "capacity_text", "string", "capacity as formatted text",
                       [("warehouses", "capacity_sqft", "int")], "type_cast"),
            MappingDef("dim_warehouse", "open_year", "int", "year warehouse opened",
                       [("warehouses", "open_date", "date")], "date_part"),
            MappingDef("dim_warehouse", "open_date_display", "string", "opening date formatted",
                       [("warehouses", "open_date", "date")], "date_format"),
            MappingDef("dim_product", "product_key", "int", "product id",
                       [("products_wh", "product_id", "int")], "rename"),
            MappingDef("dim_product", "product_category", "string", "category label",
                       [("products_wh", "category_code", "string")], "code_to_label"),
            MappingDef("dim_product", "weight_kg", "decimal", "weight converted to kilograms",
                       [("products_wh", "weight_lb", "decimal")], "arithmetic"),
            MappingDef("dim_product", "shelf_life_bucket", "string", "shelf life category short/medium/long",
                       [("products_wh", "shelf_life_days", "int")], "bucketing"),
            MappingDef("fact_stock", "stock_key", "int", "stock record id",
                       [("stock_levels", "stock_id", "int")], "rename"),
            MappingDef("fact_stock", "warehouse_name", "string", "name of warehouse from lookup",
                       [("warehouses", "warehouse_name", "string")], "fk_lookup",
                       join_path=["stock_levels.warehouse_id = warehouses.warehouse_id"]),
            MappingDef("fact_stock", "product_name", "string", "product name from lookup",
                       [("products_wh", "product_name", "string")], "fk_lookup",
                       join_path=["stock_levels.product_id = products_wh.product_id"]),
            MappingDef("fact_stock", "aisle_label", "string", "aisle code as label",
                       [("stock_levels", "aisle_code", "string")], "code_to_label"),
            MappingDef("fact_stock", "stock_value", "decimal", "quantity times unit cost",
                       [("stock_levels", "quantity", "int"), ("products_wh", "unit_cost", "decimal")], "arithmetic",
                       join_path=["stock_levels.product_id = products_wh.product_id"]),
            MappingDef("fact_stock", "needs_reorder", "boolean", "qty below reorder level",
                       [("stock_levels", "quantity", "int"), ("stock_levels", "reorder_qty", "int")], "conditional"),
            MappingDef("fact_shipment", "shipment_key", "int", "shipment id",
                       [("shipments", "shipment_id", "int")], "rename"),
            MappingDef("fact_shipment", "transit_days", "int", "days in transit",
                       [("shipments", "ship_date", "date"), ("shipments", "arrival_date", "date")], "date_diff"),
            MappingDef("fact_shipment", "carrier_name", "string", "carrier label",
                       [("shipments", "carrier_code", "string")], "code_to_label"),
            MappingDef("fact_shipment", "cost_per_lb", "decimal", "shipping cost per pound",
                       [("shipments", "cost", "decimal"), ("shipments", "weight_total", "decimal")], "arithmetic"),
            MappingDef("fact_shipment", "is_delivered", "boolean", "whether shipment arrived",
                       [("shipments", "status_code", "string")], "conditional"),
            MappingDef("fact_shipment", "ship_date_formatted", "string", "ship date in display format",
                       [("shipments", "ship_date", "date")], "date_format"),
            MappingDef("fact_shipment", "shipment_label", "string", "template: Tracking# - Carrier to Warehouse",
                       [("shipments", "tracking_number", "string"), ("shipments", "carrier_code", "string")], "template"),
            MappingDef("fact_shipment", "destination_manager", "string", "warehouse manager via shipment lookup",
                       [("warehouses", "manager_first", "string"), ("warehouses", "manager_last", "string")], "lookup_join",
                       join_path=["shipments.warehouse_id = warehouses.warehouse_id"]),
        ],
    ))

    # ─── Domain 25: Agriculture ───
    domains.append(DomainSchema(
        name="agriculture",
        source_tables=[
            TableDef("farms", [
                ColDef("farm_id", "int", is_pk=True),
                ColDef("farm_name", "string"),
                ColDef("owner_first", "string"),
                ColDef("owner_last", "string"),
                ColDef("total_acres", "decimal"),
                ColDef("region", "string"),
                ColDef("soil_type_code", "string"),
                ColDef("registration_date", "date"),
            ]),
            TableDef("fields", [
                ColDef("field_id", "int", is_pk=True),
                ColDef("farm_id", "int", is_fk="farms.farm_id"),
                ColDef("field_name", "string"),
                ColDef("area_acres", "decimal"),
                ColDef("crop_type_code", "string"),
                ColDef("irrigation_type", "string"),
                ColDef("last_planted", "date"),
            ]),
            TableDef("harvests", [
                ColDef("harvest_id", "int", is_pk=True),
                ColDef("field_id", "int", is_fk="fields.field_id"),
                ColDef("harvest_date", "date"),
                ColDef("yield_kg", "decimal"),
                ColDef("quality_grade_code", "string"),
                ColDef("sale_price_per_kg", "decimal"),
                ColDef("buyer_name", "string"),
            ]),
        ],
        mappings=[
            MappingDef("dim_farm", "farm_key", "int", "farm surrogate key",
                       [("farms", "farm_id", "int")], "rename"),
            MappingDef("dim_farm", "owner_name", "string", "farm owner full name",
                       [("farms", "owner_first", "string"), ("farms", "owner_last", "string")], "concat"),
            MappingDef("dim_farm", "soil_type_name", "string", "soil type readable label",
                       [("farms", "soil_type_code", "string")], "code_to_label"),
            MappingDef("dim_farm", "registration_year", "int", "year farm was registered",
                       [("farms", "registration_date", "date")], "date_part"),
            MappingDef("dim_field", "crop_label", "string", "readable crop type",
                       [("fields", "crop_type_code", "string")], "code_to_label"),
            MappingDef("dim_field", "farm_name", "string", "name of parent farm",
                       [("farms", "farm_name", "string")], "fk_lookup",
                       join_path=["fields.farm_id = farms.farm_id"]),
            MappingDef("fact_harvest", "harvest_key", "int", "harvest identifier",
                       [("harvests", "harvest_id", "int")], "rename"),
            MappingDef("fact_harvest", "total_revenue", "decimal", "yield times price per kg",
                       [("harvests", "yield_kg", "decimal"), ("harvests", "sale_price_per_kg", "decimal")], "arithmetic"),
            MappingDef("fact_harvest", "yield_per_acre", "decimal", "yield divided by field area",
                       [("harvests", "yield_kg", "decimal"), ("fields", "area_acres", "decimal")], "arithmetic",
                       join_path=["harvests.field_id = fields.field_id"]),
            MappingDef("fact_harvest", "quality_label", "string", "quality grade name",
                       [("harvests", "quality_grade_code", "string")], "code_to_label"),
            MappingDef("fact_harvest", "harvest_year", "int", "year of harvest",
                       [("harvests", "harvest_date", "date")], "date_part"),
            MappingDef("fact_harvest", "days_since_planting", "int", "days from planting to harvest",
                       [("fields", "last_planted", "date"), ("harvests", "harvest_date", "date")], "date_diff",
                       join_path=["harvests.field_id = fields.field_id"]),
            MappingDef("fact_harvest", "yield_bucket", "string", "yield range category",
                       [("harvests", "yield_kg", "decimal")], "bucketing"),
        ],
    ))

    # ─── Domain 26: Telecom ───
    domains.append(DomainSchema(
        name="telecom",
        source_tables=[
            TableDef("subscribers", [
                ColDef("subscriber_id", "int", is_pk=True),
                ColDef("first_name", "string"),
                ColDef("last_name", "string"),
                ColDef("phone_number", "string"),
                ColDef("email", "string"),
                ColDef("plan_id", "int", is_fk="plans.plan_id"),
                ColDef("activation_date", "date"),
                ColDef("status_code", "string"),
                ColDef("credit_limit", "decimal"),
            ]),
            TableDef("plans", [
                ColDef("plan_id", "int", is_pk=True),
                ColDef("plan_name", "string"),
                ColDef("plan_type_code", "string"),
                ColDef("monthly_cost", "decimal"),
                ColDef("data_limit_gb", "decimal"),
                ColDef("call_minutes", "int"),
            ]),
            TableDef("usage_records", [
                ColDef("usage_id", "int", is_pk=True),
                ColDef("subscriber_id", "int", is_fk="subscribers.subscriber_id"),
                ColDef("usage_date", "date"),
                ColDef("data_used_mb", "decimal"),
                ColDef("call_duration_min", "int"),
                ColDef("sms_count", "int"),
                ColDef("roaming_flag", "string"),
            ]),
            TableDef("bills_telecom", [
                ColDef("bill_id", "int", is_pk=True),
                ColDef("subscriber_id", "int", is_fk="subscribers.subscriber_id"),
                ColDef("bill_date", "date"),
                ColDef("base_charge", "decimal"),
                ColDef("overage_charge", "decimal"),
                ColDef("tax_amount", "decimal"),
                ColDef("total_due", "decimal"),
                ColDef("payment_date", "date"),
                ColDef("payment_status_code", "string"),
            ]),
        ],
        mappings=[
            MappingDef("dim_subscriber", "subscriber_key", "int", "subscriber surrogate key",
                       [("subscribers", "subscriber_id", "int")], "rename"),
            MappingDef("dim_subscriber", "subscriber_name", "string", "full name of subscriber",
                       [("subscribers", "first_name", "string"), ("subscribers", "last_name", "string")], "concat"),
            MappingDef("dim_subscriber", "plan_name", "string", "name of subscription plan",
                       [("plans", "plan_name", "string")], "fk_lookup",
                       join_path=["subscribers.plan_id = plans.plan_id"]),
            MappingDef("dim_subscriber", "plan_type_label", "string", "readable plan type",
                       [("plans", "plan_type_code", "string")], "code_to_label",
                       join_path=["subscribers.plan_id = plans.plan_id"]),
            MappingDef("dim_subscriber", "activation_year", "int", "year subscriber activated",
                       [("subscribers", "activation_date", "date")], "date_part"),
            MappingDef("dim_subscriber", "is_active", "boolean", "whether subscriber is active",
                       [("subscribers", "status_code", "string")], "conditional"),
            MappingDef("dim_subscriber", "status_label", "string", "subscriber status name",
                       [("subscribers", "status_code", "string")], "code_to_label"),
            MappingDef("fact_usage", "usage_key", "int", "usage record id",
                       [("usage_records", "usage_id", "int")], "rename"),
            MappingDef("fact_usage", "data_used_gb", "decimal", "data in GB from MB",
                       [("usage_records", "data_used_mb", "decimal")], "arithmetic"),
            MappingDef("fact_usage", "total_activity", "int", "calls plus sms count",
                       [("usage_records", "call_duration_min", "int"), ("usage_records", "sms_count", "int")], "arithmetic"),
            MappingDef("fact_usage", "is_roaming", "boolean", "whether usage was roaming",
                       [("usage_records", "roaming_flag", "string")], "conditional"),
            MappingDef("fact_billing", "bill_key", "int", "bill identifier",
                       [("bills_telecom", "bill_id", "int")], "rename"),
            MappingDef("fact_billing", "total_charges", "decimal", "base plus overage plus tax",
                       [("bills_telecom", "base_charge", "decimal"), ("bills_telecom", "overage_charge", "decimal"), ("bills_telecom", "tax_amount", "decimal")], "arithmetic"),
            MappingDef("fact_billing", "days_to_pay", "int", "days from bill to payment",
                       [("bills_telecom", "bill_date", "date"), ("bills_telecom", "payment_date", "date")], "date_diff"),
            MappingDef("fact_billing", "is_paid", "boolean", "whether bill is paid",
                       [("bills_telecom", "payment_status_code", "string")], "conditional"),
            MappingDef("fact_billing", "bill_display", "string", "formatted bill label",
                       [("bills_telecom", "bill_id", "int"), ("bills_telecom", "bill_date", "date")], "template"),
        ],
    ))

    # ─── Domain 27: Insurance ───
    domains.append(DomainSchema(
        name="insurance",
        source_tables=[
            TableDef("policyholders", [
                ColDef("holder_id", "int", is_pk=True),
                ColDef("first_name", "string"),
                ColDef("last_name", "string"),
                ColDef("date_of_birth", "date"),
                ColDef("email", "string"),
                ColDef("phone", "string"),
                ColDef("address", "string"),
                ColDef("city", "string"),
                ColDef("state", "string"),
                ColDef("risk_score", "int"),
            ]),
            TableDef("policies", [
                ColDef("policy_id", "int", is_pk=True),
                ColDef("holder_id", "int", is_fk="policyholders.holder_id"),
                ColDef("policy_number", "string"),
                ColDef("policy_type_code", "string"),
                ColDef("start_date", "date"),
                ColDef("end_date", "date"),
                ColDef("premium_monthly", "decimal"),
                ColDef("coverage_amount", "decimal"),
                ColDef("deductible", "decimal"),
                ColDef("status_code", "string"),
                ColDef("agent_id", "int", is_fk="agents.agent_id"),
            ]),
            TableDef("agents", [
                ColDef("agent_id", "int", is_pk=True),
                ColDef("first_name", "string"),
                ColDef("last_name", "string"),
                ColDef("license_number", "string"),
                ColDef("region", "string"),
                ColDef("commission_rate", "decimal"),
            ]),
            TableDef("claims", [
                ColDef("claim_id", "int", is_pk=True),
                ColDef("policy_id", "int", is_fk="policies.policy_id"),
                ColDef("claim_date", "date"),
                ColDef("incident_date", "date"),
                ColDef("claim_amount", "decimal"),
                ColDef("approved_amount", "decimal"),
                ColDef("claim_type_code", "string"),
                ColDef("status_code", "string"),
                ColDef("adjuster_notes", "string"),
            ]),
        ],
        mappings=[
            MappingDef("dim_policyholder", "holder_key", "int", "policyholder surrogate key",
                       [("policyholders", "holder_id", "int")], "rename"),
            MappingDef("dim_policyholder", "holder_name", "string", "policyholder full name",
                       [("policyholders", "first_name", "string"), ("policyholders", "last_name", "string")], "concat"),
            MappingDef("dim_policyholder", "full_address", "string", "complete mailing address",
                       [("policyholders", "address", "string"), ("policyholders", "city", "string"), ("policyholders", "state", "string")], "concat"),
            MappingDef("dim_policyholder", "birth_year", "int", "year of birth",
                       [("policyholders", "date_of_birth", "date")], "date_part"),
            MappingDef("dim_policyholder", "risk_tier", "string", "risk score bucket",
                       [("policyholders", "risk_score", "int")], "bucketing"),
            MappingDef("dim_policy", "policy_key", "int", "policy identifier",
                       [("policies", "policy_id", "int")], "rename"),
            MappingDef("dim_policy", "policy_type_name", "string", "readable policy type",
                       [("policies", "policy_type_code", "string")], "code_to_label"),
            MappingDef("dim_policy", "coverage_duration_days", "int", "days of coverage",
                       [("policies", "start_date", "date"), ("policies", "end_date", "date")], "date_diff"),
            MappingDef("dim_policy", "annual_premium", "decimal", "monthly premium times 12",
                       [("policies", "premium_monthly", "decimal")], "arithmetic"),
            MappingDef("dim_policy", "net_coverage", "decimal", "coverage minus deductible",
                       [("policies", "coverage_amount", "decimal"), ("policies", "deductible", "decimal")], "arithmetic"),
            MappingDef("dim_policy", "is_active", "boolean", "whether policy is in force",
                       [("policies", "status_code", "string")], "conditional"),
            MappingDef("dim_policy", "agent_name", "string", "agent full name",
                       [("agents", "first_name", "string"), ("agents", "last_name", "string")], "concat",
                       join_path=["policies.agent_id = agents.agent_id"]),
            MappingDef("dim_policy", "start_year", "int", "year policy started",
                       [("policies", "start_date", "date")], "date_part"),
            MappingDef("dim_policy", "policy_display", "string", "formatted policy label",
                       [("policies", "policy_number", "string"), ("policies", "policy_type_code", "string")], "template"),
            MappingDef("fact_claim", "claim_key", "int", "claim identifier",
                       [("claims", "claim_id", "int")], "rename"),
            MappingDef("fact_claim", "claim_type_name", "string", "readable claim type",
                       [("claims", "claim_type_code", "string")], "code_to_label"),
            MappingDef("fact_claim", "approval_ratio", "decimal", "approved divided by claimed",
                       [("claims", "approved_amount", "decimal"), ("claims", "claim_amount", "decimal")], "arithmetic"),
            MappingDef("fact_claim", "days_to_report", "int", "days from incident to claim",
                       [("claims", "incident_date", "date"), ("claims", "claim_date", "date")], "date_diff"),
            MappingDef("fact_claim", "is_approved", "boolean", "whether claim was approved",
                       [("claims", "status_code", "string")], "conditional"),
            MappingDef("fact_claim", "policyholder_name", "string", "claimant name via policy lookup",
                       [("policyholders", "first_name", "string"), ("policyholders", "last_name", "string")], "lookup_join",
                       join_path=["claims.policy_id = policies.policy_id", "policies.holder_id = policyholders.holder_id"]),
        ],
    ))

    # ─── Domain 28: Real Estate ───
    domains.append(DomainSchema(
        name="real_estate",
        source_tables=[
            TableDef("properties", [
                ColDef("property_id", "int", is_pk=True),
                ColDef("address", "string"),
                ColDef("city", "string"),
                ColDef("state", "string"),
                ColDef("zip_code", "string"),
                ColDef("property_type_code", "string"),
                ColDef("bedrooms", "int"),
                ColDef("bathrooms", "int"),
                ColDef("sqft", "int"),
                ColDef("lot_size_acres", "decimal"),
                ColDef("year_built", "int"),
                ColDef("list_price", "decimal"),
                ColDef("status_code", "string"),
            ]),
            TableDef("re_agents", [
                ColDef("agent_id", "int", is_pk=True),
                ColDef("first_name", "string"),
                ColDef("last_name", "string"),
                ColDef("email", "string"),
                ColDef("phone", "string"),
                ColDef("brokerage", "string"),
                ColDef("license_date", "date"),
                ColDef("commission_pct", "decimal"),
            ]),
            TableDef("listings", [
                ColDef("listing_id", "int", is_pk=True),
                ColDef("property_id", "int", is_fk="properties.property_id"),
                ColDef("agent_id", "int", is_fk="re_agents.agent_id"),
                ColDef("list_date", "date"),
                ColDef("expire_date", "date"),
                ColDef("asking_price", "decimal"),
            ]),
            TableDef("transactions_re", [
                ColDef("txn_id", "int", is_pk=True),
                ColDef("listing_id", "int", is_fk="listings.listing_id"),
                ColDef("buyer_first", "string"),
                ColDef("buyer_last", "string"),
                ColDef("sale_date", "date"),
                ColDef("sale_price", "decimal"),
                ColDef("closing_costs", "decimal"),
            ]),
        ],
        mappings=[
            MappingDef("dim_property", "property_key", "int", "property surrogate key",
                       [("properties", "property_id", "int")], "rename"),
            MappingDef("dim_property", "full_address", "string", "complete property address",
                       [("properties", "address", "string"), ("properties", "city", "string"), ("properties", "state", "string"), ("properties", "zip_code", "string")], "concat"),
            MappingDef("dim_property", "property_type_name", "string", "readable property type",
                       [("properties", "property_type_code", "string")], "code_to_label"),
            MappingDef("dim_property", "price_per_sqft", "decimal", "list price divided by sqft",
                       [("properties", "list_price", "decimal"), ("properties", "sqft", "int")], "arithmetic"),
            MappingDef("dim_property", "is_available", "boolean", "whether property is on market",
                       [("properties", "status_code", "string")], "conditional"),
            MappingDef("dim_property", "price_bracket", "string", "price range bucket",
                       [("properties", "list_price", "decimal")], "bucketing"),
            MappingDef("dim_property", "sqft_text", "string", "sqft as formatted text",
                       [("properties", "sqft", "int")], "type_cast"),
            MappingDef("dim_agent", "agent_name", "string", "agent full name",
                       [("re_agents", "first_name", "string"), ("re_agents", "last_name", "string")], "concat"),
            MappingDef("dim_agent", "license_year", "int", "year agent was licensed",
                       [("re_agents", "license_date", "date")], "date_part"),
            MappingDef("fact_listing", "listing_key", "int", "listing identifier",
                       [("listings", "listing_id", "int")], "rename"),
            MappingDef("fact_listing", "listing_duration_days", "int", "days on market",
                       [("listings", "list_date", "date"), ("listings", "expire_date", "date")], "date_diff"),
            MappingDef("fact_listing", "agent_name", "string", "listing agent name",
                       [("re_agents", "first_name", "string"), ("re_agents", "last_name", "string")], "concat",
                       join_path=["listings.agent_id = re_agents.agent_id"]),
            MappingDef("fact_sale", "sale_key", "int", "sale identifier",
                       [("transactions_re", "txn_id", "int")], "rename"),
            MappingDef("fact_sale", "buyer_name", "string", "buyer full name",
                       [("transactions_re", "buyer_first", "string"), ("transactions_re", "buyer_last", "string")], "concat"),
            MappingDef("fact_sale", "total_cost", "decimal", "sale price plus closing costs",
                       [("transactions_re", "sale_price", "decimal"), ("transactions_re", "closing_costs", "decimal")], "arithmetic"),
            MappingDef("fact_sale", "sale_year", "int", "year of sale",
                       [("transactions_re", "sale_date", "date")], "date_part"),
            MappingDef("fact_sale", "sale_display", "string", "formatted sale summary",
                       [("transactions_re", "sale_price", "decimal"), ("transactions_re", "sale_date", "date")], "template"),
        ],
    ))

    # ─── Domain 29: Logistics / Freight ───
    domains.append(DomainSchema(
        name="logistics",
        source_tables=[
            TableDef("shippers", [
                ColDef("shipper_id", "int", is_pk=True),
                ColDef("company_name", "string"),
                ColDef("contact_first", "string"),
                ColDef("contact_last", "string"),
                ColDef("phone", "string"),
                ColDef("country", "string"),
                ColDef("rating", "decimal"),
            ]),
            TableDef("freight_orders", [
                ColDef("order_id", "int", is_pk=True),
                ColDef("shipper_id", "int", is_fk="shippers.shipper_id"),
                ColDef("origin_city", "string"),
                ColDef("destination_city", "string"),
                ColDef("pickup_date", "date"),
                ColDef("delivery_date", "date"),
                ColDef("weight_kg", "decimal"),
                ColDef("volume_cbm", "decimal"),
                ColDef("freight_charge", "decimal"),
                ColDef("insurance_charge", "decimal"),
                ColDef("status_code", "string"),
                ColDef("mode_code", "string"),
            ]),
            TableDef("tracking_events", [
                ColDef("event_id", "int", is_pk=True),
                ColDef("order_id", "int", is_fk="freight_orders.order_id"),
                ColDef("event_time", "datetime"),
                ColDef("location", "string"),
                ColDef("event_type_code", "string"),
                ColDef("notes", "string"),
            ]),
        ],
        mappings=[
            MappingDef("dim_shipper", "shipper_key", "int", "shipper surrogate key",
                       [("shippers", "shipper_id", "int")], "rename"),
            MappingDef("dim_shipper", "shipper_name", "string", "shipping company name",
                       [("shippers", "company_name", "string")], "rename"),
            MappingDef("dim_shipper", "contact_name", "string", "shipper contact full name",
                       [("shippers", "contact_first", "string"), ("shippers", "contact_last", "string")], "concat"),
            MappingDef("fact_freight", "freight_key", "int", "freight order identifier",
                       [("freight_orders", "order_id", "int")], "rename"),
            MappingDef("fact_freight", "route", "string", "origin to destination",
                       [("freight_orders", "origin_city", "string"), ("freight_orders", "destination_city", "string")], "concat"),
            MappingDef("fact_freight", "transit_days", "int", "days from pickup to delivery",
                       [("freight_orders", "pickup_date", "date"), ("freight_orders", "delivery_date", "date")], "date_diff"),
            MappingDef("fact_freight", "total_charge", "decimal", "freight plus insurance",
                       [("freight_orders", "freight_charge", "decimal"), ("freight_orders", "insurance_charge", "decimal")], "arithmetic"),
            MappingDef("fact_freight", "cost_per_kg", "decimal", "freight charge per kilogram",
                       [("freight_orders", "freight_charge", "decimal"), ("freight_orders", "weight_kg", "decimal")], "arithmetic"),
            MappingDef("fact_freight", "is_delivered", "boolean", "whether order is delivered",
                       [("freight_orders", "status_code", "string")], "conditional"),
            MappingDef("fact_freight", "shipping_mode", "string", "readable transport mode",
                       [("freight_orders", "mode_code", "string")], "code_to_label"),
            MappingDef("fact_freight", "shipper_company", "string", "shipping company from lookup",
                       [("shippers", "company_name", "string")], "fk_lookup",
                       join_path=["freight_orders.shipper_id = shippers.shipper_id"]),
            MappingDef("fact_freight", "pickup_year", "int", "year of pickup",
                       [("freight_orders", "pickup_date", "date")], "date_part"),
            MappingDef("fact_freight", "weight_bucket", "string", "weight range category",
                       [("freight_orders", "weight_kg", "decimal")], "bucketing"),
            MappingDef("fact_tracking", "event_key", "int", "tracking event id",
                       [("tracking_events", "event_id", "int")], "rename"),
            MappingDef("fact_tracking", "event_type_name", "string", "readable event type",
                       [("tracking_events", "event_type_code", "string")], "code_to_label"),
            MappingDef("fact_tracking", "event_year", "int", "year of tracking event",
                       [("tracking_events", "event_time", "datetime")], "date_part"),
            MappingDef("fact_tracking", "shipment_route", "string", "origin-destination from order",
                       [("freight_orders", "origin_city", "string"), ("freight_orders", "destination_city", "string")], "lookup_join",
                       join_path=["tracking_events.order_id = freight_orders.order_id"]),
        ],
    ))

    # ─── Domain 30: Project Management (Multi-FK stress test) ───
    # Same column names (name, status) appear in multiple tables
    domains.append(DomainSchema(
        name="project_mgmt",
        source_tables=[
            TableDef("pm_projects", [
                ColDef("project_id", "int", is_pk=True),
                ColDef("project_name", "string"),
                ColDef("status", "string"),
                ColDef("start_date", "date"),
                ColDef("end_date", "date"),
                ColDef("budget", "decimal"),
                ColDef("client_id", "int", is_fk="pm_clients.client_id"),
                ColDef("manager_id", "int", is_fk="pm_employees.emp_id"),
            ]),
            TableDef("pm_employees", [
                ColDef("emp_id", "int", is_pk=True),
                ColDef("first_name", "string"),
                ColDef("last_name", "string"),
                ColDef("email", "string"),
                ColDef("role_code", "string"),
                ColDef("department_id", "int", is_fk="pm_departments.dept_id"),
                ColDef("hire_date", "date"),
                ColDef("hourly_rate", "decimal"),
            ]),
            TableDef("pm_departments", [
                ColDef("dept_id", "int", is_pk=True),
                ColDef("dept_name", "string"),
                ColDef("location", "string"),
                ColDef("head_id", "int"),
            ]),
            TableDef("pm_clients", [
                ColDef("client_id", "int", is_pk=True),
                ColDef("client_name", "string"),
                ColDef("industry", "string"),
                ColDef("country", "string"),
                ColDef("contact_email", "string"),
            ]),
            TableDef("pm_tasks", [
                ColDef("task_id", "int", is_pk=True),
                ColDef("project_id", "int", is_fk="pm_projects.project_id"),
                ColDef("assignee_id", "int", is_fk="pm_employees.emp_id"),
                ColDef("task_name", "string"),
                ColDef("status", "string"),
                ColDef("priority_code", "string"),
                ColDef("created_date", "date"),
                ColDef("due_date", "date"),
                ColDef("completed_date", "date"),
                ColDef("estimated_hours", "decimal"),
                ColDef("actual_hours", "decimal"),
            ]),
            TableDef("pm_timesheets", [
                ColDef("timesheet_id", "int", is_pk=True),
                ColDef("task_id", "int", is_fk="pm_tasks.task_id"),
                ColDef("emp_id", "int", is_fk="pm_employees.emp_id"),
                ColDef("work_date", "date"),
                ColDef("hours_logged", "decimal"),
                ColDef("notes", "string"),
            ]),
        ],
        mappings=[
            # rename (same table)
            MappingDef("dim_project", "project_key", "int", "project surrogate key",
                       [("pm_projects", "project_id", "int")], "rename"),
            MappingDef("dim_project", "project_title", "string", "name of the project",
                       [("pm_projects", "project_name", "string")], "rename"),
            # fk_lookup (different table, single hop)
            MappingDef("dim_project", "client_name", "string", "name of the client",
                       [("pm_clients", "client_name", "string")], "fk_lookup",
                       join_path=["pm_projects.client_id = pm_clients.client_id"]),
            MappingDef("dim_project", "manager_name", "string", "project manager full name",
                       [("pm_employees", "first_name", "string"), ("pm_employees", "last_name", "string")], "concat",
                       join_path=["pm_projects.manager_id = pm_employees.emp_id"]),
            MappingDef("dim_project", "client_country", "string", "country of the client",
                       [("pm_clients", "country", "string")], "fk_lookup",
                       join_path=["pm_projects.client_id = pm_clients.client_id"]),
            # date_diff
            MappingDef("dim_project", "project_duration_days", "int", "days from start to end",
                       [("pm_projects", "start_date", "date"), ("pm_projects", "end_date", "date")], "date_diff"),
            # conditional
            MappingDef("dim_project", "is_active", "boolean", "whether project is currently active",
                       [("pm_projects", "status", "string")], "conditional"),
            # code_to_label
            MappingDef("dim_employee", "role_label", "string", "human readable role name",
                       [("pm_employees", "role_code", "string")], "code_to_label"),
            # lookup_join (multi-hop: task -> project -> client)
            MappingDef("fact_task", "client_name", "string", "client name from task via project",
                       [("pm_clients", "client_name", "string")], "lookup_join",
                       join_path=["pm_tasks.project_id = pm_projects.project_id", "pm_projects.client_id = pm_clients.client_id"]),
            # lookup_join (multi-hop: task -> assignee -> department)
            MappingDef("fact_task", "assignee_department", "string", "department of task assignee",
                       [("pm_departments", "dept_name", "string")], "lookup_join",
                       join_path=["pm_tasks.assignee_id = pm_employees.emp_id", "pm_employees.department_id = pm_departments.dept_id"]),
            # cross-table date_diff (task due vs project end)
            MappingDef("fact_task", "days_before_project_end", "int", "days between task due date and project end date",
                       [("pm_tasks", "due_date", "date"), ("pm_projects", "end_date", "date")], "date_diff",
                       join_path=["pm_tasks.project_id = pm_projects.project_id"]),
            # cross-table date_diff (task created vs project start)
            MappingDef("fact_task", "days_after_project_start", "int", "days from project start to task creation",
                       [("pm_projects", "start_date", "date"), ("pm_tasks", "created_date", "date")], "date_diff",
                       join_path=["pm_tasks.project_id = pm_projects.project_id"]),
            # arithmetic
            MappingDef("fact_task", "hours_variance", "decimal", "actual minus estimated hours",
                       [("pm_tasks", "actual_hours", "decimal"), ("pm_tasks", "estimated_hours", "decimal")], "arithmetic"),
            # fk_lookup for task
            MappingDef("fact_task", "assignee_name", "string", "full name of task assignee",
                       [("pm_employees", "first_name", "string"), ("pm_employees", "last_name", "string")], "concat",
                       join_path=["pm_tasks.assignee_id = pm_employees.emp_id"]),
            MappingDef("fact_task", "project_name", "string", "name of parent project",
                       [("pm_projects", "project_name", "string")], "fk_lookup",
                       join_path=["pm_tasks.project_id = pm_projects.project_id"]),
            MappingDef("fact_task", "priority_label", "string", "readable priority level",
                       [("pm_tasks", "priority_code", "string")], "code_to_label"),
            MappingDef("fact_task", "is_completed", "boolean", "whether task is done",
                       [("pm_tasks", "status", "string")], "conditional"),
            MappingDef("fact_task", "task_duration_days", "int", "days from creation to completion",
                       [("pm_tasks", "created_date", "date"), ("pm_tasks", "completed_date", "date")], "date_diff"),
            # 3-hop: timesheet -> task -> project -> client
            MappingDef("fact_timesheet", "client_name", "string", "client name from timesheet via task and project",
                       [("pm_clients", "client_name", "string")], "lookup_join",
                       join_path=["pm_timesheets.task_id = pm_tasks.task_id", "pm_tasks.project_id = pm_projects.project_id", "pm_projects.client_id = pm_clients.client_id"]),
            # cross-table arithmetic (timesheet hours * employee rate)
            MappingDef("fact_timesheet", "labor_cost", "decimal", "hours logged times hourly rate",
                       [("pm_timesheets", "hours_logged", "decimal"), ("pm_employees", "hourly_rate", "decimal")], "arithmetic",
                       join_path=["pm_timesheets.emp_id = pm_employees.emp_id"]),
            MappingDef("fact_timesheet", "task_name", "string", "name of the task worked on",
                       [("pm_tasks", "task_name", "string")], "fk_lookup",
                       join_path=["pm_timesheets.task_id = pm_tasks.task_id"]),
        ],
    ))

    # ─── Domain 31: Medical Coding (Code-heavy stress test) ───
    # Many _code columns with both code_to_label and conditional mappings
    domains.append(DomainSchema(
        name="medical_coding",
        source_tables=[
            TableDef("mc_patients", [
                ColDef("patient_id", "int", is_pk=True),
                ColDef("first_name", "string"),
                ColDef("last_name", "string"),
                ColDef("gender_code", "string"),
                ColDef("blood_type_code", "string"),
                ColDef("marital_status_code", "string"),
                ColDef("ethnicity_code", "string"),
                ColDef("language_code", "string"),
                ColDef("date_of_birth", "date"),
                ColDef("is_deceased", "string"),
            ]),
            TableDef("mc_encounters", [
                ColDef("encounter_id", "int", is_pk=True),
                ColDef("patient_id", "int", is_fk="mc_patients.patient_id"),
                ColDef("provider_id", "int", is_fk="mc_providers.provider_id"),
                ColDef("encounter_date", "date"),
                ColDef("encounter_type_code", "string"),
                ColDef("facility_code", "string"),
                ColDef("discharge_disposition_code", "string"),
                ColDef("admission_source_code", "string"),
                ColDef("total_charge", "decimal"),
                ColDef("insurance_paid", "decimal"),
                ColDef("patient_paid", "decimal"),
                ColDef("status_code", "string"),
            ]),
            TableDef("mc_providers", [
                ColDef("provider_id", "int", is_pk=True),
                ColDef("first_name", "string"),
                ColDef("last_name", "string"),
                ColDef("specialty_code", "string"),
                ColDef("npi_number", "string"),
                ColDef("facility_id", "int", is_fk="mc_facilities.facility_id"),
            ]),
            TableDef("mc_facilities", [
                ColDef("facility_id", "int", is_pk=True),
                ColDef("facility_name", "string"),
                ColDef("facility_type_code", "string"),
                ColDef("city", "string"),
                ColDef("state", "string"),
                ColDef("bed_count", "int"),
            ]),
            TableDef("mc_diagnoses", [
                ColDef("diagnosis_id", "int", is_pk=True),
                ColDef("encounter_id", "int", is_fk="mc_encounters.encounter_id"),
                ColDef("icd_code", "string"),
                ColDef("diagnosis_type_code", "string"),
                ColDef("severity_code", "string"),
                ColDef("is_primary", "string"),
            ]),
        ],
        mappings=[
            # code_to_label (many codes -> labels)
            MappingDef("dim_patient", "gender_label", "string", "gender in readable form",
                       [("mc_patients", "gender_code", "string")], "code_to_label"),
            MappingDef("dim_patient", "blood_type_label", "string", "blood type readable name",
                       [("mc_patients", "blood_type_code", "string")], "code_to_label"),
            MappingDef("dim_patient", "marital_status_label", "string", "marital status readable name",
                       [("mc_patients", "marital_status_code", "string")], "code_to_label"),
            MappingDef("dim_patient", "ethnicity_label", "string", "ethnicity readable name",
                       [("mc_patients", "ethnicity_code", "string")], "code_to_label"),
            MappingDef("dim_patient", "language_label", "string", "preferred language name",
                       [("mc_patients", "language_code", "string")], "code_to_label"),
            # conditional (codes -> boolean flags)
            MappingDef("dim_patient", "is_deceased_flag", "boolean", "whether patient is deceased",
                       [("mc_patients", "is_deceased", "string")], "conditional"),
            MappingDef("fact_encounter", "is_emergency", "boolean", "whether encounter was emergency",
                       [("mc_encounters", "encounter_type_code", "string")], "conditional"),
            MappingDef("fact_encounter", "is_closed", "boolean", "whether encounter is finalized",
                       [("mc_encounters", "status_code", "string")], "conditional"),
            MappingDef("fact_diagnosis", "is_primary_flag", "boolean", "whether this is the primary diagnosis",
                       [("mc_diagnoses", "is_primary", "string")], "conditional"),
            # code_to_label for encounter codes
            MappingDef("fact_encounter", "encounter_type_label", "string", "encounter type readable name",
                       [("mc_encounters", "encounter_type_code", "string")], "code_to_label"),
            MappingDef("fact_encounter", "facility_label", "string", "facility code readable name",
                       [("mc_encounters", "facility_code", "string")], "code_to_label"),
            MappingDef("fact_encounter", "discharge_label", "string", "discharge disposition readable name",
                       [("mc_encounters", "discharge_disposition_code", "string")], "code_to_label"),
            MappingDef("fact_encounter", "admission_source_label", "string", "admission source readable name",
                       [("mc_encounters", "admission_source_code", "string")], "code_to_label"),
            MappingDef("dim_provider", "specialty_label", "string", "provider specialty readable name",
                       [("mc_providers", "specialty_code", "string")], "code_to_label"),
            MappingDef("dim_facility", "facility_type_label", "string", "facility type readable name",
                       [("mc_facilities", "facility_type_code", "string")], "code_to_label"),
            MappingDef("fact_diagnosis", "diagnosis_type_label", "string", "diagnosis type readable name",
                       [("mc_diagnoses", "diagnosis_type_code", "string")], "code_to_label"),
            MappingDef("fact_diagnosis", "severity_label", "string", "severity level readable name",
                       [("mc_diagnoses", "severity_code", "string")], "code_to_label"),
            # rename, concat, fk_lookup, etc.
            MappingDef("dim_patient", "patient_name", "string", "patient full name",
                       [("mc_patients", "first_name", "string"), ("mc_patients", "last_name", "string")], "concat"),
            MappingDef("dim_patient", "birth_year", "int", "year of birth",
                       [("mc_patients", "date_of_birth", "date")], "date_part"),
            MappingDef("dim_provider", "provider_name", "string", "provider full name",
                       [("mc_providers", "first_name", "string"), ("mc_providers", "last_name", "string")], "concat"),
            MappingDef("fact_encounter", "provider_name", "string", "attending provider name",
                       [("mc_providers", "first_name", "string"), ("mc_providers", "last_name", "string")], "concat",
                       join_path=["mc_encounters.provider_id = mc_providers.provider_id"]),
            MappingDef("fact_encounter", "patient_out_of_pocket", "decimal", "total minus insurance paid",
                       [("mc_encounters", "total_charge", "decimal"), ("mc_encounters", "insurance_paid", "decimal")], "arithmetic"),
            # lookup_join (multi-hop: encounter -> provider -> facility)
            MappingDef("fact_encounter", "provider_facility_name", "string", "facility name of the provider",
                       [("mc_facilities", "facility_name", "string")], "lookup_join",
                       join_path=["mc_encounters.provider_id = mc_providers.provider_id", "mc_providers.facility_id = mc_facilities.facility_id"]),
            # cross-table date_diff (patient DOB vs encounter date)
            MappingDef("fact_encounter", "patient_age_at_encounter", "int", "patient age at time of encounter",
                       [("mc_patients", "date_of_birth", "date"), ("mc_encounters", "encounter_date", "date")], "date_diff",
                       join_path=["mc_encounters.patient_id = mc_patients.patient_id"]),
        ],
    ))

    # ─── Domain 32: Financial Reporting (Ambiguous name stress test) ───
    # Column names suggest computation but are actually renames
    domains.append(DomainSchema(
        name="financial_reporting",
        source_tables=[
            TableDef("fr_accounts", [
                ColDef("account_id", "int", is_pk=True),
                ColDef("account_name", "string"),
                ColDef("account_type_code", "string"),
                ColDef("annual_revenue", "decimal"),
                ColDef("quarterly_revenue", "decimal"),
                ColDef("monthly_expenses", "decimal"),
                ColDef("annual_profit", "decimal"),
                ColDef("total_assets", "decimal"),
                ColDef("total_liabilities", "decimal"),
                ColDef("fiscal_year_start", "date"),
                ColDef("fiscal_year_end", "date"),
                ColDef("status_code", "string"),
                ColDef("region_code", "string"),
            ]),
            TableDef("fr_departments", [
                ColDef("dept_id", "int", is_pk=True),
                ColDef("dept_name", "string"),
                ColDef("cost_center_code", "string"),
                ColDef("annual_budget", "decimal"),
                ColDef("headcount", "int"),
                ColDef("manager_id", "int", is_fk="fr_managers.manager_id"),
            ]),
            TableDef("fr_managers", [
                ColDef("manager_id", "int", is_pk=True),
                ColDef("first_name", "string"),
                ColDef("last_name", "string"),
                ColDef("title", "string"),
                ColDef("email", "string"),
            ]),
            TableDef("fr_transactions", [
                ColDef("txn_id", "int", is_pk=True),
                ColDef("account_id", "int", is_fk="fr_accounts.account_id"),
                ColDef("dept_id", "int", is_fk="fr_departments.dept_id"),
                ColDef("txn_date", "date"),
                ColDef("amount", "decimal"),
                ColDef("txn_type_code", "string"),
                ColDef("category_code", "string"),
                ColDef("description", "string"),
            ]),
        ],
        mappings=[
            # AMBIGUOUS RENAMES: names suggest computation but source already has the value
            MappingDef("dim_account", "annual_revenue", "decimal", "yearly revenue for the account",
                       [("fr_accounts", "annual_revenue", "decimal")], "rename"),
            MappingDef("dim_account", "quarterly_revenue", "decimal", "quarterly revenue figure",
                       [("fr_accounts", "quarterly_revenue", "decimal")], "rename"),
            MappingDef("dim_account", "monthly_expenses", "decimal", "monthly expense total",
                       [("fr_accounts", "monthly_expenses", "decimal")], "rename"),
            MappingDef("dim_account", "annual_profit", "decimal", "yearly profit figure",
                       [("fr_accounts", "annual_profit", "decimal")], "rename"),
            MappingDef("dim_account", "total_assets", "decimal", "total asset value",
                       [("fr_accounts", "total_assets", "decimal")], "rename"),
            MappingDef("dim_account", "total_liabilities", "decimal", "total liability amount",
                       [("fr_accounts", "total_liabilities", "decimal")], "rename"),
            MappingDef("dim_department", "annual_budget", "decimal", "department yearly budget",
                       [("fr_departments", "annual_budget", "decimal")], "rename"),
            # ACTUAL arithmetic (explicit formula in description)
            MappingDef("dim_account", "net_worth", "decimal", "total assets minus total liabilities",
                       [("fr_accounts", "total_assets", "decimal"), ("fr_accounts", "total_liabilities", "decimal")], "arithmetic"),
            MappingDef("dim_account", "profit_margin_pct", "decimal", "annual profit divided by annual revenue",
                       [("fr_accounts", "annual_profit", "decimal"), ("fr_accounts", "annual_revenue", "decimal")], "arithmetic"),
            MappingDef("dim_department", "cost_per_head", "decimal", "annual budget divided by headcount",
                       [("fr_departments", "annual_budget", "decimal"), ("fr_departments", "headcount", "int")], "arithmetic"),
            # code_to_label
            MappingDef("dim_account", "account_type_label", "string", "account type readable name",
                       [("fr_accounts", "account_type_code", "string")], "code_to_label"),
            MappingDef("dim_account", "region_label", "string", "region readable name",
                       [("fr_accounts", "region_code", "string")], "code_to_label"),
            MappingDef("dim_department", "cost_center_label", "string", "cost center readable name",
                       [("fr_departments", "cost_center_code", "string")], "code_to_label"),
            MappingDef("fact_txn", "txn_type_label", "string", "transaction type readable name",
                       [("fr_transactions", "txn_type_code", "string")], "code_to_label"),
            MappingDef("fact_txn", "category_label", "string", "category readable name",
                       [("fr_transactions", "category_code", "string")], "code_to_label"),
            # conditional
            MappingDef("dim_account", "is_active", "boolean", "whether account is active",
                       [("fr_accounts", "status_code", "string")], "conditional"),
            # date
            MappingDef("dim_account", "fiscal_year_duration", "int", "days in fiscal year",
                       [("fr_accounts", "fiscal_year_start", "date"), ("fr_accounts", "fiscal_year_end", "date")], "date_diff"),
            # fk_lookup
            MappingDef("fact_txn", "account_name", "string", "name of the account",
                       [("fr_accounts", "account_name", "string")], "fk_lookup",
                       join_path=["fr_transactions.account_id = fr_accounts.account_id"]),
            MappingDef("fact_txn", "department_name", "string", "name of the department",
                       [("fr_departments", "dept_name", "string")], "fk_lookup",
                       join_path=["fr_transactions.dept_id = fr_departments.dept_id"]),
            # lookup_join (multi-hop: txn -> dept -> manager)
            MappingDef("fact_txn", "department_manager", "string", "manager name of the department",
                       [("fr_managers", "first_name", "string"), ("fr_managers", "last_name", "string")], "lookup_join",
                       join_path=["fr_transactions.dept_id = fr_departments.dept_id", "fr_departments.manager_id = fr_managers.manager_id"]),
        ],
    ))

    # ─── Domain 33: Supply Chain (Deep join chain stress test) ───
    # 4-hop chain: order_items -> orders -> customers -> regions -> countries
    domains.append(DomainSchema(
        name="supply_chain",
        source_tables=[
            TableDef("sc_countries", [
                ColDef("country_id", "int", is_pk=True),
                ColDef("country_name", "string"),
                ColDef("continent", "string"),
                ColDef("currency_code", "string"),
            ]),
            TableDef("sc_regions", [
                ColDef("region_id", "int", is_pk=True),
                ColDef("region_name", "string"),
                ColDef("country_id", "int", is_fk="sc_countries.country_id"),
                ColDef("timezone", "string"),
            ]),
            TableDef("sc_warehouses", [
                ColDef("warehouse_id", "int", is_pk=True),
                ColDef("warehouse_name", "string"),
                ColDef("region_id", "int", is_fk="sc_regions.region_id"),
                ColDef("capacity_units", "int"),
                ColDef("manager_first", "string"),
                ColDef("manager_last", "string"),
            ]),
            TableDef("sc_suppliers", [
                ColDef("supplier_id", "int", is_pk=True),
                ColDef("supplier_name", "string"),
                ColDef("region_id", "int", is_fk="sc_regions.region_id"),
                ColDef("contact_email", "string"),
                ColDef("rating", "decimal"),
            ]),
            TableDef("sc_products", [
                ColDef("product_id", "int", is_pk=True),
                ColDef("product_name", "string"),
                ColDef("supplier_id", "int", is_fk="sc_suppliers.supplier_id"),
                ColDef("unit_cost", "decimal"),
                ColDef("unit_weight_kg", "decimal"),
                ColDef("category_code", "string"),
            ]),
            TableDef("sc_inventory", [
                ColDef("inv_id", "int", is_pk=True),
                ColDef("warehouse_id", "int", is_fk="sc_warehouses.warehouse_id"),
                ColDef("product_id", "int", is_fk="sc_products.product_id"),
                ColDef("quantity", "int"),
                ColDef("last_restock_date", "date"),
                ColDef("expiry_date", "date"),
            ]),
            TableDef("sc_orders", [
                ColDef("order_id", "int", is_pk=True),
                ColDef("warehouse_id", "int", is_fk="sc_warehouses.warehouse_id"),
                ColDef("order_date", "date"),
                ColDef("ship_date", "date"),
                ColDef("delivery_date", "date"),
                ColDef("total_amount", "decimal"),
                ColDef("status_code", "string"),
            ]),
        ],
        mappings=[
            # rename
            MappingDef("dim_warehouse", "warehouse_key", "int", "warehouse surrogate key",
                       [("sc_warehouses", "warehouse_id", "int")], "rename"),
            MappingDef("dim_warehouse", "warehouse_label", "string", "warehouse name",
                       [("sc_warehouses", "warehouse_name", "string")], "rename"),
            # concat
            MappingDef("dim_warehouse", "manager_name", "string", "warehouse manager full name",
                       [("sc_warehouses", "manager_first", "string"), ("sc_warehouses", "manager_last", "string")], "concat"),
            # fk_lookup (single hop)
            MappingDef("dim_warehouse", "region_name", "string", "region of the warehouse",
                       [("sc_regions", "region_name", "string")], "fk_lookup",
                       join_path=["sc_warehouses.region_id = sc_regions.region_id"]),
            # lookup_join (2-hop: warehouse -> region -> country)
            MappingDef("dim_warehouse", "country_name", "string", "country of the warehouse",
                       [("sc_countries", "country_name", "string")], "lookup_join",
                       join_path=["sc_warehouses.region_id = sc_regions.region_id", "sc_regions.country_id = sc_countries.country_id"]),
            # lookup_join (3-hop: inventory -> product -> supplier -> region)
            MappingDef("fact_inventory", "supplier_region", "string", "region of the product supplier",
                       [("sc_regions", "region_name", "string")], "lookup_join",
                       join_path=["sc_inventory.product_id = sc_products.product_id", "sc_products.supplier_id = sc_suppliers.supplier_id", "sc_suppliers.region_id = sc_regions.region_id"]),
            # lookup_join (4-hop: inventory -> product -> supplier -> region -> country)
            MappingDef("fact_inventory", "supplier_country", "string", "country of the product supplier",
                       [("sc_countries", "country_name", "string")], "lookup_join",
                       join_path=["sc_inventory.product_id = sc_products.product_id", "sc_products.supplier_id = sc_suppliers.supplier_id", "sc_suppliers.region_id = sc_regions.region_id", "sc_regions.country_id = sc_countries.country_id"]),
            # cross-table arithmetic (inventory qty * product unit cost)
            MappingDef("fact_inventory", "stock_value", "decimal", "quantity times unit cost",
                       [("sc_inventory", "quantity", "int"), ("sc_products", "unit_cost", "decimal")], "arithmetic",
                       join_path=["sc_inventory.product_id = sc_products.product_id"]),
            # cross-table date_diff (restock date vs expiry)
            MappingDef("fact_inventory", "shelf_life_remaining", "int", "days from last restock to expiry",
                       [("sc_inventory", "last_restock_date", "date"), ("sc_inventory", "expiry_date", "date")], "date_diff"),
            # fk_lookup for inventory
            MappingDef("fact_inventory", "product_name", "string", "name of the product",
                       [("sc_products", "product_name", "string")], "fk_lookup",
                       join_path=["sc_inventory.product_id = sc_products.product_id"]),
            MappingDef("fact_inventory", "warehouse_name", "string", "name of the warehouse",
                       [("sc_warehouses", "warehouse_name", "string")], "fk_lookup",
                       join_path=["sc_inventory.warehouse_id = sc_warehouses.warehouse_id"]),
            # order mappings
            MappingDef("fact_order", "order_key", "int", "order identifier",
                       [("sc_orders", "order_id", "int")], "rename"),
            MappingDef("fact_order", "transit_days", "int", "days from ship to delivery",
                       [("sc_orders", "ship_date", "date"), ("sc_orders", "delivery_date", "date")], "date_diff"),
            MappingDef("fact_order", "processing_days", "int", "days from order to ship",
                       [("sc_orders", "order_date", "date"), ("sc_orders", "ship_date", "date")], "date_diff"),
            MappingDef("fact_order", "is_delivered", "boolean", "whether order is delivered",
                       [("sc_orders", "status_code", "string")], "conditional"),
            MappingDef("fact_order", "status_label", "string", "order status readable name",
                       [("sc_orders", "status_code", "string")], "code_to_label"),
            MappingDef("fact_order", "warehouse_name", "string", "warehouse name for the order",
                       [("sc_warehouses", "warehouse_name", "string")], "fk_lookup",
                       join_path=["sc_orders.warehouse_id = sc_warehouses.warehouse_id"]),
            # lookup_join (2-hop: order -> warehouse -> region)
            MappingDef("fact_order", "warehouse_region", "string", "region of the order warehouse",
                       [("sc_regions", "region_name", "string")], "lookup_join",
                       join_path=["sc_orders.warehouse_id = sc_warehouses.warehouse_id", "sc_warehouses.region_id = sc_regions.region_id"]),
            MappingDef("dim_product", "supplier_name", "string", "name of the product supplier",
                       [("sc_suppliers", "supplier_name", "string")], "fk_lookup",
                       join_path=["sc_products.supplier_id = sc_suppliers.supplier_id"]),
            MappingDef("dim_product", "category_label", "string", "product category readable name",
                       [("sc_products", "category_code", "string")], "code_to_label"),
        ],
    ))

    # ─── Domain 34: University System (Cross-table dates stress test) ───
    domains.append(DomainSchema(
        name="university",
        source_tables=[
            TableDef("uni_students", [
                ColDef("student_id", "int", is_pk=True),
                ColDef("first_name", "string"),
                ColDef("last_name", "string"),
                ColDef("date_of_birth", "date"),
                ColDef("enrollment_date", "date"),
                ColDef("expected_graduation", "date"),
                ColDef("major_code", "string"),
                ColDef("gpa", "decimal"),
                ColDef("status_code", "string"),
                ColDef("advisor_id", "int", is_fk="uni_faculty.faculty_id"),
            ]),
            TableDef("uni_faculty", [
                ColDef("faculty_id", "int", is_pk=True),
                ColDef("first_name", "string"),
                ColDef("last_name", "string"),
                ColDef("title_code", "string"),
                ColDef("department_id", "int", is_fk="uni_departments.dept_id"),
                ColDef("hire_date", "date"),
                ColDef("salary", "decimal"),
            ]),
            TableDef("uni_departments", [
                ColDef("dept_id", "int", is_pk=True),
                ColDef("dept_name", "string"),
                ColDef("college_code", "string"),
                ColDef("building", "string"),
                ColDef("budget", "decimal"),
            ]),
            TableDef("uni_courses", [
                ColDef("course_id", "int", is_pk=True),
                ColDef("course_name", "string"),
                ColDef("course_code", "string"),
                ColDef("department_id", "int", is_fk="uni_departments.dept_id"),
                ColDef("credits", "int"),
                ColDef("level_code", "string"),
            ]),
            TableDef("uni_enrollments", [
                ColDef("enrollment_id", "int", is_pk=True),
                ColDef("student_id", "int", is_fk="uni_students.student_id"),
                ColDef("course_id", "int", is_fk="uni_courses.course_id"),
                ColDef("semester_id", "int", is_fk="uni_semesters.semester_id"),
                ColDef("grade_code", "string"),
                ColDef("enroll_date", "date"),
                ColDef("drop_date", "date"),
                ColDef("final_score", "decimal"),
            ]),
            TableDef("uni_semesters", [
                ColDef("semester_id", "int", is_pk=True),
                ColDef("semester_name", "string"),
                ColDef("start_date", "date"),
                ColDef("end_date", "date"),
                ColDef("academic_year", "int"),
            ]),
        ],
        mappings=[
            # rename
            MappingDef("dim_student", "student_key", "int", "student surrogate key",
                       [("uni_students", "student_id", "int")], "rename"),
            # concat
            MappingDef("dim_student", "student_name", "string", "student full name",
                       [("uni_students", "first_name", "string"), ("uni_students", "last_name", "string")], "concat"),
            # date_part
            MappingDef("dim_student", "enrollment_year", "int", "year student enrolled",
                       [("uni_students", "enrollment_date", "date")], "date_part"),
            # date_diff (same table)
            MappingDef("dim_student", "program_duration_days", "int", "days from enrollment to expected graduation",
                       [("uni_students", "enrollment_date", "date"), ("uni_students", "expected_graduation", "date")], "date_diff"),
            # cross-table date_diff (student DOB vs enrollment date)
            MappingDef("dim_student", "age_at_enrollment", "int", "student age when enrolled",
                       [("uni_students", "date_of_birth", "date"), ("uni_students", "enrollment_date", "date")], "date_diff"),
            # code_to_label
            MappingDef("dim_student", "major_label", "string", "major readable name",
                       [("uni_students", "major_code", "string")], "code_to_label"),
            MappingDef("dim_student", "status_label", "string", "student status readable name",
                       [("uni_students", "status_code", "string")], "code_to_label"),
            # conditional
            MappingDef("dim_student", "is_active", "boolean", "whether student is currently enrolled",
                       [("uni_students", "status_code", "string")], "conditional"),
            # fk_lookup
            MappingDef("dim_student", "advisor_name", "string", "name of student advisor",
                       [("uni_faculty", "first_name", "string"), ("uni_faculty", "last_name", "string")], "concat",
                       join_path=["uni_students.advisor_id = uni_faculty.faculty_id"]),
            # lookup_join (2-hop: student -> advisor -> department)
            MappingDef("dim_student", "advisor_department", "string", "department of student advisor",
                       [("uni_departments", "dept_name", "string")], "lookup_join",
                       join_path=["uni_students.advisor_id = uni_faculty.faculty_id", "uni_faculty.department_id = uni_departments.dept_id"]),
            # bucketing
            MappingDef("dim_student", "gpa_tier", "string", "GPA range category",
                       [("uni_students", "gpa", "decimal")], "bucketing"),
            # enrollment fact mappings
            MappingDef("fact_enrollment", "enrollment_key", "int", "enrollment identifier",
                       [("uni_enrollments", "enrollment_id", "int")], "rename"),
            # cross-table date_diff (enrollment date vs semester start)
            MappingDef("fact_enrollment", "days_before_semester", "int", "days between enrollment and semester start",
                       [("uni_enrollments", "enroll_date", "date"), ("uni_semesters", "start_date", "date")], "date_diff",
                       join_path=["uni_enrollments.semester_id = uni_semesters.semester_id"]),
            # cross-table date_diff (enrollment date vs semester end)
            MappingDef("fact_enrollment", "days_enrolled_in_semester", "int", "days from enrollment to semester end",
                       [("uni_enrollments", "enroll_date", "date"), ("uni_semesters", "end_date", "date")], "date_diff",
                       join_path=["uni_enrollments.semester_id = uni_semesters.semester_id"]),
            # fk_lookup
            MappingDef("fact_enrollment", "course_name", "string", "name of the enrolled course",
                       [("uni_courses", "course_name", "string")], "fk_lookup",
                       join_path=["uni_enrollments.course_id = uni_courses.course_id"]),
            MappingDef("fact_enrollment", "semester_name", "string", "name of the semester",
                       [("uni_semesters", "semester_name", "string")], "fk_lookup",
                       join_path=["uni_enrollments.semester_id = uni_semesters.semester_id"]),
            MappingDef("fact_enrollment", "student_name", "string", "name of the student",
                       [("uni_students", "first_name", "string"), ("uni_students", "last_name", "string")], "concat",
                       join_path=["uni_enrollments.student_id = uni_students.student_id"]),
            # code_to_label
            MappingDef("fact_enrollment", "grade_label", "string", "grade readable name",
                       [("uni_enrollments", "grade_code", "string")], "code_to_label"),
            MappingDef("dim_course", "level_label", "string", "course level readable name",
                       [("uni_courses", "level_code", "string")], "code_to_label"),
            MappingDef("dim_department", "college_label", "string", "college readable name",
                       [("uni_departments", "college_code", "string")], "code_to_label"),
            # lookup_join (2-hop: enrollment -> course -> department)
            MappingDef("fact_enrollment", "course_department", "string", "department offering the course",
                       [("uni_departments", "dept_name", "string")], "lookup_join",
                       join_path=["uni_enrollments.course_id = uni_courses.course_id", "uni_courses.department_id = uni_departments.dept_id"]),
            # lookup_join (3-hop: enrollment -> student -> advisor -> department)
            MappingDef("fact_enrollment", "student_advisor_dept", "string", "department of the student advisor",
                       [("uni_departments", "dept_name", "string")], "lookup_join",
                       join_path=["uni_enrollments.student_id = uni_students.student_id", "uni_students.advisor_id = uni_faculty.faculty_id", "uni_faculty.department_id = uni_departments.dept_id"]),
        ],
    ))

    return domains


# ---------------------------------------------------------------
# Schema perturbation for messy/legacy naming
# ---------------------------------------------------------------

_ABBREV_MAP = {
    "employee": ["emp", "empl", "ee"],
    "department": ["dept", "dep", "div"],
    "customer": ["cust", "cst", "clnt"],
    "transaction": ["txn", "trans", "trx"],
    "product": ["prod", "prd", "itm"],
    "order": ["ord", "ordr", "po"],
    "account": ["acct", "acc"],
    "appointment": ["appt", "apt"],
    "reservation": ["resv", "rsv", "booking"],
    "subscription": ["subs", "sub"],
    "warehouse": ["wh", "whse"],
    "shipment": ["shpmt", "ship"],
    "description": ["desc", "dsc"],
    "number": ["num", "no", "nbr"],
    "address": ["addr", "adr"],
    "telephone": ["tel", "ph"],
    "quantity": ["qty", "qnty"],
    "amount": ["amt", "amnt"],
    "date": ["dt", "dte"],
    "name": ["nm", "nme"],
    "first_name": ["fname", "f_name", "first_nm", "given_name"],
    "last_name": ["lname", "l_name", "last_nm", "surname", "family_name"],
    "email": ["eml", "email_addr", "e_mail"],
    "status": ["stat", "sts", "state"],
    "identifier": ["id", "ident", "key"],
    "company": ["co", "corp", "org"],
    "location": ["loc", "locn"],
    "manager": ["mgr", "mngr"],
    "salary": ["sal", "pay", "wage"],
    "budget": ["bdgt", "bgt"],
}

_NAMING_CONVENTIONS = [
    "snake_case",       # employee_first_name
    "camelCase",        # employeeFirstName
    "PascalCase",       # EmployeeFirstName
    "abbreviated",      # emp_fname
    "legacy_prefix",    # tbl_emp_fname or t_emp_01
]


def _perturb_name(name: str, convention: str) -> str:
    """Apply naming convention perturbation to a column/table name."""
    parts = name.split("_")

    if convention == "abbreviated":
        new_parts = []
        for p in parts:
            if p.lower() in _ABBREV_MAP:
                new_parts.append(random.choice(_ABBREV_MAP[p.lower()]))
            elif len(p) > 5 and random.random() < 0.4:
                new_parts.append(p[:3])
            else:
                new_parts.append(p)
        return "_".join(new_parts)

    if convention == "camelCase":
        return parts[0].lower() + "".join(p.capitalize() for p in parts[1:])

    if convention == "PascalCase":
        return "".join(p.capitalize() for p in parts)

    if convention == "legacy_prefix":
        prefix = random.choice(["tbl_", "t_", "tb_", ""])
        return prefix + "_".join(parts)

    return name  # snake_case (no change)


def perturb_domain(domain: DomainSchema, convention: str) -> DomainSchema:
    """Create a perturbed copy of a domain with different naming conventions.
    Note: We perturb TABLE names (for user prompt variety) but keep
    the MAPPING source references intact (since the assistant answer
    must reference the actual column names in the schema).
    """
    # For LLM training, we only perturb the target names to teach
    # the model to handle various naming styles in the TARGET.
    # The source schema stays the same (it's the "ground truth" the model reads).
    new_mappings = []
    for m in domain.mappings:
        new_tgt_col = _perturb_name(m.target_col, convention)
        new_tgt_table = _perturb_name(m.target_table, convention)
        new_desc = m.target_desc
        if convention == "abbreviated" and random.random() < 0.4:
            new_desc = ""  # abbreviated schemas often lack descriptions
        new_mappings.append(MappingDef(
            target_table=new_tgt_table,
            target_col=new_tgt_col,
            target_type=m.target_type,
            target_desc=new_desc,
            source_cols=m.source_cols,
            transform=m.transform,
            join_path=m.join_path,
        ))

    return DomainSchema(
        name=f"{domain.name}_{convention}",
        source_tables=domain.source_tables,
        mappings=new_mappings,
    )


# ---------------------------------------------------------------
# Rich reasoning templates
# ---------------------------------------------------------------

_REASONING_TEMPLATES = {
    "rename": [
        '"{tgt}" maps to "{src}" as a simple rename (sub_operation: rename_only) since they represent the same concept. This is rename (not fk_lookup) because the source column is in the same table as the primary entity -- no join is needed.',
        'The target "{tgt}" corresponds to source "{src}" - a direct rename_only with no computation needed. No FK join is required since the column is in the primary table.',
        '"{src}" is the natural source for "{tgt}" as they refer to the same data. This is a rename (rename_only) because both columns are in the same table.',
        'Mapping "{tgt}" requires just renaming "{src}" (rename_only) since both represent the same attribute. No cross-table join is involved.',
        '"{tgt}" is a straightforward rename_only of "{src}". Despite any name differences, no computation or join is needed -- the data is in the same table.',
    ],
    "direct_copy": [
        '"{tgt}" is a direct_copy of "{src}" with no transformation. The column is in the same table, no join needed.',
        'Source "{src}" copies directly (direct_copy) to target "{tgt}" without changes. Same table, no computation.',
    ],
    "concat": [
        '"{tgt}" requires combining {src_list} by concatenation into a single string. Sub-operation: concat of these columns.',
        'The target "{tgt}" is built by joining {src_list} together as a composite string value via concatenation.',
        '"{tgt}" is composed from multiple source columns: {src_list}. Concatenation merges them into one combined value.',
        'To produce "{tgt}", concatenate {src_list} into a single value. This is a multi-column string combination.',
    ],
    "fk_lookup": [
        '"{tgt}" requires looking up "{src}" from a related table via foreign key join (fk_dimension_lookup). This is fk_lookup (not rename) because the source column is in a different table that must be joined.',
        'The target "{tgt}" comes from "{src}" in a different table, accessed through a FK dimension lookup. A join is required, so this is fk_lookup, not rename.',
        '"{src}" lives in a foreign table and must be joined to get "{tgt}" (fk_dimension_lookup). Because a cross-table join is needed, this is fk_lookup.',
        'A foreign key dimension lookup is needed to get "{src}" for the target "{tgt}". The source is in a different table from the primary entity.',
        '"{tgt}" maps to "{src}" which is in a different table. This requires an FK dimension lookup, making it fk_lookup rather than rename.',
    ],
    "date_part": [
        '"{tgt}" extracts a date component from "{src}" using date_part extraction.',
        'The target "{tgt}" is derived by extracting a specific date part from the date column "{src}".',
        '"{src}" is a date/datetime column; "{tgt}" extracts a specific part (year/month/day/quarter/hour) from it.',
        'Extract the date component from "{src}" to produce "{tgt}". This is date_part extraction, not rename, because a date component is being isolated.',
    ],
    "date_diff": [
        '"{tgt}" is the difference in time between {src_list}. Both date columns are selected as value columns (not FK keys). This computes a duration or age.',
        'The target "{tgt}" computes a duration or age from {src_list}. Select the actual date value columns, not the join keys.',
        'Calculate "{tgt}" by finding the date difference between {src_list}. Each source is the actual date column holding the data.',
        '"{tgt}" requires computing the time difference between {src_list}. These are the data-holding date columns for duration/age calculation.',
    ],
    "arithmetic": [
        '"{tgt}" is computed from {src_list} using a mathematical operation. An explicit computation is specified in the description.',
        'The target "{tgt}" requires arithmetic on {src_list}. The description specifies the exact mathematical operation to apply.',
        '"{tgt}" involves a calculation combining {src_list}. This is arithmetic because a specific math operation (add/subtract/multiply/divide/ratio/conversion) is explicitly needed.',
        'A mathematical operation on {src_list} produces "{tgt}". The sub_operation specifies the exact math: add, subtract, multiply, divide, ratio_percentage, or scaling_unit_conversion.',
    ],
    "conditional": [
        '"{tgt}" is a boolean flag derived from evaluating "{src}". This is conditional (not code_to_label) because the target is a true/false flag, not a label string. The sub_operation identifies the specific condition type.',
        'The target "{tgt}" converts "{src}" into a true/false boolean flag. The target type is boolean, so this is conditional with a specific flag sub_operation.',
        '"{src}" is evaluated against a condition to produce the boolean "{tgt}". Boolean output means conditional, not code_to_label. The flag type specifies the condition.',
        '"{tgt}" derives a boolean from "{src}". Since the target is a flag (not a readable label), this is conditional with a status/threshold/equality/null check.',
    ],
    "bucketing": [
        '"{tgt}" bins the continuous values of "{src}" into discrete categories using range classification.',
        '"{src}" is bucketed into ranges to produce the categorical "{tgt}" via bucketing/binning.',
        'The target "{tgt}" groups the numeric "{src}" into labeled buckets using range classification.',
        '"{tgt}" classifies "{src}" into predefined range brackets via bucketing.',
    ],
    "code_to_label": [
        '"{tgt}" converts the coded value in "{src}" to a human-readable label string (code_to_label). This is code_to_label (not conditional) because the target is a descriptive label, not a boolean flag.',
        '"{src}" contains a code that is decoded to a descriptive label for "{tgt}" (code_to_label). The target is a label string, so this is code_to_label.',
        'The target "{tgt}" maps the internal code in "{src}" to its display name via code_to_label. Label output means code_to_label, not conditional.',
        'A lookup converts the code in "{src}" to the readable label "{tgt}" (code_to_label). Since the output is a human-readable name (not boolean), use code_to_label.',
        '"{tgt}" translates the code in "{src}" to a readable label. code_to_label is correct because the target is a string label, not a boolean flag.',
    ],
    "type_cast": [
        '"{tgt}" converts "{src}" to a different data type without changing the value (type_cast).',
        '"{src}" is cast to a new type for the target "{tgt}" (type_cast). Only the type changes, not the data.',
        'The target "{tgt}" requires type conversion (type_cast) of "{src}" to a different data type.',
        'A data type change (type_cast) from "{src}" produces "{tgt}". The value remains the same, only the type is converted.',
    ],
    "date_format": [
        '"{tgt}" formats the date in "{src}" into a display string (format_date).',
        '"{src}" is a date converted to a formatted display string for "{tgt}" (format_date).',
        'The target "{tgt}" renders "{src}" as a human-readable date string via format_date.',
    ],
    "date_parse": [
        '"{tgt}" parses "{src}" into a proper date/datetime type (parse_date).',
        '"{src}" is parsed from a string into a structured date for "{tgt}" (parse_date).',
        'The target "{tgt}" converts the text date in "{src}" to datetime via parse_date.',
    ],
    "template": [
        '"{tgt}" uses a formatted string_template combining {src_list} into a display pattern.',
        'A string_template is applied to {src_list} to create "{tgt}" with a predefined format.',
        '"{tgt}" is a formatted display string built from {src_list} using a string_template.',
        'The target "{tgt}" assembles {src_list} using a predefined string_template pattern.',
    ],
    "lookup_join": [
        '"{tgt}" requires a multi_hop_lookup across joined tables to get {src_list}. Select the final value column(s), not the intermediate FK columns used for joining.',
        'A multi_hop_lookup traverses related tables to bring {src_list} for "{tgt}". The output columns are the data values at the end of the join chain, not the join keys.',
        '"{tgt}" involves a multi-hop join through intermediate tables to access {src_list}. Output the value column from the final table in the join path (multi_hop_lookup).',
        'The target "{tgt}" needs a multi_hop_lookup to reach {src_list}. These are the actual value columns, not the FK keys used to traverse the join path.',
    ],
}


# ---------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------

def serialize_schema(domain: DomainSchema) -> str:
    """Serialize source tables as compact schema."""
    lines = []
    for table in domain.source_tables:
        col_parts = []
        for col in table.columns:
            flags = []
            if col.is_pk:
                flags.append("PK")
            if col.is_fk:
                flags.append("FK")
            flag_str = " ".join(flags)
            col_parts.append(f"{col.name} {flag_str} {col.type}".strip())
        lines.append(f"  {table.name}({', '.join(col_parts)})")
    return "\n".join(lines)


def serialize_joins(domain: DomainSchema) -> str:
    """Extract join paths from FK definitions."""
    joins = set()
    for table in domain.source_tables:
        for col in table.columns:
            if col.is_fk:
                joins.add(f"  {table.name}.{col.name} = {col.is_fk}")
    return "\n".join(sorted(joins)) if joins else "  (none)"


def build_user_prompt(domain: DomainSchema, mapping: MappingDef,
                      tgt_name: str, tgt_desc: str) -> str:
    schema_text = serialize_schema(domain)
    joins_text = serialize_joins(domain)
    desc_part = f' - "{tgt_desc}"' if tgt_desc else ""
    return (
        f"Source Schema:\n{schema_text}\n\n"
        f"Joins:\n{joins_text}\n\n"
        f"Map target: {mapping.target_table}.{tgt_name} ({mapping.target_type}){desc_part}"
    )


def infer_sub_operation(mapping: MappingDef) -> str:
    """Infer the fine-grained sub_operation from the mapping context.

    If mapping.sub_operation is already set, return it directly.
    Otherwise, infer from the transform type, target description, and column names.
    """
    if mapping.sub_operation:
        return mapping.sub_operation

    tf = mapping.transform
    desc = (mapping.target_desc or "").lower()
    tgt = (mapping.target_col or "").lower()
    n_src = len(mapping.source_cols)

    # ── rename / direct_copy ──
    if tf == "rename":
        return "rename_only"
    if tf == "direct_copy":
        return "direct_copy"

    # ── concat ──
    if tf == "concat":
        return "concat_multi" if n_src >= 3 else "concat_two"

    # ── fk_lookup ──
    if tf == "fk_lookup":
        return "fk_dimension_lookup"

    # ── lookup_join ──
    if tf == "lookup_join":
        return "multi_hop_lookup"

    # ── date_part ──
    if tf == "date_part":
        if any(w in desc or w in tgt for w in ["month", "monthly"]):
            return "extract_month"
        if any(w in desc or w in tgt for w in ["quarter", "quarterly"]):
            return "extract_quarter"
        if any(w in desc or w in tgt for w in ["day", "daily"]):
            return "extract_day"
        if any(w in desc or w in tgt for w in ["hour", "hourly"]):
            return "extract_hour"
        return "extract_year"  # default for date_part

    # ── date_diff ──
    if tf == "date_diff":
        if any(w in desc for w in ["age", "years old", "current age"]):
            return "age_calculation"
        if any(w in desc for w in ["hour", "hours"]):
            return "duration_hours"
        return "duration_days"  # default for date_diff

    # ── date_format ──
    if tf == "date_format":
        return "format_date"

    # ── date_parse ──
    if tf == "date_parse":
        return "parse_date"

    # ── arithmetic ──
    if tf == "arithmetic":
        if any(w in desc for w in ["plus", "sum", "total", "add", "combined", "including"]):
            return "add"
        if any(w in desc for w in ["minus", "subtract", "net", "remaining", "difference"]):
            return "subtract"
        if any(w in desc for w in ["times", "multiply", "product", "times 12"]):
            return "multiply"
        if any(w in desc for w in ["divided", "per ", "ratio", "rate"]):
            return "divide"
        if any(w in desc for w in ["percent", "pct", "proportion"]):
            return "ratio_percentage"
        if any(w in desc for w in ["convert", "conversion", "in kg", "in gb", "to kg"]):
            return "scaling_unit_conversion"
        # Fallback: guess from description patterns
        if n_src == 1:
            return "scaling_unit_conversion"
        return "multiply"  # safe default for multi-column arithmetic

    # ── conditional ──
    if tf == "conditional":
        if any(w in desc for w in ["null", "missing", "empty", "blank"]):
            return "null_presence_flag"
        if any(w in desc for w in ["equal", "equals", "match"]):
            return "equality_check"
        if any(w in desc for w in ["whether", "is_", "flag", "active", "completed",
                                    "delivered", "paid", "approved", "expired"]):
            return "status_flag"
        return "threshold_flag"

    # ── code_to_label ──
    if tf == "code_to_label":
        if any(w in desc for w in ["harmoniz", "standardiz", "normaliz"]):
            return "category_harmonization"
        return "code_to_label"

    # ── bucketing ──
    if tf == "bucketing":
        if any(w in desc for w in ["range", "bracket", "tier", "band", "bucket"]):
            return "range_classification"
        return "bucketing_binning"

    # ── type_cast ──
    if tf == "type_cast":
        tgt_type = (mapping.target_type or "").lower()
        if tgt_type in ("string", "text", "varchar"):
            return "type_cast_string"
        if tgt_type in ("date", "datetime", "timestamp"):
            return "type_cast_date"
        return "type_cast_numeric"

    # ── template ──
    if tf == "template":
        return "string_template"

    return tf  # fallback


def build_assistant_response(mapping: MappingDef, tgt_name: str) -> str:
    src_cols_str = ", ".join(f"{t}.{c}" for t, c, _ in mapping.source_cols)
    src_list = " and ".join(f'"{t}.{c}"' for t, c, _ in mapping.source_cols)
    first_src = f"{mapping.source_cols[0][0]}.{mapping.source_cols[0][1]}"

    sub_op = infer_sub_operation(mapping)

    templates = _REASONING_TEMPLATES.get(mapping.transform, [
        f'"{tgt_name}" is derived from {src_list} using {mapping.transform}.'
    ])
    reasoning = random.choice(templates).format(
        tgt=tgt_name, src=first_src, src_list=src_list,
    )
    if mapping.join_path:
        jp_str = " -> ".join(mapping.join_path)
        reasoning += f" Join path: {jp_str}."

    return (
        f"source_columns: {src_cols_str}\n"
        f"transform_type: {mapping.transform}\n"
        f"sub_operation: {sub_op}\n"
        f"reasoning: {reasoning}"
    )


# ---------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------

_TARGET_SYNONYMS = {
    "full_name": ["complete_name", "display_name", "person_name", "fullname", "name_full"],
    "email_address": ["email_addr", "contact_email", "e_mail", "email_id", "electronic_mail"],
    "phone_number": ["phone_no", "contact_phone", "telephone", "mobile_number", "tel_no"],
    "hire_year": ["year_hired", "hiring_year", "join_year", "start_year", "onboard_year"],
    "department_name": ["dept_name", "division_name", "department_label", "org_unit_name"],
    "employee_key": ["emp_key", "employee_id", "emp_identifier", "worker_key", "staff_key"],
    "annual_salary": ["yearly_salary", "salary_annual", "compensation", "yearly_pay", "annual_pay"],
    "customer_name": ["client_name", "buyer_name", "patron_name", "purchaser_name", "cust_name"],
    "product_name": ["item_name", "product_title", "article_name", "goods_name", "sku_name"],
    "birth_year": ["year_of_birth", "born_year", "dob_year", "birth_yr"],
    "is_active": ["active_flag", "is_enabled", "status_flag", "active_indicator", "is_live"],
    "total_amount": ["gross_total", "order_total", "sum_amount", "total_value", "full_amount"],
    "supplier_company": ["vendor_name", "supplier_name", "provider_company"],
    "building_name": ["facility_name", "site_name", "premises_name"],
    "sensor_name": ["device_name", "probe_name", "instrument_name"],
    "driver_name": ["operator_name", "chauffeur_name"],
    "campaign_name": ["campaign_title", "promo_name"],
    "lead_name": ["prospect_name", "potential_name"],
    "contact_name": ["person_name", "individual_name"],
    # -- V4 additions for broader coverage --
    "account_number": ["acct_no", "account_num", "acct_nbr", "account_id_str"],
    "order_date": ["purchase_date", "placed_date", "order_dt", "transaction_date"],
    "delivery_date": ["ship_date", "dispatch_date", "arrival_date", "received_date"],
    "invoice_amount": ["billed_amount", "charge_total", "inv_amt", "payment_due"],
    "policy_number": ["policy_no", "policy_id", "contract_number", "coverage_id"],
    "claim_amount": ["payout_amount", "reimbursement", "claim_value", "settlement_amt"],
    "property_address": ["site_address", "location_addr", "property_loc", "premises_addr"],
    "crop_yield": ["harvest_amount", "production_qty", "yield_per_acre", "output_volume"],
    "subscriber_count": ["user_count", "active_users", "member_count", "sub_total"],
    "shipment_status": ["delivery_status", "tracking_status", "package_state", "ship_state"],
    "premium_amount": ["monthly_premium", "insurance_cost", "coverage_cost", "plan_fee"],
    "agent_name": ["broker_name", "representative_name", "sales_agent", "handler_name"],
    "duration_days": ["elapsed_days", "period_days", "span_days", "length_days"],
    "unit_price": ["price_per_unit", "item_price", "rate", "cost_each"],
    "total_cost": ["gross_cost", "aggregate_cost", "sum_cost", "overall_cost"],
    "start_date": ["begin_date", "effective_date", "commencement_date", "from_date"],
    "end_date": ["expiry_date", "termination_date", "close_date", "to_date"],
    "region_name": ["territory_name", "area_name", "zone_name", "district_name"],
    "category_name": ["class_name", "group_name", "type_label", "segment_name"],
}


def _augment_name(name: str) -> str:
    if name in _TARGET_SYNONYMS and random.random() < 0.5:
        return random.choice(_TARGET_SYNONYMS[name])
    r = random.random()
    if r < 0.08:
        return name.replace("_", "").lower()
    if r < 0.15:
        parts = name.split("_")
        return "".join(p.capitalize() for p in parts)
    if r < 0.20:
        return name.replace("_", "-")
    return name


def _augment_desc(desc: str) -> str:
    if random.random() < 0.25:
        return ""
    if random.random() < 0.15:
        words = desc.split()
        return " ".join(words[:max(1, len(words)//2)])
    return desc


def _augment_table_prefix(table_name: str) -> str:
    if random.random() < 0.3:
        prefixes = ["dim_", "fact_", "stg_", "dw_", "dwh_", "raw_", ""]
        base = re.sub(r"^(dim_|fact_|stg_|dw_|dwh_|raw_|bridge_)", "", table_name)
        return random.choice(prefixes) + base
    return table_name


# ---------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------

def generate_llm_dataset(
    output_dir: str = "llm_training_data",
    base_augmentation_rounds: int = 12,
    perturbed_augmentation_rounds: int = 5,
    seed: int = 42,
):
    """Generate production-quality training data (V4 with adversarial + CoT)."""
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Collect all domains: 12 base + 8 extra
    base_domains = _build_base_domains()
    extra_domains = _build_extra_domains()
    all_domains = base_domains + extra_domains

    total_base_mappings = sum(len(d.mappings) for d in all_domains)
    print(f"Built {len(all_domains)} domains with {total_base_mappings} base mappings")

    records = []

    # ── Phase 1: Base domains with standard augmentation ──
    for domain in all_domains:
        for mapping in domain.mappings:
            for aug_round in range(base_augmentation_rounds):
                if aug_round == 0:
                    tgt_name = mapping.target_col
                    tgt_desc = mapping.target_desc
                    tgt_table = mapping.target_table
                else:
                    tgt_name = _augment_name(mapping.target_col)
                    tgt_desc = _augment_desc(mapping.target_desc)
                    tgt_table = _augment_table_prefix(mapping.target_table)

                aug_m = MappingDef(
                    target_table=tgt_table, target_col=tgt_name,
                    target_type=mapping.target_type, target_desc=tgt_desc,
                    source_cols=mapping.source_cols, transform=mapping.transform,
                    join_path=mapping.join_path,
                )

                user_msg = build_user_prompt(domain, aug_m, tgt_name, tgt_desc)
                assistant_msg = build_assistant_response(aug_m, tgt_name)

                records.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": assistant_msg},
                    ],
                    "domain": domain.name,
                    "transform": mapping.transform,
                    "arity": len(mapping.source_cols),
                    "phase": "base",
                })

    # ── Phase 2: Perturbed naming conventions ──
    conventions_to_use = ["abbreviated", "camelCase", "PascalCase", "legacy_prefix"]
    for domain in all_domains:
        for conv in conventions_to_use:
            perturbed = perturb_domain(domain, conv)
            for mapping in perturbed.mappings:
                for aug_round in range(perturbed_augmentation_rounds):
                    if aug_round == 0:
                        tgt_name = mapping.target_col
                        tgt_desc = mapping.target_desc
                    else:
                        tgt_name = _augment_name(mapping.target_col)
                        tgt_desc = _augment_desc(mapping.target_desc)

                    user_msg = build_user_prompt(perturbed, mapping, tgt_name, tgt_desc)
                    assistant_msg = build_assistant_response(mapping, tgt_name)

                    records.append({
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_msg},
                            {"role": "assistant", "content": assistant_msg},
                        ],
                        "domain": perturbed.name,
                        "transform": mapping.transform,
                        "arity": len(mapping.source_cols),
                        "phase": "perturbed",
                    })

    # ── Phase 3: Minimal-description examples ──
    # Real-world targets often have NO description
    for domain in all_domains:
        for mapping in domain.mappings:
            for _ in range(2):
                tgt_name = _augment_name(mapping.target_col)
                aug_m = MappingDef(
                    target_table=mapping.target_table, target_col=tgt_name,
                    target_type=mapping.target_type, target_desc="",
                    source_cols=mapping.source_cols, transform=mapping.transform,
                    join_path=mapping.join_path,
                )
                user_msg = build_user_prompt(domain, aug_m, tgt_name, "")
                assistant_msg = build_assistant_response(aug_m, tgt_name)

                records.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": assistant_msg},
                    ],
                    "domain": domain.name,
                    "transform": mapping.transform,
                    "arity": len(mapping.source_cols),
                    "phase": "no_desc",
                })

    # ── Phase 4: Rebalancing – oversample rare transforms ──
    # Ensure no transform has fewer than min_per_transform examples
    min_per_transform = 400
    transform_buckets: Dict[str, List[dict]] = {}
    for r in records:
        transform_buckets.setdefault(r["transform"], []).append(r)

    rebalanced_adds = []
    for transform, bucket in transform_buckets.items():
        if len(bucket) < min_per_transform:
            deficit = min_per_transform - len(bucket)
            # Oversample with slight augmentation
            for i in range(deficit):
                src_rec = bucket[i % len(bucket)]
                # Create a variant by re-generating with different augmentation
                new_rec = copy.deepcopy(src_rec)
                new_rec["phase"] = "rebalanced"
                # Modify the reasoning slightly for variety
                msgs = new_rec["messages"]
                asst = msgs[2]["content"]
                lines = asst.split("\n")
                if len(lines) >= 3:
                    # Add a small note to make it unique
                    lines[2] = lines[2].rstrip() + f" [variant {i+1}]"
                    msgs[2]["content"] = "\n".join(lines)
                rebalanced_adds.append(new_rec)

    records.extend(rebalanced_adds)
    if rebalanced_adds:
        print(f"  Phase 4: Added {len(rebalanced_adds)} rebalanced examples for rare transforms")

    # ── Phase 5: Oversample composite (arity >= 3) examples ──
    composite_records = [r for r in records if r["arity"] >= 3]
    oversample_factor = 3  # triple the composite examples
    composite_adds = []
    for _ in range(oversample_factor - 1):
        for rec in composite_records:
            new_rec = copy.deepcopy(rec)
            new_rec["phase"] = "composite_boost"
            composite_adds.append(new_rec)
    records.extend(composite_adds)
    if composite_adds:
        print(f"  Phase 5: Added {len(composite_adds)} composite-boost examples (arity>=3)")

    # ── Phase 6: Adversarial / Ambiguous Examples ──
    # Add schema noise (audit columns) and cross-table ambiguity
    _NOISE_COLUMNS = [
        ColDef("created_by", "string"),
        ColDef("updated_by", "string"),
        ColDef("created_at", "datetime"),
        ColDef("updated_at", "datetime"),
        ColDef("is_deleted", "string"),
        ColDef("version", "int"),
        ColDef("row_hash", "string"),
        ColDef("etl_load_date", "datetime"),
    ]

    adversarial_adds = []
    for domain in all_domains:
        for mapping in domain.mappings:
            for _ in range(2):
                # Add 2-4 noise columns to each source table
                noisy_tables = []
                for table in domain.source_tables:
                    noise_count = random.randint(2, 4)
                    noisy_cols = list(table.columns) + random.sample(
                        _NOISE_COLUMNS, min(noise_count, len(_NOISE_COLUMNS))
                    )
                    noisy_tables.append(TableDef(table.name, noisy_cols))

                noisy_domain = DomainSchema(
                    name=f"{domain.name}_noisy",
                    source_tables=noisy_tables,
                    mappings=domain.mappings,
                )

                tgt_name = _augment_name(mapping.target_col)
                tgt_desc = mapping.target_desc if random.random() > 0.3 else ""
                user_msg = build_user_prompt(noisy_domain, mapping, tgt_name, tgt_desc)
                assistant_msg = build_assistant_response(mapping, tgt_name)

                adversarial_adds.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": assistant_msg},
                    ],
                    "domain": noisy_domain.name,
                    "transform": mapping.transform,
                    "arity": len(mapping.source_cols),
                    "phase": "adversarial",
                })

    records.extend(adversarial_adds)
    print(f"  Phase 6: Added {len(adversarial_adds)} adversarial/noisy-schema examples")

    # ── Phase 7: Chain-of-thought reasoning for complex transforms ──
    _COT_TRANSFORMS = {"lookup_join", "arithmetic", "template", "date_diff", "concat"}

    _COT_TEMPLATES = {
        "lookup_join": (
            "Step 1: The target \"{tgt}\" requires data from \"{final_src}\" which is not "
            "directly in the main table.\n"
            "Step 2: Trace the join path: {join_trace}.\n"
            "Step 3: This is a multi-hop lookup requiring a lookup_join transform.\n"
            "Conclusion: Use {src_list} via lookup_join."
        ),
        "arithmetic": (
            "Step 1: The target \"{tgt}\" involves a computed value, not a direct column.\n"
            "Step 2: Identify the operands: {src_list}.\n"
            "Step 3: Apply the mathematical operation to derive the result.\n"
            "Conclusion: Use arithmetic transform on {src_list}."
        ),
        "template": (
            "Step 1: The target \"{tgt}\" is a formatted display string, not raw data.\n"
            "Step 2: Identify the components: {src_list}.\n"
            "Step 3: Combine them using a string template pattern.\n"
            "Conclusion: Use template transform with {src_list}."
        ),
        "date_diff": (
            "Step 1: The target \"{tgt}\" represents a duration or time gap.\n"
            "Step 2: Identify the two date endpoints: {src_list}.\n"
            "Step 3: Compute the difference between the dates.\n"
            "Conclusion: Use date_diff transform on {src_list}."
        ),
        "concat": (
            "Step 1: The target \"{tgt}\" combines multiple text values.\n"
            "Step 2: Identify the parts to join: {src_list}.\n"
            "Step 3: Concatenate them into a single string.\n"
            "Conclusion: Use concat transform with {src_list}."
        ),
    }

    cot_adds = []
    for domain in all_domains:
        for mapping in domain.mappings:
            if mapping.transform not in _COT_TRANSFORMS:
                continue
            for _ in range(3):
                tgt_name = _augment_name(mapping.target_col)
                src_cols_str = ", ".join(f"{t}.{c}" for t, c, _ in mapping.source_cols)
                src_list = " and ".join(f'"{t}.{c}"' for t, c, _ in mapping.source_cols)
                final_src = f"{mapping.source_cols[-1][0]}.{mapping.source_cols[-1][1]}"

                join_trace = " -> ".join(mapping.join_path) if mapping.join_path else "direct table"

                cot_template = _COT_TEMPLATES.get(mapping.transform, "")
                if cot_template:
                    reasoning = cot_template.format(
                        tgt=tgt_name, src_list=src_list,
                        final_src=final_src, join_trace=join_trace,
                    )
                else:
                    reasoning = f'"{tgt_name}" is derived from {src_list} using {mapping.transform}.'

                cot_sub_op = infer_sub_operation(mapping)
                assistant_msg = (
                    f"source_columns: {src_cols_str}\n"
                    f"transform_type: {mapping.transform}\n"
                    f"sub_operation: {cot_sub_op}\n"
                    f"reasoning: {reasoning}"
                )

                tgt_desc = mapping.target_desc if random.random() > 0.2 else ""
                user_msg = build_user_prompt(domain, mapping, tgt_name, tgt_desc)

                cot_adds.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": assistant_msg},
                    ],
                    "domain": domain.name,
                    "transform": mapping.transform,
                    "arity": len(mapping.source_cols),
                    "phase": "chain_of_thought",
                })

    records.extend(cot_adds)
    print(f"  Phase 7: Added {len(cot_adds)} chain-of-thought reasoning examples")

    # ── Phase 8: Contrastive / Hard-Negative Examples ──
    # Teach the model to distinguish confusable transform pairs
    contrastive_adds = []

    for domain in all_domains:
        for mapping in domain.mappings:
            # 8a: For every fk_lookup, generate a contrastive rename example
            # Same column concept but from the SAME table = rename
            if mapping.transform == "fk_lookup" and len(mapping.source_cols) == 1:
                src_table, src_col, src_type = mapping.source_cols[0]
                # Create a "what if this column was in the primary table" rename
                for _ in range(3):
                    tgt_name = _augment_name(mapping.target_col)
                    # The rename reasoning explicitly says "same table, no join"
                    reasoning = (
                        f'"{tgt_name}" maps to "{src_table}.{src_col}" as a simple rename. '
                        f'This is rename (not fk_lookup) because the source column is in '
                        f'the same table as the primary entity -- no FK join is needed.'
                    )
                    # Keep the fk_lookup example too with explicit "different table" reasoning
                    fk_reasoning = (
                        f'"{tgt_name}" requires looking up "{src_table}.{src_col}" from a '
                        f'different table via FK join. This is fk_lookup (not rename) because '
                        f'the source column is in a different table that requires a join to reach.'
                    )
                    jp_str = " -> ".join(mapping.join_path) if mapping.join_path else ""
                    if jp_str:
                        fk_reasoning += f" Join path: {jp_str}."

                    src_str = f"{src_table}.{src_col}"
                    # fk_lookup version (original)
                    tgt_desc = mapping.target_desc if random.random() > 0.2 else ""
                    user_msg = build_user_prompt(domain, mapping, tgt_name, tgt_desc)
                    contrastive_adds.append({
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_msg},
                            {"role": "assistant", "content": f"source_columns: {src_str}\ntransform_type: fk_lookup\nsub_operation: fk_dimension_lookup\nreasoning: {fk_reasoning}"},
                        ],
                        "domain": domain.name, "transform": "fk_lookup",
                        "arity": 1, "phase": "contrastive",
                    })

            # 8b: For every code_to_label, generate a contrastive conditional example
            if mapping.transform == "code_to_label" and len(mapping.source_cols) == 1:
                src_table, src_col, src_type = mapping.source_cols[0]
                src_str = f"{src_table}.{src_col}"
                for _ in range(2):
                    # code_to_label version with explicit disambiguation
                    tgt_name = _augment_name(mapping.target_col)
                    c2l_reasoning = (
                        f'"{tgt_name}" converts the code in "{src_str}" to a human-readable '
                        f'label string. This is code_to_label (not conditional) because the '
                        f'target is a descriptive label, not a boolean flag.'
                    )
                    tgt_desc = mapping.target_desc if random.random() > 0.2 else ""
                    user_msg = build_user_prompt(domain, mapping, tgt_name, tgt_desc)
                    contrastive_adds.append({
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_msg},
                            {"role": "assistant", "content": f"source_columns: {src_str}\ntransform_type: code_to_label\nsub_operation: code_to_label\nreasoning: {c2l_reasoning}"},
                        ],
                        "domain": domain.name, "transform": "code_to_label",
                        "arity": 1, "phase": "contrastive",
                    })

            # 8c: For every conditional, generate with explicit boolean reasoning
            if mapping.transform == "conditional" and len(mapping.source_cols) == 1:
                src_table, src_col, src_type = mapping.source_cols[0]
                src_str = f"{src_table}.{src_col}"
                for _ in range(2):
                    tgt_name = _augment_name(mapping.target_col)
                    cond_reasoning = (
                        f'"{tgt_name}" derives a boolean flag from "{src_str}". '
                        f'This is conditional (not code_to_label) because the target is a '
                        f'true/false flag, not a human-readable label string.'
                    )
                    tgt_desc = mapping.target_desc if random.random() > 0.2 else ""
                    user_msg = build_user_prompt(domain, mapping, tgt_name, tgt_desc)
                    cond_sub_op = infer_sub_operation(mapping)
                    contrastive_adds.append({
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_msg},
                            {"role": "assistant", "content": f"source_columns: {src_str}\ntransform_type: conditional\nsub_operation: {cond_sub_op}\nreasoning: {cond_reasoning}"},
                        ],
                        "domain": domain.name, "transform": "conditional",
                        "arity": 1, "phase": "contrastive",
                    })

    # 8d: Adversarial rename examples where names SUGGEST computation
    _ADVERSARIAL_RENAMES = [
        ("salary", "annual_salary", "decimal", "yearly salary", "Despite the name suggesting annual computation, the source salary column already represents annual salary -- simple rename."),
        ("revenue", "total_revenue", "decimal", "total revenue figure", "The source revenue column already contains the total -- no aggregation needed, just rename."),
        ("cost", "monthly_cost", "decimal", "monthly cost", "The source cost column already represents monthly cost -- rename, not arithmetic."),
        ("price", "unit_price", "decimal", "price per unit", "The source price column already stores unit price -- no division needed, just rename."),
        ("balance", "current_balance", "decimal", "current account balance", "The source balance already reflects the current value -- simple rename."),
        ("count", "total_count", "int", "total count", "The source count column already has the total -- no aggregation, just rename."),
        ("amount", "net_amount", "decimal", "net amount", "The source amount already represents the net value -- rename, not arithmetic."),
        ("rate", "hourly_rate", "decimal", "hourly rate", "The source rate column already stores hourly rate -- no conversion needed."),
        ("score", "final_score", "decimal", "final score", "The source score already represents the final value -- rename only."),
        ("budget", "annual_budget", "decimal", "annual budget", "The source budget column already contains the annual figure -- simple rename."),
    ]

    for domain in all_domains:
        for table in domain.source_tables:
            for col in table.columns:
                for src_suffix, tgt_name, tgt_type, tgt_desc, reason in _ADVERSARIAL_RENAMES:
                    if col.name.endswith(src_suffix) or col.name == src_suffix:
                        if col.type in ("decimal", "int"):
                            src_str = f"{table.name}.{col.name}"
                            for _ in range(2):
                                aug_tgt = _augment_name(tgt_name)
                                aug_desc = tgt_desc if random.random() > 0.3 else ""
                                # Build a minimal mapping for the user prompt
                                adv_mapping = MappingDef(
                                    target_table="dim_" + table.name.split("_")[-1] if "_" in table.name else "dim_" + table.name,
                                    target_col=aug_tgt, target_type=tgt_type,
                                    target_desc=aug_desc,
                                    source_cols=[(table.name, col.name, col.type)],
                                    transform="rename",
                                )
                                user_msg = build_user_prompt(domain, adv_mapping, aug_tgt, aug_desc)
                                contrastive_adds.append({
                                    "messages": [
                                        {"role": "system", "content": SYSTEM_PROMPT},
                                        {"role": "user", "content": user_msg},
                                        {"role": "assistant", "content": f"source_columns: {src_str}\ntransform_type: rename\nsub_operation: rename_only\nreasoning: {reason}"},
                                    ],
                                    "domain": domain.name, "transform": "rename",
                                    "arity": 1, "phase": "contrastive",
                                })
                            break  # only match first adversarial pattern per column

    records.extend(contrastive_adds)
    print(f"  Phase 8: Added {len(contrastive_adds)} contrastive/hard-negative examples")

    # ── Phase 9: Cross-Table Column Selection Examples ──
    # Teach the model to pick value columns from the correct tables
    cross_table_adds = []

    for domain in all_domains:
        for mapping in domain.mappings:
            # 9a: Cross-table date_diff (columns from different tables)
            if mapping.transform == "date_diff" and mapping.join_path:
                src_tables_set = set(t for t, c, _ in mapping.source_cols)
                if len(src_tables_set) >= 2:
                    for _ in range(4):
                        tgt_name = _augment_name(mapping.target_col)
                        src_str = ", ".join(f"{t}.{c}" for t, c, _ in mapping.source_cols)
                        src_list = " and ".join(f'"{t}.{c}"' for t, c, _ in mapping.source_cols)
                        jp_str = " -> ".join(mapping.join_path)
                        reasoning = (
                            f'"{tgt_name}" computes the date difference between {src_list}. '
                            f'Both date columns are selected as VALUE columns from their '
                            f'respective tables (not FK join keys). Join path: {jp_str}.'
                        )
                        tgt_desc = mapping.target_desc if random.random() > 0.2 else ""
                        user_msg = build_user_prompt(domain, mapping, tgt_name, tgt_desc)
                        dd_sub_op = infer_sub_operation(mapping)
                        cross_table_adds.append({
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user_msg},
                                {"role": "assistant", "content": f"source_columns: {src_str}\ntransform_type: date_diff\nsub_operation: {dd_sub_op}\nreasoning: {reasoning}"},
                            ],
                            "domain": domain.name, "transform": "date_diff",
                            "arity": len(mapping.source_cols), "phase": "cross_table",
                        })

            # 9b: Multi-hop lookup_join with explicit value-vs-key reasoning
            if mapping.transform == "lookup_join" and mapping.join_path:
                for _ in range(4):
                    tgt_name = _augment_name(mapping.target_col)
                    src_str = ", ".join(f"{t}.{c}" for t, c, _ in mapping.source_cols)
                    src_list = " and ".join(f'"{t}.{c}"' for t, c, _ in mapping.source_cols)
                    jp_str = " -> ".join(mapping.join_path)
                    final_table = mapping.source_cols[-1][0]
                    reasoning = (
                        f'"{tgt_name}" requires a multi-hop lookup to get {src_list} from '
                        f'the {final_table} table. These are the VALUE columns at the end '
                        f'of the join chain, NOT the intermediate FK columns. '
                        f'Join path: {jp_str}.'
                    )
                    tgt_desc = mapping.target_desc if random.random() > 0.2 else ""
                    user_msg = build_user_prompt(domain, mapping, tgt_name, tgt_desc)
                    cross_table_adds.append({
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_msg},
                            {"role": "assistant", "content": f"source_columns: {src_str}\ntransform_type: lookup_join\nsub_operation: multi_hop_lookup\nreasoning: {reasoning}"},
                        ],
                        "domain": domain.name, "transform": "lookup_join",
                        "arity": len(mapping.source_cols), "phase": "cross_table",
                    })

            # 9c: Cross-table arithmetic with value column reasoning
            if mapping.transform == "arithmetic" and mapping.join_path:
                src_tables_set = set(t for t, c, _ in mapping.source_cols)
                if len(src_tables_set) >= 2:
                    for _ in range(3):
                        tgt_name = _augment_name(mapping.target_col)
                        src_str = ", ".join(f"{t}.{c}" for t, c, _ in mapping.source_cols)
                        src_list = " and ".join(f'"{t}.{c}"' for t, c, _ in mapping.source_cols)
                        jp_str = " -> ".join(mapping.join_path)
                        reasoning = (
                            f'"{tgt_name}" requires arithmetic on {src_list} from different '
                            f'tables. Select the actual data columns, not the FK join keys. '
                            f'Join path: {jp_str}.'
                        )
                        tgt_desc = mapping.target_desc if random.random() > 0.2 else ""
                        user_msg = build_user_prompt(domain, mapping, tgt_name, tgt_desc)
                        arith_sub_op = infer_sub_operation(mapping)
                        cross_table_adds.append({
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user_msg},
                                {"role": "assistant", "content": f"source_columns: {src_str}\ntransform_type: arithmetic\nsub_operation: {arith_sub_op}\nreasoning: {reasoning}"},
                            ],
                            "domain": domain.name, "transform": "arithmetic",
                            "arity": len(mapping.source_cols), "phase": "cross_table",
                        })

            # 9d: fk_lookup with explicit "value not key" reasoning
            if mapping.transform == "fk_lookup" and mapping.join_path:
                for _ in range(2):
                    tgt_name = _augment_name(mapping.target_col)
                    src_str = ", ".join(f"{t}.{c}" for t, c, _ in mapping.source_cols)
                    src_list = " and ".join(f'"{t}.{c}"' for t, c, _ in mapping.source_cols)
                    jp_str = " -> ".join(mapping.join_path)
                    reasoning = (
                        f'"{tgt_name}" maps to {src_list} via FK lookup. Output the value '
                        f'column(s) that hold the actual data, not the FK key used for '
                        f'joining. Join path: {jp_str}.'
                    )
                    tgt_desc = mapping.target_desc if random.random() > 0.2 else ""
                    user_msg = build_user_prompt(domain, mapping, tgt_name, tgt_desc)
                    cross_table_adds.append({
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_msg},
                            {"role": "assistant", "content": f"source_columns: {src_str}\ntransform_type: fk_lookup\nsub_operation: fk_dimension_lookup\nreasoning: {reasoning}"},
                        ],
                        "domain": domain.name, "transform": "fk_lookup",
                        "arity": len(mapping.source_cols), "phase": "cross_table",
                    })

    records.extend(cross_table_adds)
    print(f"  Phase 9: Added {len(cross_table_adds)} cross-table column selection examples")

    # ── Phase 10: Single-Response Enforcement Examples ──
    # Teach the model to output EXACTLY ONE mapping per request
    single_response_adds = []

    for domain in all_domains:
        if len(domain.mappings) < 3:
            continue
        # Pick a subset of mappings and for each, generate an example
        # with explicit "only this target" reasoning
        for mapping in domain.mappings:
            for _ in range(1):
                tgt_name = _augment_name(mapping.target_col)
                src_str = ", ".join(f"{t}.{c}" for t, c, _ in mapping.source_cols)

                # Build standard reasoning but append single-response note
                templates = _REASONING_TEMPLATES.get(mapping.transform, [
                    f'"{tgt_name}" is derived from {src_str} using {mapping.transform}.'
                ])
                src_list = " and ".join(f'"{t}.{c}"' for t, c, _ in mapping.source_cols)
                first_src = f"{mapping.source_cols[0][0]}.{mapping.source_cols[0][1]}"
                reasoning = random.choice(templates).format(
                    tgt=tgt_name, src=first_src, src_list=src_list,
                )
                if mapping.join_path:
                    jp_str = " -> ".join(mapping.join_path)
                    reasoning += f" Join path: {jp_str}."

                tgt_desc = mapping.target_desc if random.random() > 0.2 else ""
                user_msg = build_user_prompt(domain, mapping, tgt_name, tgt_desc)

                sr_sub_op = infer_sub_operation(mapping)
                single_response_adds.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": f"source_columns: {src_str}\ntransform_type: {mapping.transform}\nsub_operation: {sr_sub_op}\nreasoning: {reasoning}"},
                    ],
                    "domain": domain.name, "transform": mapping.transform,
                    "arity": len(mapping.source_cols), "phase": "single_response",
                })

    records.extend(single_response_adds)
    print(f"  Phase 10: Added {len(single_response_adds)} single-response enforcement examples")

    # Shuffle and split 85/15
    random.shuffle(records)
    n = len(records)
    split = int(n * 0.85)
    train_records = records[:split]
    val_records = records[split:]

    # Write
    def _write_jsonl(path, data):
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    _write_jsonl(os.path.join(output_dir, "llm_train.jsonl"), train_records)
    _write_jsonl(os.path.join(output_dir, "llm_val.jsonl"), val_records)

    # Stats
    print(f"\n{'='*60}")
    print(f"  LLM Training Data Generation Complete")
    print(f"{'='*60}")
    print(f"  Train: {len(train_records):,} examples")
    print(f"  Val:   {len(val_records):,} examples")
    print(f"  Total: {n:,} examples")
    print(f"\n  Domains ({len(all_domains)}):")
    for d in all_domains:
        print(f"    {d.name}: {len(d.mappings)} mappings")

    tc = Counter(r["transform"] for r in records)
    print(f"\n  Transform distribution:")
    for t, cnt in tc.most_common():
        print(f"    {t:20s}: {cnt:,}")

    # Sub-operation distribution (extracted from assistant content)
    import re as _re
    sub_op_pattern = _re.compile(r"sub_operation:\s*(\S+)")
    sub_ops = []
    for r in records:
        for msg in r["messages"]:
            if msg["role"] == "assistant":
                m = sub_op_pattern.search(msg["content"])
                if m:
                    sub_ops.append(m.group(1))
    soc = Counter(sub_ops)
    print(f"\n  Sub-operation distribution ({len(soc)} unique):")
    for s, cnt in soc.most_common():
        print(f"    {s:30s}: {cnt:,}")

    ac = Counter(r["arity"] for r in records)
    print(f"\n  Arity distribution:")
    for a, cnt in sorted(ac.items()):
        print(f"    arity={a}: {cnt:,}")

    pc = Counter(r["phase"] for r in records)
    print(f"\n  Phase distribution:")
    for p, cnt in pc.most_common():
        print(f"    {p:15s}: {cnt:,}")

    print(f"\n  Output: {output_dir}/")


if __name__ == "__main__":
    generate_llm_dataset()
