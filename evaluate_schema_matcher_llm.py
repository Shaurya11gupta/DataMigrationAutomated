#!/usr/bin/env python3
"""
Evaluate the Schema Matcher LLM (V4 - Production)
====================================================
Loads the Phi-3-mini base model + LoRA adapter and runs a comprehensive
evaluation covering all 15 transform types.

Production features:
  - 32 hand-crafted test cases across all transform types
  - Fuzzy column matching with partial-credit scoring
  - Per-transform-type accuracy breakdown
  - Difficulty tiers (Easy / Medium / Hard)
  - Structured output validation with retry logic
  - JSON export of detailed results
  - Greedy decoding (temperature=0) for deterministic outputs
"""

import argparse
import json
import os
import re
import time
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# ── Check dependencies ──
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("Install with: pip install torch transformers peft accelerate")
    sys.exit(1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTER_PATH = os.path.join(BASE_DIR, "schema_matcher_llm", "adapter")
MERGED_PATH = os.path.join(BASE_DIR, "schema_matcher_llm", "merged")
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"


# =====================================================================
# LOAD MODEL
# =====================================================================

def load_model(prefer_merged: bool = True):
    """Load model -- prefer merged (no peft needed) if available."""
    from transformers import AutoConfig

    # Try merged model first (faster, no peft dependency at inference)
    if prefer_merged and os.path.isdir(MERGED_PATH):
        print(f"[1/2] Loading merged model from: {MERGED_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(MERGED_PATH, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            MERGED_PATH,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto" if DEVICE == "cuda" else None,
            trust_remote_code=False,
            attn_implementation="eager",
        )
        model.eval()
        if DEVICE == "cpu":
            model = model.to("cpu")
        print(f"[OK] Merged model loaded on {DEVICE}")
        print(f"     Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model, tokenizer

    # Fall back to base + LoRA adapter
    print(f"[1/3] Loading tokenizer from adapter...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[2/3] Loading base model: {BASE_MODEL} ...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=False,
        attn_implementation="eager",
    )

    print(f"[3/3] Loading LoRA adapter from: {ADAPTER_PATH} ...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    if DEVICE == "cpu":
        model = model.to("cpu")

    print(f"[OK] Model loaded on {DEVICE}")
    print(f"     Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


# =====================================================================
# SYSTEM PROMPT
# =====================================================================

SYSTEM_PROMPT = """You are a schema mapping expert. Given a source database schema and a target column, identify which source column(s) should be mapped to the target and what transformation is needed.

Rules:
- Use ONLY columns that exist in the source schema
- Choose the MINIMUM number of source columns needed
- For surrogate keys, use the primary key of the matching entity table
- For person names combining first and last, use concat transform
- For cross-table lookups via foreign key, use fk_lookup transform
- For extracting year/month from a date, use date_part transform
- For computing differences between dates, use date_diff transform
- For math operations between columns, use arithmetic transform
- For boolean flags derived from status columns, use conditional transform
- For simple column renames or copies, use rename transform

Valid transform types: rename, direct_copy, concat, fk_lookup, date_part, date_diff, date_format, date_parse, arithmetic, conditional, bucketing, code_to_label, type_cast, lookup_join, template

Respond in EXACTLY this format:
source_columns: <table.column>, <table.column>, ...
transform_type: <transform>
reasoning: <brief explanation>"""


# =====================================================================
# INFERENCE WITH PRODUCTION FEATURES
# =====================================================================

def generate_response(
    model, tokenizer, user_prompt: str,
    max_new_tokens: int = 512,
    max_retries: int = 2,
) -> str:
    """Generate a schema mapping response with retry logic and validation."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        input_text = (
            f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
            f"<|user|>\n{user_prompt}<|end|>\n"
            f"<|assistant|>\n"
        )

    for attempt in range(max_retries + 1):
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=1.0,   # greedy: use do_sample=False
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Validate structured output
        if validate_response_format(response):
            return response

        if attempt < max_retries:
            # Retry with a nudge
            input_text += "\nPlease respond in the exact format: source_columns, transform_type, reasoning."

    return response  # Return best effort even if malformed


def validate_response_format(response: str) -> bool:
    """Check that the response contains all required structured fields."""
    has_source = bool(re.search(r'source_columns?:', response, re.IGNORECASE))
    has_transform = bool(re.search(r'transform_type:', response, re.IGNORECASE))
    has_reasoning = bool(re.search(r'reasoning:', response, re.IGNORECASE))
    return has_source and has_transform and has_reasoning


def parse_response(response: str) -> Dict:
    """Parse model response into structured fields."""
    result = {"source_columns": [], "transform_type": "", "reasoning": "", "raw": response}

    sc_match = re.search(r'source_columns?:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
    if sc_match:
        cols_str = sc_match.group(1).strip()
        result["source_columns"] = [c.strip() for c in cols_str.split(",") if c.strip()]

    tt_match = re.search(r'transform_type:\s*(\S+)', response, re.IGNORECASE)
    if tt_match:
        result["transform_type"] = tt_match.group(1).strip().lower()

    r_match = re.search(r'reasoning:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
    if r_match:
        result["reasoning"] = r_match.group(1).strip()

    return result


# =====================================================================
# FUZZY MATCHING
# =====================================================================

def fuzzy_column_score(predicted: List[str], expected: List[str]) -> float:
    """Compute column match score with partial credit.

    Returns:
        1.0 = exact match
        0.5 = correct table(s) but wrong column(s)
        0.0 = completely wrong
    """
    pred = sorted([c.lower().strip() for c in predicted])
    exp = sorted([c.lower().strip() for c in expected])

    if pred == exp:
        return 1.0

    # Partial credit: check table-level matches
    pred_tables = sorted([c.split(".")[0] for c in pred if "." in c])
    exp_tables = sorted([c.split(".")[0] for c in exp if "." in c])

    if pred_tables == exp_tables and pred_tables:
        return 0.5  # Right tables, wrong columns

    # Check if at least some overlap
    pred_set = set(pred)
    exp_set = set(exp)
    overlap = len(pred_set & exp_set)
    if overlap > 0:
        precision = overlap / max(len(pred_set), 1)
        recall = overlap / max(len(exp_set), 1)
        return (precision + recall) / 4.0  # Max partial = 0.5

    return 0.0


# =====================================================================
# TEST CASES (32 cases covering all 15 transform types)
# =====================================================================

# Difficulty tiers
EASY = "easy"
MEDIUM = "medium"
HARD = "hard"

TEST_CASES = [
    # ── RENAME (Easy) ──
    {
        "prompt": """Source Schema:
  customers(customer_id PK int, first_name string, last_name string, email string, phone string, city string)

Joins: (none)

Map target: dim_customer.cust_email (string) - "customer email address\"""",
        "expected_columns": ["customers.email"],
        "expected_transform": "rename",
        "description": "Simple rename: email -> cust_email",
        "difficulty": EASY,
    },
    {
        "prompt": """Source Schema:
  products(product_id PK int, product_name string, sku string, unit_price decimal, category string, weight_grams int)

Joins: (none)

Map target: dim_product.item_name (string) - "name of the product\"""",
        "expected_columns": ["products.product_name"],
        "expected_transform": "rename",
        "description": "Rename with synonym: product_name -> item_name",
        "difficulty": EASY,
    },

    # ── DIRECT_COPY (Easy) ──
    {
        "prompt": """Source Schema:
  employees(emp_id PK int, first_name string, last_name string, email string, badge_number string)

Joins: (none)

Map target: stg_employee.badge_number (string) - "employee badge ID\"""",
        "expected_columns": ["employees.badge_number"],
        "expected_transform": "direct_copy",
        "description": "Direct copy: badge_number unchanged",
        "difficulty": EASY,
    },

    # ── CONCAT (Easy-Medium) ──
    {
        "prompt": """Source Schema:
  employees(emp_id PK int, first_name string, last_name string, department_id FK int, hire_date date, salary decimal)
  departments(department_id PK int, dept_name string, location string)

Joins:
  employees.department_id = departments.department_id

Map target: dim_employee.full_name (string) - "employee full name\"""",
        "expected_columns": ["employees.first_name", "employees.last_name"],
        "expected_transform": "concat",
        "description": "Concat first + last name",
        "difficulty": EASY,
    },
    {
        "prompt": """Source Schema:
  citizens(citizen_id PK int, first_name string, last_name string, address string, city string, state string, zip_code string)

Joins: (none)

Map target: dim_citizen.full_address (string) - "complete mailing address\"""",
        "expected_columns": ["citizens.address", "citizens.city", "citizens.state", "citizens.zip_code"],
        "expected_transform": "concat",
        "description": "Concat 4-column address",
        "difficulty": MEDIUM,
    },

    # ── FK_LOOKUP (Medium) ──
    {
        "prompt": """Source Schema:
  orders(order_id PK int, customer_id FK int, order_date date, total_amount decimal, status string)
  customers(customer_id PK int, first_name string, last_name string, email string, city string, country string)

Joins:
  orders.customer_id = customers.customer_id

Map target: fact_order.customer_country (string) - "country of the customer\"""",
        "expected_columns": ["customers.country"],
        "expected_transform": "fk_lookup",
        "description": "FK lookup: customer country via orders",
        "difficulty": MEDIUM,
    },
    {
        "prompt": """Source Schema:
  enrollments(enrollment_id PK int, student_id FK int, course_id FK int, enroll_date date, score decimal)
  courses(course_id PK int, course_title string, instructor_id FK int, category_code string, price decimal)
  instructors(instructor_id PK int, first_name string, last_name string, email string)

Joins:
  enrollments.course_id = courses.course_id
  courses.instructor_id = instructors.instructor_id

Map target: fact_enrollment.course_title (string) - "name of enrolled course\"""",
        "expected_columns": ["courses.course_title"],
        "expected_transform": "fk_lookup",
        "description": "FK lookup: course title from enrollments",
        "difficulty": MEDIUM,
    },

    # ── DATE_PART (Easy) ──
    {
        "prompt": """Source Schema:
  transactions(txn_id PK int, account_id FK int, txn_date date, amount decimal, txn_type string)
  accounts(account_id PK int, customer_id FK int, account_type string, balance decimal)

Joins:
  transactions.account_id = accounts.account_id

Map target: fact_transaction.txn_month (int) - "month of transaction\"""",
        "expected_columns": ["transactions.txn_date"],
        "expected_transform": "date_part",
        "description": "Date part: extract month from txn_date",
        "difficulty": EASY,
    },
    {
        "prompt": """Source Schema:
  campaigns(campaign_id PK int, campaign_name string, start_date date, end_date date, budget decimal, spend decimal)

Joins: (none)

Map target: dim_campaign.start_year (int) - "year campaign started\"""",
        "expected_columns": ["campaigns.start_date"],
        "expected_transform": "date_part",
        "description": "Date part: extract year from start_date",
        "difficulty": EASY,
    },

    # ── DATE_DIFF (Medium) ──
    {
        "prompt": """Source Schema:
  projects(project_id PK int, project_name string, start_date date, end_date date, budget decimal, manager_id FK int)
  employees(emp_id PK int, first_name string, last_name string)

Joins:
  projects.manager_id = employees.emp_id

Map target: fact_project.duration_days (int) - "project duration in days\"""",
        "expected_columns": ["projects.start_date", "projects.end_date"],
        "expected_transform": "date_diff",
        "description": "Date diff: end_date - start_date",
        "difficulty": MEDIUM,
    },
    {
        "prompt": """Source Schema:
  bills(bill_id PK int, cust_id FK int, bill_date date, amount decimal, payment_date date, status string)

Joins: (none)

Map target: fact_billing.days_to_pay (int) - "days between bill and payment\"""",
        "expected_columns": ["bills.bill_date", "bills.payment_date"],
        "expected_transform": "date_diff",
        "description": "Date diff: payment_date - bill_date",
        "difficulty": MEDIUM,
    },

    # ── DATE_FORMAT (Easy) ──
    {
        "prompt": """Source Schema:
  bank_accounts(account_id PK int, account_number string, open_date date, balance decimal, status_code string)

Joins: (none)

Map target: dim_account.open_date_formatted (string) - "opening date in display format\"""",
        "expected_columns": ["bank_accounts.open_date"],
        "expected_transform": "date_format",
        "description": "Date format: date to display string",
        "difficulty": EASY,
    },

    # ── DATE_PARSE (Easy) ──
    {
        "prompt": """Source Schema:
  bank_customers(customer_id PK int, first_name string, last_name string, kyc_date date, credit_score int)

Joins: (none)

Map target: dim_customer.kyc_timestamp (datetime) - "KYC date parsed to datetime\"""",
        "expected_columns": ["bank_customers.kyc_date"],
        "expected_transform": "date_parse",
        "description": "Date parse: date -> datetime",
        "difficulty": EASY,
    },

    # ── ARITHMETIC (Medium) ──
    {
        "prompt": """Source Schema:
  order_items(item_id PK int, order_id FK int, product_id FK int, quantity int, unit_price decimal, discount_pct decimal)
  orders(order_id PK int, order_date date, customer_id FK int)

Joins:
  order_items.order_id = orders.order_id

Map target: fact_sales.line_total (decimal) - "quantity times unit price\"""",
        "expected_columns": ["order_items.quantity", "order_items.unit_price"],
        "expected_transform": "arithmetic",
        "description": "Arithmetic: quantity * unit_price",
        "difficulty": MEDIUM,
    },
    {
        "prompt": """Source Schema:
  store_items(item_id PK int, item_name string, unit_price decimal, cost_price decimal, category_code string)

Joins: (none)

Map target: dim_product.profit_margin (decimal) - "selling price minus cost\"""",
        "expected_columns": ["store_items.unit_price", "store_items.cost_price"],
        "expected_transform": "arithmetic",
        "description": "Arithmetic: unit_price - cost_price",
        "difficulty": MEDIUM,
    },
    {
        "prompt": """Source Schema:
  posts(post_id PK int, user_id FK int, like_count int, comment_count int, share_count int, post_date datetime)

Joins: (none)

Map target: fact_post.total_engagement (int) - "likes plus comments plus shares\"""",
        "expected_columns": ["posts.like_count", "posts.comment_count", "posts.share_count"],
        "expected_transform": "arithmetic",
        "description": "Arithmetic: 3-column sum",
        "difficulty": HARD,
    },

    # ── CONDITIONAL (Easy) ──
    {
        "prompt": """Source Schema:
  subscriptions(sub_id PK int, user_id FK int, plan string, status string, start_date date, end_date date, monthly_fee decimal)
  users(user_id PK int, username string, email string, created_at datetime)

Joins:
  subscriptions.user_id = users.user_id

Map target: dim_subscription.is_active (boolean) - "whether subscription is currently active\"""",
        "expected_columns": ["subscriptions.status"],
        "expected_transform": "conditional",
        "description": "Conditional: status -> is_active flag",
        "difficulty": EASY,
    },
    {
        "prompt": """Source Schema:
  freight_orders(order_id PK int, shipper_id FK int, pickup_date date, delivery_date date, weight_kg decimal, status_code string, mode_code string)

Joins: (none)

Map target: fact_freight.is_delivered (boolean) - "whether order is delivered\"""",
        "expected_columns": ["freight_orders.status_code"],
        "expected_transform": "conditional",
        "description": "Conditional: status_code -> is_delivered",
        "difficulty": EASY,
    },

    # ── TYPE_CAST (Easy) ──
    {
        "prompt": """Source Schema:
  sensors(sensor_id PK int, sensor_name string, location string, installed_date date)
  readings(reading_id PK int, sensor_id FK int, timestamp datetime, value string, unit string)

Joins:
  readings.sensor_id = sensors.sensor_id

Map target: fact_reading.numeric_value (decimal) - "sensor reading as number\"""",
        "expected_columns": ["readings.value"],
        "expected_transform": "type_cast",
        "description": "Type cast: string value -> decimal",
        "difficulty": EASY,
    },
    {
        "prompt": """Source Schema:
  properties(property_id PK int, address string, sqft int, list_price decimal, status_code string)

Joins: (none)

Map target: dim_property.sqft_text (string) - "sqft as formatted text\"""",
        "expected_columns": ["properties.sqft"],
        "expected_transform": "type_cast",
        "description": "Type cast: int -> string",
        "difficulty": EASY,
    },

    # ── CODE_TO_LABEL (Easy) ──
    {
        "prompt": """Source Schema:
  bank_accounts(account_id PK int, account_number string, account_type_code string, balance decimal, status_code string)

Joins: (none)

Map target: dim_account.account_type_label (string) - "human readable account type\"""",
        "expected_columns": ["bank_accounts.account_type_code"],
        "expected_transform": "code_to_label",
        "description": "Code to label: account_type_code -> label",
        "difficulty": EASY,
    },
    {
        "prompt": """Source Schema:
  patients(patient_id PK int, first_name string, last_name string, gender_code string, blood_type string)

Joins: (none)

Map target: dim_patient.gender_label (string) - "gender in readable form\"""",
        "expected_columns": ["patients.gender_code"],
        "expected_transform": "code_to_label",
        "description": "Code to label: gender_code -> label",
        "difficulty": EASY,
    },

    # ── BUCKETING (Medium) ──
    {
        "prompt": """Source Schema:
  bank_customers(customer_id PK int, first_name string, last_name string, income_annual decimal, credit_score int, risk_category_code string)

Joins: (none)

Map target: dim_customer.income_bracket (string) - "income range bucket\"""",
        "expected_columns": ["bank_customers.income_annual"],
        "expected_transform": "bucketing",
        "description": "Bucketing: income -> bracket",
        "difficulty": MEDIUM,
    },
    {
        "prompt": """Source Schema:
  courses(course_id PK int, course_title string, price decimal, duration_hours decimal, publish_date date)

Joins: (none)

Map target: dim_course.price_tier (string) - "price bracket: free, low, mid, high\"""",
        "expected_columns": ["courses.price"],
        "expected_transform": "bucketing",
        "description": "Bucketing: price -> tier",
        "difficulty": MEDIUM,
    },

    # ── TEMPLATE (Hard) ──
    {
        "prompt": """Source Schema:
  bank_accounts(account_id PK int, account_number string, account_type_code string, balance decimal, status_code string)

Joins: (none)

Map target: dim_account.account_display (string) - "formatted account label like Account #12345 (Savings)\"""",
        "expected_columns": ["bank_accounts.account_number", "bank_accounts.account_type_code"],
        "expected_transform": "template",
        "description": "Template: formatted account display string",
        "difficulty": HARD,
    },
    {
        "prompt": """Source Schema:
  doctors(doctor_id PK int, first_name string, last_name string, specialization_code string, department_id FK int)

Joins: (none)

Map target: dim_doctor.doctor_display (string) - "formatted: Dr. FirstName LastName - Specialty\"""",
        "expected_columns": ["doctors.first_name", "doctors.last_name", "doctors.specialization_code"],
        "expected_transform": "template",
        "description": "Template: doctor display with 3 columns",
        "difficulty": HARD,
    },

    # ── LOOKUP_JOIN (Hard) ──
    {
        "prompt": """Source Schema:
  invoices(invoice_id PK int, order_id FK int, invoice_date date, amount decimal, paid boolean)
  orders(order_id PK int, customer_id FK int, order_date date, status string)
  customers(customer_id PK int, company_name string, email string, region string)

Joins:
  invoices.order_id = orders.order_id
  orders.customer_id = customers.customer_id

Map target: fact_invoice.customer_region (string) - "region of the customer\"""",
        "expected_columns": ["customers.region"],
        "expected_transform": "fk_lookup",
        "description": "Multi-hop FK: invoice -> order -> customer.region",
        "difficulty": HARD,
    },
    {
        "prompt": """Source Schema:
  visits(visit_id PK int, patient_id FK int, doctor_id FK int, visit_date date, diagnosis_code string, visit_cost decimal)
  doctors(doctor_id PK int, first_name string, last_name string, specialization_code string, department_id FK int)
  hospital_departments(dept_id PK int, dept_name string, floor int, bed_count int)

Joins:
  visits.doctor_id = doctors.doctor_id
  doctors.department_id = hospital_departments.dept_id

Map target: fact_visit.department_name (string) - "department from doctor lookup\"""",
        "expected_columns": ["hospital_departments.dept_name"],
        "expected_transform": "lookup_join",
        "description": "Lookup join: visit -> doctor -> department",
        "difficulty": HARD,
    },
    {
        "prompt": """Source Schema:
  claims(claim_id PK int, policy_id FK int, claim_date date, claim_amount decimal, status_code string)
  policies(policy_id PK int, holder_id FK int, policy_number string, start_date date, premium_monthly decimal)
  policyholders(holder_id PK int, first_name string, last_name string, email string, risk_score int)

Joins:
  claims.policy_id = policies.policy_id
  policies.holder_id = policyholders.holder_id

Map target: fact_claim.policyholder_name (string) - "claimant name via policy lookup\"""",
        "expected_columns": ["policyholders.first_name", "policyholders.last_name"],
        "expected_transform": "lookup_join",
        "description": "Lookup join: claim -> policy -> policyholder name",
        "difficulty": HARD,
    },

    # ── CROSS-DOMAIN: Healthcare age calculation (Hard) ──
    {
        "prompt": """Source Schema:
  patients(patient_id PK int, first_name string, last_name string, dob date, gender string, blood_type string)
  visits(visit_id PK int, patient_id FK int, doctor_id FK int, visit_date date, diagnosis string, treatment string, cost decimal)
  doctors(doctor_id PK int, doctor_name string, specialty string, department string)

Joins:
  visits.patient_id = patients.patient_id
  visits.doctor_id = doctors.doctor_id

Map target: fact_visit.patient_age_at_visit (int) - "patient age at time of visit\"""",
        "expected_columns": ["patients.dob", "visits.visit_date"],
        "expected_transform": "date_diff",
        "description": "Healthcare: age = visit_date - dob (cross-table)",
        "difficulty": HARD,
    },

    # ── CROSS-DOMAIN: Agriculture yield calculation (Hard) ──
    {
        "prompt": """Source Schema:
  fields(field_id PK int, farm_id FK int, field_name string, area_acres decimal, crop_type_code string, last_planted date)
  harvests(harvest_id PK int, field_id FK int, harvest_date date, yield_kg decimal, quality_grade_code string, sale_price_per_kg decimal)

Joins:
  harvests.field_id = fields.field_id

Map target: fact_harvest.yield_per_acre (decimal) - "yield divided by field area\"""",
        "expected_columns": ["harvests.yield_kg", "fields.area_acres"],
        "expected_transform": "arithmetic",
        "description": "Arithmetic: cross-table yield/area calculation",
        "difficulty": HARD,
    },
]


# =====================================================================
# EVALUATE
# =====================================================================

def evaluate(model, tokenizer, export_path: Optional[str] = None) -> Dict:
    """Run all test cases and compute comprehensive metrics."""
    n = len(TEST_CASES)
    print(f"\n{'='*80}")
    print(f"  SCHEMA MATCHER LLM EVALUATION ({n} test cases)")
    print(f"{'='*80}\n")

    results = []
    metrics = {
        "total": n,
        "col_exact": 0,
        "col_fuzzy_sum": 0.0,
        "transform_correct": 0,
        "both_correct": 0,
        "format_ok": 0,
        "total_time": 0.0,
        "by_transform": defaultdict(lambda: {"total": 0, "col_ok": 0, "tf_ok": 0, "both_ok": 0}),
        "by_difficulty": defaultdict(lambda: {"total": 0, "col_ok": 0, "tf_ok": 0, "both_ok": 0}),
    }

    for i, tc in enumerate(TEST_CASES):
        desc = tc["description"]
        difficulty = tc.get("difficulty", "unknown")
        exp_transform = tc["expected_transform"]

        print(f"--- Test {i+1}/{n}: [{difficulty.upper()}] {desc} ---")

        t0 = time.time()
        raw_response = generate_response(model, tokenizer, tc["prompt"])
        elapsed = time.time() - t0
        metrics["total_time"] += elapsed

        parsed = parse_response(raw_response)

        # Evaluate source columns
        pred_cols = sorted([c.lower().strip() for c in parsed["source_columns"]])
        exp_cols = sorted([c.lower().strip() for c in tc["expected_columns"]])
        cols_exact = pred_cols == exp_cols
        cols_fuzzy = fuzzy_column_score(parsed["source_columns"], tc["expected_columns"])

        # Evaluate transform type
        pred_tf = parsed["transform_type"].lower().strip()
        exp_tf = exp_transform.lower().strip()
        tf_match = pred_tf == exp_tf

        # Format check
        has_format = validate_response_format(raw_response)

        # Update counters
        if cols_exact:
            metrics["col_exact"] += 1
        metrics["col_fuzzy_sum"] += cols_fuzzy
        if tf_match:
            metrics["transform_correct"] += 1
        if cols_exact and tf_match:
            metrics["both_correct"] += 1
        if has_format:
            metrics["format_ok"] += 1

        # Per-transform breakdown
        tf_bucket = metrics["by_transform"][exp_tf]
        tf_bucket["total"] += 1
        if cols_exact:
            tf_bucket["col_ok"] += 1
        if tf_match:
            tf_bucket["tf_ok"] += 1
        if cols_exact and tf_match:
            tf_bucket["both_ok"] += 1

        # Per-difficulty breakdown
        diff_bucket = metrics["by_difficulty"][difficulty]
        diff_bucket["total"] += 1
        if cols_exact:
            diff_bucket["col_ok"] += 1
        if tf_match:
            diff_bucket["tf_ok"] += 1
        if cols_exact and tf_match:
            diff_bucket["both_ok"] += 1

        status = "PASS" if (cols_exact and tf_match) else "PARTIAL" if (cols_fuzzy > 0 or tf_match) else "FAIL"
        print(f"  Response ({elapsed:.1f}s):")
        print(f"    {raw_response[:200]}{'...' if len(raw_response) > 200 else ''}")
        print(f"  Columns:   expected={exp_cols}, got={pred_cols} {'EXACT' if cols_exact else f'fuzzy={cols_fuzzy:.2f}'}")
        print(f"  Transform: expected={exp_tf}, got={pred_tf} {'OK' if tf_match else 'WRONG'}")
        print(f"  [{status}]\n")

        results.append({
            "test_id": i + 1,
            "description": desc,
            "difficulty": difficulty,
            "transform": exp_tf,
            "cols_exact": cols_exact,
            "cols_fuzzy_score": cols_fuzzy,
            "tf_match": tf_match,
            "both_correct": cols_exact and tf_match,
            "format_ok": has_format,
            "elapsed_s": round(elapsed, 2),
            "predicted_columns": pred_cols,
            "expected_columns": exp_cols,
            "predicted_transform": pred_tf,
            "expected_transform": exp_tf,
            "raw_response": raw_response,
        })

    # ── Summary ──
    print(f"\n{'='*80}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"  Source columns (exact):  {metrics['col_exact']}/{n} = {metrics['col_exact']/n:.1%}")
    print(f"  Source columns (fuzzy):  {metrics['col_fuzzy_sum']/n:.2f} avg score")
    print(f"  Transform type:          {metrics['transform_correct']}/{n} = {metrics['transform_correct']/n:.1%}")
    print(f"  Both correct:            {metrics['both_correct']}/{n} = {metrics['both_correct']/n:.1%}")
    print(f"  Format compliance:       {metrics['format_ok']}/{n} = {metrics['format_ok']/n:.1%}")
    print(f"  Average response time:   {metrics['total_time']/n:.1f}s")

    # Per-transform breakdown
    print(f"\n  PER-TRANSFORM BREAKDOWN:")
    print(f"  {'Transform':<18} {'Total':>5} {'ColOK':>6} {'TfOK':>6} {'Both':>6} {'Accuracy':>9}")
    print(f"  {'-'*55}")
    for tf_name in sorted(metrics["by_transform"].keys()):
        b = metrics["by_transform"][tf_name]
        acc = b["both_ok"] / b["total"] if b["total"] > 0 else 0
        print(f"  {tf_name:<18} {b['total']:>5} {b['col_ok']:>6} {b['tf_ok']:>6} {b['both_ok']:>6} {acc:>8.0%}")

    # Per-difficulty breakdown
    print(f"\n  PER-DIFFICULTY BREAKDOWN:")
    print(f"  {'Difficulty':<10} {'Total':>5} {'ColOK':>6} {'TfOK':>6} {'Both':>6} {'Accuracy':>9}")
    print(f"  {'-'*48}")
    for diff in [EASY, MEDIUM, HARD]:
        if diff in metrics["by_difficulty"]:
            b = metrics["by_difficulty"][diff]
            acc = b["both_ok"] / b["total"] if b["total"] > 0 else 0
            print(f"  {diff:<10} {b['total']:>5} {b['col_ok']:>6} {b['tf_ok']:>6} {b['both_ok']:>6} {acc:>8.0%}")

    # Failed cases
    failures = [r for r in results if not r["both_correct"]]
    if failures:
        print(f"\n  FAILED CASES ({len(failures)}):")
        for r in failures:
            print(f"    [{r['difficulty'].upper()}] {r['description']}:")
            if not r["cols_exact"]:
                print(f"      Cols: expected {r['expected_columns']}, got {r['predicted_columns']} (fuzzy={r['cols_fuzzy_score']:.2f})")
            if not r["tf_match"]:
                print(f"      Transform: expected {r['expected_transform']}, got {r['predicted_transform']}")

    print(f"{'='*80}")

    # ── Export to JSON ──
    if export_path:
        export_data = {
            "summary": {
                "total_tests": n,
                "col_exact_accuracy": round(metrics["col_exact"] / n, 4),
                "col_fuzzy_avg": round(metrics["col_fuzzy_sum"] / n, 4),
                "transform_accuracy": round(metrics["transform_correct"] / n, 4),
                "both_correct_accuracy": round(metrics["both_correct"] / n, 4),
                "format_compliance": round(metrics["format_ok"] / n, 4),
                "avg_response_time_s": round(metrics["total_time"] / n, 2),
                "total_time_s": round(metrics["total_time"], 2),
            },
            "by_transform": {
                k: {
                    "total": v["total"],
                    "accuracy": round(v["both_ok"] / v["total"], 4) if v["total"] > 0 else 0,
                }
                for k, v in metrics["by_transform"].items()
            },
            "by_difficulty": {
                k: {
                    "total": v["total"],
                    "accuracy": round(v["both_ok"] / v["total"], 4) if v["total"] > 0 else 0,
                }
                for k, v in metrics["by_difficulty"].items()
            },
            "results": results,
        }
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"\n  Results exported to: {export_path}")

    return {"metrics": metrics, "results": results}


# =====================================================================
# VALIDATION DATA EVALUATION
# =====================================================================

def evaluate_on_val_data(model, tokenizer, val_path=None, max_samples=20):
    """Evaluate on held-out validation data from the training set."""
    if val_path is None:
        val_path = os.path.join(BASE_DIR, "llm_training_data", "llm_val.jsonl")

    if not os.path.exists(val_path):
        print(f"\n[SKIP] Validation file not found: {val_path}")
        return

    print(f"\n{'='*80}")
    print(f"  VALIDATION DATA EVALUATION (up to {max_samples} samples)")
    print(f"{'='*80}\n")

    records = []
    with open(val_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    import random
    random.seed(42)
    if len(records) > max_samples:
        records = random.sample(records, max_samples)

    col_correct = 0
    tf_correct = 0
    both_correct = 0
    col_fuzzy_sum = 0.0

    for i, rec in enumerate(records):
        messages = rec["messages"]
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        expected_response = next((m["content"] for m in messages if m["role"] == "assistant"), "")

        expected_parsed = parse_response(expected_response)

        t0 = time.time()
        pred_response = generate_response(model, tokenizer, user_msg)
        elapsed = time.time() - t0
        pred_parsed = parse_response(pred_response)

        exp_cols = sorted([c.lower().strip() for c in expected_parsed["source_columns"]])
        pred_cols = sorted([c.lower().strip() for c in pred_parsed["source_columns"]])
        cols_ok = exp_cols == pred_cols
        fuzzy = fuzzy_column_score(pred_parsed["source_columns"], expected_parsed["source_columns"])

        exp_tf = expected_parsed["transform_type"].lower().strip()
        pred_tf = pred_parsed["transform_type"].lower().strip()
        tf_ok = exp_tf == pred_tf

        if cols_ok:
            col_correct += 1
        col_fuzzy_sum += fuzzy
        if tf_ok:
            tf_correct += 1
        if cols_ok and tf_ok:
            both_correct += 1

        status = "PASS" if (cols_ok and tf_ok) else "FAIL"
        domain = rec.get("domain", "?")
        transform = rec.get("transform", "?")
        print(f"  [{status}] Sample {i+1} ({domain}/{transform}) [{elapsed:.1f}s] "
              f"cols={'EXACT' if cols_ok else f'fuzzy={fuzzy:.2f}'} tf={'OK' if tf_ok else 'WRONG'}")

        if not (cols_ok and tf_ok):
            if not cols_ok:
                print(f"      Expected cols: {exp_cols}, Got: {pred_cols}")
            if not tf_ok:
                print(f"      Expected tf: {exp_tf}, Got: {pred_tf}")

    n = len(records)
    print(f"\n  Val Results:")
    print(f"    Columns (exact): {col_correct}/{n} ({col_correct/n:.1%})")
    print(f"    Columns (fuzzy): {col_fuzzy_sum/n:.2f} avg")
    print(f"    Transform:       {tf_correct}/{n} ({tf_correct/n:.1%})")
    print(f"    Both correct:    {both_correct}/{n} ({both_correct/n:.1%})")


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Schema Matcher LLM")
    parser.add_argument("--export", type=str, default=None,
                        help="Export detailed results to JSON file (e.g., eval_results.json)")
    parser.add_argument("--val-samples", type=int, default=15,
                        help="Number of validation samples to test (default: 15)")
    parser.add_argument("--no-val", action="store_true",
                        help="Skip validation data evaluation (faster)")
    parser.add_argument("--no-merged", action="store_true",
                        help="Force loading base+adapter even if merged model exists")
    args = parser.parse_args()

    # Default export path if not specified
    export_path = args.export or os.path.join(BASE_DIR, "eval_results.json")

    model, tokenizer = load_model(prefer_merged=not args.no_merged)
    result = evaluate(model, tokenizer, export_path=export_path)

    if not args.no_val:
        evaluate_on_val_data(model, tokenizer, max_samples=args.val_samples)
