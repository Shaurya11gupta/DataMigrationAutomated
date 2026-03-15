"""
Test Schema Matcher LLM V4 on candidate selection + transformation declaration.
Runs diverse real-world-style schema mapping scenarios and checks:
  1. Does the model pick the correct source column(s)?
  2. Does it declare the right transformation type?
  3. Does it handle cross-table joins, ambiguity, and complex transforms?
"""
import json
import os
import re
import sys
import time

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
except ImportError as e:
    print(f"[ERROR] Missing: {e}")
    print("pip install torch transformers peft accelerate")
    sys.exit(1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTER_PATH = os.path.join(BASE_DIR, "schema_matcher_llm1", "adapter")
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"

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


def load_model():
    print(f"[1/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[2/3] Loading base model: {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=False,
        attn_implementation="eager",
    )

    print(f"[3/3] Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    if DEVICE == "cpu":
        model = model.to("cpu")
    print(f"[OK] Model loaded on {DEVICE} ({sum(p.numel() for p in model.parameters()):,} params)")
    return model, tokenizer


def generate(model, tokenizer, prompt):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>\n"

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=256, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse(response):
    result = {"source_columns": [], "transform_type": "", "reasoning": ""}
    m = re.search(r'source_columns?:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
    if m:
        result["source_columns"] = [c.strip() for c in m.group(1).split(",") if c.strip()]
    m = re.search(r'transform_type:\s*(\S+)', response, re.IGNORECASE)
    if m:
        result["transform_type"] = m.group(1).strip().lower()
    m = re.search(r'reasoning:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
    if m:
        result["reasoning"] = m.group(1).strip()[:200]
    return result


# ── Test scenarios: realistic candidate selection + transform tasks ──
TESTS = [
    # === SCENARIO 1: HR / Employee Management ===
    {
        "schema": """Source Schema:
  employees(emp_id PK int, first_name string, last_name string, email string, hire_date date, salary decimal, department_id FK int)
  departments(dept_id PK int, dept_name string, location string, budget decimal)

Joins:
  employees.department_id = departments.dept_id""",
        "targets": [
            ("dim_employee.employee_key (int) - \"surrogate key\"",
             ["employees.emp_id"], "rename"),
            ("dim_employee.full_name (string) - \"employee full name\"",
             ["employees.first_name", "employees.last_name"], "concat"),
            ("dim_employee.hire_year (int) - \"year hired\"",
             ["employees.hire_date"], "date_part"),
            ("dim_employee.department_name (string) - \"department name\"",
             ["departments.dept_name"], "fk_lookup"),
            ("dim_employee.annual_salary (decimal) - \"yearly salary\"",
             ["employees.salary"], "rename"),
        ],
        "name": "HR Schema",
    },

    # === SCENARIO 2: E-commerce / Orders ===
    {
        "schema": """Source Schema:
  orders(order_id PK int, customer_id FK int, order_date date, total_amount decimal, status string)
  customers(customer_id PK int, first_name string, last_name string, email string, country string)
  order_items(item_id PK int, order_id FK int, product_id FK int, quantity int, unit_price decimal)
  products(product_id PK int, product_name string, category_code string, weight_grams int)

Joins:
  orders.customer_id = customers.customer_id
  order_items.order_id = orders.order_id
  order_items.product_id = products.product_id""",
        "targets": [
            ("fact_order.customer_country (string) - \"country of the customer\"",
             ["customers.country"], "fk_lookup"),
            ("fact_order.customer_name (string) - \"full name of customer\"",
             ["customers.first_name", "customers.last_name"], "concat"),
            ("fact_order.order_year (int) - \"year the order was placed\"",
             ["orders.order_date"], "date_part"),
            ("fact_order.is_completed (boolean) - \"whether order is complete\"",
             ["orders.status"], "conditional"),
            ("fact_sales.line_total (decimal) - \"quantity times unit price\"",
             ["order_items.quantity", "order_items.unit_price"], "arithmetic"),
            ("dim_product.category_label (string) - \"human readable category\"",
             ["products.category_code"], "code_to_label"),
        ],
        "name": "E-commerce Schema",
    },

    # === SCENARIO 3: Healthcare ===
    {
        "schema": """Source Schema:
  patients(patient_id PK int, first_name string, last_name string, dob date, gender_code string, insurance_id FK int)
  insurance_plans(plan_id PK int, plan_name string, provider_name string, coverage_type_code string)
  visits(visit_id PK int, patient_id FK int, doctor_id FK int, visit_date date, diagnosis_code string, visit_cost decimal)
  doctors(doctor_id PK int, first_name string, last_name string, specialization_code string, department_id FK int)
  hospital_departments(dept_id PK int, dept_name string, floor int)

Joins:
  patients.insurance_id = insurance_plans.plan_id
  visits.patient_id = patients.patient_id
  visits.doctor_id = doctors.doctor_id
  doctors.department_id = hospital_departments.dept_id""",
        "targets": [
            ("dim_patient.patient_name (string) - \"full name of patient\"",
             ["patients.first_name", "patients.last_name"], "concat"),
            ("dim_patient.birth_year (int) - \"year patient was born\"",
             ["patients.dob"], "date_part"),
            ("dim_patient.gender_label (string) - \"gender in readable form\"",
             ["patients.gender_code"], "code_to_label"),
            ("dim_patient.insurance_plan (string) - \"name of patient insurance plan\"",
             ["insurance_plans.plan_name"], "fk_lookup"),
            ("fact_visit.patient_age_at_visit (int) - \"patient age at time of visit\"",
             ["patients.dob", "visits.visit_date"], "date_diff"),
            ("fact_visit.department_name (string) - \"department from doctor lookup\"",
             ["hospital_departments.dept_name"], "lookup_join"),
        ],
        "name": "Healthcare Schema",
    },

    # === SCENARIO 4: Banking (with noisy audit columns) ===
    {
        "schema": """Source Schema:
  bank_accounts(account_id PK int, account_number string, account_type_code string, customer_id FK int, open_date date, balance decimal, status_code string, created_by string, updated_at datetime)
  bank_customers(customer_id PK int, first_name string, last_name string, email string, credit_score int, income_annual decimal, risk_category_code string, created_by string, updated_at datetime)
  bank_transactions(txn_id PK int, account_id FK int, txn_date date, amount decimal, txn_type_code string, description string, created_by string)

Joins:
  bank_accounts.customer_id = bank_customers.customer_id
  bank_transactions.account_id = bank_accounts.account_id""",
        "targets": [
            ("dim_customer.customer_name (string) - \"customer full name\"",
             ["bank_customers.first_name", "bank_customers.last_name"], "concat"),
            ("dim_account.account_type_label (string) - \"human readable account type\"",
             ["bank_accounts.account_type_code"], "code_to_label"),
            ("dim_account.is_active (boolean) - \"whether account is currently open\"",
             ["bank_accounts.status_code"], "conditional"),
            ("dim_customer.income_bracket (string) - \"income range bucket\"",
             ["bank_customers.income_annual"], "bucketing"),
            ("fact_txn.transaction_year (int) - \"year of transaction\"",
             ["bank_transactions.txn_date"], "date_part"),
        ],
        "name": "Banking Schema (with noise columns)",
    },
]


def run_all_tests(model, tokenizer):
    print(f"\n{'='*80}")
    print(f"  CANDIDATE SELECTION + TRANSFORMATION TEST")
    print(f"  ({sum(len(t['targets']) for t in TESTS)} mappings across {len(TESTS)} schemas)")
    print(f"{'='*80}\n")

    total = 0
    col_correct = 0
    tf_correct = 0
    both_correct = 0
    total_time = 0
    failures = []

    for scenario in TESTS:
        print(f"\n--- {scenario['name']} ---")
        for target_str, exp_cols, exp_tf in scenario["targets"]:
            total += 1
            prompt = f"{scenario['schema']}\n\nMap target: {target_str}"

            t0 = time.time()
            response = generate(model, tokenizer, prompt)
            elapsed = time.time() - t0
            total_time += elapsed

            parsed = parse(response)
            pred_cols = sorted([c.lower().strip() for c in parsed["source_columns"]])
            expected = sorted([c.lower().strip() for c in exp_cols])
            pred_tf = parsed["transform_type"]

            cols_ok = pred_cols == expected
            tf_ok = pred_tf == exp_tf

            if cols_ok:
                col_correct += 1
            if tf_ok:
                tf_correct += 1
            if cols_ok and tf_ok:
                both_correct += 1

            status = "PASS" if (cols_ok and tf_ok) else "FAIL"
            target_short = target_str.split(" - ")[0] if " - " in target_str else target_str[:50]

            print(f"  [{status}] {target_short}")
            print(f"         Cols: {pred_cols} {'OK' if cols_ok else f'EXPECTED {expected}'}")
            print(f"         Transform: {pred_tf} {'OK' if tf_ok else f'EXPECTED {exp_tf}'}")
            if parsed.get("reasoning"):
                print(f"         Reasoning: {parsed['reasoning'][:120]}")
            print(f"         Time: {elapsed:.1f}s")

            if not (cols_ok and tf_ok):
                failures.append({
                    "scenario": scenario["name"],
                    "target": target_short,
                    "expected_cols": expected,
                    "predicted_cols": pred_cols,
                    "expected_tf": exp_tf,
                    "predicted_tf": pred_tf,
                })

    # ── Summary ──
    n = total
    print(f"\n{'='*80}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"  Source column accuracy:  {col_correct}/{n} = {col_correct/n:.1%}")
    print(f"  Transform accuracy:      {tf_correct}/{n} = {tf_correct/n:.1%}")
    print(f"  Both correct:            {both_correct}/{n} = {both_correct/n:.1%}")
    print(f"  Average response time:   {total_time/n:.1f}s")
    print(f"  Total time:              {total_time:.0f}s")

    if failures:
        print(f"\n  FAILURES ({len(failures)}):")
        for f in failures:
            print(f"    [{f['scenario']}] {f['target']}")
            if f["predicted_cols"] != f["expected_cols"]:
                print(f"      Cols: got {f['predicted_cols']}, expected {f['expected_cols']}")
            if f["predicted_tf"] != f["expected_tf"]:
                print(f"      Transform: got {f['predicted_tf']}, expected {f['expected_tf']}")
    else:
        print(f"\n  ALL TESTS PASSED!")

    print(f"{'='*80}")
    return {"total": n, "col_correct": col_correct, "tf_correct": tf_correct, "both": both_correct}


if __name__ == "__main__":
    model, tokenizer = load_model()
    run_all_tests(model, tokenizer)
