"""
Retail Analytics Schema — Compare schema_matcher_llm vs schema_matcher_llm1
===========================================================================
Runs the Retail Demo target columns through both models and prints
side-by-side comparison of predicted source columns.
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
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Retail Analytics schema (matches app's EXAMPLE_RETAIL_DEMO)
RETAIL_SCHEMA = """Source Schema:
  stores(store_id PK int, store_name string, city string, region_id FK int)
  regions(region_id PK int, region_code string, region_name string)
  products(product_id PK int, product_name string, category_id FK int, unit_cost decimal)
  categories(category_id PK int, category_code string, category_name string)
  customers(customer_id PK int, first_name string, last_name string, email string, signup_date date)
  orders(order_id PK int, customer_id FK int, store_id FK int, order_date date, status_code string)
  order_lines(line_id PK int, order_id FK int, product_id FK int, quantity int, unit_price decimal)

Joins:
  stores.region_id = regions.region_id
  products.category_id = categories.category_id
  orders.customer_id = customers.customer_id
  orders.store_id = stores.store_id
  order_lines.order_id = orders.order_id
  order_lines.product_id = products.product_id"""

# Target columns with expected source columns (for reference)
RETAIL_TARGETS = [
    ("dim_store.store_key (int) - surrogate key for store", ["stores.store_id"]),
    ("dim_store.store_name (string) - store display name", ["stores.store_name"]),
    ("dim_store.city (string) - store city", ["stores.city"]),
    ("dim_store.region_label (string) - human-readable region name from region lookup", ["regions.region_name"]),
    ("dim_product.product_key (int) - surrogate key for product", ["products.product_id"]),
    ("dim_product.product_name (string) - product name", ["products.product_name"]),
    ("dim_product.category_label (string) - human-readable category from category lookup", ["categories.category_name"]),
    ("dim_product.unit_cost (decimal) - product unit cost", ["products.unit_cost"]),
    ("dim_customer.customer_key (int) - surrogate key for customer", ["customers.customer_id"]),
    ("dim_customer.customer_name (string) - full name combining first and last name", ["customers.first_name", "customers.last_name"]),
    ("dim_customer.email (string) - customer email", ["customers.email"]),
    ("dim_customer.signup_year (int) - year customer signed up", ["customers.signup_date"]),
    ("fact_sales.order_key (int) - order identifier", ["orders.order_id"]),
    ("fact_sales.customer_name (string) - name of customer who placed the order", ["customers.first_name", "customers.last_name"]),
    ("fact_sales.store_name (string) - store where order was placed", ["stores.store_name"]),
    ("fact_sales.order_year (int) - year the order was placed", ["orders.order_date"]),
    ("fact_sales.line_total (decimal) - quantity multiplied by unit price for the line", ["order_lines.quantity", "order_lines.unit_price"]),
    ("fact_sales.is_shipped (boolean) - whether order status is shipped", ["orders.status_code"]),
]

SYSTEM_PROMPT = """You are a schema mapping expert. Given a source database schema and a target column, identify which source column(s) should be mapped to the target and what transformation is needed.

COLUMN SELECTION RULES:
- Use ONLY columns that exist in the source schema
- Choose the MINIMUM number of source columns needed
- ALWAYS output the VALUE column(s) that hold the actual data, NOT the FK/join key columns
- For surrogate keys, use the primary key of the matching entity table

TRANSFORM RULES:
- rename: same table, no computation
- fk_lookup: column in different table reached via single FK join
- concat: combining 2+ columns into one string
- date_part: extracting year/month from date
- arithmetic: qty * price, etc.
- conditional: boolean from status/code
- code_to_label: code to human-readable label

Valid transform types: rename, direct_copy, concat, fk_lookup, date_part, date_diff, arithmetic, conditional, code_to_label, lookup_join, template

Respond in EXACTLY this format:
source_columns: <table.column>, <table.column>, ...
transform_type: <transform>
reasoning: <brief explanation>"""


def load_model(adapter_path: str):
    """Load model and tokenizer from adapter path."""
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float32 if DEVICE == "cpu" else torch.float16
    device_map = None if DEVICE == "cpu" else "auto"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=False,
        attn_implementation="eager",
    )
    if DEVICE == "cpu":
        model = model.to("cpu")

    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str) -> str:
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
            **inputs, max_new_tokens=200, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse(response: str) -> dict:
    result = {"source_columns": [], "transform_type": ""}
    m = re.search(r'source_columns?:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
    if m:
        result["source_columns"] = [c.strip().lower() for c in m.group(1).split(",") if c.strip()]
    m = re.search(r'transform_type:\s*(\S+)', response, re.IGNORECASE)
    if m:
        result["transform_type"] = m.group(1).strip().lower()
    return result


def run_retail_test(model, tokenizer, model_name: str) -> list:
    """Run all Retail targets and return list of (target_short, predicted_cols, expected_cols, match)."""
    results = []
    for target_str, exp_cols in RETAIL_TARGETS:
        prompt = f"{RETAIL_SCHEMA}\n\nMap target: {target_str}"
        response = generate(model, tokenizer, prompt)
        parsed = parse(response)
        pred = sorted(parsed["source_columns"])
        exp = sorted([c.lower() for c in exp_cols])
        match = pred == exp
        target_short = target_str.split(" - ")[0] if " - " in target_str else target_str[:55]
        results.append((target_short, pred, exp, match, parsed["transform_type"]))
    return results


def main():
    paths = [
        ("schema_matcher_llm", os.path.join(BASE_DIR, "schema_matcher_llm", "adapter")),
        ("schema_matcher_llm1", os.path.join(BASE_DIR, "schema_matcher_llm1", "adapter")),
    ]

    # Check which adapters exist
    available = []
    for name, path in paths:
        cfg = os.path.join(path, "adapter_config.json")
        if os.path.exists(cfg):
            available.append((name, path))
        else:
            print(f"[SKIP] {name}: adapter not found at {path}")

    if not available:
        print("No adapters found. Exiting.")
        sys.exit(1)

    print(f"\n{'='*100}")
    print("  RETAIL ANALYTICS SCHEMA — Model Comparison")
    print(f"  Device: {DEVICE}  |  Targets: {len(RETAIL_TARGETS)}")
    print(f"{'='*100}\n")

    all_results = {}
    for model_name, adapter_path in available:
        print(f"[Loading] {model_name}...")
        t0 = time.time()
        model, tokenizer = load_model(adapter_path)
        load_time = time.time() - t0
        print(f"  Loaded in {load_time:.1f}s\n")

        print(f"[Running] {model_name} on {len(RETAIL_TARGETS)} targets...")
        t0 = time.time()
        results = run_retail_test(model, tokenizer, model_name)
        run_time = time.time() - t0
        all_results[model_name] = results
        print(f"  Done in {run_time:.1f}s ({run_time/len(RETAIL_TARGETS):.2f}s per target)\n")

        # Free memory for next model
        del model, tokenizer
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # Print comparison table
    print(f"\n{'='*100}")
    print("  COLUMN-BY-COLUMN COMPARISON (source_columns)")
    print(f"{'='*100}\n")

    models = list(all_results.keys())
    col_width = 50
    pred_width = 35

    header = f"{'Target':<{col_width}}"
    for m in models:
        header += f" | {m} (predicted)"
    header += f" | Expected"
    print(header)
    print("-" * len(header))

    col_correct = {m: 0 for m in models}
    for i in range(len(RETAIL_TARGETS)):
        target_short = RETAIL_TARGETS[i][0].split(" - ")[0] if " - " in RETAIL_TARGETS[i][0] else RETAIL_TARGETS[i][0][:col_width-2]
        expected = RETAIL_TARGETS[i][1]

        row = f"{target_short[:col_width-2]:<{col_width}}"
        for m in models:
            pred = all_results[m][i][1]
            match = all_results[m][i][3]
            if match:
                col_correct[m] += 1
            pred_str = ", ".join(pred) if pred else "(none)"
            if len(pred_str) > pred_width - 2:
                pred_str = pred_str[:pred_width - 5] + "..."
            icon = "✓" if match else "✗"
            row += f" | {icon} {pred_str:<{pred_width-2}}"
        exp_str = ", ".join(expected)
        if len(exp_str) > pred_width - 2:
            exp_str = exp_str[:pred_width - 5] + "..."
        row += f" | {exp_str}"
        print(row)

    print(f"\n{'='*100}")
    print("  SUMMARY — Source Column Accuracy")
    print(f"{'='*100}")
    for m in models:
        n = len(RETAIL_TARGETS)
        acc = col_correct[m] / n * 100
        print(f"  {m}: {col_correct[m]}/{n} = {acc:.1f}%")
    print(f"{'='*100}\n")

    # Save results to JSON
    out_path = os.path.join(BASE_DIR, "retail_comparison_results.json")
    export = {
        "schema": "Retail Analytics",
        "device": DEVICE,
        "targets": RETAIL_TARGETS,
        "results": {
            m: [
                {
                    "target": r[0],
                    "predicted": r[1],
                    "expected": r[2],
                    "match": r[3],
                    "transform": r[4],
                }
                for r in all_results[m]
            ]
            for m in models
        },
        "summary": {m: {"correct": col_correct[m], "total": len(RETAIL_TARGETS), "accuracy": col_correct[m] / len(RETAIL_TARGETS) * 100} for m in models},
    }
    with open(out_path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"Results saved to: {out_path}\n")


if __name__ == "__main__":
    main()
