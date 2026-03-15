"""
Test: schema_matcher_llm1 on Retail Analytics Demo
==================================================
Runs the Retail Analytics source + target (same as app) through schema_matcher_llm1
and checks if predicted source columns match expected mappings.

Run: python test_retail_schema_matcher_llm1.py

Note: Uses CPU by default; takes several minutes for 18 target columns.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_bridge import json_to_tables, discover_joins
from candidate_generation_v3 import SchemaMatcherLLM

# Reuse Retail source from test_retail_joins
from test_retail_joins import RETAIL_SOURCE_TABLES

# Retail target (matches app templates/target_ingestion.html TARGET_EXAMPLE_RETAIL_DEMO)
RETAIL_TARGET_TABLES = [
    {
        "name": "dim_store",
        "columns": [
            {"name": "store_key", "type": "int", "description": "surrogate key for store"},
            {"name": "store_name", "type": "string", "description": "store display name"},
            {"name": "city", "type": "string", "description": "store city"},
            {"name": "region_label", "type": "string", "description": "human-readable region name from region lookup"},
        ],
    },
    {
        "name": "dim_product",
        "columns": [
            {"name": "product_key", "type": "int", "description": "surrogate key for product"},
            {"name": "product_name", "type": "string", "description": "product name"},
            {"name": "category_label", "type": "string", "description": "human-readable category from category lookup"},
            {"name": "unit_cost", "type": "decimal", "description": "product unit cost"},
        ],
    },
    {
        "name": "dim_customer",
        "columns": [
            {"name": "customer_key", "type": "int", "description": "surrogate key for customer"},
            {"name": "customer_name", "type": "string", "description": "full name combining first and last name"},
            {"name": "email", "type": "string", "description": "customer email"},
            {"name": "signup_year", "type": "int", "description": "year customer signed up"},
        ],
    },
    {
        "name": "fact_sales",
        "columns": [
            {"name": "order_key", "type": "int", "description": "order identifier"},
            {"name": "customer_name", "type": "string", "description": "name of customer who placed the order"},
            {"name": "store_name", "type": "string", "description": "store where order was placed"},
            {"name": "order_year", "type": "int", "description": "year the order was placed"},
            {"name": "line_total", "type": "decimal", "description": "quantity multiplied by unit price for the line"},
            {"name": "is_shipped", "type": "boolean", "description": "whether order status is shipped"},
        ],
    },
]

# Expected: (target_table, target_column) -> sorted list of expected source columns (lowercase)
EXPECTED_MAPPINGS = {
    ("dim_store", "store_key"): ["stores.store_id"],
    ("dim_store", "store_name"): ["stores.store_name"],
    ("dim_store", "city"): ["stores.city"],
    ("dim_store", "region_label"): ["regions.region_name"],
    ("dim_product", "product_key"): ["products.product_id"],
    ("dim_product", "product_name"): ["products.product_name"],
    ("dim_product", "category_label"): ["categories.category_name"],
    ("dim_product", "unit_cost"): ["products.unit_cost"],
    ("dim_customer", "customer_key"): ["customers.customer_id"],
    ("dim_customer", "customer_name"): ["customers.first_name", "customers.last_name"],
    ("dim_customer", "email"): ["customers.email"],
    ("dim_customer", "signup_year"): ["customers.signup_date"],
    ("fact_sales", "order_key"): ["orders.order_id"],
    ("fact_sales", "customer_name"): ["customers.first_name", "customers.last_name"],
    ("fact_sales", "store_name"): ["stores.store_name"],
    ("fact_sales", "order_year"): ["orders.order_date"],
    ("fact_sales", "line_total"): ["order_lines.quantity", "order_lines.unit_price"],
    ("fact_sales", "is_shipped"): ["orders.status_code"],
}


def _norm(cols):
    """Normalize column refs for comparison (lowercase, sorted)."""
    return sorted(c.strip().lower() for c in cols if c)


def run_test():
    print("=" * 70)
    print("  schema_matcher_llm1 on Retail Analytics Demo")
    print("  (Same source + target as app)")
    print("=" * 70)

    tables = json_to_tables(RETAIL_SOURCE_TABLES)
    edges, _ = discover_joins(tables)

    print("\n[1/2] Loading schema_matcher_llm1 (device=cpu)...")
    t0 = time.time()
    engine = SchemaMatcherLLM(
        source_tables=tables,
        join_edges=edges,
        model_dir="schema_matcher_llm1",
        device="cpu",
    )
    load_time = time.time() - t0
    print(f"      Loaded in {load_time:.1f}s")

    print("\n[2/2] Running mapping on 18 target columns...")
    t0 = time.time()
    result = engine.rank_all_targets(RETAIL_TARGET_TABLES, top_k=5)
    run_time = time.time() - t0
    print(f"      Done in {run_time:.1f}s ({run_time / 18:.2f}s per column)")

    # Compare results to expected
    print("\n--- Results: expected vs predicted ---")
    correct = 0
    total = 0
    for tbl in result["tables"]:
        tname = tbl["name"]
        for col in tbl["columns"]:
            cname = col["name"]
            predicted = col.get("final_source", [])
            expected = EXPECTED_MAPPINGS.get((tname, cname), [])

            pred_norm = _norm(predicted)
            exp_norm = _norm(expected)
            match = pred_norm == exp_norm
            if match:
                correct += 1
            total += 1

            status = "OK" if match else "FAIL"
            pred_str = ", ".join(predicted) if predicted else "(unmapped)"
            exp_str = ", ".join(expected) if expected else "(none)"
            print(f"  [{status}] {tname}.{cname}")
            print(f"         expected: {exp_str}")
            print(f"         predicted: {pred_str}")

    acc = correct / total * 100 if total else 0
    print("\n" + "=" * 70)
    print(f"  RESULT: {correct}/{total} correct ({acc:.1f}%)")
    if correct == total:
        print("  All mappings are correct.")
    else:
        print(f"  {total - correct} mapping(s) incorrect or unmapped.")
    print("=" * 70 + "\n")

    # Optionally save full results
    out_path = os.path.join(os.path.dirname(__file__), "retail_llm1_test_results.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "model": "schema_matcher_llm1",
                "schema": "Retail Analytics Demo",
                "correct": correct,
                "total": total,
                "accuracy": acc,
                "load_time_sec": load_time,
                "run_time_sec": run_time,
                "results": result,
            },
            f,
            indent=2,
        )
    print(f"Full results saved to: {out_path}\n")

    return correct == total


if __name__ == "__main__":
    ok = run_test()
    sys.exit(0 if ok else 1)
