"""
Debug script: Compare what the pipeline sends for app examples vs test scenarios.
Run: python debug_app_examples.py

This helps identify why Load 5-Table and Retail Analytics Demo give wrong answers
while candidate_selection_test_results.json shows composite columns working.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_bridge import json_to_tables, discover_joins
from candidate_generation_v3 import serialize_source_schema, serialize_join_edges, build_mapping_prompt

# ── Retail Demo (from templates/source_ingestion.html EXAMPLE_RETAIL_DEMO) ──
RETAIL_SOURCE = {
    "tables": [
        {"name": "stores", "columns": [
            {"name": "store_id", "type": "int", "nullable": False, "is_pk": True, "is_fk": False},
            {"name": "store_name", "type": "string", "nullable": False, "is_pk": False, "is_fk": False},
            {"name": "city", "type": "string", "nullable": True, "is_pk": False, "is_fk": False},
            {"name": "region_id", "type": "int", "nullable": True, "is_pk": False, "is_fk": True},
        ], "sample_data": [
            {"store_id": 1, "store_name": "Downtown Flagship", "city": "New York", "region_id": 100},
            {"store_id": 2, "store_name": "Westside Mall", "city": "Los Angeles", "region_id": 200},
        ]},
        {"name": "regions", "columns": [
            {"name": "region_id", "type": "int", "nullable": False, "is_pk": True, "is_fk": False},
            {"name": "region_code", "type": "string", "nullable": False, "is_pk": False, "is_fk": False},
            {"name": "region_name", "type": "string", "nullable": True, "is_pk": False, "is_fk": False},
        ], "sample_data": [
            {"region_id": 100, "region_code": "NE", "region_name": "Northeast"},
            {"region_id": 200, "region_code": "SW", "region_name": "Southwest"},
        ]},
        {"name": "products", "columns": [
            {"name": "product_id", "type": "int", "nullable": False, "is_pk": True, "is_fk": False},
            {"name": "product_name", "type": "string", "nullable": False, "is_pk": False, "is_fk": False},
            {"name": "category_id", "type": "int", "nullable": True, "is_pk": False, "is_fk": True},
            {"name": "unit_cost", "type": "decimal", "nullable": True, "is_pk": False, "is_fk": False},
        ], "sample_data": [
            {"product_id": 501, "product_name": "Wireless Headphones", "category_id": 1, "unit_cost": 49.99},
            {"product_id": 502, "product_name": "USB-C Hub", "category_id": 2, "unit_cost": 29.99},
        ]},
        {"name": "categories", "columns": [
            {"name": "category_id", "type": "int", "nullable": False, "is_pk": True, "is_fk": False},
            {"name": "category_code", "type": "string", "nullable": False, "is_pk": False, "is_fk": False},
            {"name": "category_name", "type": "string", "nullable": True, "is_pk": False, "is_fk": False},
        ], "sample_data": [
            {"category_id": 1, "category_code": "AUD", "category_name": "Audio"},
            {"category_id": 2, "category_code": "ACC", "category_name": "Accessories"},
        ]},
        {"name": "customers", "columns": [
            {"name": "customer_id", "type": "int", "nullable": False, "is_pk": True, "is_fk": False},
            {"name": "first_name", "type": "string", "nullable": False, "is_pk": False, "is_fk": False},
            {"name": "last_name", "type": "string", "nullable": False, "is_pk": False, "is_fk": False},
            {"name": "email", "type": "string", "nullable": True, "is_pk": False, "is_fk": False},
            {"name": "signup_date", "type": "date", "nullable": True, "is_pk": False, "is_fk": False},
        ], "sample_data": [
            {"customer_id": 1001, "first_name": "Emma", "last_name": "Wilson", "email": "emma@example.com", "signup_date": "2023-01-15"},
            {"customer_id": 1002, "first_name": "James", "last_name": "Taylor", "email": "james@example.com", "signup_date": "2023-06-20"},
        ]},
        {"name": "orders", "columns": [
            {"name": "order_id", "type": "int", "nullable": False, "is_pk": True, "is_fk": False},
            {"name": "customer_id", "type": "int", "nullable": False, "is_pk": False, "is_fk": True},
            {"name": "store_id", "type": "int", "nullable": False, "is_pk": False, "is_fk": True},
            {"name": "order_date", "type": "date", "nullable": False, "is_pk": False, "is_fk": False},
            {"name": "status_code", "type": "string", "nullable": True, "is_pk": False, "is_fk": False},
        ], "sample_data": [
            {"order_id": 9001, "customer_id": 1001, "store_id": 1, "order_date": "2024-02-10", "status_code": "SHIPPED"},
            {"order_id": 9002, "customer_id": 1002, "store_id": 2, "order_date": "2024-02-15", "status_code": "PENDING"},
        ]},
        {"name": "order_lines", "columns": [
            {"name": "line_id", "type": "int", "nullable": False, "is_pk": True, "is_fk": False},
            {"name": "order_id", "type": "int", "nullable": False, "is_pk": False, "is_fk": True},
            {"name": "product_id", "type": "int", "nullable": False, "is_pk": False, "is_fk": True},
            {"name": "quantity", "type": "int", "nullable": False, "is_pk": False, "is_fk": False},
            {"name": "unit_price", "type": "decimal", "nullable": False, "is_pk": False, "is_fk": False},
        ], "sample_data": [
            {"line_id": 1, "order_id": 9001, "product_id": 501, "quantity": 2, "unit_price": 59.99},
            {"line_id": 2, "order_id": 9001, "product_id": 502, "quantity": 1, "unit_price": 34.99},
        ]},
    ]
}

RETAIL_TARGET_SAMPLE = [
    ("fact_sales", "line_total", "decimal", "quantity multiplied by unit price for the line", ["order_lines.quantity", "order_lines.unit_price"]),
    ("dim_customer", "customer_name", "string", "full name combining first and last name", ["customers.first_name", "customers.last_name"]),
    ("fact_sales", "customer_name", "string", "name of customer who placed the order", ["customers.first_name", "customers.last_name"]),
]


def main():
    print("=" * 80)
    print("  DEBUG: App Example Pipeline vs Test")
    print("=" * 80)

    # 1. Retail Demo
    print("\n--- RETAIL DEMO ---")
    tables = json_to_tables(RETAIL_SOURCE["tables"])
    edges, elapsed = discover_joins(tables)

    schema_text = serialize_source_schema(tables)
    joins_text = serialize_join_edges(edges)

    print(f"\nDiscovered {len(edges)} joins in {elapsed:.2f}s")
    print("\nSchema (as seen by LLM):")
    print(schema_text)
    print("\nJoins (as seen by LLM):")
    print(joins_text)

    # Check if order_lines join exists (critical for line_total)
    order_lines_join = any("order_lines" in str(e) for e in edges)
    print(f"\n[CHECK] order_lines join present: {order_lines_join} (required for fact_sales.line_total)")

    # Expected joins for Retail
    expected_joins = [
        "stores.region_id = regions.region_id",
        "products.category_id = categories.category_id",
        "orders.customer_id = customers.customer_id",
        "orders.store_id = stores.store_id",
        "order_lines.order_id = orders.order_id",
        "order_lines.product_id = products.product_id",
    ]
    found_joins = set(joins_text.replace("  ", "").split("\n"))
    for ej in expected_joins:
        found = any(ej in j for j in found_joins)
        print(f"  {ej}: {'OK' if found else 'MISSING'}")

    # Sample prompt for line_total
    print("\n--- Sample prompt for fact_sales.line_total ---")
    msgs = build_mapping_prompt(tables, edges, "fact_sales", "line_total", "decimal",
                                "quantity multiplied by unit price for the line")
    user_content = msgs[1]["content"]
    print(user_content[-1200:] if len(user_content) > 1200 else user_content)  # last part has target

    print("\n" + "=" * 80)
    print("  Model: pipeline uses schema_matcher_llm1, test results used schema_matcher_llm")
    print("  If joins are correct but answers wrong, try schema_matcher_llm in pipeline.")
    print("=" * 80)


if __name__ == "__main__":
    main()
