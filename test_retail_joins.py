"""
Test: Retail Analytics Demo — Join Discovery
=============================================
Uses the same source schema as the app's "★ Retail Analytics Demo" (EXAMPLE_RETAIL_DEMO).
Runs join discovery and checks that all expected FK-based joins are found.

Run: python test_retail_joins.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_bridge import json_to_tables, discover_joins

# Retail Analytics source schema (matches app templates/source_ingestion.html EXAMPLE_RETAIL_DEMO)
RETAIL_SOURCE_TABLES = [
    {
        "name": "stores",
        "columns": [
            {"name": "store_id", "type": "int", "nullable": False, "is_pk": True, "is_fk": False},
            {"name": "store_name", "type": "string", "nullable": False, "is_pk": False, "is_fk": False},
            {"name": "city", "type": "string", "nullable": True, "is_pk": False, "is_fk": False},
            {"name": "region_id", "type": "int", "nullable": True, "is_pk": False, "is_fk": True},
        ],
        "sample_data": [
            {"store_id": 1, "store_name": "Downtown Flagship", "city": "New York", "region_id": 100},
            {"store_id": 2, "store_name": "Westside Mall", "city": "Los Angeles", "region_id": 200},
        ],
    },
    {
        "name": "regions",
        "columns": [
            {"name": "region_id", "type": "int", "nullable": False, "is_pk": True, "is_fk": False},
            {"name": "region_code", "type": "string", "nullable": False, "is_pk": False, "is_fk": False},
            {"name": "region_name", "type": "string", "nullable": True, "is_pk": False, "is_fk": False},
        ],
        "sample_data": [
            {"region_id": 100, "region_code": "NE", "region_name": "Northeast"},
            {"region_id": 200, "region_code": "SW", "region_name": "Southwest"},
        ],
    },
    {
        "name": "products",
        "columns": [
            {"name": "product_id", "type": "int", "nullable": False, "is_pk": True, "is_fk": False},
            {"name": "product_name", "type": "string", "nullable": False, "is_pk": False, "is_fk": False},
            {"name": "category_id", "type": "int", "nullable": True, "is_pk": False, "is_fk": True},
            {"name": "unit_cost", "type": "decimal", "nullable": True, "is_pk": False, "is_fk": False},
        ],
        "sample_data": [
            {"product_id": 501, "product_name": "Wireless Headphones", "category_id": 1, "unit_cost": 49.99},
            {"product_id": 502, "product_name": "USB-C Hub", "category_id": 2, "unit_cost": 29.99},
        ],
    },
    {
        "name": "categories",
        "columns": [
            {"name": "category_id", "type": "int", "nullable": False, "is_pk": True, "is_fk": False},
            {"name": "category_code", "type": "string", "nullable": False, "is_pk": False, "is_fk": False},
            {"name": "category_name", "type": "string", "nullable": True, "is_pk": False, "is_fk": False},
        ],
        "sample_data": [
            {"category_id": 1, "category_code": "AUD", "category_name": "Audio"},
            {"category_id": 2, "category_code": "ACC", "category_name": "Accessories"},
        ],
    },
    {
        "name": "customers",
        "columns": [
            {"name": "customer_id", "type": "int", "nullable": False, "is_pk": True, "is_fk": False},
            {"name": "first_name", "type": "string", "nullable": False, "is_pk": False, "is_fk": False},
            {"name": "last_name", "type": "string", "nullable": False, "is_pk": False, "is_fk": False},
            {"name": "email", "type": "string", "nullable": True, "is_pk": False, "is_fk": False},
            {"name": "signup_date", "type": "date", "nullable": True, "is_pk": False, "is_fk": False},
        ],
        "sample_data": [
            {"customer_id": 1001, "first_name": "Emma", "last_name": "Wilson", "email": "emma@example.com", "signup_date": "2023-01-15"},
            {"customer_id": 1002, "first_name": "James", "last_name": "Taylor", "email": "james@example.com", "signup_date": "2023-06-20"},
        ],
    },
    {
        "name": "orders",
        "columns": [
            {"name": "order_id", "type": "int", "nullable": False, "is_pk": True, "is_fk": False},
            {"name": "customer_id", "type": "int", "nullable": False, "is_pk": False, "is_fk": True},
            {"name": "store_id", "type": "int", "nullable": False, "is_pk": False, "is_fk": True},
            {"name": "order_date", "type": "date", "nullable": False, "is_pk": False, "is_fk": False},
            {"name": "status_code", "type": "string", "nullable": True, "is_pk": False, "is_fk": False},
        ],
        "sample_data": [
            {"order_id": 9001, "customer_id": 1001, "store_id": 1, "order_date": "2024-02-10", "status_code": "SHIPPED"},
            {"order_id": 9002, "customer_id": 1002, "store_id": 2, "order_date": "2024-02-15", "status_code": "PENDING"},
        ],
    },
    {
        "name": "order_lines",
        "columns": [
            {"name": "line_id", "type": "int", "nullable": False, "is_pk": True, "is_fk": False},
            {"name": "order_id", "type": "int", "nullable": False, "is_pk": False, "is_fk": True},
            {"name": "product_id", "type": "int", "nullable": False, "is_pk": False, "is_fk": True},
            {"name": "quantity", "type": "int", "nullable": False, "is_pk": False, "is_fk": False},
            {"name": "unit_price", "type": "decimal", "nullable": False, "is_pk": False, "is_fk": False},
        ],
        "sample_data": [
            {"line_id": 1, "order_id": 9001, "product_id": 501, "quantity": 2, "unit_price": 59.99},
            {"line_id": 2, "order_id": 9001, "product_id": 502, "quantity": 1, "unit_price": 34.99},
        ],
    },
]

# Expected joins for Retail Analytics (from FK relationships and schema design).
# Format: (left_table, left_col, right_table, right_col) — order normalized for comparison.
EXPECTED_JOINS = [
    ("stores", "region_id", "regions", "region_id"),
    ("products", "category_id", "categories", "category_id"),
    ("orders", "customer_id", "customers", "customer_id"),
    ("orders", "store_id", "stores", "store_id"),
    ("order_lines", "order_id", "orders", "order_id"),
    ("order_lines", "product_id", "products", "product_id"),
]


def _normalize_join_edge(lt, lc, rt, rc) -> tuple:
    """Normalize to (table_a, col_a, table_b, col_b) with (table_a, col_a) <= (table_b, col_b) for comparison."""
    if (lt, lc) <= (rt, rc):
        return (lt, lc, rt, rc)
    return (rt, rc, lt, lc)


def _normalize_join(e) -> tuple:
    """Normalize JoinEdge to same tuple form."""
    return _normalize_join_edge(e.left_table, e.left_cols[0], e.right_table, e.right_cols[0])


def run_test():
    print("=" * 70)
    print("  Retail Analytics Demo — Join Discovery Test")
    print("  (Same source as app: Retail Analytics Demo)")
    print("=" * 70)

    tables = json_to_tables(RETAIL_SOURCE_TABLES)
    edges, elapsed = discover_joins(tables)

    # Normalize discovered joins (single-column joins only for expected comparison)
    discovered_set = set()
    for e in edges:
        if len(e.left_cols) == 1 and len(e.right_cols) == 1:
            discovered_set.add(_normalize_join(e))

    expected_set = {_normalize_join_edge(lt, lc, rt, rc) for (lt, lc, rt, rc) in EXPECTED_JOINS}

    # Report: all discovered joins
    print(f"\nDiscovered {len(edges)} join(s) in {elapsed:.2f}s\n")
    print("--- Discovered joins (table.col = table.col, confidence) ---")
    for e in sorted(edges, key=lambda x: (-x.confidence, x.left_table, x.right_table)):
        for lc, rc in zip(e.left_cols, e.right_cols):
            print(f"  {e.left_table}.{lc} = {e.right_table}.{rc}  conf={e.confidence:.3f}")

    # Check expected
    print("\n--- Expected vs discovered ---")
    all_ok = True
    for lt, lc, rt, rc in EXPECTED_JOINS:
        key = _normalize_join_edge(lt, lc, rt, rc)
        found = key in discovered_set
        status = "OK" if found else "MISSING"
        if not found:
            all_ok = False
        print(f"  {lt}.{lc} = {rt}.{rc}  -> {status}")

    # Extra joins (discovered but not in expected list — may be valid alternatives)
    extra = discovered_set - expected_set
    if extra:
        print("\n--- Extra joins (discovered, not in expected list) ---")
        for t in sorted(extra):
            print(f"  {t[0]}.{t[1]} = {t[2]}.{t[3]}")

    print("\n" + "=" * 70)
    if all_ok:
        print("  RESULT: All expected joins were discovered. Joins are correct.")
    else:
        print("  RESULT: Some expected joins were MISSING. Check join discovery / sample data.")
    print("=" * 70 + "\n")
    return all_ok


if __name__ == "__main__":
    ok = run_test()
    sys.exit(0 if ok else 1)
