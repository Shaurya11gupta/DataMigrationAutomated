#!/usr/bin/env python3
"""
Combined Pipeline Validation  (Stage A + Stage B)
==================================================

End-to-end demonstration:

  Stage A  CandidateGenerationEngine  (rule-based, heuristic)
     |        generates & scores candidates
     v
  Bridge    converts CandidateSet -> text pairs
     |
     v
  Stage B  CandidateSelectorStage1   (ML: bi-encoder + cross-encoder)
              reranks candidates with learned scores

Usage:
  python validate_combined_pipeline.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
from constraint_similarity_engine import ColumnConstraints
from join_graph_builder_v2 import Column, ColumnType, JoinEdge, Table
from value_similarity_engine import ColumnStats

from candidate_generation_algorithm import CandidateGenerationEngine, TargetSpec
from candidate_selector_stage1 import (
    CandidateSelectorStage1,
    CandidateSet as SelectorCandidateSet,
    SourceColumn,
    TargetColumn,
    _parse_candidate,
    _parse_target,
    target_to_text,
    candidate_to_text,
)


# ===================================================================
# Helper
# ===================================================================

def make_col(name: str, rows: list, typ: str, constraints=None) -> Column:
    values = [r.get(name) for r in rows]
    return Column(
        name=name,
        col_type=ColumnType(typ),
        constraints=constraints or ColumnConstraints(),
        stats=ColumnStats(values),
    )


# ===================================================================
# Realistic multi-table schema
# ===================================================================

def build_test_schema():
    customers_rows = [
        {"customer_id": 1, "first_name": "Alice", "last_name": "Smith",
         "email": "alice.smith@acme.com", "country_code": "US",
         "city": "Boston", "state_code": "MA", "postal_code": "02108",
         "phone_number": "6175551234", "signup_date": "2021-01-15",
         "risk_score": 82.3, "status_code": "ACTIVE"},
        {"customer_id": 2, "first_name": "Bob", "last_name": "Jones",
         "email": "bob.jones@acme.com", "country_code": "IN",
         "city": "Bengaluru", "state_code": "KA", "postal_code": "560001",
         "phone_number": "8044412345", "signup_date": "2020-07-02",
         "risk_score": 45.1, "status_code": "ACTIVE"},
        {"customer_id": 3, "first_name": "Carol", "last_name": "Brown",
         "email": "carol.brown@acme.com", "country_code": "UK",
         "city": "London", "state_code": "LN", "postal_code": "EC1A1BB",
         "phone_number": "2071234567", "signup_date": "2019-03-21",
         "risk_score": 67.7, "status_code": "INACTIVE"},
        {"customer_id": 4, "first_name": "David", "last_name": "Miller",
         "email": "david.miller@acme.com", "country_code": "US",
         "city": "Seattle", "state_code": "WA", "postal_code": "98101",
         "phone_number": "4253339988", "signup_date": "2022-10-18",
         "risk_score": 91.4, "status_code": "ACTIVE"},
        {"customer_id": 5, "first_name": "Emma", "last_name": "Wilson",
         "email": "emma.wilson@acme.com", "country_code": "IN",
         "city": "Pune", "state_code": "MH", "postal_code": "411001",
         "phone_number": "2044008877", "signup_date": "2023-02-01",
         "risk_score": 38.9, "status_code": "ACTIVE"},
        {"customer_id": 6, "first_name": "Frank", "last_name": "Taylor",
         "email": "frank.taylor@acme.com", "country_code": "UK",
         "city": "Manchester", "state_code": "MN", "postal_code": "M11AE",
         "phone_number": "1610098765", "signup_date": "2021-08-29",
         "risk_score": 72.0, "status_code": "INACTIVE"},
    ]

    orders_rows = [
        {"order_id": 1001, "customer_id": 1, "amount_local": 120.50,
         "currency_code": "USD", "event_date": "2024-01-05",
         "service_code": "SVC_A", "quantity": 2, "discount_amount": 10.5,
         "tax_amount": 5.2},
        {"order_id": 1002, "customer_id": 2, "amount_local": 9500.00,
         "currency_code": "INR", "event_date": "2024-01-08",
         "service_code": "SVC_B", "quantity": 4, "discount_amount": 150.0,
         "tax_amount": 180.0},
        {"order_id": 1003, "customer_id": 3, "amount_local": 88.30,
         "currency_code": "GBP", "event_date": "2024-01-11",
         "service_code": "SVC_A", "quantity": 1, "discount_amount": 4.0,
         "tax_amount": 3.5},
        {"order_id": 1004, "customer_id": 4, "amount_local": 230.00,
         "currency_code": "USD", "event_date": "2024-01-13",
         "service_code": "SVC_C", "quantity": 5, "discount_amount": 15.0,
         "tax_amount": 9.3},
        {"order_id": 1005, "customer_id": 5, "amount_local": 4100.00,
         "currency_code": "INR", "event_date": "2024-01-18",
         "service_code": "SVC_B", "quantity": 8, "discount_amount": 125.0,
         "tax_amount": 80.0},
        {"order_id": 1006, "customer_id": 6, "amount_local": 67.20,
         "currency_code": "GBP", "event_date": "2024-01-21",
         "service_code": "SVC_C", "quantity": 2, "discount_amount": 2.2,
         "tax_amount": 2.0},
    ]

    country_rows = [
        {"country_code": "US", "country_name": "United States", "fx_rate": 1.0},
        {"country_code": "IN", "country_name": "India", "fx_rate": 0.012},
        {"country_code": "UK", "country_name": "United Kingdom", "fx_rate": 1.27},
    ]

    service_rows = [
        {"service_code": "SVC_A", "service_category": "Consulting"},
        {"service_code": "SVC_B", "service_category": "Support"},
        {"service_code": "SVC_C", "service_category": "Platform"},
    ]

    documents_rows = [
        {"doc_id": 501, "customer_id": 1, "file_path": "/docs/us/alice_profile.pdf",
         "created_at_text": "2024-01-05 10:11:00"},
        {"doc_id": 502, "customer_id": 2, "file_path": "/docs/in/bob_statement.csv",
         "created_at_text": "2024-01-08 09:00:12"},
        {"doc_id": 503, "customer_id": 3, "file_path": "/docs/uk/carol_report.xlsx",
         "created_at_text": "2024-01-11 15:31:22"},
        {"doc_id": 504, "customer_id": 4, "file_path": "/docs/us/david_id.png",
         "created_at_text": "2024-01-13 07:22:44"},
        {"doc_id": 505, "customer_id": 5, "file_path": "/docs/in/emma_contract.docx",
         "created_at_text": "2024-01-18 14:07:09"},
        {"doc_id": 506, "customer_id": 6, "file_path": "/docs/uk/frank_note.txt",
         "created_at_text": "2024-01-21 19:20:02"},
    ]

    customers = Table(
        name="src_customers",
        columns={
            "customer_id": make_col("customer_id", customers_rows, "int",
                                    ColumnConstraints(nullable=False, is_primary_key=True, is_unique=True)),
            "first_name": make_col("first_name", customers_rows, "string"),
            "last_name": make_col("last_name", customers_rows, "string"),
            "email": make_col("email", customers_rows, "string",
                              ColumnConstraints(is_unique=True)),
            "country_code": make_col("country_code", customers_rows, "string"),
            "city": make_col("city", customers_rows, "string"),
            "state_code": make_col("state_code", customers_rows, "string"),
            "postal_code": make_col("postal_code", customers_rows, "string"),
            "phone_number": make_col("phone_number", customers_rows, "string"),
            "signup_date": make_col("signup_date", customers_rows, "date"),
            "risk_score": make_col("risk_score", customers_rows, "decimal"),
            "status_code": make_col("status_code", customers_rows, "string"),
        },
        row_count=len(customers_rows), rows=customers_rows,
    )
    orders = Table(
        name="src_orders",
        columns={
            "order_id": make_col("order_id", orders_rows, "int",
                                 ColumnConstraints(nullable=False, is_primary_key=True, is_unique=True)),
            "customer_id": make_col("customer_id", orders_rows, "int",
                                    ColumnConstraints(is_foreign_key=True)),
            "amount_local": make_col("amount_local", orders_rows, "decimal"),
            "currency_code": make_col("currency_code", orders_rows, "string"),
            "event_date": make_col("event_date", orders_rows, "date"),
            "service_code": make_col("service_code", orders_rows, "string"),
            "quantity": make_col("quantity", orders_rows, "int"),
            "discount_amount": make_col("discount_amount", orders_rows, "decimal"),
            "tax_amount": make_col("tax_amount", orders_rows, "decimal"),
        },
        row_count=len(orders_rows), rows=orders_rows,
    )
    dim_country = Table(
        name="dim_country",
        columns={
            "country_code": make_col("country_code", country_rows, "string",
                                     ColumnConstraints(nullable=False, is_primary_key=True, is_unique=True)),
            "country_name": make_col("country_name", country_rows, "string"),
            "fx_rate": make_col("fx_rate", country_rows, "decimal"),
        },
        row_count=len(country_rows), rows=country_rows,
    )
    dim_service = Table(
        name="dim_service",
        columns={
            "service_code": make_col("service_code", service_rows, "string",
                                     ColumnConstraints(nullable=False, is_primary_key=True, is_unique=True)),
            "service_category": make_col("service_category", service_rows, "string"),
        },
        row_count=len(service_rows), rows=service_rows,
    )
    src_documents = Table(
        name="src_documents",
        columns={
            "doc_id": make_col("doc_id", documents_rows, "int",
                               ColumnConstraints(nullable=False, is_primary_key=True, is_unique=True)),
            "customer_id": make_col("customer_id", documents_rows, "int",
                                    ColumnConstraints(is_foreign_key=True)),
            "file_path": make_col("file_path", documents_rows, "string"),
            "created_at_text": make_col("created_at_text", documents_rows, "string"),
        },
        row_count=len(documents_rows), rows=documents_rows,
    )

    source_tables = {
        "src_customers": customers, "src_orders": orders,
        "dim_country": dim_country, "dim_service": dim_service,
        "src_documents": src_documents,
    }
    join_edges = [
        JoinEdge("src_customers", "src_orders", ["customer_id"], ["customer_id"],
                 "1:N", 0.96, ["pk_fk"], {"name": 1.0, "value": 1.0}, "left_parent"),
        JoinEdge("src_customers", "dim_country", ["country_code"], ["country_code"],
                 "N:1", 0.93, ["lookup"], {"name": 0.93, "value": 1.0}, "right_parent"),
        JoinEdge("src_orders", "dim_service", ["service_code"], ["service_code"],
                 "N:1", 0.91, ["lookup"], {"name": 0.88, "value": 1.0}, "right_parent"),
        JoinEdge("src_customers", "src_documents", ["customer_id"], ["customer_id"],
                 "1:N", 0.92, ["pk_fk"], {"name": 0.95, "value": 1.0}, "left_parent"),
    ]
    return source_tables, join_edges, customers_rows, orders_rows, country_rows, service_rows, documents_rows


# ===================================================================
# Target specs  (ground truth in comments)
# ===================================================================

def build_targets(cust, orders, countries, services, docs):
    targets = []

    # 1. CONCAT
    targets.append({
        "title": "CONCAT: first_name + last_name -> full_name",
        "expected": "src_customers.first_name, src_customers.last_name",
        "expected_family": "concat",
        "spec": TargetSpec(
            table="dim_customer", name="full_name", col_type=ColumnType("string"),
            constraints=ColumnConstraints(),
            stats=ColumnStats([f"{r['first_name']} {r['last_name']}" for r in cust]),
            description="Customer full name",
            sample_values=[f"{r['first_name']} {r['last_name']}" for r in cust],
        ),
    })
    # 2. EMAIL EXTRACT
    targets.append({
        "title": "EMAIL EXTRACT: email -> email_username",
        "expected": "src_customers.email",
        "expected_family": "email_username_extract",
        "spec": TargetSpec(
            table="dim_customer", name="email_username", col_type=ColumnType("string"),
            constraints=ColumnConstraints(),
            stats=ColumnStats([r["email"].split("@")[0] for r in cust]),
            description="Local part from email",
            sample_values=[r["email"].split("@")[0] for r in cust],
        ),
    })
    # 3. LOOKUP
    cc_map = {c["country_code"]: c["country_name"] for c in countries}
    targets.append({
        "title": "LOOKUP: country_code -> country_name",
        "expected": "dim_country.country_name",
        "expected_family": "identity",
        "spec": TargetSpec(
            table="dim_customer", name="country_name", col_type=ColumnType("string"),
            constraints=ColumnConstraints(),
            stats=ColumnStats([cc_map[r["country_code"]] for r in cust]),
            description="Country full name lookup",
        ),
    })
    # 4. ARITHMETIC
    targets.append({
        "title": "ARITHMETIC: amount + tax -> gross_total",
        "expected": "src_orders.amount_local, src_orders.tax_amount",
        "expected_family": "arithmetic",
        "spec": TargetSpec(
            table="fct_order", name="gross_total", col_type=ColumnType("decimal"),
            constraints=ColumnConstraints(),
            stats=ColumnStats([round(r["amount_local"] + r["tax_amount"], 4) for r in orders]),
            description="Tax-inclusive total",
        ),
    })
    # 5. CONDITIONAL
    targets.append({
        "title": "CONDITIONAL: risk_score -> is_high_risk",
        "expected": "src_customers.risk_score",
        "expected_family": "conditional",
        "spec": TargetSpec(
            table="dim_customer", name="is_high_risk", col_type=ColumnType("boolean"),
            constraints=ColumnConstraints(),
            stats=ColumnStats([r["risk_score"] >= 70.0 for r in cust]),
            description="High risk threshold flag",
            sample_values=[r["risk_score"] >= 70.0 for r in cust],
        ),
    })
    # 6. DATE PART
    targets.append({
        "title": "DATE PART: signup_date -> signup_year",
        "expected": "src_customers.signup_date",
        "expected_family": "date_part",
        "spec": TargetSpec(
            table="dim_customer", name="signup_year", col_type=ColumnType("int"),
            constraints=ColumnConstraints(),
            stats=ColumnStats([int(r["signup_date"][:4]) for r in cust]),
            description="Year extracted from signup date",
            sample_values=[int(r["signup_date"][:4]) for r in cust],
        ),
    })
    # 7. PARSE DATE
    targets.append({
        "title": "PARSE DATE: created_at_text -> created_date",
        "expected": "src_documents.created_at_text",
        "expected_family": "parse_date",
        "spec": TargetSpec(
            table="dim_document", name="created_date", col_type=ColumnType("date"),
            constraints=ColumnConstraints(),
            stats=ColumnStats([r["created_at_text"][:10] for r in docs]),
            description="Parse timestamp text to date",
            sample_values=[r["created_at_text"][:10] for r in docs],
        ),
    })
    return targets


# ===================================================================
# Stage A -> Stage B bridge
# ===================================================================

def bridge_stage_a_to_b(target_spec: TargetSpec, stage_a_result: dict):
    """
    Convert Stage A CandidateSet output into Stage B dataclasses
    for the CandidateSelectorStage1.rank() API.
    """
    target_col = TargetColumn(
        table=target_spec.table,
        column=target_spec.name,
        type=getattr(target_spec.col_type, "base_type", "string"),
        description=target_spec.description or "",
    )

    candidate_sets: List[SelectorCandidateSet] = []
    for i, cand in enumerate(stage_a_result.get("top_candidates", [])):
        columns: List[SourceColumn] = []
        for col_ref in cand.get("candidate_columns", []):
            parts = col_ref.split(".", 1)
            if len(parts) == 2:
                columns.append(SourceColumn(table=parts[0], column=parts[1], type="string"))
        candidate_sets.append(SelectorCandidateSet(
            id=f"cand_{i:03d}",
            columns=columns,
            join_path=cand.get("join_path", []),
            transform_hint=cand.get("best_transform_family", ""),
        ))
    return target_col, candidate_sets


# ===================================================================
# Pretty printers
# ===================================================================

SEP = "=" * 100
SUBSEP = "-" * 100


def hdr(text):
    print(f"\n{SEP}\n  {text}\n{SEP}")


def bar(score, width=30):
    filled = int(score * width)
    return "#" * filled + "." * (width - filled)


# ===================================================================
# Load Stage B models
# ===================================================================

def load_stage_b(artifacts_root: Path) -> Optional[CandidateSelectorStage1]:
    """
    Try to load from local artifact dirs first.
    If weight files are missing, fall back to the base pretrained models
    from the same architecture that was used for training.
    """
    import json as _json

    meta_path = artifacts_root / "training_metadata.json"
    bi_dir = artifacts_root / "biencoder"
    cross_dir = artifacts_root / "cross_encoder"

    # Read training metadata for base model names
    bi_base = "sentence-transformers/all-MiniLM-L6-v2"
    cross_base = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    if meta_path.exists():
        meta = _json.loads(meta_path.read_text(encoding="utf-8"))
        bi_base = meta.get("bi_model_name", bi_base)
        cross_base = meta.get("cross_model_name", cross_base)

    # Check if local weight files exist
    bi_has_weights = any((bi_dir / f).exists() for f in
                         ["model.safetensors", "pytorch_model.bin"])
    cross_has_weights = any((cross_dir / f).exists() for f in
                            ["model.safetensors", "pytorch_model.bin"])

    if bi_has_weights and cross_has_weights:
        print(f"  Loading FINE-TUNED models from {artifacts_root}")
        bi_path = str(bi_dir)
        cross_path = str(cross_dir)
    else:
        missing = []
        if not bi_has_weights:
            missing.append("bi-encoder")
        if not cross_has_weights:
            missing.append("cross-encoder")
        print(f"  NOTE: Weight files missing for {', '.join(missing)}.")
        print(f"  Loading BASE pretrained models: {bi_base}, {cross_base}")
        print(f"  (Pipeline will work but with pretrained weights, not fine-tuned.)")
        print(f"  To get fine-tuned weights, retrain with:")
        print(f"    python candidate_selector_stage1.py train \\")
        print(f"      --input-jsonl stage1_training_input_sample.jsonl")
        bi_path = bi_base
        cross_path = cross_base

    print(f"\n  Loading bi-encoder:    {bi_path}")
    print(f"  Loading cross-encoder: {cross_path}")

    selector = CandidateSelectorStage1(
        biencoder_path=bi_path,
        cross_encoder_path=cross_path,
    )
    return selector


# ===================================================================
# Main
# ===================================================================

def main():
    hdr("FULL END-TO-END PIPELINE VALIDATION  (Stage A + Stage B)")

    # ------------------------------------------------------------------
    # 1. Build schema
    # ------------------------------------------------------------------
    hdr("STEP 1  |  Build Source Schema")
    tables, edges, cust, ords, countries, svcs, docs = build_test_schema()
    print(f"  Tables: {len(tables)}")
    for tn, t in tables.items():
        print(f"    {tn}: {len(t.columns)} columns, {t.row_count} rows")
    print(f"  Join edges: {len(edges)}")
    for e in edges:
        print(f"    {e.left_table}.{e.left_cols} <-> {e.right_table}.{e.right_cols} "
              f"[{e.cardinality}, conf={e.confidence}]")

    # ------------------------------------------------------------------
    # 2. Init Stage A
    # ------------------------------------------------------------------
    hdr("STEP 2  |  Initialize Stage A (CandidateGenerationEngine)")
    t0 = time.time()
    engine = CandidateGenerationEngine(source_tables=tables, join_edges=edges)
    print(f"  Initialized in {time.time() - t0:.3f}s")
    print(f"  Indexed {sum(len(v) for v in engine._table_to_ids.values())} columns "
          f"across buckets: {list(engine._index_by_bucket.keys())}")

    # ------------------------------------------------------------------
    # 3. Init Stage B
    # ------------------------------------------------------------------
    hdr("STEP 3  |  Initialize Stage B (CandidateSelectorStage1)")
    artifacts = Path("artifacts/stage1_candidate_selector")
    selector = load_stage_b(artifacts)
    print(f"  Stage B ready.")

    # ------------------------------------------------------------------
    # 4. Build targets
    # ------------------------------------------------------------------
    hdr("STEP 4  |  Define Target Columns")
    targets = build_targets(cust, ords, countries, svcs, docs)
    for i, t in enumerate(targets, 1):
        print(f"  {i}. {t['title']}")
        print(f"     Expected: {t['expected']} [{t['expected_family']}]")

    # ------------------------------------------------------------------
    # 5. Run full pipeline per target
    # ------------------------------------------------------------------
    hdr("STEP 5  |  Run Full Pipeline (Stage A -> Bridge -> Stage B)")
    print(f"  Stage A: Coarse -> Fine -> Composite -> Feasibility -> QuickScore -> Rank")
    print(f"  Bridge:  CandidateSet -> (target_text, candidate_text) pairs")
    print(f"  Stage B: Bi-encoder retrieval -> Cross-encoder reranking")
    print()

    results = []
    for tgt_info in targets:
        title = tgt_info["title"]
        expected = tgt_info["expected"]
        expected_family = tgt_info["expected_family"]
        target = tgt_info["spec"]

        print(f"\n{SUBSEP}")
        print(f"  TARGET: {title}")
        print(f"  Expected answer: {expected} [{expected_family}]")
        print(SUBSEP)

        # --- Stage A ---
        t0 = time.time()
        stage_a = engine.rank_candidates(
            target=target, coarse_top_m=50, fine_top_m=25,
            max_arity=4, max_hops=3, top_k=8, abstain_threshold=0.58,
        )
        stage_a_time = time.time() - t0

        print(f"\n  STAGE A results ({stage_a_time:.2f}s)")
        print(f"    Coarse: {stage_a['debug']['coarse_count']} | "
              f"Fine: {stage_a['debug']['fine_count']} | "
              f"Total combos: {stage_a['debug']['candidate_count_total']}")
        print(f"    Abstain: {stage_a['abstain']}")

        for i, c in enumerate(stage_a["top_candidates"][:6], 1):
            cols = ", ".join(c["candidate_columns"])
            print(f"    {i}. [{c['confidence']:.3f}] {bar(c['confidence'])} "
                  f"{cols} | {c['best_transform_family']}")

        # --- Bridge ---
        target_col, candidate_sets = bridge_stage_a_to_b(target, stage_a)

        print(f"\n  BRIDGE -> {len(candidate_sets)} candidates serialized for Stage B")
        print(f"    Target text: {target_to_text(target_col)}")
        for cs in candidate_sets[:3]:
            print(f"    {cs.id}: {candidate_to_text(cs)}")

        # --- Stage B ---
        t1 = time.time()
        stage_b = selector.rank(
            target=target_col,
            candidate_sets=candidate_sets,
            retrieval_k=50,
            top_k=8,
        )
        stage_b_time = time.time() - t1

        print(f"\n  STAGE B results ({stage_b_time:.2f}s)")
        # Map candidate IDs back to column info
        cand_map = {cs.id: cs for cs in candidate_sets}
        for r in stage_b[:6]:
            cid = r["candidate_id"]
            cs = cand_map.get(cid)
            cols_str = ", ".join(f"{sc.table}.{sc.column}" for sc in cs.columns) if cs else "?"
            hint = cs.transform_hint if cs else "?"
            bi = r["bi_encoder_similarity"]
            ce = r["cross_encoder_score"]
            comb = r.get("combined_score", 0.0)
            print(f"    Rank {r['rank']:2d}. [bi={bi:+.4f} ce={ce:+.4f} comb={comb:.4f}] "
                  f"{cols_str} | {hint}")

        # --- Combined evaluation ---
        stage_a_top1 = ", ".join(stage_a["top_candidates"][0]["candidate_columns"]) if stage_a["top_candidates"] else "NONE"
        stage_a_family = stage_a["top_candidates"][0]["best_transform_family"] if stage_a["top_candidates"] else "NONE"
        if stage_b:
            b_cid = stage_b[0]["candidate_id"]
            b_cs = cand_map.get(b_cid)
            stage_b_top1 = ", ".join(f"{sc.table}.{sc.column}" for sc in b_cs.columns) if b_cs else "NONE"
            stage_b_family = b_cs.transform_hint if b_cs else "NONE"
        else:
            stage_b_top1 = "NONE"
            stage_b_family = "NONE"

        a_correct = expected in stage_a_top1 or stage_a_top1 in expected
        b_correct = expected in stage_b_top1 or stage_b_top1 in expected

        print(f"\n  COMPARISON:")
        print(f"    Expected:      {expected} [{expected_family}]")
        print(f"    Stage A Top-1: {stage_a_top1} [{stage_a_family}]  "
              f"{'CORRECT' if a_correct else 'WRONG'}")
        print(f"    Stage B Top-1: {stage_b_top1} [{stage_b_family}]  "
              f"{'CORRECT' if b_correct else 'WRONG'}")

        results.append({
            "title": title,
            "expected": expected,
            "stage_a_top1": stage_a_top1,
            "stage_a_family": stage_a_family,
            "stage_a_correct": a_correct,
            "stage_a_time": stage_a_time,
            "stage_b_top1": stage_b_top1,
            "stage_b_family": stage_b_family,
            "stage_b_correct": b_correct,
            "stage_b_time": stage_b_time,
            "stage_a_conf": stage_a["top_candidates"][0]["confidence"] if stage_a["top_candidates"] else 0,
            "stage_b_ce": stage_b[0]["cross_encoder_score"] if stage_b else 0,
            "stage_b_comb": stage_b[0].get("combined_score", 0) if stage_b else 0,
        })

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    hdr("VALIDATION SUMMARY")

    a_correct_count = sum(1 for r in results if r["stage_a_correct"])
    b_correct_count = sum(1 for r in results if r["stage_b_correct"])
    total = len(results)

    print(f"\n  {'Target':<50} {'Stage A':<22} {'Stage B':<22}")
    print(f"  {'='*50} {'='*22} {'='*22}")
    for r in results:
        a_label = "CORRECT" if r["stage_a_correct"] else "WRONG"
        b_label = "CORRECT" if r["stage_b_correct"] else "WRONG"
        a_str = f"{a_label} ({r['stage_a_conf']:.3f})"
        b_str = f"{b_label} ({r['stage_b_ce']:+.3f})"
        print(f"  {r['title']:<50} {a_str:<22} {b_str:<22}")

    print(f"\n  Stage A accuracy: {a_correct_count}/{total} ({a_correct_count/total*100:.0f}%)")
    print(f"  Stage B accuracy: {b_correct_count}/{total} ({b_correct_count/total*100:.0f}%)")

    total_a_time = sum(r["stage_a_time"] for r in results)
    total_b_time = sum(r["stage_b_time"] for r in results)
    print(f"\n  Stage A total time: {total_a_time:.2f}s (avg {total_a_time/total:.2f}s)")
    print(f"  Stage B total time: {total_b_time:.2f}s (avg {total_b_time/total:.2f}s)")

    print(f"\n  HOW THEY WORK TOGETHER:")
    print(f"  {'='*70}")
    print(f"  Stage A (Rule-based) provides HIGH RECALL:")
    print(f"    - Explores all possible column combinations + join paths")
    print(f"    - Uses heuristics: name similarity, type compat, value overlap")
    print(f"    - Executes quick sample-based verification")
    print(f"    - Returns top-K candidates (usually 8) per target")
    print(f"")
    print(f"  Stage B (ML-based) provides HIGH PRECISION:")
    print(f"    - Bi-encoder embeds target+candidate text, retrieves by cosine sim")
    print(f"    - Cross-encoder re-scores (target, candidate) pairs")
    print(f"    - Learns semantic patterns from training data")
    print(f"    - Corrects Stage A mistakes by understanding context")
    print(f"")
    print(f"  Pipeline: The two stages are COMPLEMENTARY:")
    print(f"    Stage A ensures the correct answer is IN the candidate list")
    print(f"    Stage B ensures the correct answer is at the TOP of the list")

    # Check if using base models
    bi_dir = Path("artifacts/stage1_candidate_selector/biencoder")
    has_weights = any((bi_dir / f).exists() for f in ["model.safetensors", "pytorch_model.bin"])
    if not has_weights:
        print(f"\n  NOTE: Stage B used BASE pretrained model weights (not fine-tuned).")
        print(f"  With fine-tuned weights, Stage B accuracy would significantly improve.")
        print(f"  The model weight files (.safetensors/.bin) need to be added to:")
        print(f"    artifacts/stage1_candidate_selector/biencoder/")
        print(f"    artifacts/stage1_candidate_selector/cross_encoder/")

    print(f"\n{SEP}")
    print(f"  VALIDATION COMPLETE")
    print(SEP)


if __name__ == "__main__":
    main()
