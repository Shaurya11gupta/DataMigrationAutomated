"""
Value Similarity Engine - Interactive Visualization Web App
=============================================================
A beautiful, immersive web app that visualizes how two columns
of data values are compared through statistical profiling:

  Raw Values -> Normalization -> Type Detection -> Column Profiling
            -> Signal Computation -> Weighted Score

Run:
  python value_similarity_app.py
  Open http://localhost:5002
"""
from __future__ import annotations

import json
import math
import time
from flask import Flask, jsonify, render_template, request

from value_similarity_engine import ColumnStats, ValueSimilarity

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("value_similarity.html")


@app.route("/api/compare", methods=["POST"])
def compare():
    """
    Compare two columns of values.
    Expects JSON: { "values_a": [...], "values_b": [...], "name_a": "...", "name_b": "..." }
    """
    data = request.get_json(force=True)
    raw_a = data.get("values_a", [])
    raw_b = data.get("values_b", [])
    name_a = data.get("name_a", "Column A")
    name_b = data.get("name_b", "Column B")

    if not raw_a or not raw_b:
        return jsonify({"error": "Both columns must have values"}), 400

    t0 = time.time()

    # Build stats
    stats_a = ColumnStats(raw_a)
    stats_b = ColumnStats(raw_b)

    # Compute similarity
    engine = ValueSimilarity(stats_a, stats_b)
    result = engine.compute_score()

    elapsed = (time.time() - t0) * 1000

    # Build profile summaries for frontend
    def _profile(stats, name, raw):
        p = {
            "name": name,
            "total_rows": stats.total_rows,
            "valid_count": stats.valid_count,
            "null_frac": round(stats.null_frac, 4),
            "detected_type": stats.type,
            "n_distinct": stats.n_distinct,
            "distinct_ratio": round(stats.distinct_ratio, 4),
        }
        if stats.type == "numeric":
            p["min"] = stats.min
            p["max"] = stats.max
            p["q05"] = round(stats.q05, 4)
            p["q50"] = round(stats.q50, 4)
            p["q95"] = round(stats.q95, 4)
            p["integer_ratio"] = round(stats.integer_ratio, 4)
            p["monotonic"] = round(stats.monotonic, 4)
            p["outlier_ratio"] = round(stats.outlier_ratio, 4)
            # Build histogram data for chart
            import numpy as np
            if stats.data_for_hist is not None and len(stats.data_for_hist) >= 2:
                arr = stats.data_for_hist
                lo, hi = float(np.quantile(arr, 0.01)), float(np.quantile(arr, 0.99))
                if hi <= lo:
                    hi = lo + 1
                bins = list(map(float, np.linspace(lo, hi, 21)))
                hist, _ = np.histogram(np.clip(arr, lo, hi), bins=bins)
                p["histogram"] = {
                    "bins": [round(b, 4) for b in bins],
                    "counts": [int(c) for c in hist],
                }
        elif stats.type == "categorical":
            p["entropy"] = round(stats.entropy, 4)
            p["avg_len"] = round(stats.avg_len, 2)
            p["std_len"] = round(stats.std_len, 2)
            # Top 10 most common values
            mcv_sorted = sorted(stats.mcv.items(), key=lambda x: -x[1])[:10]
            p["top_values"] = [{"value": str(k), "freq": round(v, 4)} for k, v in mcv_sorted]

        # Sample preview (first 8 values)
        p["sample_preview"] = [str(v) for v in raw[:8]]
        return p

    profile_a = _profile(stats_a, name_a, raw_a)
    profile_b = _profile(stats_b, name_b, raw_b)

    # Determine which signals were used
    details = result.get("details", {})
    is_numeric = stats_a.type == "numeric" and stats_b.type == "numeric"
    is_categorical = stats_a.type == "categorical" and stats_b.type == "categorical"

    # Build signal breakdown with weights
    signals = []
    if is_categorical:
        weights = {
            "value_evidence": 0.36, "mcv_cos": 0.16, "weighted_jacc": 0.14,
            "contain": 0.10, "jacc": 0.08, "n_distinct": 0.07,
            "null": 0.04, "entropy": 0.03, "len_profile": 0.02,
        }
        labels = {
            "value_evidence": "Value Evidence (max overlap)",
            "mcv_cos": "MCV Cosine Similarity",
            "weighted_jacc": "Weighted Jaccard",
            "contain": "Set Containment",
            "jacc": "Set Jaccard",
            "n_distinct": "Distinct Count Similarity",
            "null": "Null Compatibility",
            "entropy": "Entropy Similarity",
            "len_profile": "Length Profile Similarity",
        }
    elif is_numeric:
        weights = {
            "value_evidence": 0.32, "hist": 0.16, "cdf": 0.12,
            "exact_overlap": 0.10, "range": 0.08, "n_distinct": 0.08,
            "null": 0.05, "int_ratio": 0.03, "monotonic": 0.02,
            "outliers": 0.02, "scale": 0.02,
        }
        labels = {
            "value_evidence": "Value Evidence (distribution overlap)",
            "hist": "Histogram Intersection",
            "cdf": "CDF Similarity",
            "exact_overlap": "Exact Value Overlap",
            "range": "Range Containment",
            "n_distinct": "Distinct Count Similarity",
            "null": "Null Compatibility",
            "int_ratio": "Integer Ratio Match",
            "monotonic": "Monotonicity Match",
            "outliers": "Outlier Similarity",
            "scale": "Scale Similarity",
        }
    else:
        weights = {}
        labels = {}

    for key in weights:
        val = details.get(key, 0)
        signals.append({
            "key": key,
            "label": labels.get(key, key),
            "value": val,
            "weight": weights[key],
            "contribution": round(val * weights[key], 4),
        })

    return jsonify({
        "profile_a": profile_a,
        "profile_b": profile_b,
        "final_score": result.get("final", 0),
        "reason": result.get("reason", None),
        "type_match": stats_a.type == stats_b.type,
        "detected_type": stats_a.type if stats_a.type == stats_b.type else f"{stats_a.type} vs {stats_b.type}",
        "signals": signals,
        "runtime_ms": round(elapsed, 2),
    })


@app.route("/api/examples", methods=["GET"])
def get_examples():
    """Predefined examples for quick testing — covers all major comparison scenarios."""
    return jsonify({"examples": [
        {
            "label": "Exact ID Match",
            "desc": "Identical integer PKs",
            "name_a": "employee_id",
            "name_b": "emp_id",
            "values_a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            "values_b": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        },
        {
            "label": "Similar Salaries",
            "desc": "Close numeric ranges with slight variance",
            "name_a": "salary",
            "name_b": "annual_pay",
            "values_a": [55000, 62000, 48000, 71000, 59000, 45000, 83000, 67000, 52000, 74000, 61000, 56000],
            "values_b": [54500, 61800, 47500, 70800, 58700, 44800, 82500, 66500, 51800, 73500, 60800, 55500],
        },
        {
            "label": "Different Scales",
            "desc": "Age (20-50) vs salary (45k-83k)",
            "name_a": "age",
            "name_b": "salary",
            "values_a": [25, 30, 35, 28, 42, 38, 22, 31, 27, 45, 33, 29],
            "values_b": [55000, 62000, 48000, 71000, 59000, 45000, 83000, 67000, 52000, 74000, 61000, 56000],
        },
        {
            "label": "Same Departments",
            "desc": "Identical category sets, different order",
            "name_a": "dept_name",
            "name_b": "department",
            "values_a": ["Engineering", "Sales", "Engineering", "HR", "Sales", "Marketing", "Engineering", "HR", "Sales", "Marketing"],
            "values_b": ["Engineering", "Sales", "HR", "Engineering", "Marketing", "Sales", "HR", "Engineering", "Sales", "Marketing"],
        },
        {
            "label": "Disjoint Categories",
            "desc": "Departments vs countries (no overlap)",
            "name_a": "department",
            "name_b": "country",
            "values_a": ["Engineering", "Sales", "HR", "Marketing", "Finance", "Engineering", "Sales", "HR"],
            "values_b": ["USA", "Canada", "UK", "Germany", "France", "USA", "Canada", "UK"],
        },
        {
            "label": "Partial Status Overlap",
            "desc": "Order vs shipment statuses (some shared)",
            "name_a": "order_status",
            "name_b": "shipment_status",
            "values_a": ["pending", "shipped", "delivered", "pending", "cancelled", "shipped", "delivered", "pending", "shipped", "delivered"],
            "values_b": ["in_transit", "delivered", "pending", "delivered", "in_transit", "pending", "delivered", "in_transit", "pending", "delivered"],
        },
        {
            "label": "Type Mismatch",
            "desc": "Numeric quantity vs categorical product names",
            "name_a": "quantity",
            "name_b": "product_name",
            "values_a": [10, 25, 5, 15, 30, 8, 12, 20],
            "values_b": ["Widget", "Gadget", "Widget", "Doohickey", "Gadget", "Widget", "Thingamajig", "Doohickey"],
        },
        {
            "label": "Null Imbalance",
            "desc": "70% nulls vs 10% nulls",
            "name_a": "phone",
            "name_b": "email",
            "values_a": ["555-1234", None, None, "555-5678", None, None, None, "555-9012", None, None],
            "values_b": ["a@b.com", "c@d.com", "e@f.com", "g@h.com", "i@j.com", "k@l.com", "m@n.com", "o@p.com", None, "q@r.com"],
        },
        {
            "label": "Boolean Flags",
            "desc": "Two binary columns with similar distributions",
            "name_a": "is_active",
            "name_b": "is_enabled",
            "values_a": ["true", "true", "false", "true", "false", "true", "true", "false", "true", "true"],
            "values_b": ["true", "false", "true", "true", "true", "false", "true", "true", "false", "true"],
        },
        {
            "label": "Subset Containment",
            "desc": "Column B values are a subset of column A",
            "name_a": "all_colors",
            "name_b": "primary_colors",
            "values_a": ["red", "blue", "green", "yellow", "orange", "purple", "red", "blue", "green", "yellow", "orange", "purple"],
            "values_b": ["red", "blue", "yellow", "red", "blue", "yellow", "red", "blue"],
        },
        {
            "label": "High Cardinality",
            "desc": "Many unique emails vs many unique usernames",
            "name_a": "email",
            "name_b": "username",
            "values_a": ["alice@co.com", "bob@co.com", "carol@co.com", "dave@co.com", "eve@co.com", "frank@co.com", "grace@co.com", "hank@co.com"],
            "values_b": ["alice_01", "bob_99", "carol_x", "dave_z", "eve_42", "frank_7", "grace_abc", "hank_def"],
        },
        {
            "label": "Monotonic Sequences",
            "desc": "Both columns are strictly increasing",
            "name_a": "order_id",
            "name_b": "invoice_number",
            "values_a": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010],
            "values_b": [5001, 5002, 5003, 5004, 5005, 5006, 5007, 5008, 5009, 5010],
        },
        {
            "label": "Percentage vs Decimal",
            "desc": "Same data, different scale (0-100 vs 0.0-1.0)",
            "name_a": "completion_pct",
            "name_b": "progress_ratio",
            "values_a": [25, 50, 75, 100, 10, 90, 60, 40, 80, 33],
            "values_b": [0.25, 0.50, 0.75, 1.0, 0.10, 0.90, 0.60, 0.40, 0.80, 0.33],
        },
        {
            "label": "Gender Codes vs Labels",
            "desc": "Coded values vs human-readable labels",
            "name_a": "gender_code",
            "name_b": "gender_label",
            "values_a": ["M", "F", "M", "F", "M", "F", "M", "M", "F", "M"],
            "values_b": ["Male", "Female", "Male", "Female", "Male", "Female", "Male", "Male", "Female", "Male"],
        },
    ]})


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Value Similarity Engine - Interactive Visualization")
    print("=" * 60)
    print("  Open: http://localhost:5002")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5002)
