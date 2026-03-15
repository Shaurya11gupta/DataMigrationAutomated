"""
Name Similarity Engine - Interactive Visualization Web App
============================================================
A beautiful, immersive web application that visualizes the full
name similarity pipeline:

  Input -> Regex Split -> Segmentation Model -> Dictionary Lookup
       -> Expand Classifier -> Abbreviation Expander -> Embedding
       -> Cosine Similarity -> Conflict Detection -> Final Score

Run:
  python name_similarity_app.py
  Open http://localhost:5001
"""
from __future__ import annotations

import json
import time
import re
import math
from flask import Flask, jsonify, render_template, request

# ── Lazy model loading (avoid crashing if models aren't present) ──
_MODELS_LOADED = False
_LOAD_ERROR = None

# Pre-define the abbreviation dictionary (always available, no model needed)
ABBREV_DICT = {
    "id": "identifier", "pk": "primary", "fk": "foreign", "ref": "reference",
    "key": "key", "code": "code", "cust": "customer", "usr": "user",
    "acct": "account", "acc": "account", "dept": "department",
    "org": "organization", "emp": "employee", "mgr": "manager",
    "mngr": "manager", "adm": "admin", "cfg": "configuration",
    "conf": "configuration", "param": "parameter", "attr": "attribute",
    "meta": "metadata", "db": "database", "tbl": "table", "col": "column",
    "idx": "index", "seq": "sequence", "repo": "repository",
    "amt": "amount", "bal": "balance", "qty": "quantity",
    "txn": "transaction", "inv": "invoice", "ord": "order",
    "pay": "payment", "price": "price", "cost": "cost",
    "ship": "shipment", "addr": "address", "dest": "destination",
    "src": "source", "ts": "timestamp", "dt": "date", "tm": "time",
    "dob": "birthdate", "svc": "service", "api": "interface",
    "ctrl": "controller", "proc": "process", "exec": "execution",
    "cfgmgr": "configuration manager", "stat": "status", "sts": "status",
    "flag": "flag", "type": "type", "grp": "group",
    "grpmap": "group mapping", "map": "mapping", "rel": "relation",
    "desc": "description", "msg": "message", "err": "error",
    "num": "number", "val": "value", "cnt": "count",
    "min": "minimum", "max": "maximum",
}


def _try_load_models():
    """Attempt to load the ML models. Returns True if successful."""
    global _MODELS_LOADED, _LOAD_ERROR
    if _MODELS_LOADED:
        return True
    try:
        # This will load segmentation, classifier, expander, and embedding models
        import seg_classf_abbrev_test  # noqa: F401
        _MODELS_LOADED = True
        return True
    except Exception as e:
        _LOAD_ERROR = str(e)
        return False


# ── Regex split (always available, no model needed) ──
CAMEL = re.compile(r'[A-Z]?[a-z]+|[A-Z]+|\d+')


def regex_split(name: str) -> list:
    name = name.replace("_", " ").replace("-", " ")
    out = []
    for part in name.split():
        out.extend(CAMEL.findall(part))
    return out


# ── Flask app ──
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("name_similarity.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Full pipeline analysis of two column names.
    Returns detailed step-by-step results for visualization.
    """
    data = request.get_json(force=True)
    name_a = data.get("name_a", "").strip()
    name_b = data.get("name_b", "").strip()

    if not name_a or not name_b:
        return jsonify({"error": "Both names are required"}), 400

    t_start = time.time()
    models_available = _try_load_models()

    # ── Step 1: Regex Split ──
    tokens_a = regex_split(name_a)
    tokens_b = regex_split(name_b)

    # ── Step 2: Segmentation ──
    seg_details_a = []
    seg_details_b = []
    seg_tokens_a = []
    seg_tokens_b = []

    if models_available:
        from seg_classf_abbrev_test import segment_token
        for t in tokens_a:
            parts = list(segment_token(t))
            seg_details_a.append({"input": t, "output": parts})
            seg_tokens_a.extend(parts)
        for t in tokens_b:
            parts = list(segment_token(t))
            seg_details_b.append({"input": t, "output": parts})
            seg_tokens_b.extend(parts)
    else:
        # Fallback: just pass through
        for t in tokens_a:
            seg_details_a.append({"input": t, "output": [t.lower()]})
            seg_tokens_a.append(t.lower())
        for t in tokens_b:
            seg_details_b.append({"input": t, "output": [t.lower()]})
            seg_tokens_b.append(t.lower())

    # ── Step 3: Dictionary + Classifier + Expander ──
    expansion_details_a = []
    expansion_details_b = []
    final_tokens_a = []
    final_tokens_b = []

    if models_available:
        from seg_classf_abbrev_test import (
            expand_with_dictionary_then_model,
            should_expand,
        )
        for t in seg_tokens_a:
            expanded, source, prob = expand_with_dictionary_then_model(t)
            need_expand, clf_prob = should_expand(t) if source != "dict" else (False, 0.0)
            expansion_details_a.append({
                "input": t,
                "dict_lookup": ABBREV_DICT.get(t.lower(), None),
                "classifier_expand": need_expand if source != "dict" else None,
                "classifier_prob": round(clf_prob, 4) if source != "dict" else None,
                "model_output": expanded if source == "model" else None,
                "final": expanded,
                "source": source,
            })
            final_tokens_a.append(expanded)
        for t in seg_tokens_b:
            expanded, source, prob = expand_with_dictionary_then_model(t)
            need_expand, clf_prob = should_expand(t) if source != "dict" else (False, 0.0)
            expansion_details_b.append({
                "input": t,
                "dict_lookup": ABBREV_DICT.get(t.lower(), None),
                "classifier_expand": need_expand if source != "dict" else None,
                "classifier_prob": round(clf_prob, 4) if source != "dict" else None,
                "model_output": expanded if source == "model" else None,
                "final": expanded,
                "source": source,
            })
            final_tokens_b.append(expanded)
    else:
        # Fallback: dictionary only
        for t in seg_tokens_a:
            tl = t.lower()
            d = ABBREV_DICT.get(tl)
            expansion_details_a.append({
                "input": t,
                "dict_lookup": d,
                "classifier_expand": None,
                "classifier_prob": None,
                "model_output": None,
                "final": d if d else tl,
                "source": "dict" if d else "none",
            })
            final_tokens_a.append(d if d else tl)
        for t in seg_tokens_b:
            tl = t.lower()
            d = ABBREV_DICT.get(tl)
            expansion_details_b.append({
                "input": t,
                "dict_lookup": d,
                "classifier_expand": None,
                "classifier_prob": None,
                "model_output": None,
                "final": d if d else tl,
                "source": "dict" if d else "none",
            })
            final_tokens_b.append(d if d else tl)

    # ── Step 4: Embedding + Similarity ──
    phrase_a = " ".join(final_tokens_a)
    phrase_b = " ".join(final_tokens_b)

    similarity_result = {}
    if models_available:
        from seg_classf_abbrev_test import name_similarity
        similarity_result = name_similarity(name_a, name_b)
    else:
        # Basic Jaccard fallback
        set_a = set(t.lower() for t in final_tokens_a)
        set_b = set(t.lower() for t in final_tokens_b)
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        sim = inter / union if union > 0 else 0.0
        similarity_result = {
            "similarity": round(sim, 4),
            "confidence": "HIGH" if sim > 0.85 else ("MEDIUM" if sim > 0.65 else "LOW"),
            "base_similarity": round(sim, 4),
            "penalty": 0.0,
            "tokens_A": final_tokens_a,
            "tokens_B": final_tokens_b,
            "conflicts": [],
            "expansion_sources": {},
            "runtime_ms": 0,
        }

    elapsed = (time.time() - t_start) * 1000

    return jsonify({
        "name_a": name_a,
        "name_b": name_b,
        "models_loaded": models_available,
        "model_load_error": _LOAD_ERROR if not models_available else None,

        "step1_regex": {
            "a": {"input": name_a, "tokens": tokens_a},
            "b": {"input": name_b, "tokens": tokens_b},
        },
        "step2_segmentation": {
            "a": seg_details_a,
            "b": seg_details_b,
        },
        "step3_expansion": {
            "a": expansion_details_a,
            "b": expansion_details_b,
        },
        "step4_similarity": {
            "phrase_a": phrase_a,
            "phrase_b": phrase_b,
            "base_similarity": similarity_result.get("base_similarity", 0),
            "penalty": similarity_result.get("penalty", 0),
            "final_similarity": similarity_result.get("similarity", 0),
            "confidence": similarity_result.get("confidence", "N/A"),
            "conflicts": similarity_result.get("conflicts", []),
            "conflict_details": similarity_result.get("conflict_details", []),
            "token_conflicts_a": similarity_result.get("token_conflicts_a", {}),
            "token_conflicts_b": similarity_result.get("token_conflicts_b", {}),
        },
        "runtime_ms": round(elapsed, 2),

        # Dictionary info for the UI
        "dictionary_size": len(ABBREV_DICT),
    })


@app.route("/api/dictionary", methods=["GET"])
def get_dictionary():
    """Return the abbreviation dictionary for display."""
    return jsonify({"dictionary": ABBREV_DICT, "size": len(ABBREV_DICT)})


@app.route("/api/examples", methods=["GET"])
def get_examples():
    """Return predefined examples for quick testing."""
    return jsonify({"examples": [
        {"a": "custId", "b": "customer_id", "label": "Abbreviation match"},
        {"a": "acctBal", "b": "account_balance", "label": "Multi-abbreviation"},
        {"a": "empFirstNm", "b": "employee_first_name", "label": "Segmentation + expansion"},
        {"a": "shipAddr", "b": "shipment_address", "label": "Compound abbreviation"},
        {"a": "usrGrpMap", "b": "user_group_mapping", "label": "Triple abbreviation"},
        {"a": "cfgMgr", "b": "configuration_manager", "label": "Concatenated abbreviations"},
        {"a": "minPrice", "b": "maxPrice", "label": "Conflict detection (directional)"},
        {"a": "created_date", "b": "deleted_date", "label": "Conflict detection (temporal)"},
        {"a": "buyer_name", "b": "seller_name", "label": "Conflict detection (role)"},
        {"a": "price_usd", "b": "price_eur", "label": "Conflict detection (unit)"},
        {"a": "deptName", "b": "department_name", "label": "Simple abbreviation"},
        {"a": "txnAmt", "b": "transaction_amount", "label": "Finance terms"},
    ]})


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Name Similarity Engine - Interactive Visualization")
    print("=" * 60)
    print("  Open: http://localhost:5001")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5001)
