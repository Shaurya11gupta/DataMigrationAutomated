"""
DataMigrationCrane – Main Web Application
==========================================
Flask app for the full schema mapping pipeline:

  Step 1: Source Ingestion    → Define source tables (JSON or visual builder)
  Step 2: Source Visualization → Join graph discovery & display
  Step 3: Target Ingestion    → Define target tables
  Step 4: Mapping Results     → Candidate selection (schema_matcher_llm1 on CPU)

Pipeline flow:
  Source JSON → json_to_tables → discover_joins → tables_to_summary
  Target JSON → run_full_mapping (V3 LLM: schema_matcher_llm1 on CPU) → candidate selection results

Run:
  python app.py
  Open http://127.0.0.1:5000
"""
from __future__ import annotations

import json
from flask import Flask, jsonify, redirect, render_template, request, url_for

from pipeline_bridge import (
    discover_joins,
    edges_to_json,
    json_to_tables,
    run_full_mapping,
    tables_to_summary,
)

app = Flask(__name__)
app.secret_key = "datamigrationcrane-2026"

# In-memory state (single-user dev mode)
STATE: dict = {
    "source_tables": None,       # Dict[str, Table]
    "source_tables_json": [],    # raw JSON for re-display
    "join_edges": [],            # List[JoinEdge]
    "join_edges_json": [],       # serialized for frontend
    "tables_summary": [],        # frontend-friendly summary
    "join_time": 0,
    "target_tables_json": [],    # raw target JSON
    "mapping_results": None,     # full mapping output (candidate selection)
}


# -----------------------------------------------------------
# Routes – Page views
# -----------------------------------------------------------

@app.route("/")
def index():
    """Redirect to Step 1: Source Ingestion."""
    return redirect(url_for("source_ingestion"))


@app.route("/source/ingest")
def source_ingestion():
    """Step 1: Define source schema (tables, columns, optional sample data)."""
    return render_template("source_ingestion.html")


@app.route("/source/visualize")
def source_visualization():
    """Step 2: Visualize source schema and discovered join graph."""
    if not STATE["source_tables"]:
        return redirect(url_for("source_ingestion"))
    return render_template(
        "source_visualization.html",
        tables=json.dumps(STATE["tables_summary"]),
        edges=json.dumps(STATE["join_edges_json"]),
        join_time=STATE["join_time"],
    )


@app.route("/target/ingest")
def target_ingestion():
    """Step 3: Define target schema. Source tables must be loaded first."""
    if not STATE["source_tables"]:
        return redirect(url_for("source_ingestion"))
    return render_template(
        "target_ingestion.html",
        source_tables=json.dumps(STATE["tables_summary"]),
    )


@app.route("/target/visualize")
def target_visualization():
    """Step 4: Display candidate selection results."""
    if not STATE["mapping_results"]:
        return redirect(url_for("target_ingestion"))
    return render_template(
        "target_visualization.html",
        results=json.dumps(STATE["mapping_results"]),
    )


# -----------------------------------------------------------
# API endpoints
# -----------------------------------------------------------

@app.route("/api/source/submit", methods=["POST"])
def api_source_submit():
    """
    Process source schema: convert JSON → tables, discover joins.
    Returns: tables_count, edges_count, time, redirect URL.
    """
    try:
        data = request.get_json(force=True)
        tables_json = data.get("tables", [])
        if not tables_json:
            return jsonify({"error": "No tables provided"}), 400

        # Convert JSON to domain objects
        tables = json_to_tables(tables_json)
        STATE["source_tables"] = tables
        STATE["source_tables_json"] = tables_json
        STATE["tables_summary"] = tables_to_summary(tables)

        # Discover join relationships
        edges, elapsed = discover_joins(tables)
        STATE["join_edges"] = edges
        STATE["join_edges_json"] = edges_to_json(edges)
        STATE["join_time"] = round(elapsed, 2)

        return jsonify({
            "success": True,
            "tables_count": len(tables),
            "edges_count": len(edges),
            "time": round(elapsed, 2),
            "redirect": url_for("source_visualization"),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/target/map", methods=["POST"])
def api_target_map():
    """
    Run candidate selection pipeline: schema_matcher_llm1 (V3) on CPU.
    Maps each target column to source column candidates.
    Returns: stats (total_columns, mapped_columns, mapping_rate, total_time), redirect URL.
    """
    try:
        if not STATE["source_tables"]:
            return jsonify({"error": "No source schema loaded"}), 400

        data = request.get_json(force=True)
        target_json = data.get("tables", [])
        if not target_json:
            return jsonify({"error": "No target tables provided"}), 400

        STATE["target_tables_json"] = target_json

        # Run full mapping (V3 LLM on CPU, V2 bi-encoder disabled)
        results = run_full_mapping(
            source_tables=STATE["source_tables"],
            join_edges=STATE["join_edges"],
            target_tables_json=target_json,
        )
        STATE["mapping_results"] = results

        return jsonify({
            "success": True,
            "stats": results["stats"],
            "redirect": url_for("target_visualization"),
        })
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


# -----------------------------------------------------------
# Run
# -----------------------------------------------------------

if __name__ == "__main__":
    print("\n  DataMigrationCrane Web App")
    print("  Pipeline: Source → Join Graph → Target → Candidate Selection (schema_matcher_llm1 on CPU)")
    print("  http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
