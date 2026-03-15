#!/usr/bin/env python3
"""
Robustness Test Suite for Stage A + Stage B Pipeline
=====================================================

Tests across 6 categories:
  1. Cross-type transformations (date->int, string->date, numeric->bool, etc.)
  2. Multi-hop join lookups (2-3 hop traversals)
  3. Complex multi-column composites (concat, arithmetic, conditional)
  4. Edge cases (ambiguous names, similar columns, sparse data)
  5. String transformations (split, regex, case, trim, email)
  6. Semantic name matching (synonyms, abbreviations, unusual naming)

Each test case has:
  - A target spec with expected ground truth
  - Expected source columns and transformation family

Usage:
  python robustness_test.py
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from constraint_similarity_engine import ColumnConstraints
from join_graph_builder_v2 import Column, ColumnType, JoinEdge, Table
from value_similarity_engine import ColumnStats

from candidate_generation_algorithm import CandidateGenerationEngine, NameEmbedder, TargetSpec
from candidate_selector_stage1 import (
    CandidateSelectorStage1,
    CandidateSet as SelectorCandidateSet,
    SourceColumn,
    TargetColumn,
    target_to_text,
    candidate_to_text,
)


# ===================================================================
# Helpers
# ===================================================================

def make_col(name, rows, typ, constraints=None):
    values = [r.get(name) for r in rows]
    return Column(name=name, col_type=ColumnType(typ),
                  constraints=constraints or ColumnConstraints(),
                  stats=ColumnStats(values))


def bridge(target_spec, stage_a_result):
    target_col = TargetColumn(
        table=target_spec.table, column=target_spec.name,
        type=getattr(target_spec.col_type, "base_type", "string"),
        description=target_spec.description or "",
    )
    candidate_sets = []
    for i, cand in enumerate(stage_a_result.get("top_candidates", [])):
        columns = []
        for col_ref in cand.get("candidate_columns", []):
            parts = col_ref.split(".", 1)
            if len(parts) == 2:
                columns.append(SourceColumn(table=parts[0], column=parts[1], type="string"))
        candidate_sets.append(SelectorCandidateSet(
            id=f"cand_{i:03d}", columns=columns,
            join_path=cand.get("join_path", []),
            transform_hint=cand.get("best_transform_family", ""),
        ))
    return target_col, candidate_sets


SEP = "=" * 105
SUB = "-" * 105


# ===================================================================
# Build a rich, realistic multi-table schema
# ===================================================================

def build_schema():
    # -- EMPLOYEES --
    emp_rows = [
        {"emp_id": i, "first_name": fn, "last_name": ln,
         "email_address": f"{fn.lower()}.{ln.lower()}@corp.com",
         "hire_date": hd, "department_id": did, "salary_amount": sal,
         "manager_id": mid, "phone_num": ph, "job_title": jt,
         "birth_date": bd, "address_line1": addr, "zip_code": zc,
         "performance_rating": pr, "is_active": act}
        for i, (fn, ln, hd, did, sal, mid, ph, jt, bd, addr, zc, pr, act) in enumerate([
            ("Alice", "Johnson", "2019-03-15", 10, 95000.0, None, "+1-617-555-0101", "Senior Engineer", "1990-05-12", "123 Main St", "02108", 4.2, True),
            ("Bob", "Williams", "2020-07-22", 10, 82000.0, 0, "+1-617-555-0102", "Engineer", "1992-08-30", "456 Oak Ave", "02109", 3.8, True),
            ("Carol", "Davis", "2018-01-10", 20, 110000.0, None, "+44-20-7946-0958", "VP Marketing", "1985-11-03", "789 High St", "EC1A1BB", 4.5, True),
            ("David", "Garcia", "2021-11-05", 30, 75000.0, 2, "+91-80-4412-3456", "Analyst", "1995-02-18", "101 MG Road", "560001", 3.5, True),
            ("Emma", "Martinez", "2017-06-30", 20, 98000.0, 2, "+1-206-555-0199", "Marketing Manager", "1988-09-25", "222 Pine Blvd", "98101", 4.0, False),
            ("Frank", "Anderson", "2022-02-14", 10, 78000.0, 0, "+1-415-555-0177", "Junior Engineer", "1997-12-01", "333 Elm St", "94102", 3.2, True),
            ("Grace", "Thomas", "2016-09-01", 40, 125000.0, None, "+1-212-555-0133", "CFO", "1980-04-14", "444 Broadway", "10001", 4.8, True),
            ("Henry", "Jackson", "2023-04-18", 30, 68000.0, 3, "+91-22-2345-6789", "Junior Analyst", "1998-07-07", "555 Link Rd", "400001", 3.0, True),
        ], start=0)
    ]

    # -- DEPARTMENTS --
    dept_rows = [
        {"department_id": 10, "dept_name": "Engineering", "dept_budget": 500000.0, "location_city": "Boston"},
        {"department_id": 20, "dept_name": "Marketing", "dept_budget": 350000.0, "location_city": "London"},
        {"department_id": 30, "dept_name": "Analytics", "dept_budget": 200000.0, "location_city": "Bengaluru"},
        {"department_id": 40, "dept_name": "Finance", "dept_budget": 400000.0, "location_city": "New York"},
    ]

    # -- PROJECTS --
    proj_rows = [
        {"project_id": 100, "project_name": "Platform Rewrite", "department_id": 10,
         "start_date": "2023-01-15", "end_date": "2024-06-30", "budget_usd": 250000.0, "status": "active"},
        {"project_id": 101, "project_name": "Brand Campaign", "department_id": 20,
         "start_date": "2023-06-01", "end_date": "2023-12-31", "budget_usd": 150000.0, "status": "completed"},
        {"project_id": 102, "project_name": "Data Lake Migration", "department_id": 30,
         "start_date": "2024-01-01", "end_date": "2025-03-31", "budget_usd": 180000.0, "status": "active"},
        {"project_id": 103, "project_name": "Financial Reporting", "department_id": 40,
         "start_date": "2022-07-01", "end_date": "2023-06-30", "budget_usd": 120000.0, "status": "completed"},
    ]

    # -- TIMESHEETS --
    ts_rows = [
        {"timesheet_id": 1, "emp_id": 0, "project_id": 100, "log_date": "2024-01-15", "hours_worked": 8.0, "description": "API redesign"},
        {"timesheet_id": 2, "emp_id": 1, "project_id": 100, "log_date": "2024-01-15", "hours_worked": 7.5, "description": "Frontend refactor"},
        {"timesheet_id": 3, "emp_id": 2, "project_id": 101, "log_date": "2023-09-10", "hours_worked": 6.0, "description": "Campaign review"},
        {"timesheet_id": 4, "emp_id": 3, "project_id": 102, "log_date": "2024-02-20", "hours_worked": 9.0, "description": "Schema design"},
        {"timesheet_id": 5, "emp_id": 4, "project_id": 101, "log_date": "2023-08-05", "hours_worked": 5.5, "description": "Content creation"},
        {"timesheet_id": 6, "emp_id": 5, "project_id": 100, "log_date": "2024-01-16", "hours_worked": 8.0, "description": "Unit testing"},
        {"timesheet_id": 7, "emp_id": 6, "project_id": 103, "log_date": "2023-01-20", "hours_worked": 4.0, "description": "Budget planning"},
        {"timesheet_id": 8, "emp_id": 7, "project_id": 102, "log_date": "2024-03-01", "hours_worked": 7.0, "description": "Data profiling"},
    ]

    # -- LOCATIONS (for 2-hop joins) --
    loc_rows = [
        {"location_city": "Boston", "country_code": "US", "timezone": "America/New_York", "currency": "USD"},
        {"location_city": "London", "country_code": "UK", "timezone": "Europe/London", "currency": "GBP"},
        {"location_city": "Bengaluru", "country_code": "IN", "timezone": "Asia/Kolkata", "currency": "INR"},
        {"location_city": "New York", "country_code": "US", "timezone": "America/New_York", "currency": "USD"},
    ]

    # Build Table objects
    employees = Table("src_employees", {
        "emp_id": make_col("emp_id", emp_rows, "int", ColumnConstraints(nullable=False, is_primary_key=True, is_unique=True)),
        "first_name": make_col("first_name", emp_rows, "string"),
        "last_name": make_col("last_name", emp_rows, "string"),
        "email_address": make_col("email_address", emp_rows, "string", ColumnConstraints(is_unique=True)),
        "hire_date": make_col("hire_date", emp_rows, "date"),
        "department_id": make_col("department_id", emp_rows, "int", ColumnConstraints(is_foreign_key=True)),
        "salary_amount": make_col("salary_amount", emp_rows, "decimal"),
        "manager_id": make_col("manager_id", emp_rows, "int"),
        "phone_num": make_col("phone_num", emp_rows, "string"),
        "job_title": make_col("job_title", emp_rows, "string"),
        "birth_date": make_col("birth_date", emp_rows, "date"),
        "address_line1": make_col("address_line1", emp_rows, "string"),
        "zip_code": make_col("zip_code", emp_rows, "string"),
        "performance_rating": make_col("performance_rating", emp_rows, "decimal"),
        "is_active": make_col("is_active", emp_rows, "boolean"),
    }, row_count=len(emp_rows), rows=emp_rows)

    departments = Table("dim_department", {
        "department_id": make_col("department_id", dept_rows, "int", ColumnConstraints(nullable=False, is_primary_key=True, is_unique=True)),
        "dept_name": make_col("dept_name", dept_rows, "string"),
        "dept_budget": make_col("dept_budget", dept_rows, "decimal"),
        "location_city": make_col("location_city", dept_rows, "string"),
    }, row_count=len(dept_rows), rows=dept_rows)

    projects = Table("src_projects", {
        "project_id": make_col("project_id", proj_rows, "int", ColumnConstraints(nullable=False, is_primary_key=True, is_unique=True)),
        "project_name": make_col("project_name", proj_rows, "string"),
        "department_id": make_col("department_id", proj_rows, "int", ColumnConstraints(is_foreign_key=True)),
        "start_date": make_col("start_date", proj_rows, "date"),
        "end_date": make_col("end_date", proj_rows, "date"),
        "budget_usd": make_col("budget_usd", proj_rows, "decimal"),
        "status": make_col("status", proj_rows, "string"),
    }, row_count=len(proj_rows), rows=proj_rows)

    timesheets = Table("src_timesheets", {
        "timesheet_id": make_col("timesheet_id", ts_rows, "int", ColumnConstraints(nullable=False, is_primary_key=True, is_unique=True)),
        "emp_id": make_col("emp_id", ts_rows, "int", ColumnConstraints(is_foreign_key=True)),
        "project_id": make_col("project_id", ts_rows, "int", ColumnConstraints(is_foreign_key=True)),
        "log_date": make_col("log_date", ts_rows, "date"),
        "hours_worked": make_col("hours_worked", ts_rows, "decimal"),
        "description": make_col("description", ts_rows, "string"),
    }, row_count=len(ts_rows), rows=ts_rows)

    locations = Table("dim_location", {
        "location_city": make_col("location_city", loc_rows, "string", ColumnConstraints(nullable=False, is_primary_key=True, is_unique=True)),
        "country_code": make_col("country_code", loc_rows, "string"),
        "timezone": make_col("timezone", loc_rows, "string"),
        "currency": make_col("currency", loc_rows, "string"),
    }, row_count=len(loc_rows), rows=loc_rows)

    tables = {
        "src_employees": employees, "dim_department": departments,
        "src_projects": projects, "src_timesheets": timesheets,
        "dim_location": locations,
    }

    edges = [
        JoinEdge("src_employees", "dim_department", ["department_id"], ["department_id"],
                 "N:1", 0.95, ["pk_fk"], {"name": 1.0, "value": 1.0}, "right_parent"),
        JoinEdge("src_employees", "src_timesheets", ["emp_id"], ["emp_id"],
                 "1:N", 0.94, ["pk_fk"], {"name": 0.95, "value": 1.0}, "left_parent"),
        JoinEdge("src_projects", "src_timesheets", ["project_id"], ["project_id"],
                 "1:N", 0.93, ["pk_fk"], {"name": 0.92, "value": 1.0}, "left_parent"),
        JoinEdge("src_projects", "dim_department", ["department_id"], ["department_id"],
                 "N:1", 0.92, ["pk_fk"], {"name": 1.0, "value": 1.0}, "right_parent"),
        JoinEdge("dim_department", "dim_location", ["location_city"], ["location_city"],
                 "N:1", 0.90, ["lookup"], {"name": 0.90, "value": 1.0}, "right_parent"),
    ]

    return tables, edges, emp_rows, dept_rows, proj_rows, ts_rows, loc_rows


# ===================================================================
# Test cases organized by category
# ===================================================================

def build_test_cases(emp, dept, proj, ts, loc):
    dept_map = {d["department_id"]: d for d in dept}
    loc_map = {l["location_city"]: l for l in loc}
    cases = []

    # ---------------------------------------------------------------
    # CATEGORY 1: Cross-type transformations
    # ---------------------------------------------------------------

    # 1.1 date -> int (extract year from hire_date)
    cases.append({
        "cat": "Cross-Type", "title": "hire_date -> hire_year (date->int)",
        "expected_cols": "src_employees.hire_date", "expected_family": "date_part",
        "spec": TargetSpec("dim_employee", "hire_year", ColumnType("int"), ColumnConstraints(),
                           ColumnStats([int(e["hire_date"][:4]) for e in emp]),
                           "Year employee was hired",
                           [int(e["hire_date"][:4]) for e in emp]),
    })

    # 1.2 date -> int (extract month from birth_date)
    cases.append({
        "cat": "Cross-Type", "title": "birth_date -> birth_month (date->int)",
        "expected_cols": "src_employees.birth_date", "expected_family": "date_part",
        "spec": TargetSpec("dim_employee", "birth_month", ColumnType("int"), ColumnConstraints(),
                           ColumnStats([int(e["birth_date"][5:7]) for e in emp]),
                           "Month of birth",
                           [int(e["birth_date"][5:7]) for e in emp]),
    })

    # 1.3 string -> date (parse text date)
    cases.append({
        "cat": "Cross-Type", "title": "hire_date text -> parsed_hire_date (str->date)",
        "expected_cols": "src_employees.hire_date", "expected_family": "identity",
        "spec": TargetSpec("dim_employee", "hire_date_parsed", ColumnType("date"), ColumnConstraints(),
                           ColumnStats([e["hire_date"] for e in emp]),
                           "Parsed hire date"),
    })

    # 1.4 numeric -> boolean (salary threshold)
    cases.append({
        "cat": "Cross-Type", "title": "salary_amount -> is_senior_salary (numeric->bool)",
        "expected_cols": "src_employees.salary_amount", "expected_family": "conditional",
        "spec": TargetSpec("dim_employee", "is_senior_salary", ColumnType("boolean"), ColumnConstraints(),
                           ColumnStats([e["salary_amount"] > 90000 for e in emp]),
                           "Whether salary exceeds senior threshold",
                           [e["salary_amount"] > 90000 for e in emp]),
    })

    # 1.5 numeric -> string (cast rating to label)
    cases.append({
        "cat": "Cross-Type", "title": "performance_rating -> rating_text (num->str)",
        "expected_cols": "src_employees.performance_rating", "expected_family": "cast_to_string",
        "spec": TargetSpec("dim_employee", "rating_text", ColumnType("string"), ColumnConstraints(),
                           ColumnStats([str(e["performance_rating"]) for e in emp]),
                           "Performance rating as text"),
    })

    # ---------------------------------------------------------------
    # CATEGORY 2: Lookup / join-based transformations
    # ---------------------------------------------------------------

    # 2.1 Simple lookup: dept_id -> dept_name
    cases.append({
        "cat": "Lookup", "title": "department_id -> department_name (lookup)",
        "expected_cols": "dim_department.dept_name", "expected_family": "identity",
        "spec": TargetSpec("dim_employee", "department_name", ColumnType("string"), ColumnConstraints(),
                           ColumnStats([dept_map[e["department_id"]]["dept_name"] for e in emp]),
                           "Department name from lookup"),
    })

    # 2.2 2-hop lookup: emp -> dept -> location -> timezone
    cases.append({
        "cat": "Lookup", "title": "emp -> dept -> location -> timezone (2-hop)",
        "expected_cols": "dim_location.timezone", "expected_family": "identity",
        "spec": TargetSpec("dim_employee", "work_timezone", ColumnType("string"), ColumnConstraints(),
                           ColumnStats([loc_map[dept_map[e["department_id"]]["location_city"]]["timezone"] for e in emp]),
                           "Employee work timezone via department location"),
    })

    # 2.3 2-hop lookup: emp -> dept -> location -> currency
    cases.append({
        "cat": "Lookup", "title": "emp -> dept -> location -> currency (2-hop)",
        "expected_cols": "dim_location.currency", "expected_family": "identity",
        "spec": TargetSpec("dim_employee", "pay_currency", ColumnType("string"), ColumnConstraints(),
                           ColumnStats([loc_map[dept_map[e["department_id"]]["location_city"]]["currency"] for e in emp]),
                           "Currency used for employee pay"),
    })

    # ---------------------------------------------------------------
    # CATEGORY 3: String transformations
    # ---------------------------------------------------------------

    # 3.1 Email username extraction
    cases.append({
        "cat": "String", "title": "email_address -> email_local_part (extract)",
        "expected_cols": "src_employees.email_address", "expected_family": "email_username_extract",
        "spec": TargetSpec("dim_employee", "email_local_part", ColumnType("string"), ColumnConstraints(),
                           ColumnStats([e["email_address"].split("@")[0] for e in emp]),
                           "Username from email",
                           [e["email_address"].split("@")[0] for e in emp]),
    })

    # 3.2 Lowercase job title
    cases.append({
        "cat": "String", "title": "job_title -> job_title_lower (lowercase)",
        "expected_cols": "src_employees.job_title", "expected_family": "lower",
        "spec": TargetSpec("dim_employee", "job_title_lower", ColumnType("string"), ColumnConstraints(),
                           ColumnStats([e["job_title"].lower() for e in emp]),
                           "Job title in lowercase",
                           [e["job_title"].lower() for e in emp]),
    })

    # 3.3 Concat first + last -> full name
    cases.append({
        "cat": "String", "title": "first_name + last_name -> employee_full_name (concat)",
        "expected_cols": "src_employees.first_name, src_employees.last_name",
        "expected_family": "concat",
        "spec": TargetSpec("dim_employee", "employee_full_name", ColumnType("string"), ColumnConstraints(),
                           ColumnStats([f"{e['first_name']} {e['last_name']}" for e in emp]),
                           "Employee full name",
                           [f"{e['first_name']} {e['last_name']}" for e in emp]),
    })

    # 3.4 Trim/normalize address
    cases.append({
        "cat": "String", "title": "address_line1 -> clean_address (trim)",
        "expected_cols": "src_employees.address_line1", "expected_family": "trim",
        "spec": TargetSpec("dim_employee", "clean_address", ColumnType("string"), ColumnConstraints(),
                           ColumnStats([e["address_line1"].strip() for e in emp]),
                           "Cleaned address text",
                           [e["address_line1"].strip() for e in emp]),
    })

    # ---------------------------------------------------------------
    # CATEGORY 4: Numeric transformations
    # ---------------------------------------------------------------

    # 4.1 Round salary
    cases.append({
        "cat": "Numeric", "title": "salary_amount -> salary_rounded (round)",
        "expected_cols": "src_employees.salary_amount", "expected_family": "round",
        "spec": TargetSpec("dim_employee", "salary_rounded", ColumnType("decimal"), ColumnConstraints(),
                           ColumnStats([round(e["salary_amount"], -3) for e in emp]),
                           "Salary rounded to nearest thousand"),
    })

    # 4.2 Arithmetic: budget_usd / 12 = monthly budget (single col scale)
    cases.append({
        "cat": "Numeric", "title": "budget_usd -> monthly_budget (scale /12)",
        "expected_cols": "src_projects.budget_usd", "expected_family": "scale",
        "spec": TargetSpec("dim_project", "monthly_budget", ColumnType("decimal"), ColumnConstraints(),
                           ColumnStats([round(p["budget_usd"] / 12, 2) for p in proj]),
                           "Monthly project budget"),
    })

    # 4.3 Identity: hours_worked direct copy
    cases.append({
        "cat": "Numeric", "title": "hours_worked -> total_hours (identity)",
        "expected_cols": "src_timesheets.hours_worked", "expected_family": "identity",
        "spec": TargetSpec("fct_timesheet", "total_hours", ColumnType("decimal"), ColumnConstraints(),
                           ColumnStats([t["hours_worked"] for t in ts]),
                           "Hours worked copied"),
    })

    # ---------------------------------------------------------------
    # CATEGORY 5: Semantic / tricky name matching
    # ---------------------------------------------------------------

    # 5.1 "emp_tenure_years" should find hire_date (semantic: tenure = years since hire)
    import datetime
    cases.append({
        "cat": "Semantic", "title": "hire_date -> emp_tenure_years (semantic date->int)",
        "expected_cols": "src_employees.hire_date", "expected_family": "date_part",
        "spec": TargetSpec("dim_employee", "emp_tenure_years", ColumnType("int"), ColumnConstraints(),
                           ColumnStats([2025 - int(e["hire_date"][:4]) for e in emp]),
                           "Years since employee was hired",
                           [2025 - int(e["hire_date"][:4]) for e in emp]),
    })

    # 5.2 "dept_location" should find location_city via department
    cases.append({
        "cat": "Semantic", "title": "department -> dept_location (semantic lookup)",
        "expected_cols": "dim_department.location_city", "expected_family": "identity",
        "spec": TargetSpec("dim_employee", "dept_location", ColumnType("string"), ColumnConstraints(),
                           ColumnStats([dept_map[e["department_id"]]["location_city"] for e in emp]),
                           "Location of employee department"),
    })

    # 5.3 "proj_duration_days" should find start_date + end_date (date diff)
    cases.append({
        "cat": "Semantic", "title": "start_date + end_date -> proj_duration_days (date_diff)",
        "expected_cols": "src_projects.start_date, src_projects.end_date",
        "expected_family": "date_diff",
        "spec": TargetSpec("dim_project", "proj_duration_days", ColumnType("int"), ColumnConstraints(),
                           ColumnStats([365, 214, 455, 366]),  # approx durations
                           "Project duration in days"),
    })

    # 5.4 Unusual abbreviation: "empl_dob" should find birth_date
    cases.append({
        "cat": "Semantic", "title": "birth_date -> empl_dob (abbreviation)",
        "expected_cols": "src_employees.birth_date", "expected_family": "identity",
        "spec": TargetSpec("dim_employee", "empl_dob", ColumnType("date"), ColumnConstraints(),
                           ColumnStats([e["birth_date"] for e in emp]),
                           "Employee date of birth"),
    })

    # ---------------------------------------------------------------
    # CATEGORY 6: Edge cases
    # ---------------------------------------------------------------

    # 6.1 Column with same name in multiple tables: department_id
    cases.append({
        "cat": "Edge Case", "title": "department_id -> dept_key (ambiguous multi-table)",
        "expected_cols": "src_employees.department_id", "expected_family": "identity",
        "spec": TargetSpec("fct_employee", "dept_key", ColumnType("int"), ColumnConstraints(),
                           ColumnStats([e["department_id"] for e in emp]),
                           "Department key for employee fact"),
    })

    # 6.2 project_name direct copy (identity, should not pick project_id)
    cases.append({
        "cat": "Edge Case", "title": "project_name -> project_title (identity rename)",
        "expected_cols": "src_projects.project_name", "expected_family": "identity",
        "spec": TargetSpec("dim_project", "project_title", ColumnType("string"), ColumnConstraints(),
                           ColumnStats([p["project_name"] for p in proj]),
                           "Project title (renamed from project_name)",
                           [p["project_name"] for p in proj]),
    })

    # 6.3 status -> is_active_project (string -> bool)
    cases.append({
        "cat": "Edge Case", "title": "status -> is_active_project (str->bool flag)",
        "expected_cols": "src_projects.status", "expected_family": "conditional",
        "spec": TargetSpec("dim_project", "is_active_project", ColumnType("boolean"), ColumnConstraints(),
                           ColumnStats([p["status"] == "active" for p in proj]),
                           "Whether project is currently active",
                           [p["status"] == "active" for p in proj]),
    })

    return cases


# ===================================================================
# Load Stage B
# ===================================================================

def load_stage_b(artifacts_root):
    import json as _json
    meta_path = artifacts_root / "training_metadata.json"
    bi_dir = artifacts_root / "biencoder"
    cross_dir = artifacts_root / "cross_encoder"

    bi_base = "sentence-transformers/all-MiniLM-L6-v2"
    cross_base = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    if meta_path.exists():
        meta = _json.loads(meta_path.read_text(encoding="utf-8"))
        bi_base = meta.get("bi_model_name", bi_base)
        cross_base = meta.get("cross_model_name", cross_base)

    bi_has = any((bi_dir / f).exists() for f in ["model.safetensors", "pytorch_model.bin"])
    cross_has = any((cross_dir / f).exists() for f in ["model.safetensors", "pytorch_model.bin"])
    bi_path = str(bi_dir) if bi_has else bi_base
    cross_path = str(cross_dir) if cross_has else cross_base

    return CandidateSelectorStage1(biencoder_path=bi_path, cross_encoder_path=cross_path)


# ===================================================================
# Main
# ===================================================================

def main():
    print(f"\n{SEP}")
    print(f"  ROBUSTNESS TEST SUITE  -  Stage A + Stage B Pipeline")
    print(f"{SEP}")

    # Build schema
    tables, edges, emp, dept, proj, ts, loc = build_schema()
    print(f"\n  Schema: {len(tables)} tables, {len(edges)} join edges")
    for tn, t in tables.items():
        print(f"    {tn}: {len(t.columns)} cols, {t.row_count} rows")

    # Init engines - use SentenceTransformer for semantic understanding if available
    try:
        embedder = NameEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print(f"  SemanticEngine: using SentenceTransformer (embedding-based concept matching)")
    except Exception:
        embedder = NameEmbedder(model_name=None)
        print(f"  SemanticEngine: using hashed fallback (no SentenceTransformer)")
    engine = CandidateGenerationEngine(source_tables=tables, join_edges=edges, embedder=embedder)
    print(f"  Stage A: {sum(len(v) for v in engine._table_to_ids.values())} indexed columns")

    selector = load_stage_b(Path("artifacts/stage1_candidate_selector"))
    print(f"  Stage B: loaded\n")

    # Build test cases
    cases = build_test_cases(emp, dept, proj, ts, loc)
    print(f"  Test cases: {len(cases)}")

    # Group by category
    by_cat = defaultdict(list)
    for c in cases:
        by_cat[c["cat"]].append(c)

    results = []
    total_a_time = 0.0
    total_b_time = 0.0

    for cat_name, cat_cases in by_cat.items():
        print(f"\n{SEP}")
        print(f"  CATEGORY: {cat_name} ({len(cat_cases)} tests)")
        print(SEP)

        for tc in cat_cases:
            title = tc["title"]
            expected = tc["expected_cols"]
            expected_fam = tc["expected_family"]
            target = tc["spec"]

            # Stage A
            t0 = time.time()
            sa = engine.rank_candidates(target=target, coarse_top_m=50, fine_top_m=25,
                                        max_arity=4, max_hops=3, top_k=8, abstain_threshold=0.45)
            a_time = time.time() - t0
            total_a_time += a_time

            # Bridge
            target_col, cand_sets = bridge(target, sa)

            # Stage B
            t1 = time.time()
            sb = selector.rank(target=target_col, candidate_sets=cand_sets, retrieval_k=50, top_k=8)
            b_time = time.time() - t1
            total_b_time += b_time

            # Evaluate
            cand_map = {cs.id: cs for cs in cand_sets}
            a_top1_cols = ", ".join(sa["top_candidates"][0]["candidate_columns"]) if sa["top_candidates"] else "NONE"
            a_top1_fam = sa["top_candidates"][0]["best_transform_family"] if sa["top_candidates"] else "NONE"
            a_conf = sa["top_candidates"][0]["confidence"] if sa["top_candidates"] else 0

            if sb:
                b_cs = cand_map.get(sb[0]["candidate_id"])
                b_top1_cols = ", ".join(f"{sc.table}.{sc.column}" for sc in b_cs.columns) if b_cs else "NONE"
                b_top1_fam = b_cs.transform_hint if b_cs else "NONE"
                b_comb = sb[0].get("combined_score", 0)
            else:
                b_top1_cols = "NONE"
                b_top1_fam = "NONE"
                b_comb = 0

            # Check correctness (flexible: expected should be contained in top1)
            a_correct = all(ec.strip() in a_top1_cols for ec in expected.split(","))
            b_correct = all(ec.strip() in b_top1_cols for ec in expected.split(","))

            # Also check in top-3 for recall
            a_top3_all = set()
            for c in sa["top_candidates"][:3]:
                for col in c["candidate_columns"]:
                    a_top3_all.add(col)
            a_in_top3 = all(ec.strip() in a_top3_all for ec in expected.split(","))

            b_top3_all = set()
            for r in sb[:3]:
                cs = cand_map.get(r["candidate_id"])
                if cs:
                    for sc in cs.columns:
                        b_top3_all.add(f"{sc.table}.{sc.column}")
            b_in_top3 = all(ec.strip() in b_top3_all for ec in expected.split(","))

            status_a = "PASS" if a_correct else ("TOP3" if a_in_top3 else "FAIL")
            status_b = "PASS" if b_correct else ("TOP3" if b_in_top3 else "FAIL")

            print(f"\n  {SUB}")
            print(f"  {title}")
            print(f"    Expected: {expected} [{expected_fam}]")
            print(f"    Stage A [{status_a}] conf={a_conf:.3f}: {a_top1_cols} [{a_top1_fam}]")
            if not a_correct and a_in_top3:
                # Show top-3
                for i, c in enumerate(sa["top_candidates"][:3], 1):
                    print(f"      A-{i}: {', '.join(c['candidate_columns'])} [{c['best_transform_family']}] conf={c['confidence']:.3f}")
            print(f"    Stage B [{status_b}] comb={b_comb:.3f}: {b_top1_cols} [{b_top1_fam}]")
            if not b_correct and b_in_top3:
                for r in sb[:3]:
                    cs = cand_map.get(r["candidate_id"])
                    cols = ", ".join(f"{sc.table}.{sc.column}" for sc in cs.columns) if cs else "?"
                    print(f"      B-{r['rank']}: {cols} [{cs.transform_hint if cs else '?'}] comb={r.get('combined_score',0):.3f}")

            results.append({
                "cat": tc["cat"], "title": title,
                "a_status": status_a, "b_status": status_b,
                "a_conf": a_conf, "b_comb": b_comb,
                "a_time": a_time, "b_time": b_time,
            })

    # Summary
    print(f"\n\n{SEP}")
    print(f"  ROBUSTNESS TEST SUMMARY")
    print(SEP)

    total = len(results)
    a_pass = sum(1 for r in results if r["a_status"] == "PASS")
    a_top3 = sum(1 for r in results if r["a_status"] in ("PASS", "TOP3"))
    b_pass = sum(1 for r in results if r["b_status"] == "PASS")
    b_top3 = sum(1 for r in results if r["b_status"] in ("PASS", "TOP3"))

    # Per-category breakdown
    print(f"\n  {'Category':<20} {'Tests':>5} {'A-Top1':>7} {'A-Top3':>7} {'B-Top1':>7} {'B-Top3':>7}")
    print(f"  {'='*20} {'='*5} {'='*7} {'='*7} {'='*7} {'='*7}")
    for cat in by_cat:
        cat_results = [r for r in results if r["cat"] == cat]
        ct = len(cat_results)
        ca1 = sum(1 for r in cat_results if r["a_status"] == "PASS")
        ca3 = sum(1 for r in cat_results if r["a_status"] in ("PASS", "TOP3"))
        cb1 = sum(1 for r in cat_results if r["b_status"] == "PASS")
        cb3 = sum(1 for r in cat_results if r["b_status"] in ("PASS", "TOP3"))
        print(f"  {cat:<20} {ct:>5} {ca1:>4}/{ct:<2} {ca3:>4}/{ct:<2} {cb1:>4}/{ct:<2} {cb3:>4}/{ct:<2}")

    print(f"\n  {'OVERALL':<20} {total:>5} {a_pass:>4}/{total:<2} {a_top3:>4}/{total:<2} {b_pass:>4}/{total:<2} {b_top3:>4}/{total:<2}")

    print(f"\n  Stage A: Top-1 accuracy = {a_pass}/{total} ({a_pass/total*100:.0f}%), "
          f"Top-3 recall = {a_top3}/{total} ({a_top3/total*100:.0f}%)")
    print(f"  Stage B: Top-1 accuracy = {b_pass}/{total} ({b_pass/total*100:.0f}%), "
          f"Top-3 recall = {b_top3}/{total} ({b_top3/total*100:.0f}%)")

    print(f"\n  Avg confidence - Stage A: {sum(r['a_conf'] for r in results)/total:.3f}")
    print(f"  Avg combined   - Stage B: {sum(r['b_comb'] for r in results)/total:.3f}")
    print(f"\n  Total time: Stage A = {total_a_time:.1f}s, Stage B = {total_b_time:.1f}s")

    # Detailed per-test status
    print(f"\n  {'#':<3} {'Status':<10} {'Title':<55} {'A-Conf':>7} {'B-Comb':>7}")
    print(f"  {'='*3} {'='*10} {'='*55} {'='*7} {'='*7}")
    for i, r in enumerate(results, 1):
        combined_status = "PASS" if r["a_status"] == "PASS" and r["b_status"] == "PASS" else \
                         "PARTIAL" if r["a_status"] == "PASS" or r["b_status"] == "PASS" else \
                         "TOP3" if r["a_status"] == "TOP3" or r["b_status"] == "TOP3" else "FAIL"
        print(f"  {i:<3} {combined_status:<10} {r['title']:<55} {r['a_conf']:>7.3f} {r['b_comb']:>7.3f}")

    print(f"\n{SEP}")
    print(f"  ROBUSTNESS TEST COMPLETE")
    print(SEP)


if __name__ == "__main__":
    main()
