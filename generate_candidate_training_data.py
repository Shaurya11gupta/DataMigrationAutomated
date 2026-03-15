#!/usr/bin/env python3
"""
Generate Training Data for Bi-Encoder + Cross-Encoder Candidate Generation
============================================================================
Creates rich (query, positive, negatives) triplets from 12+ domain schemas.

Each training record:
  - query:     serialized target column text
  - positive:  serialized correct source candidate text
  - negatives: list of serialized wrong candidates

Output: JSONL files for bi-encoder (triplet) and cross-encoder (pairs).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ───────────────────────────────────────────────────────
# Serialization (shared with the inference engine)
# ───────────────────────────────────────────────────────

_SPLIT_RE = re.compile(r"[_\-.\s]+|(?<=[a-z])(?=[A-Z])")

def tokenize_name(name: str) -> str:
    """Convert column name to readable tokens: emp_first_name -> 'emp first name'."""
    return " ".join(t.lower() for t in _SPLIT_RE.split(name) if t)


def serialize_target(
    table: str, column: str, col_type: str,
    description: str = "", constraints: str = "",
) -> str:
    """Serialize a target column spec as rich text for the encoder."""
    parts = [
        f"target: {table}.{column}",
        f"[{col_type}]",
        f"meaning: {tokenize_name(column)}",
    ]
    if description:
        parts.append(f"desc: {description}")
    if constraints:
        parts.append(f"constraints: {constraints}")
    return " | ".join(parts)


def serialize_candidate(
    columns: List[Tuple[str, str, str]],  # (table, column, type)
    join_path: List[str] = None,
    transform_hint: str = "",
    table_context: str = "",
) -> str:
    """
    Serialize a source candidate set as rich text.
    columns: list of (table, column, type)
    """
    col_parts = []
    for tbl, col, ctype in columns:
        col_parts.append(f"{tbl}.{col} [{ctype}]")

    parts = [f"source: {', '.join(col_parts)}"]

    arity = len(columns)
    parts.append(f"arity: {arity}")

    # Table membership
    tables = list(dict.fromkeys(c[0] for c in columns))
    if len(tables) == 1:
        parts.append("same_table")
    else:
        parts.append(f"cross_table: {' + '.join(tables)}")

    # Column name meanings
    meanings = [tokenize_name(c[1]) for c in columns]
    parts.append(f"col_meanings: {'; '.join(meanings)}")

    if join_path:
        parts.append(f"join: {' -> '.join(join_path)}")

    if transform_hint:
        parts.append(f"transform: {transform_hint}")

    if table_context:
        parts.append(f"context: {table_context}")

    return " | ".join(parts)


# ───────────────────────────────────────────────────────
# Domain schema definitions
# ───────────────────────────────────────────────────────

@dataclass
class ColDef:
    name: str
    type: str
    is_pk: bool = False
    is_fk: str = ""  # "table.col" if FK
    desc: str = ""

@dataclass
class TableDef:
    name: str
    columns: List[ColDef]

@dataclass
class MappingDef:
    """One target column and its correct source mapping."""
    target_table: str
    target_col: str
    target_type: str
    target_desc: str
    source_cols: List[Tuple[str, str, str]]  # (table, col, type)
    transform: str
    join_path: List[str] = field(default_factory=list)
    sub_operation: str = ""  # Fine-grained operation (e.g. "subtract", "extract_year")

@dataclass
class DomainSchema:
    name: str
    source_tables: List[TableDef]
    mappings: List[MappingDef]


def _build_domains() -> List[DomainSchema]:
    """Build 12+ domain schemas with diverse mapping patterns."""
    domains = []

    # ─── Domain 1: HR ───
    domains.append(DomainSchema(
        name="hr",
        source_tables=[
            TableDef("employees", [
                ColDef("emp_id", "int", is_pk=True),
                ColDef("first_name", "string"),
                ColDef("last_name", "string"),
                ColDef("middle_name", "string"),
                ColDef("email", "string"),
                ColDef("phone", "string"),
                ColDef("hire_date", "date"),
                ColDef("birth_date", "date"),
                ColDef("salary", "decimal"),
                ColDef("bonus_pct", "decimal"),
                ColDef("department_id", "int", is_fk="departments.dept_id"),
                ColDef("manager_id", "int", is_fk="employees.emp_id"),
                ColDef("position_id", "int", is_fk="positions.position_id"),
                ColDef("status", "string"),
                ColDef("gender", "string"),
            ]),
            TableDef("departments", [
                ColDef("dept_id", "int", is_pk=True),
                ColDef("dept_name", "string"),
                ColDef("dept_code", "string"),
                ColDef("location", "string"),
                ColDef("budget", "decimal"),
                ColDef("head_emp_id", "int", is_fk="employees.emp_id"),
            ]),
            TableDef("positions", [
                ColDef("position_id", "int", is_pk=True),
                ColDef("title", "string"),
                ColDef("level", "int"),
                ColDef("min_salary", "decimal"),
                ColDef("max_salary", "decimal"),
            ]),
            TableDef("salary_history", [
                ColDef("history_id", "int", is_pk=True),
                ColDef("emp_id", "int", is_fk="employees.emp_id"),
                ColDef("effective_date", "date"),
                ColDef("old_salary", "decimal"),
                ColDef("new_salary", "decimal"),
                ColDef("reason", "string"),
            ]),
        ],
        mappings=[
            MappingDef("dim_employee", "employee_key", "int", "surrogate key",
                       [("employees", "emp_id", "int")], "rename"),
            MappingDef("dim_employee", "full_name", "string", "employee full name first and last",
                       [("employees", "first_name", "string"), ("employees", "last_name", "string")], "concat"),
            MappingDef("dim_employee", "complete_name", "string", "full name with middle",
                       [("employees", "first_name", "string"), ("employees", "middle_name", "string"), ("employees", "last_name", "string")], "concat"),
            MappingDef("dim_employee", "email_address", "string", "employee corporate email",
                       [("employees", "email", "string")], "rename"),
            MappingDef("dim_employee", "hire_year", "int", "year employee was hired",
                       [("employees", "hire_date", "date")], "date_part"),
            MappingDef("dim_employee", "hire_month", "int", "month employee was hired",
                       [("employees", "hire_date", "date")], "date_part"),
            MappingDef("dim_employee", "tenure_days", "int", "days since hire",
                       [("employees", "hire_date", "date")], "date_diff"),
            MappingDef("dim_employee", "age", "int", "employee age in years",
                       [("employees", "birth_date", "date")], "date_diff"),
            MappingDef("dim_employee", "annual_salary", "decimal", "yearly salary",
                       [("employees", "salary", "decimal")], "rename"),
            MappingDef("dim_employee", "total_compensation", "decimal", "salary plus bonus",
                       [("employees", "salary", "decimal"), ("employees", "bonus_pct", "decimal")], "arithmetic"),
            MappingDef("dim_employee", "department_name", "string", "name of the department",
                       [("departments", "dept_name", "string")], "fk_lookup",
                       join_path=["employees.department_id = departments.dept_id"]),
            MappingDef("dim_employee", "position_title", "string", "job title",
                       [("positions", "title", "string")], "fk_lookup",
                       join_path=["employees.position_id = positions.position_id"]),
            MappingDef("dim_employee", "manager_name", "string", "name of the manager",
                       [("employees", "first_name", "string"), ("employees", "last_name", "string")], "concat",
                       join_path=["employees.manager_id = employees.emp_id"]),
            MappingDef("dim_employee", "is_active", "boolean", "whether employee is currently active",
                       [("employees", "status", "string")], "conditional"),
            MappingDef("dim_employee", "salary_band", "string", "salary level band",
                       [("employees", "salary", "decimal")], "bucketing"),
            MappingDef("dim_employee", "phone_number", "string", "contact phone",
                       [("employees", "phone", "string")], "rename"),
            MappingDef("dim_employee", "gender_label", "string", "gender description",
                       [("employees", "gender", "string")], "code_to_label"),
        ],
    ))

    # ─── Domain 2: E-commerce ───
    domains.append(DomainSchema(
        name="ecommerce",
        source_tables=[
            TableDef("customers", [
                ColDef("customer_id", "int", is_pk=True),
                ColDef("first_name", "string"),
                ColDef("last_name", "string"),
                ColDef("email", "string"),
                ColDef("phone", "string"),
                ColDef("registration_date", "date"),
                ColDef("country", "string"),
                ColDef("city", "string"),
                ColDef("postal_code", "string"),
                ColDef("street_address", "string"),
                ColDef("loyalty_points", "int"),
                ColDef("tier", "string"),
            ]),
            TableDef("orders", [
                ColDef("order_id", "int", is_pk=True),
                ColDef("customer_id", "int", is_fk="customers.customer_id"),
                ColDef("order_date", "date"),
                ColDef("ship_date", "date"),
                ColDef("total_amount", "decimal"),
                ColDef("discount_amount", "decimal"),
                ColDef("tax_amount", "decimal"),
                ColDef("status", "string"),
                ColDef("payment_method", "string"),
                ColDef("shipping_method", "string"),
                ColDef("currency_code", "string"),
            ]),
            TableDef("order_items", [
                ColDef("item_id", "int", is_pk=True),
                ColDef("order_id", "int", is_fk="orders.order_id"),
                ColDef("product_id", "int", is_fk="products.product_id"),
                ColDef("quantity", "int"),
                ColDef("unit_price", "decimal"),
                ColDef("discount_pct", "decimal"),
            ]),
            TableDef("products", [
                ColDef("product_id", "int", is_pk=True),
                ColDef("product_name", "string"),
                ColDef("sku", "string"),
                ColDef("category_id", "int", is_fk="categories.category_id"),
                ColDef("brand", "string"),
                ColDef("unit_cost", "decimal"),
                ColDef("list_price", "decimal"),
                ColDef("weight_kg", "decimal"),
                ColDef("description", "string"),
            ]),
            TableDef("categories", [
                ColDef("category_id", "int", is_pk=True),
                ColDef("category_name", "string"),
                ColDef("parent_category_id", "int"),
            ]),
        ],
        mappings=[
            MappingDef("dim_customer", "customer_key", "int", "surrogate key",
                       [("customers", "customer_id", "int")], "rename"),
            MappingDef("dim_customer", "full_name", "string", "customer full name",
                       [("customers", "first_name", "string"), ("customers", "last_name", "string")], "concat"),
            MappingDef("dim_customer", "email_address", "string", "customer email",
                       [("customers", "email", "string")], "rename"),
            MappingDef("dim_customer", "registration_year", "int", "year of registration",
                       [("customers", "registration_date", "date")], "date_part"),
            MappingDef("dim_customer", "full_address", "string", "complete mailing address",
                       [("customers", "street_address", "string"), ("customers", "city", "string"),
                        ("customers", "country", "string")], "concat"),
            MappingDef("dim_customer", "customer_tier", "string", "loyalty tier label",
                       [("customers", "tier", "string")], "rename"),
            MappingDef("dim_customer", "loyalty_score", "int", "accumulated loyalty points",
                       [("customers", "loyalty_points", "int")], "rename"),
            MappingDef("fact_order", "order_key", "int", "order surrogate key",
                       [("orders", "order_id", "int")], "rename"),
            MappingDef("fact_order", "customer_name", "string", "name of ordering customer",
                       [("customers", "first_name", "string"), ("customers", "last_name", "string")], "concat",
                       join_path=["orders.customer_id = customers.customer_id"]),
            MappingDef("fact_order", "order_year", "int", "year order was placed",
                       [("orders", "order_date", "date")], "date_part"),
            MappingDef("fact_order", "order_month", "int", "month order was placed",
                       [("orders", "order_date", "date")], "date_part"),
            MappingDef("fact_order", "days_to_ship", "int", "days between order and shipment",
                       [("orders", "order_date", "date"), ("orders", "ship_date", "date")], "date_diff"),
            MappingDef("fact_order", "net_amount", "decimal", "total minus discount",
                       [("orders", "total_amount", "decimal"), ("orders", "discount_amount", "decimal")], "arithmetic"),
            MappingDef("fact_order", "gross_amount", "decimal", "total including tax",
                       [("orders", "total_amount", "decimal"), ("orders", "tax_amount", "decimal")], "arithmetic"),
            MappingDef("fact_order", "payment_type", "string", "payment method used",
                       [("orders", "payment_method", "string")], "rename"),
            MappingDef("fact_order", "is_completed", "boolean", "whether order is completed",
                       [("orders", "status", "string")], "conditional"),
            MappingDef("fact_order", "product_name", "string", "name of the ordered product",
                       [("products", "product_name", "string")], "fk_lookup",
                       join_path=["order_items.product_id = products.product_id"]),
            MappingDef("fact_order", "category_name", "string", "product category",
                       [("categories", "category_name", "string")], "fk_lookup",
                       join_path=["products.category_id = categories.category_id"]),
            MappingDef("fact_order", "item_revenue", "decimal", "quantity times unit price",
                       [("order_items", "quantity", "int"), ("order_items", "unit_price", "decimal")], "arithmetic"),
            MappingDef("fact_order", "profit_margin", "decimal", "price minus cost per unit",
                       [("products", "list_price", "decimal"), ("products", "unit_cost", "decimal")], "arithmetic"),
        ],
    ))

    # ─── Domain 3: Finance ───
    domains.append(DomainSchema(
        name="finance",
        source_tables=[
            TableDef("accounts", [
                ColDef("account_id", "int", is_pk=True),
                ColDef("account_number", "string"),
                ColDef("account_name", "string"),
                ColDef("account_type", "string"),
                ColDef("currency", "string"),
                ColDef("balance", "decimal"),
                ColDef("opened_date", "date"),
                ColDef("closed_date", "date"),
                ColDef("branch_id", "int", is_fk="branches.branch_id"),
                ColDef("customer_id", "int", is_fk="customers.customer_id"),
                ColDef("status", "string"),
                ColDef("interest_rate", "decimal"),
            ]),
            TableDef("transactions", [
                ColDef("txn_id", "int", is_pk=True),
                ColDef("account_id", "int", is_fk="accounts.account_id"),
                ColDef("txn_date", "date"),
                ColDef("txn_time", "string"),
                ColDef("amount", "decimal"),
                ColDef("txn_type", "string"),
                ColDef("description", "string"),
                ColDef("counterparty", "string"),
                ColDef("reference_no", "string"),
                ColDef("channel", "string"),
            ]),
            TableDef("branches", [
                ColDef("branch_id", "int", is_pk=True),
                ColDef("branch_name", "string"),
                ColDef("branch_code", "string"),
                ColDef("city", "string"),
                ColDef("region", "string"),
                ColDef("manager_name", "string"),
            ]),
            TableDef("customers", [
                ColDef("customer_id", "int", is_pk=True),
                ColDef("first_name", "string"),
                ColDef("last_name", "string"),
                ColDef("date_of_birth", "date"),
                ColDef("ssn_hash", "string"),
                ColDef("credit_score", "int"),
                ColDef("annual_income", "decimal"),
                ColDef("risk_rating", "string"),
            ]),
        ],
        mappings=[
            MappingDef("dim_account", "account_key", "int", "account surrogate key",
                       [("accounts", "account_id", "int")], "rename"),
            MappingDef("dim_account", "account_number", "string", "public account number",
                       [("accounts", "account_number", "string")], "direct_copy"),
            MappingDef("dim_account", "account_type_label", "string", "readable account type",
                       [("accounts", "account_type", "string")], "code_to_label"),
            MappingDef("dim_account", "opening_year", "int", "year account opened",
                       [("accounts", "opened_date", "date")], "date_part"),
            MappingDef("dim_account", "account_age_days", "int", "days since opening",
                       [("accounts", "opened_date", "date")], "date_diff"),
            MappingDef("dim_account", "branch_name", "string", "branch where account resides",
                       [("branches", "branch_name", "string")], "fk_lookup",
                       join_path=["accounts.branch_id = branches.branch_id"]),
            MappingDef("dim_account", "branch_city", "string", "city of the branch",
                       [("branches", "city", "string")], "fk_lookup",
                       join_path=["accounts.branch_id = branches.branch_id"]),
            MappingDef("dim_account", "customer_full_name", "string", "name of account holder",
                       [("customers", "first_name", "string"), ("customers", "last_name", "string")], "concat",
                       join_path=["accounts.customer_id = customers.customer_id"]),
            MappingDef("dim_account", "customer_age", "int", "age of customer",
                       [("customers", "date_of_birth", "date")], "date_diff",
                       join_path=["accounts.customer_id = customers.customer_id"]),
            MappingDef("dim_account", "is_active", "boolean", "whether account is open",
                       [("accounts", "status", "string")], "conditional"),
            MappingDef("fact_transaction", "transaction_key", "int", "txn surrogate key",
                       [("transactions", "txn_id", "int")], "rename"),
            MappingDef("fact_transaction", "transaction_date", "date", "date of transaction",
                       [("transactions", "txn_date", "date")], "direct_copy"),
            MappingDef("fact_transaction", "transaction_year", "int", "year of transaction",
                       [("transactions", "txn_date", "date")], "date_part"),
            MappingDef("fact_transaction", "transaction_amount", "decimal", "monetary amount",
                       [("transactions", "amount", "decimal")], "rename"),
            MappingDef("fact_transaction", "transaction_type", "string", "type of transaction",
                       [("transactions", "txn_type", "string")], "rename"),
            MappingDef("fact_transaction", "account_name", "string", "name of the related account",
                       [("accounts", "account_name", "string")], "fk_lookup",
                       join_path=["transactions.account_id = accounts.account_id"]),
            MappingDef("fact_transaction", "account_holder", "string", "full name of account holder",
                       [("customers", "first_name", "string"), ("customers", "last_name", "string")], "concat",
                       join_path=["transactions.account_id = accounts.account_id",
                                  "accounts.customer_id = customers.customer_id"]),
        ],
    ))

    # ─── Domain 4: Healthcare ───
    domains.append(DomainSchema(
        name="healthcare",
        source_tables=[
            TableDef("patients", [
                ColDef("patient_id", "int", is_pk=True),
                ColDef("first_name", "string"),
                ColDef("last_name", "string"),
                ColDef("dob", "date"),
                ColDef("gender", "string"),
                ColDef("blood_type", "string"),
                ColDef("phone_number", "string"),
                ColDef("email", "string"),
                ColDef("address", "string"),
                ColDef("insurance_id", "int", is_fk="insurance.insurance_id"),
                ColDef("primary_doctor_id", "int", is_fk="doctors.doctor_id"),
            ]),
            TableDef("doctors", [
                ColDef("doctor_id", "int", is_pk=True),
                ColDef("first_name", "string"),
                ColDef("last_name", "string"),
                ColDef("specialization", "string"),
                ColDef("license_number", "string"),
                ColDef("department_id", "int", is_fk="hospital_departments.dept_id"),
            ]),
            TableDef("appointments", [
                ColDef("appointment_id", "int", is_pk=True),
                ColDef("patient_id", "int", is_fk="patients.patient_id"),
                ColDef("doctor_id", "int", is_fk="doctors.doctor_id"),
                ColDef("appointment_date", "date"),
                ColDef("appointment_time", "string"),
                ColDef("duration_minutes", "int"),
                ColDef("reason", "string"),
                ColDef("status", "string"),
                ColDef("notes", "string"),
                ColDef("fee", "decimal"),
            ]),
            TableDef("insurance", [
                ColDef("insurance_id", "int", is_pk=True),
                ColDef("provider_name", "string"),
                ColDef("plan_name", "string"),
                ColDef("coverage_type", "string"),
                ColDef("expiry_date", "date"),
            ]),
            TableDef("hospital_departments", [
                ColDef("dept_id", "int", is_pk=True),
                ColDef("dept_name", "string"),
                ColDef("floor_number", "int"),
                ColDef("wing", "string"),
            ]),
        ],
        mappings=[
            MappingDef("dim_patient", "patient_key", "int", "patient surrogate key",
                       [("patients", "patient_id", "int")], "rename"),
            MappingDef("dim_patient", "patient_name", "string", "patient full name",
                       [("patients", "first_name", "string"), ("patients", "last_name", "string")], "concat"),
            MappingDef("dim_patient", "date_of_birth", "date", "patient birth date",
                       [("patients", "dob", "date")], "rename"),
            MappingDef("dim_patient", "birth_year", "int", "year of birth",
                       [("patients", "dob", "date")], "date_part"),
            MappingDef("dim_patient", "patient_age", "int", "current age",
                       [("patients", "dob", "date")], "date_diff"),
            MappingDef("dim_patient", "blood_group", "string", "blood type",
                       [("patients", "blood_type", "string")], "rename"),
            MappingDef("dim_patient", "insurance_provider", "string", "insurance company name",
                       [("insurance", "provider_name", "string")], "fk_lookup",
                       join_path=["patients.insurance_id = insurance.insurance_id"]),
            MappingDef("dim_patient", "insurance_plan", "string", "insurance plan details",
                       [("insurance", "plan_name", "string")], "fk_lookup",
                       join_path=["patients.insurance_id = insurance.insurance_id"]),
            MappingDef("dim_patient", "primary_doctor_name", "string", "assigned doctor full name",
                       [("doctors", "first_name", "string"), ("doctors", "last_name", "string")], "concat",
                       join_path=["patients.primary_doctor_id = doctors.doctor_id"]),
            MappingDef("dim_patient", "doctor_specialty", "string", "primary doctor specialization",
                       [("doctors", "specialization", "string")], "fk_lookup",
                       join_path=["patients.primary_doctor_id = doctors.doctor_id"]),
            MappingDef("fact_appointment", "appointment_key", "int", "appointment id",
                       [("appointments", "appointment_id", "int")], "rename"),
            MappingDef("fact_appointment", "visit_date", "date", "date of the visit",
                       [("appointments", "appointment_date", "date")], "rename"),
            MappingDef("fact_appointment", "visit_year", "int", "year of visit",
                       [("appointments", "appointment_date", "date")], "date_part"),
            MappingDef("fact_appointment", "patient_full_name", "string", "patient name for the visit",
                       [("patients", "first_name", "string"), ("patients", "last_name", "string")], "concat",
                       join_path=["appointments.patient_id = patients.patient_id"]),
            MappingDef("fact_appointment", "attending_doctor", "string", "doctor name",
                       [("doctors", "first_name", "string"), ("doctors", "last_name", "string")], "concat",
                       join_path=["appointments.doctor_id = doctors.doctor_id"]),
            MappingDef("fact_appointment", "visit_reason", "string", "reason for appointment",
                       [("appointments", "reason", "string")], "rename"),
            MappingDef("fact_appointment", "consultation_fee", "decimal", "fee charged",
                       [("appointments", "fee", "decimal")], "rename"),
            MappingDef("fact_appointment", "is_completed", "boolean", "whether appointment was completed",
                       [("appointments", "status", "string")], "conditional"),
        ],
    ))

    # ─── Domain 5: Education ───
    domains.append(DomainSchema(
        name="education",
        source_tables=[
            TableDef("students", [
                ColDef("student_id", "int", is_pk=True),
                ColDef("first_name", "string"),
                ColDef("last_name", "string"),
                ColDef("email", "string"),
                ColDef("enrollment_date", "date"),
                ColDef("graduation_date", "date"),
                ColDef("gpa", "decimal"),
                ColDef("major_id", "int", is_fk="majors.major_id"),
                ColDef("advisor_id", "int", is_fk="faculty.faculty_id"),
            ]),
            TableDef("courses", [
                ColDef("course_id", "int", is_pk=True),
                ColDef("course_code", "string"),
                ColDef("course_name", "string"),
                ColDef("credits", "int"),
                ColDef("department_id", "int", is_fk="academic_departments.dept_id"),
            ]),
            TableDef("enrollments", [
                ColDef("enrollment_id", "int", is_pk=True),
                ColDef("student_id", "int", is_fk="students.student_id"),
                ColDef("course_id", "int", is_fk="courses.course_id"),
                ColDef("semester", "string"),
                ColDef("grade", "string"),
                ColDef("score", "decimal"),
            ]),
            TableDef("faculty", [
                ColDef("faculty_id", "int", is_pk=True),
                ColDef("first_name", "string"),
                ColDef("last_name", "string"),
                ColDef("title", "string"),
                ColDef("department_id", "int", is_fk="academic_departments.dept_id"),
            ]),
            TableDef("majors", [
                ColDef("major_id", "int", is_pk=True),
                ColDef("major_name", "string"),
                ColDef("degree_type", "string"),
            ]),
            TableDef("academic_departments", [
                ColDef("dept_id", "int", is_pk=True),
                ColDef("dept_name", "string"),
                ColDef("building", "string"),
            ]),
        ],
        mappings=[
            MappingDef("dim_student", "student_key", "int", "student surrogate key",
                       [("students", "student_id", "int")], "rename"),
            MappingDef("dim_student", "student_name", "string", "student full name",
                       [("students", "first_name", "string"), ("students", "last_name", "string")], "concat"),
            MappingDef("dim_student", "student_email", "string", "student email address",
                       [("students", "email", "string")], "rename"),
            MappingDef("dim_student", "enrollment_year", "int", "year of enrollment",
                       [("students", "enrollment_date", "date")], "date_part"),
            MappingDef("dim_student", "years_enrolled", "int", "duration in years",
                       [("students", "enrollment_date", "date"), ("students", "graduation_date", "date")], "date_diff"),
            MappingDef("dim_student", "grade_point_avg", "decimal", "cumulative GPA",
                       [("students", "gpa", "decimal")], "rename"),
            MappingDef("dim_student", "major_name", "string", "name of declared major",
                       [("majors", "major_name", "string")], "fk_lookup",
                       join_path=["students.major_id = majors.major_id"]),
            MappingDef("dim_student", "advisor_name", "string", "academic advisor full name",
                       [("faculty", "first_name", "string"), ("faculty", "last_name", "string")], "concat",
                       join_path=["students.advisor_id = faculty.faculty_id"]),
            MappingDef("fact_enrollment", "enrollment_key", "int", "enrollment id",
                       [("enrollments", "enrollment_id", "int")], "rename"),
            MappingDef("fact_enrollment", "student_full_name", "string", "enrolled student name",
                       [("students", "first_name", "string"), ("students", "last_name", "string")], "concat",
                       join_path=["enrollments.student_id = students.student_id"]),
            MappingDef("fact_enrollment", "course_title", "string", "name of the course",
                       [("courses", "course_name", "string")], "fk_lookup",
                       join_path=["enrollments.course_id = courses.course_id"]),
            MappingDef("fact_enrollment", "course_credits", "int", "credit hours",
                       [("courses", "credits", "int")], "fk_lookup",
                       join_path=["enrollments.course_id = courses.course_id"]),
            MappingDef("fact_enrollment", "letter_grade", "string", "final grade",
                       [("enrollments", "grade", "string")], "rename"),
            MappingDef("fact_enrollment", "numeric_score", "decimal", "final score",
                       [("enrollments", "score", "decimal")], "rename"),
            MappingDef("fact_enrollment", "department_name", "string", "academic department of the course",
                       [("academic_departments", "dept_name", "string")], "fk_lookup",
                       join_path=["courses.department_id = academic_departments.dept_id"]),
        ],
    ))

    # ─── Domain 6: Logistics ───
    domains.append(DomainSchema(
        name="logistics",
        source_tables=[
            TableDef("warehouses", [
                ColDef("warehouse_id", "int", is_pk=True),
                ColDef("warehouse_name", "string"),
                ColDef("city", "string"),
                ColDef("state", "string"),
                ColDef("country", "string"),
                ColDef("capacity_sqft", "int"),
                ColDef("manager_name", "string"),
            ]),
            TableDef("shipments", [
                ColDef("shipment_id", "int", is_pk=True),
                ColDef("origin_warehouse_id", "int", is_fk="warehouses.warehouse_id"),
                ColDef("destination_city", "string"),
                ColDef("carrier_id", "int", is_fk="carriers.carrier_id"),
                ColDef("ship_date", "date"),
                ColDef("delivery_date", "date"),
                ColDef("weight_kg", "decimal"),
                ColDef("cost", "decimal"),
                ColDef("status", "string"),
                ColDef("tracking_number", "string"),
            ]),
            TableDef("carriers", [
                ColDef("carrier_id", "int", is_pk=True),
                ColDef("carrier_name", "string"),
                ColDef("carrier_type", "string"),
                ColDef("contact_email", "string"),
            ]),
        ],
        mappings=[
            MappingDef("dim_warehouse", "warehouse_key", "int", "warehouse id",
                       [("warehouses", "warehouse_id", "int")], "rename"),
            MappingDef("dim_warehouse", "warehouse_label", "string", "warehouse name",
                       [("warehouses", "warehouse_name", "string")], "rename"),
            MappingDef("dim_warehouse", "warehouse_location", "string", "city and state combined",
                       [("warehouses", "city", "string"), ("warehouses", "state", "string")], "concat"),
            MappingDef("dim_warehouse", "full_location", "string", "city state country",
                       [("warehouses", "city", "string"), ("warehouses", "state", "string"),
                        ("warehouses", "country", "string")], "concat"),
            MappingDef("fact_shipment", "shipment_key", "int", "shipment id",
                       [("shipments", "shipment_id", "int")], "rename"),
            MappingDef("fact_shipment", "origin_warehouse_name", "string", "name of source warehouse",
                       [("warehouses", "warehouse_name", "string")], "fk_lookup",
                       join_path=["shipments.origin_warehouse_id = warehouses.warehouse_id"]),
            MappingDef("fact_shipment", "carrier_company", "string", "name of shipping carrier",
                       [("carriers", "carrier_name", "string")], "fk_lookup",
                       join_path=["shipments.carrier_id = carriers.carrier_id"]),
            MappingDef("fact_shipment", "transit_days", "int", "days in transit",
                       [("shipments", "ship_date", "date"), ("shipments", "delivery_date", "date")], "date_diff"),
            MappingDef("fact_shipment", "ship_year", "int", "year of shipment",
                       [("shipments", "ship_date", "date")], "date_part"),
            MappingDef("fact_shipment", "shipment_cost", "decimal", "shipping cost",
                       [("shipments", "cost", "decimal")], "rename"),
            MappingDef("fact_shipment", "is_delivered", "boolean", "delivery status flag",
                       [("shipments", "status", "string")], "conditional"),
            MappingDef("fact_shipment", "cost_per_kg", "decimal", "cost divided by weight",
                       [("shipments", "cost", "decimal"), ("shipments", "weight_kg", "decimal")], "arithmetic"),
        ],
    ))

    # ─── Domain 7: Real Estate ───
    domains.append(DomainSchema(
        name="realestate",
        source_tables=[
            TableDef("properties", [
                ColDef("property_id", "int", is_pk=True),
                ColDef("address", "string"),
                ColDef("city", "string"),
                ColDef("state", "string"),
                ColDef("zip_code", "string"),
                ColDef("property_type", "string"),
                ColDef("bedrooms", "int"),
                ColDef("bathrooms", "int"),
                ColDef("sqft", "int"),
                ColDef("year_built", "int"),
                ColDef("lot_size_acres", "decimal"),
                ColDef("listing_price", "decimal"),
                ColDef("agent_id", "int", is_fk="agents.agent_id"),
            ]),
            TableDef("agents", [
                ColDef("agent_id", "int", is_pk=True),
                ColDef("first_name", "string"),
                ColDef("last_name", "string"),
                ColDef("license_no", "string"),
                ColDef("agency_name", "string"),
                ColDef("phone", "string"),
            ]),
            TableDef("sales", [
                ColDef("sale_id", "int", is_pk=True),
                ColDef("property_id", "int", is_fk="properties.property_id"),
                ColDef("buyer_name", "string"),
                ColDef("sale_date", "date"),
                ColDef("sale_price", "decimal"),
                ColDef("closing_costs", "decimal"),
            ]),
        ],
        mappings=[
            MappingDef("dim_property", "property_key", "int", "property identifier",
                       [("properties", "property_id", "int")], "rename"),
            MappingDef("dim_property", "full_address", "string", "complete property address",
                       [("properties", "address", "string"), ("properties", "city", "string"),
                        ("properties", "state", "string"), ("properties", "zip_code", "string")], "concat"),
            MappingDef("dim_property", "property_city", "string", "city where property is located",
                       [("properties", "city", "string")], "rename"),
            MappingDef("dim_property", "property_category", "string", "type of property",
                       [("properties", "property_type", "string")], "rename"),
            MappingDef("dim_property", "total_rooms", "int", "bedrooms plus bathrooms",
                       [("properties", "bedrooms", "int"), ("properties", "bathrooms", "int")], "arithmetic"),
            MappingDef("dim_property", "price_per_sqft", "decimal", "listing price per square foot",
                       [("properties", "listing_price", "decimal"), ("properties", "sqft", "int")], "arithmetic"),
            MappingDef("dim_property", "listing_agent", "string", "name of the listing agent",
                       [("agents", "first_name", "string"), ("agents", "last_name", "string")], "concat",
                       join_path=["properties.agent_id = agents.agent_id"]),
            MappingDef("dim_property", "agency", "string", "real estate agency name",
                       [("agents", "agency_name", "string")], "fk_lookup",
                       join_path=["properties.agent_id = agents.agent_id"]),
            MappingDef("fact_sale", "sale_key", "int", "sale id",
                       [("sales", "sale_id", "int")], "rename"),
            MappingDef("fact_sale", "sale_year", "int", "year of sale",
                       [("sales", "sale_date", "date")], "date_part"),
            MappingDef("fact_sale", "net_proceeds", "decimal", "sale price minus closing costs",
                       [("sales", "sale_price", "decimal"), ("sales", "closing_costs", "decimal")], "arithmetic"),
            MappingDef("fact_sale", "sold_property_address", "string", "address of sold property",
                       [("properties", "address", "string")], "fk_lookup",
                       join_path=["sales.property_id = properties.property_id"]),
        ],
    ))

    # ─── Domain 8: Manufacturing ───
    domains.append(DomainSchema(
        name="manufacturing",
        source_tables=[
            TableDef("products", [
                ColDef("product_id", "int", is_pk=True),
                ColDef("product_name", "string"),
                ColDef("product_code", "string"),
                ColDef("category", "string"),
                ColDef("unit_weight", "decimal"),
                ColDef("unit_cost", "decimal"),
            ]),
            TableDef("production_orders", [
                ColDef("po_id", "int", is_pk=True),
                ColDef("product_id", "int", is_fk="products.product_id"),
                ColDef("order_qty", "int"),
                ColDef("produced_qty", "int"),
                ColDef("start_date", "date"),
                ColDef("end_date", "date"),
                ColDef("line_id", "int", is_fk="production_lines.line_id"),
                ColDef("status", "string"),
            ]),
            TableDef("production_lines", [
                ColDef("line_id", "int", is_pk=True),
                ColDef("line_name", "string"),
                ColDef("plant_id", "int", is_fk="plants.plant_id"),
                ColDef("capacity_per_hour", "int"),
            ]),
            TableDef("plants", [
                ColDef("plant_id", "int", is_pk=True),
                ColDef("plant_name", "string"),
                ColDef("location", "string"),
                ColDef("country", "string"),
            ]),
        ],
        mappings=[
            MappingDef("fact_production", "production_key", "int", "production order key",
                       [("production_orders", "po_id", "int")], "rename"),
            MappingDef("fact_production", "product_name", "string", "manufactured product name",
                       [("products", "product_name", "string")], "fk_lookup",
                       join_path=["production_orders.product_id = products.product_id"]),
            MappingDef("fact_production", "yield_rate", "decimal", "produced divided by ordered",
                       [("production_orders", "produced_qty", "int"), ("production_orders", "order_qty", "int")], "arithmetic"),
            MappingDef("fact_production", "production_days", "int", "days of production",
                       [("production_orders", "start_date", "date"), ("production_orders", "end_date", "date")], "date_diff"),
            MappingDef("fact_production", "start_year", "int", "year production started",
                       [("production_orders", "start_date", "date")], "date_part"),
            MappingDef("fact_production", "line_name", "string", "production line used",
                       [("production_lines", "line_name", "string")], "fk_lookup",
                       join_path=["production_orders.line_id = production_lines.line_id"]),
            MappingDef("fact_production", "plant_location", "string", "plant location",
                       [("plants", "location", "string")], "fk_lookup",
                       join_path=["production_lines.plant_id = plants.plant_id"]),
            MappingDef("fact_production", "total_weight", "decimal", "produced qty times unit weight",
                       [("production_orders", "produced_qty", "int"), ("products", "unit_weight", "decimal")], "arithmetic",
                       join_path=["production_orders.product_id = products.product_id"]),
            MappingDef("fact_production", "is_complete", "boolean", "whether production is done",
                       [("production_orders", "status", "string")], "conditional"),
        ],
    ))

    # ─── Domain 9: Telecom ───
    domains.append(DomainSchema(
        name="telecom",
        source_tables=[
            TableDef("subscribers", [
                ColDef("subscriber_id", "int", is_pk=True),
                ColDef("first_name", "string"),
                ColDef("last_name", "string"),
                ColDef("phone_number", "string"),
                ColDef("email", "string"),
                ColDef("activation_date", "date"),
                ColDef("plan_id", "int", is_fk="plans.plan_id"),
                ColDef("address", "string"),
                ColDef("city", "string"),
                ColDef("state", "string"),
                ColDef("credit_limit", "decimal"),
            ]),
            TableDef("plans", [
                ColDef("plan_id", "int", is_pk=True),
                ColDef("plan_name", "string"),
                ColDef("monthly_fee", "decimal"),
                ColDef("data_limit_gb", "int"),
                ColDef("plan_type", "string"),
            ]),
            TableDef("usage_records", [
                ColDef("record_id", "int", is_pk=True),
                ColDef("subscriber_id", "int", is_fk="subscribers.subscriber_id"),
                ColDef("usage_date", "date"),
                ColDef("data_used_mb", "decimal"),
                ColDef("voice_minutes", "int"),
                ColDef("sms_count", "int"),
                ColDef("roaming_flag", "string"),
            ]),
            TableDef("billing", [
                ColDef("bill_id", "int", is_pk=True),
                ColDef("subscriber_id", "int", is_fk="subscribers.subscriber_id"),
                ColDef("billing_date", "date"),
                ColDef("amount_due", "decimal"),
                ColDef("amount_paid", "decimal"),
                ColDef("payment_date", "date"),
                ColDef("status", "string"),
            ]),
        ],
        mappings=[
            MappingDef("dim_subscriber", "subscriber_key", "int", "subscriber id",
                       [("subscribers", "subscriber_id", "int")], "rename"),
            MappingDef("dim_subscriber", "subscriber_name", "string", "full name of subscriber",
                       [("subscribers", "first_name", "string"), ("subscribers", "last_name", "string")], "concat"),
            MappingDef("dim_subscriber", "contact_email", "string", "subscriber email",
                       [("subscribers", "email", "string")], "rename"),
            MappingDef("dim_subscriber", "full_address", "string", "complete address with city and state",
                       [("subscribers", "address", "string"), ("subscribers", "city", "string"), ("subscribers", "state", "string")], "concat"),
            MappingDef("dim_subscriber", "activation_year", "int", "year of activation",
                       [("subscribers", "activation_date", "date")], "date_part"),
            MappingDef("dim_subscriber", "plan_name", "string", "subscribed plan",
                       [("plans", "plan_name", "string")], "fk_lookup",
                       join_path=["subscribers.plan_id = plans.plan_id"]),
            MappingDef("dim_subscriber", "monthly_charge", "decimal", "monthly fee of plan",
                       [("plans", "monthly_fee", "decimal")], "fk_lookup",
                       join_path=["subscribers.plan_id = plans.plan_id"]),
            MappingDef("fact_usage", "usage_key", "int", "usage record identifier",
                       [("usage_records", "record_id", "int")], "rename"),
            MappingDef("fact_usage", "subscriber_full_name", "string", "name of subscriber",
                       [("subscribers", "first_name", "string"), ("subscribers", "last_name", "string")], "concat",
                       join_path=["usage_records.subscriber_id = subscribers.subscriber_id"]),
            MappingDef("fact_usage", "usage_year", "int", "year of usage",
                       [("usage_records", "usage_date", "date")], "date_part"),
            MappingDef("fact_usage", "total_usage_gb", "decimal", "data used converted to GB",
                       [("usage_records", "data_used_mb", "decimal")], "arithmetic"),
            MappingDef("fact_usage", "is_roaming", "boolean", "whether usage was roaming",
                       [("usage_records", "roaming_flag", "string")], "conditional"),
            MappingDef("fact_billing", "bill_key", "int", "billing id",
                       [("billing", "bill_id", "int")], "rename"),
            MappingDef("fact_billing", "outstanding_amount", "decimal", "amount due minus paid",
                       [("billing", "amount_due", "decimal"), ("billing", "amount_paid", "decimal")], "arithmetic"),
            MappingDef("fact_billing", "days_to_pay", "int", "days between billing and payment",
                       [("billing", "billing_date", "date"), ("billing", "payment_date", "date")], "date_diff"),
            MappingDef("fact_billing", "is_paid", "boolean", "whether bill was paid",
                       [("billing", "status", "string")], "conditional"),
        ],
    ))

    # ─── Domain 10: Travel & Hospitality ───
    domains.append(DomainSchema(
        name="travel",
        source_tables=[
            TableDef("guests", [
                ColDef("guest_id", "int", is_pk=True),
                ColDef("first_name", "string"),
                ColDef("last_name", "string"),
                ColDef("email", "string"),
                ColDef("phone", "string"),
                ColDef("nationality", "string"),
                ColDef("dob", "date"),
                ColDef("passport_no", "string"),
                ColDef("loyalty_tier", "string"),
            ]),
            TableDef("hotels", [
                ColDef("hotel_id", "int", is_pk=True),
                ColDef("hotel_name", "string"),
                ColDef("city", "string"),
                ColDef("country", "string"),
                ColDef("star_rating", "int"),
                ColDef("total_rooms", "int"),
            ]),
            TableDef("reservations", [
                ColDef("reservation_id", "int", is_pk=True),
                ColDef("guest_id", "int", is_fk="guests.guest_id"),
                ColDef("hotel_id", "int", is_fk="hotels.hotel_id"),
                ColDef("check_in_date", "date"),
                ColDef("check_out_date", "date"),
                ColDef("room_type", "string"),
                ColDef("rate_per_night", "decimal"),
                ColDef("total_cost", "decimal"),
                ColDef("status", "string"),
                ColDef("booking_source", "string"),
            ]),
        ],
        mappings=[
            MappingDef("dim_guest", "guest_key", "int", "guest identifier",
                       [("guests", "guest_id", "int")], "rename"),
            MappingDef("dim_guest", "guest_name", "string", "guest full name",
                       [("guests", "first_name", "string"), ("guests", "last_name", "string")], "concat"),
            MappingDef("dim_guest", "guest_email", "string", "email of the guest",
                       [("guests", "email", "string")], "rename"),
            MappingDef("dim_guest", "birth_year", "int", "year of birth",
                       [("guests", "dob", "date")], "date_part"),
            MappingDef("dim_guest", "guest_age", "int", "current age in years",
                       [("guests", "dob", "date")], "date_diff"),
            MappingDef("fact_reservation", "reservation_key", "int", "reservation id",
                       [("reservations", "reservation_id", "int")], "rename"),
            MappingDef("fact_reservation", "guest_full_name", "string", "name of the guest",
                       [("guests", "first_name", "string"), ("guests", "last_name", "string")], "concat",
                       join_path=["reservations.guest_id = guests.guest_id"]),
            MappingDef("fact_reservation", "hotel_name", "string", "name of the hotel",
                       [("hotels", "hotel_name", "string")], "fk_lookup",
                       join_path=["reservations.hotel_id = hotels.hotel_id"]),
            MappingDef("fact_reservation", "hotel_location", "string", "hotel city and country",
                       [("hotels", "city", "string"), ("hotels", "country", "string")], "concat",
                       join_path=["reservations.hotel_id = hotels.hotel_id"]),
            MappingDef("fact_reservation", "stay_duration", "int", "nights of stay",
                       [("reservations", "check_in_date", "date"), ("reservations", "check_out_date", "date")], "date_diff"),
            MappingDef("fact_reservation", "checkin_year", "int", "year of check-in",
                       [("reservations", "check_in_date", "date")], "date_part"),
            MappingDef("fact_reservation", "nightly_rate", "decimal", "rate per night charged",
                       [("reservations", "rate_per_night", "decimal")], "rename"),
            MappingDef("fact_reservation", "is_confirmed", "boolean", "whether reservation is confirmed",
                       [("reservations", "status", "string")], "conditional"),
        ],
    ))

    # ─── Domain 11: Sports ───
    domains.append(DomainSchema(
        name="sports",
        source_tables=[
            TableDef("players", [
                ColDef("player_id", "int", is_pk=True),
                ColDef("first_name", "string"),
                ColDef("last_name", "string"),
                ColDef("dob", "date"),
                ColDef("nationality", "string"),
                ColDef("position", "string"),
                ColDef("jersey_number", "int"),
                ColDef("team_id", "int", is_fk="teams.team_id"),
                ColDef("height_cm", "int"),
                ColDef("weight_kg", "decimal"),
                ColDef("contract_start", "date"),
                ColDef("contract_end", "date"),
                ColDef("annual_salary", "decimal"),
            ]),
            TableDef("teams", [
                ColDef("team_id", "int", is_pk=True),
                ColDef("team_name", "string"),
                ColDef("city", "string"),
                ColDef("stadium", "string"),
                ColDef("founded_year", "int"),
                ColDef("league_id", "int", is_fk="leagues.league_id"),
            ]),
            TableDef("leagues", [
                ColDef("league_id", "int", is_pk=True),
                ColDef("league_name", "string"),
                ColDef("country", "string"),
                ColDef("sport", "string"),
            ]),
            TableDef("matches", [
                ColDef("match_id", "int", is_pk=True),
                ColDef("home_team_id", "int", is_fk="teams.team_id"),
                ColDef("away_team_id", "int", is_fk="teams.team_id"),
                ColDef("match_date", "date"),
                ColDef("home_score", "int"),
                ColDef("away_score", "int"),
                ColDef("venue", "string"),
                ColDef("attendance", "int"),
            ]),
        ],
        mappings=[
            MappingDef("dim_player", "player_key", "int", "player surrogate key",
                       [("players", "player_id", "int")], "rename"),
            MappingDef("dim_player", "player_name", "string", "player full name",
                       [("players", "first_name", "string"), ("players", "last_name", "string")], "concat"),
            MappingDef("dim_player", "player_age", "int", "current age",
                       [("players", "dob", "date")], "date_diff"),
            MappingDef("dim_player", "birth_year", "int", "year of birth",
                       [("players", "dob", "date")], "date_part"),
            MappingDef("dim_player", "bmi", "decimal", "weight divided by height squared",
                       [("players", "weight_kg", "decimal"), ("players", "height_cm", "int")], "arithmetic"),
            MappingDef("dim_player", "contract_duration", "int", "contract length in days",
                       [("players", "contract_start", "date"), ("players", "contract_end", "date")], "date_diff"),
            MappingDef("dim_player", "team_name", "string", "name of the team",
                       [("teams", "team_name", "string")], "fk_lookup",
                       join_path=["players.team_id = teams.team_id"]),
            MappingDef("dim_player", "league_name", "string", "name of the league",
                       [("leagues", "league_name", "string")], "fk_lookup",
                       join_path=["teams.league_id = leagues.league_id"]),
            MappingDef("fact_match", "match_key", "int", "match id",
                       [("matches", "match_id", "int")], "rename"),
            MappingDef("fact_match", "match_year", "int", "year of the match",
                       [("matches", "match_date", "date")], "date_part"),
            MappingDef("fact_match", "total_goals", "int", "home score plus away score",
                       [("matches", "home_score", "int"), ("matches", "away_score", "int")], "arithmetic"),
            MappingDef("fact_match", "goal_difference", "int", "home score minus away score",
                       [("matches", "home_score", "int"), ("matches", "away_score", "int")], "arithmetic"),
        ],
    ))

    # ─── Domain 12: Insurance ───
    domains.append(DomainSchema(
        name="insurance",
        source_tables=[
            TableDef("policyholders", [
                ColDef("holder_id", "int", is_pk=True),
                ColDef("first_name", "string"),
                ColDef("last_name", "string"),
                ColDef("date_of_birth", "date"),
                ColDef("ssn_hash", "string"),
                ColDef("email", "string"),
                ColDef("phone", "string"),
                ColDef("address", "string"),
                ColDef("city", "string"),
                ColDef("state", "string"),
                ColDef("occupation", "string"),
                ColDef("annual_income", "decimal"),
            ]),
            TableDef("policies", [
                ColDef("policy_id", "int", is_pk=True),
                ColDef("holder_id", "int", is_fk="policyholders.holder_id"),
                ColDef("policy_type", "string"),
                ColDef("start_date", "date"),
                ColDef("end_date", "date"),
                ColDef("premium", "decimal"),
                ColDef("coverage_amount", "decimal"),
                ColDef("deductible", "decimal"),
                ColDef("status", "string"),
                ColDef("agent_id", "int", is_fk="agents.agent_id"),
            ]),
            TableDef("claims", [
                ColDef("claim_id", "int", is_pk=True),
                ColDef("policy_id", "int", is_fk="policies.policy_id"),
                ColDef("claim_date", "date"),
                ColDef("incident_date", "date"),
                ColDef("claim_amount", "decimal"),
                ColDef("approved_amount", "decimal"),
                ColDef("claim_type", "string"),
                ColDef("status", "string"),
                ColDef("description", "string"),
            ]),
            TableDef("agents", [
                ColDef("agent_id", "int", is_pk=True),
                ColDef("first_name", "string"),
                ColDef("last_name", "string"),
                ColDef("license_no", "string"),
                ColDef("region", "string"),
                ColDef("commission_rate", "decimal"),
            ]),
        ],
        mappings=[
            MappingDef("dim_policyholder", "holder_key", "int", "policyholder id",
                       [("policyholders", "holder_id", "int")], "rename"),
            MappingDef("dim_policyholder", "holder_name", "string", "full name of policyholder",
                       [("policyholders", "first_name", "string"), ("policyholders", "last_name", "string")], "concat"),
            MappingDef("dim_policyholder", "holder_age", "int", "age of policyholder",
                       [("policyholders", "date_of_birth", "date")], "date_diff"),
            MappingDef("dim_policyholder", "birth_year", "int", "year of birth",
                       [("policyholders", "date_of_birth", "date")], "date_part"),
            MappingDef("dim_policyholder", "full_address", "string", "complete address",
                       [("policyholders", "address", "string"), ("policyholders", "city", "string"), ("policyholders", "state", "string")], "concat"),
            MappingDef("dim_policy", "policy_key", "int", "policy identifier",
                       [("policies", "policy_id", "int")], "rename"),
            MappingDef("dim_policy", "policy_duration_days", "int", "policy length in days",
                       [("policies", "start_date", "date"), ("policies", "end_date", "date")], "date_diff"),
            MappingDef("dim_policy", "start_year", "int", "policy start year",
                       [("policies", "start_date", "date")], "date_part"),
            MappingDef("dim_policy", "holder_full_name", "string", "name of the policyholder",
                       [("policyholders", "first_name", "string"), ("policyholders", "last_name", "string")], "concat",
                       join_path=["policies.holder_id = policyholders.holder_id"]),
            MappingDef("dim_policy", "agent_name", "string", "assigned agent full name",
                       [("agents", "first_name", "string"), ("agents", "last_name", "string")], "concat",
                       join_path=["policies.agent_id = agents.agent_id"]),
            MappingDef("dim_policy", "is_active", "boolean", "whether policy is currently active",
                       [("policies", "status", "string")], "conditional"),
            MappingDef("dim_policy", "net_coverage", "decimal", "coverage minus deductible",
                       [("policies", "coverage_amount", "decimal"), ("policies", "deductible", "decimal")], "arithmetic"),
            MappingDef("fact_claim", "claim_key", "int", "claim identifier",
                       [("claims", "claim_id", "int")], "rename"),
            MappingDef("fact_claim", "claim_year", "int", "year of claim",
                       [("claims", "claim_date", "date")], "date_part"),
            MappingDef("fact_claim", "processing_days", "int", "days from incident to claim",
                       [("claims", "incident_date", "date"), ("claims", "claim_date", "date")], "date_diff"),
            MappingDef("fact_claim", "approval_rate", "decimal", "approved divided by claimed",
                       [("claims", "approved_amount", "decimal"), ("claims", "claim_amount", "decimal")], "arithmetic"),
            MappingDef("fact_claim", "is_approved", "boolean", "whether claim was approved",
                       [("claims", "status", "string")], "conditional"),
        ],
    ))

    return domains


# ───────────────────────────────────────────────────────
# Name augmentation for diversity
# ───────────────────────────────────────────────────────

_NAME_VARIANTS = {
    "full_name": ["complete_name", "display_name", "person_name", "fullname"],
    "email_address": ["email_addr", "contact_email", "e_mail", "email_id"],
    "phone_number": ["phone_no", "contact_phone", "telephone", "mobile_number"],
    "hire_year": ["year_hired", "hiring_year", "join_year", "start_year"],
    "hire_date": ["date_hired", "joining_date", "start_date", "onboarding_date"],
    "department_name": ["dept_name", "division_name", "department_label", "org_unit"],
    "employee_key": ["emp_key", "employee_id", "emp_identifier", "worker_key"],
    "annual_salary": ["yearly_salary", "salary_annual", "compensation", "yearly_pay"],
    "customer_name": ["client_name", "buyer_name", "cust_name", "purchaser_name"],
    "order_date": ["purchase_date", "transaction_date", "order_timestamp"],
    "product_name": ["item_name", "product_title", "product_label", "article_name"],
    "total_amount": ["gross_total", "order_total", "sum_amount", "total_value"],
    "is_active": ["active_flag", "is_enabled", "status_flag", "active_indicator"],
    "registration_year": ["signup_year", "enrolled_year", "joined_year", "reg_year"],
    "birth_year": ["year_of_birth", "born_year", "dob_year"],
    "transaction_date": ["txn_date", "posting_date", "effective_date"],
    "branch_name": ["office_name", "branch_label", "location_name"],
    "patient_name": ["patient_full_name", "client_name", "member_name"],
    "student_name": ["learner_name", "pupil_name", "enrollee_name"],
    "course_title": ["course_name", "class_name", "subject_name", "module_name"],
    "sale_year": ["year_sold", "selling_year", "closing_year"],
}

_DESC_VARIANTS = [
    "", "the {thing}", "{thing} value", "{thing} from source", "derived {thing}",
]


def _augment_name(name: str) -> str:
    """Return a random variant of a column name."""
    if name in _NAME_VARIANTS and random.random() < 0.5:
        return random.choice(_NAME_VARIANTS[name])
    # Random casing variants
    r = random.random()
    if r < 0.15:
        return name.replace("_", "").lower()  # camelCase-ish
    if r < 0.25:
        parts = name.split("_")
        return "".join(p.capitalize() for p in parts)  # PascalCase
    if r < 0.35:
        return name.replace("_", "-")  # kebab-case
    return name


def _augment_desc(desc: str) -> str:
    """Randomly modify description."""
    if random.random() < 0.3:
        return ""  # sometimes no description
    if random.random() < 0.2:
        return desc.split()[0] if desc else ""
    return desc


# ───────────────────────────────────────────────────────
# Negative candidate generation
# ───────────────────────────────────────────────────────

def _generate_negatives(
    mapping: MappingDef,
    domain: DomainSchema,
    n_hard: int = 5,
    n_random: int = 5,
) -> List[Tuple[List[Tuple[str, str, str]], str, List[str]]]:
    """
    Generate negative candidates for a mapping.
    Returns list of (source_cols, transform, join_path).
    """
    negatives = []
    all_cols = []
    for table in domain.source_tables:
        for col in table.columns:
            all_cols.append((table.name, col.name, col.type))

    correct_set = set((t, c) for t, c, _ in mapping.source_cols)

    # Hard negatives: same table, similar type, but wrong column
    for t, c, ct in mapping.source_cols:
        table_cols = [(tb.name, col.name, col.type)
                      for tb in domain.source_tables if tb.name == t
                      for col in tb.columns
                      if (tb.name, col.name) not in correct_set]
        random.shuffle(table_cols)
        for neg_col in table_cols[:n_hard]:
            negatives.append(([neg_col], "rename", []))

    # Hard negatives: partial correct (only one of multiple required columns)
    if len(mapping.source_cols) > 1:
        for src in mapping.source_cols:
            negatives.append(([src], "rename", []))

    # Hard negatives: right column names from wrong tables
    correct_names = {c for _, c, _ in mapping.source_cols}
    for t, c, ct in all_cols:
        if c in correct_names and (t, c) not in correct_set:
            negatives.append(([(t, c, ct)], mapping.transform, []))

    # Random negatives
    random.shuffle(all_cols)
    for neg_col in all_cols[:n_random]:
        if (neg_col[0], neg_col[1]) not in correct_set:
            negatives.append(([neg_col], random.choice(["rename", "type_cast", "direct_copy"]), []))

    # Random pairs (wrong combinations)
    if len(all_cols) > 2:
        for _ in range(min(n_random, 5)):
            pair = random.sample(all_cols, 2)
            if not any((p[0], p[1]) in correct_set for p in pair):
                negatives.append((pair, random.choice(["concat", "arithmetic"]), []))

    # De-duplicate
    seen = set()
    unique = []
    for cols, tr, jp in negatives:
        key = tuple(sorted((c[0], c[1]) for c in cols))
        if key not in seen and key != tuple(sorted((c[0], c[1]) for c in mapping.source_cols)):
            seen.add(key)
            unique.append((cols, tr, jp))

    return unique[:n_hard + n_random]


# ───────────────────────────────────────────────────────
# Main data generation
# ───────────────────────────────────────────────────────

def generate_dataset(
    output_dir: str = "candidate_training_data",
    augmentation_rounds: int = 10,
    seed: int = 42,
):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    domains = _build_domains()
    print(f"Built {len(domains)} domains with {sum(len(d.mappings) for d in domains)} mappings")

    bi_records = []     # For bi-encoder: {query, positive, negatives}
    cross_records = []  # For cross-encoder: {query, candidate, label}

    for domain in domains:
        # Build table context string
        table_ctx = "; ".join(
            f"{t.name}({', '.join(c.name for c in t.columns)})"
            for t in domain.source_tables
        )

        for mapping in domain.mappings:
            for aug_round in range(augmentation_rounds):
                # Augment target name/desc
                tgt_name = _augment_name(mapping.target_col) if aug_round > 0 else mapping.target_col
                tgt_desc = _augment_desc(mapping.target_desc)

                query = serialize_target(
                    mapping.target_table, tgt_name, mapping.target_type, tgt_desc,
                )

                positive = serialize_candidate(
                    mapping.source_cols,
                    join_path=mapping.join_path,
                    transform_hint=mapping.transform,
                    table_context=table_ctx[:300],
                )

                negatives_raw = _generate_negatives(mapping, domain)
                negative_texts = [
                    serialize_candidate(
                        cols, join_path=jp, transform_hint=tr, table_context=table_ctx[:300],
                    )
                    for cols, tr, jp in negatives_raw
                ]

                if negative_texts:
                    bi_records.append({
                        "query": query,
                        "positive": positive,
                        "negatives": negative_texts[:10],
                        "domain": domain.name,
                        "transform": mapping.transform,
                        "arity": len(mapping.source_cols),
                    })

                    # Cross-encoder positive
                    cross_records.append({
                        "query": query,
                        "candidate": positive,
                        "label": 1,
                        "domain": domain.name,
                        "transform": mapping.transform,
                    })

                    # Cross-encoder negatives
                    for neg in negative_texts[:6]:
                        cross_records.append({
                            "query": query,
                            "candidate": neg,
                            "label": 0,
                            "domain": domain.name,
                        })

    # Shuffle and split
    random.shuffle(bi_records)
    random.shuffle(cross_records)

    n_bi = len(bi_records)
    n_cross = len(cross_records)
    bi_train = bi_records[:int(n_bi * 0.85)]
    bi_val = bi_records[int(n_bi * 0.85):]
    cross_train = cross_records[:int(n_cross * 0.85)]
    cross_val = cross_records[int(n_cross * 0.85):]

    def _write_jsonl(path, records):
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    _write_jsonl(os.path.join(output_dir, "bi_encoder_train.jsonl"), bi_train)
    _write_jsonl(os.path.join(output_dir, "bi_encoder_val.jsonl"), bi_val)
    _write_jsonl(os.path.join(output_dir, "cross_encoder_train.jsonl"), cross_train)
    _write_jsonl(os.path.join(output_dir, "cross_encoder_val.jsonl"), cross_val)

    print(f"\nGenerated:")
    print(f"  Bi-encoder:    {len(bi_train):,} train + {len(bi_val):,} val = {n_bi:,} total")
    print(f"  Cross-encoder: {len(cross_train):,} train + {len(cross_val):,} val = {n_cross:,} total")
    print(f"  Domains: {[d.name for d in domains]}")
    print(f"  Transform distribution:")
    from collections import Counter
    tc = Counter(r["transform"] for r in bi_records)
    for t, cnt in tc.most_common():
        print(f"    {t:20s}: {cnt}")
    arity_c = Counter(r["arity"] for r in bi_records)
    print(f"  Arity distribution:")
    for a, cnt in sorted(arity_c.items()):
        print(f"    arity={a}: {cnt}")
    print(f"\nOutput: {output_dir}/")


if __name__ == "__main__":
    generate_dataset()
