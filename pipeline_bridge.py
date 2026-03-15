"""
Pipeline Bridge
================
Converts web-app JSON input into the data structures expected by
the existing ML pipeline modules, and runs each stage.

Includes a smart post-processing layer that:
  - Adds direct name-matching candidates that Stage A may miss
  - Infers the correct transform type based on source->target patterns
  - Prefers simpler mappings (single-column) over noisy composites
"""
from __future__ import annotations

import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from constraint_similarity_engine import ColumnConstraints, ConstraintCompatibilityJoin
from join_graph_builder_v2 import (
    Column, ColumnType, DefaultTypeCompatibility, JoinEdge,
    JoinGraphBuilderV2, Table,
)
from value_similarity_engine import ColumnStats, ValueSimilarity

from candidate_generation_algorithm import (
    CandidateGenerationEngine, NameEmbedder, TargetSpec,
)

# V3 LLM-based engine (best accuracy)
try:
    from candidate_generation_v3 import SchemaMatcherLLM
    HAS_V3 = True
except Exception:
    HAS_V3 = False

# V2 ML-based engine (fast, good accuracy)
try:
    from candidate_generation_v2 import CandidateGeneratorV2
    HAS_V2 = True
except Exception:
    HAS_V2 = False

# Optional imports (Stage B and transform classifier)
try:
    from candidate_selector_stage1 import (
        CandidateSelectorStage1,
        CandidateSet as SelectorCandidateSet,
        SourceColumn,
        TargetColumn,
    )
    HAS_STAGE_B = True
except Exception:
    HAS_STAGE_B = False


# ---------------------------------------------------------------
# Tokenization & name similarity helpers
# ---------------------------------------------------------------

_SPLIT_RE = re.compile(r"[_\-.\s]+|(?<=[a-z])(?=[A-Z])")

def _tokenize(name: str) -> List[str]:
    """Split a column name into lowercase tokens."""
    return [t.lower() for t in _SPLIT_RE.split(name) if t]


def _token_overlap(a_tokens: List[str], b_tokens: List[str]) -> float:
    """Jaccard-like overlap between two token lists."""
    if not a_tokens or not b_tokens:
        return 0.0
    sa, sb = set(a_tokens), set(b_tokens)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / max(union, 1)


# Common abbreviation/synonym expansions for name matching
_SYNONYMS: Dict[str, List[str]] = {
    "emp": ["employee", "empl"],
    "employee": ["emp", "empl"],
    "dept": ["department", "dep"],
    "department": ["dept", "dep"],
    "mgr": ["manager"],
    "manager": ["mgr"],
    "addr": ["address"],
    "address": ["addr"],
    "num": ["number", "no"],
    "qty": ["quantity"],
    "amt": ["amount"],
    "amount": ["amt"],
    "desc": ["description"],
    "cat": ["category"],
    "org": ["organization", "organisation"],
    "loc": ["location"],
    "location": ["loc"],
    "tel": ["telephone", "phone"],
    "proj": ["project"],
    "project": ["proj"],
    "cust": ["customer"],
    "customer": ["cust"],
    "prod": ["product"],
    "product": ["prod"],
    "txn": ["transaction"],
    "acct": ["account"],
    "account": ["acct"],
    "fname": ["first_name", "firstname"],
    "lname": ["last_name", "lastname"],
    "dob": ["date_of_birth", "birth_date"],
    "title": ["name", "label"],
    "key": ["id", "pk", "identifier"],
    "id": ["key", "identifier"],
    "owning": ["owner", "owns"],
    "annual": ["yearly", "year"],
    "salary": ["pay", "compensation", "wage"],
}

def _stem(token: str) -> str:
    """Very simple stemmer: strips common suffixes for matching."""
    if token.endswith("ies"):
        return token[:-3] + "y"  # e.g., categories -> category
    if token.endswith("ses") or token.endswith("xes"):
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]  # e.g., employees -> employee
    return token


def _expand_tokens(tokens: List[str]) -> List[str]:
    """Expand abbreviations to full forms for better matching."""
    expanded = list(tokens)
    # Add stemmed forms
    for t in tokens:
        stemmed = _stem(t)
        if stemmed != t:
            expanded.append(stemmed)

    for t in list(expanded):
        for abbr, fulls in _SYNONYMS.items():
            if t == abbr or _stem(t) == abbr or t == _stem(abbr):
                for f in fulls:
                    expanded.extend(_tokenize(f))
            elif t in [ft for f in fulls for ft in _tokenize(f)]:
                expanded.append(abbr)
    return list(set(expanded))


def _name_similarity(src_name: str, tgt_name: str) -> float:
    """Compute semantic name similarity between source and target column names."""
    src_tok = _tokenize(src_name)
    tgt_tok = _tokenize(tgt_name)

    # Direct overlap
    direct = _token_overlap(src_tok, tgt_tok)

    # Expanded overlap (with synonyms)
    src_exp = _expand_tokens(src_tok)
    tgt_exp = _expand_tokens(tgt_tok)
    expanded = _token_overlap(src_exp, tgt_exp)

    # Substring match bonus
    sub_bonus = 0.0
    src_lower = src_name.lower().replace("_", "")
    tgt_lower = tgt_name.lower().replace("_", "")
    if src_lower in tgt_lower or tgt_lower in src_lower:
        sub_bonus = 0.3

    return min(1.0, max(direct, expanded) + sub_bonus)


# ---------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------

_NUM_TYPES = {"int", "integer", "bigint", "float", "double", "decimal", "number", "numeric"}
_STR_TYPES = {"string", "varchar", "text", "char"}
_DATE_TYPES = {"date", "datetime", "timestamp", "time"}


def _base_type(ct):
    return getattr(ct, "base_type", str(ct)).lower()


def _types_compatible(src_type: str, tgt_type: str) -> bool:
    """Check if source and target types are directly compatible (same family)."""
    s, t = src_type.lower(), tgt_type.lower()
    if s == t:
        return True
    if s in _NUM_TYPES and t in _NUM_TYPES:
        return True
    if s in _STR_TYPES and t in _STR_TYPES:
        return True
    if s in _DATE_TYPES and t in _DATE_TYPES:
        return True
    # Cross-type (e.g., date→int, int→string) is NOT directly compatible
    return False


# ---------------------------------------------------------------
# Transform type inference
# ---------------------------------------------------------------

_DATE_PART_TOKENS = {"year", "month", "day", "quarter", "week", "hour", "minute", "second"}
_CONCAT_HINTS = {"full", "combined", "complete", "merged", "joined", "formatted", "display", "combined"}
_NAME_TOKENS = {"name", "first", "last", "middle", "full", "given", "family", "surname"}

# Known multi-column composition patterns
# target token(s) -> list of required source token patterns
_COMPOSITE_PATTERNS: Dict[str, List[List[str]]] = {
    "full_name": [["first", "last"], ["first_name", "last_name"], ["fname", "lname"]],
    "lead_name": [["first", "last"], ["first_name", "last_name"]],
    "display_name": [["first", "last"], ["first_name", "last_name"]],
    "name_email": [["first", "last", "email"], ["name", "email"]],
    "address": [["street", "city"], ["line1", "city"], ["addr1", "city"]],
}


def infer_transform_type(
    target_name: str,
    target_type: str,
    source_cols: List[Tuple[str, str, str]],  # (table, column, type)
    join_needed: bool,
) -> str:
    """
    Infer the most likely transformation type based on source->target patterns.
    
    Returns one of: direct_copy, rename, type_cast, date_part, concat, 
                     template, fk_lookup, arithmetic, conditional, aggregate
    """
    tgt_tok = set(_tokenize(target_name))
    tgt_type_l = target_type.lower()
    n_src = len(source_cols)

    if n_src == 0:
        return "unmapped"

    if n_src == 1:
        s_table, s_col, s_type = source_cols[0]
        s_tok = set(_tokenize(s_col))
        s_type_l = s_type.lower()

        # Exact same name
        if s_col.lower() == target_name.lower():
            return "direct_copy"

        # Date -> int with date-part tokens in target name
        if s_type_l in _DATE_TYPES and tgt_type_l in _NUM_TYPES:
            if tgt_tok & _DATE_PART_TOKENS:
                return "date_part"
            return "date_part"  # likely extract year/month etc

        # Date -> string = format
        if s_type_l in _DATE_TYPES and tgt_type_l in _STR_TYPES:
            return "date_format"

        # String -> date = date_parse
        if s_type_l in _STR_TYPES and tgt_type_l in _DATE_TYPES:
            return "date_parse"

        # Same type, different name = rename
        if s_type_l == tgt_type_l or (s_type_l in _NUM_TYPES and tgt_type_l in _NUM_TYPES):
            if join_needed:
                return "fk_lookup"
            return "rename"

        # Different types
        if join_needed:
            return "fk_lookup"
        return "type_cast"

    # Multiple sources
    src_types = {s[2].lower() for s in source_cols}
    src_tables = {s[0] for s in source_cols}

    # Multiple strings -> string (likely concat)
    if tgt_type_l in _STR_TYPES and all(st in _STR_TYPES for st in src_types):
        if tgt_tok & _CONCAT_HINTS or tgt_tok & _NAME_TOKENS:
            return "concat"
        return "concat"

    # Multiple strings -> string with name tokens
    if tgt_type_l in _STR_TYPES:
        src_name_tokens = set()
        for s in source_cols:
            src_name_tokens.update(_tokenize(s[1]))
        if src_name_tokens & _NAME_TOKENS:
            return "concat"
        if tgt_tok & {"formatted", "combined", "template"}:
            return "template"

    # Numerics -> numeric
    if tgt_type_l in _NUM_TYPES and all(st in _NUM_TYPES for st in src_types):
        return "arithmetic"

    # Date + something -> numeric (date_diff)
    if any(st in _DATE_TYPES for st in src_types) and tgt_type_l in _NUM_TYPES:
        if any(t in tgt_tok for t in {"diff", "days", "duration", "tenure", "age", "elapsed"}):
            return "date_diff"
        return "date_part"

    # Mixed types with join
    if len(src_tables) > 1:
        return "lookup_join"

    return "conditional"


# ---------------------------------------------------------------
# Smart candidate finder: direct name matching
# ---------------------------------------------------------------

def _find_name_match_candidates(
    target_name: str,
    target_type: str,
    target_desc: str,
    target_table: str,
    source_tables: Dict[str, Table],
    join_edges: List[JoinEdge],
) -> List[Dict[str, Any]]:
    """
    Find source columns that match the target by name/description.
    Returns candidates sorted by relevance, each with:
      source_columns, transform_type, confidence, join_path
    """
    tgt_tok = _tokenize(target_name)
    tgt_exp = _expand_tokens(tgt_tok)
    desc_tok = _tokenize(target_desc) if target_desc else []
    all_tgt_tokens = set(tgt_exp + desc_tok)

    # Build adjacency for join path lookups
    table_joins: Dict[str, Dict[str, JoinEdge]] = defaultdict(dict)
    for e in join_edges:
        table_joins[e.left_table][e.right_table] = e
        table_joins[e.right_table][e.left_table] = e

    # Score every source column
    scored: List[Tuple[float, str, str, str]] = []  # (score, table, col, type)
    for tname, table in source_tables.items():
        for cname, col in table.columns.items():
            ctype = _base_type(col.col_type)
            sim = _name_similarity(cname, target_name)

            # Boost if description tokens match column name tokens
            col_tok = set(_tokenize(cname))
            desc_overlap = len(col_tok & set(desc_tok)) / max(len(desc_tok), 1) if desc_tok else 0
            sim = max(sim, sim + desc_overlap * 0.3)

            # Type compatibility bonus/penalty
            type_bonus = 0.15 if _types_compatible(ctype, target_type) else -0.15

            # PK bonus for "key"/"id" targets, penalty for non-key targets
            pk_bonus = 0.0
            if col.constraints and (col.constraints.is_primary_key or col.constraints.is_foreign_key):
                if any(t in all_tgt_tokens for t in {"key", "id", "pk", "identifier"}):
                    pk_bonus = 0.15
                else:
                    pk_bonus = -0.1  # IDs shouldn't map to non-ID targets

            # Table name relevance bonus
            table_tok = set(_tokenize(tname))
            table_exp = set(_expand_tokens(list(table_tok)))
            table_rel = len(table_exp & all_tgt_tokens) / max(len(all_tgt_tokens), 1)
            table_bonus = table_rel * 0.2

            total = sim + type_bonus + pk_bonus + table_bonus
            if total > 0.15:
                scored.append((total, tname, cname, ctype))

    scored.sort(key=lambda x: -x[0])

    candidates: List[Dict[str, Any]] = []

    # Determine which tables are "primary" for this target entity
    # e.g., dim_employee → employees is the primary table
    tgt_table_tok = set(_tokenize(target_table))
    tgt_table_exp = set(_expand_tokens(list(tgt_table_tok)))

    def _is_primary_table(tname: str) -> bool:
        """Check if source table is the primary entity table for this target."""
        t_tok = set(_tokenize(tname))
        t_exp = set(_expand_tokens(list(t_tok)))
        return bool(t_exp & tgt_table_exp)

    # --- Single-column candidates ---
    for score, tname, cname, ctype in scored[:10]:
        if score < 0.2:
            break
        # Determine if join is needed
        # Join needed if source table is NOT the primary entity table
        is_primary = _is_primary_table(tname)
        needs_join = not is_primary

        join_path = []
        if needs_join:
            for e in join_edges:
                if e.left_table == tname or e.right_table == tname:
                    join_path = [{
                        "from": e.left_table,
                        "to": e.right_table,
                        "left_cols": list(e.left_cols),
                        "right_cols": list(e.right_cols),
                        "confidence": round(float(e.confidence), 4),
                    }]
                    break

        src_cols = [(tname, cname, ctype)]
        transform = infer_transform_type(target_name, target_type, src_cols, needs_join)
        conf = min(0.99, score * 0.8 + 0.2)  # scale to reasonable confidence

        candidates.append({
            "source_columns": [f"{tname}.{cname}"],
            "transform_type": transform,
            "confidence": round(conf, 3),
            "join_path": join_path,
            "score": score,
            "is_single": True,
        })

    # --- Multi-column candidates (concat patterns) ---
    tgt_tok_set = set(tgt_tok)
    desc_tok_set = set(desc_tok)
    all_hint_tokens = tgt_tok_set | desc_tok_set

    # Check for concat patterns: full_name -> first_name + last_name
    if target_type.lower() in _STR_TYPES | _NUM_TYPES:
        # Group scored cols by table
        by_table: Dict[str, List[Tuple[float, str, str]]] = defaultdict(list)
        for score, tname, cname, ctype in scored[:20]:
            by_table[tname].append((score, cname, ctype))

        for tname, cols in by_table.items():
            if len(cols) >= 2:
                # Try pairs/triples of columns from same table
                for i in range(min(len(cols), 5)):
                    for j in range(i + 1, min(len(cols), 5)):
                        c1 = cols[i]
                        c2 = cols[j]
                        pair_tokens = set(_tokenize(c1[1])) | set(_tokenize(c2[1]))
                        # Good combo if their combined tokens cover target tokens
                        coverage = len(pair_tokens & all_hint_tokens) / max(len(all_hint_tokens), 1)
                        if coverage > 0.3 or (c1[0] + c2[0]) > 0.7:
                            src_cols = [
                                (tname, c1[1], c1[2]),
                                (tname, c2[1], c2[2]),
                            ]
                            transform = infer_transform_type(target_name, target_type, src_cols, False)
                            combo_score = (c1[0] + c2[0]) / 2 * 0.9
                            candidates.append({
                                "source_columns": [f"{tname}.{c1[1]}", f"{tname}.{c2[1]}"],
                                "transform_type": transform,
                                "confidence": round(min(0.99, combo_score * 0.8 + 0.15), 3),
                                "join_path": [],
                                "score": combo_score,
                                "is_single": False,
                            })

    # --- Cross-table lookup candidates ---
    # e.g., department_name needs departments.dept_name via employees->departments join
    for score, tname, cname, ctype in scored[:5]:
        if score > 0.35:
            # Check if we can reach this table via joins
            for other_table in source_tables:
                if other_table != tname and other_table in table_joins.get(tname, {}):
                    edge = table_joins[tname][other_table]
                    jp = [{
                        "from": edge.left_table,
                        "to": edge.right_table,
                        "left_cols": list(edge.left_cols),
                        "right_cols": list(edge.right_cols),
                        "confidence": round(float(edge.confidence), 4),
                    }]
                    candidates.append({
                        "source_columns": [f"{tname}.{cname}"],
                        "transform_type": "fk_lookup",
                        "confidence": round(min(0.95, score * 0.7 + 0.15), 3),
                        "join_path": jp,
                        "score": score * 0.85,
                        "is_single": True,
                    })

    # De-duplicate by source_columns key
    seen = set()
    deduped = []
    for c in candidates:
        key = tuple(sorted(c["source_columns"]))
        if key not in seen:
            seen.add(key)
            deduped.append(c)

    deduped.sort(key=lambda x: -x["confidence"])
    return deduped


# ---------------------------------------------------------------
# Smart result selection
# ---------------------------------------------------------------

def _detect_concat_need(
    target_name: str,
    target_desc: str,
    candidates: List[Dict[str, Any]],
    source_tables: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Detect if the target column needs a concat/template of multiple source columns.
    Returns the best combo candidate if so, else None.
    """
    tgt_tok = _tokenize(target_name)
    desc_tok = _tokenize(target_desc) if target_desc else []
    tgt_name_lower = target_name.lower().replace("_", " ")

    # Check known composite patterns
    for pattern_key, required_sets in _COMPOSITE_PATTERNS.items():
        if pattern_key in tgt_name_lower or all(t in tgt_tok + desc_tok for t in pattern_key.split("_")):
            for required_tokens in required_sets:
                # Find candidates that collectively have these tokens
                for c in candidates:
                    if not c.get("is_single", True) and len(c["source_columns"]) >= 2:
                        all_src_tok = set()
                        for ref in c["source_columns"]:
                            parts = ref.split(".", 1)
                            if len(parts) == 2:
                                all_src_tok.update(_tokenize(parts[1]))
                        if all(rt in all_src_tok for rt in required_tokens):
                            # Filter out irrelevant columns (keep only those with tokens in required set)
                            filtered_cols = []
                            required_set = set(required_tokens)
                            for ref in c["source_columns"]:
                                p = ref.split(".", 1)
                                if len(p) == 2:
                                    col_tok = set(_tokenize(p[1]))
                                    if col_tok & required_set:
                                        filtered_cols.append(ref)
                            if len(filtered_cols) >= 2:
                                return {**c, "source_columns": filtered_cols}
                            return c

    # Check description hints
    desc_lower = target_desc.lower() if target_desc else ""
    concat_keywords = ["combining", "combined", "concatenat", "merge", "full name",
                       "first and last", "formatted as", "template"]
    if any(kw in desc_lower for kw in concat_keywords):
        # Look for multi-column candidates
        best_combo = None
        best_score = 0
        for c in candidates:
            if not c.get("is_single", True) and len(c["source_columns"]) >= 2:
                score = c.get("confidence", 0)
                if score > best_score:
                    best_score = score
                    best_combo = c
        if best_combo:
            return best_combo

    # Check if target contains "full" + "name" tokens → need first + last
    if "full" in tgt_tok and "name" in tgt_tok:
        for c in candidates:
            if not c.get("is_single", True):
                src_tok = set()
                for ref in c["source_columns"]:
                    p = ref.split(".", 1)
                    if len(p) == 2:
                        src_tok.update(_tokenize(p[1]))
                if ("first" in src_tok or "first_name" in src_tok) and ("last" in src_tok or "last_name" in src_tok):
                    return c

    # Check for "lead_name", "manager_name" etc → same as full_name
    # BUT NOT "department_name", "project_name" etc. (those are entity.attribute, not concat)
    # Only trigger if the role token refers to a PERSON, not an entity table
    _PERSON_ROLE_TOKENS = {"lead", "manager", "mgr", "employee", "emp", "customer",
                           "cust", "user", "contact", "owner", "author", "creator",
                           "approver", "reviewer", "assignee", "supervisor"}
    _ENTITY_TOKENS = {"department", "dept", "project", "proj", "product", "prod",
                      "account", "acct", "order", "table", "category", "org",
                      "organization", "company", "team", "group", "region", "store"}

    if "name" in tgt_tok and len(tgt_tok) >= 2:
        role_tokens = [t for t in tgt_tok if t != "name"]
        # Only concat if role refers to a person (lead_name, manager_name)
        # NOT if role refers to an entity (department_name, project_name)
        role_exp = set(_expand_tokens(role_tokens))
        is_person_role = bool(role_exp & _PERSON_ROLE_TOKENS)
        is_entity_role = bool(role_exp & _ENTITY_TOKENS)

        if is_person_role and not is_entity_role:
            for c in candidates:
                if not c.get("is_single", True):
                    src_tok = set()
                    for ref in c["source_columns"]:
                        p = ref.split(".", 1)
                        if len(p) == 2:
                            src_tok.update(_tokenize(p[1]))
                    if ("first" in src_tok) and ("last" in src_tok):
                        return c

    return None


def _detect_fk_lookup(
    target_name: str,
    target_type: str,
    target_desc: str,
    candidates: List[Dict[str, Any]],
    source_tables: Dict[str, Any],
    join_edges: List[JoinEdge],
) -> Optional[Dict[str, Any]]:
    """
    Detect if the target needs a cross-table FK lookup.
    E.g., department_name -> departments.dept_name via employees.department_id join.
    """
    tgt_tok = set(_tokenize(target_name))
    tgt_exp = set(_expand_tokens(list(tgt_tok)))
    desc_tok = set(_tokenize(target_desc)) if target_desc else set()
    all_tokens = tgt_exp | desc_tok

    # Find which source table the target entity most likely comes from
    # e.g., "department_name" -> "departments" table
    best_table = None
    best_table_score = 0
    for tname in source_tables:
        table_tok = set(_tokenize(tname))
        table_exp = set(_expand_tokens(list(table_tok)))
        overlap = len(table_exp & all_tokens) / max(len(all_tokens), 1)
        if overlap > best_table_score:
            best_table_score = overlap
            best_table = tname

    if best_table and best_table_score > 0.15:
        # Look for the best column in that table
        table = source_tables[best_table]
        best_col = None
        best_col_score = 0
        for cname, col in table.columns.items():
            sim = _name_similarity(cname, target_name)
            ctype = _base_type(col.col_type)
            # Strong type compatibility bonus/penalty
            if _types_compatible(ctype, target_type):
                type_bonus = 0.2
            else:
                type_bonus = -0.2  # penalize type mismatch (e.g., int col for string target)
            # Penalize PKs/FKs for non-key targets (dept_id shouldn't map to dept_name target)
            pk_penalty = 0.0
            if col.constraints and (col.constraints.is_primary_key or col.constraints.is_foreign_key):
                tgt_tok_local = set(_tokenize(target_name))
                if not tgt_tok_local & {"key", "id", "pk", "fk", "identifier", "code"}:
                    pk_penalty = -0.15
            total = sim + type_bonus + pk_penalty
            if total > best_col_score:
                best_col_score = total
                best_col = cname

        if best_col and best_col_score > 0.3:
            # Check if this table needs a join to be reached
            # A join is needed if the target table entity doesn't directly
            # contain this source table (it's in a different table)
            join_path = []
            needs_join = False
            for e in join_edges:
                if (e.left_table == best_table or e.right_table == best_table):
                    join_path = [{
                        "from": e.left_table,
                        "to": e.right_table,
                        "left_cols": list(e.left_cols),
                        "right_cols": list(e.right_cols),
                        "confidence": round(float(e.confidence), 4),
                    }]
                    needs_join = True
                    break

            # Determine transform type
            ctype = _base_type(source_tables[best_table].columns[best_col].col_type)
            src_cols_parsed = [(best_table, best_col, ctype)]
            transform = infer_transform_type(target_name, target_type, src_cols_parsed, needs_join)
            # Override to fk_lookup if cross-table join is needed
            if needs_join and transform in ("rename", "direct_copy"):
                transform = "fk_lookup"

            return {
                "source_columns": [f"{best_table}.{best_col}"],
                "transform_type": transform,
                "confidence": round(min(0.95, best_col_score * 0.7 + 0.25), 3),
                "join_path": join_path,
                "score": best_col_score,
                "is_single": True,
            }

    return None


def _select_best_mapping(
    target_name: str,
    target_type: str,
    target_desc: str,
    stage_a_candidates: List[Dict[str, Any]],
    name_match_candidates: List[Dict[str, Any]],
    source_tables: Optional[Dict[str, Any]] = None,
    join_edges: Optional[List[JoinEdge]] = None,
) -> Dict[str, Any]:
    """
    Select the best mapping from combined Stage A + name-match candidates.
    Uses pattern detection for concat, fk_lookup, and simplicity preference.
    """
    tgt_tok = set(_tokenize(target_name))
    desc_tok = set(_tokenize(target_desc)) if target_desc else set()

    # Merge all candidates
    all_candidates = []

    # Add name-match candidates (with priority flag)
    for c in name_match_candidates:
        all_candidates.append({**c, "_source": "name_match"})

    # Add Stage A candidates (reprocessed)
    for c in stage_a_candidates:
        src_cols = c.get("candidate_columns", [])
        if not src_cols:
            continue
        parsed = []
        for ref in src_cols:
            parts = ref.split(".", 1)
            if len(parts) == 2:
                parsed.append((parts[0], parts[1], "string"))

        join_path = c.get("join_path", [])
        has_join = len(join_path) > 0
        transform = infer_transform_type(target_name, target_type, parsed, has_join)

        original_family = c.get("best_transform_family", "unknown")
        feasible = c.get("feasible_families", [])
        if original_family != "lookup_join" and original_family in feasible:
            transform = original_family

        all_candidates.append({
            "source_columns": src_cols,
            "transform_type": transform,
            "confidence": c.get("confidence", 0),
            "join_path": join_path,
            "score": c.get("final_score", 0),
            "is_single": len(src_cols) == 1,
            "_source": "stage_a",
        })

    if not all_candidates:
        return {
            "source_columns": [],
            "transform_type": "unmapped",
            "confidence": 0,
            "join_path": [],
        }

    # ===== Pattern detection =====

    # 1) Detect if target needs concat (full_name, lead_name, etc.)
    concat_result = _detect_concat_need(target_name, target_desc, all_candidates, source_tables)
    if concat_result:
        # Re-infer transform
        src_parsed = []
        for ref in concat_result["source_columns"]:
            p = ref.split(".", 1)
            if len(p) == 2:
                src_parsed.append((p[0], p[1], "string"))
        transform = infer_transform_type(target_name, target_type, src_parsed, bool(concat_result.get("join_path")))
        if transform not in ("concat", "template"):
            transform = "concat"
        return {
            "source_columns": concat_result["source_columns"],
            "transform_type": transform,
            "confidence": max(concat_result.get("confidence", 0.7), 0.75),
            "join_path": concat_result.get("join_path", []),
        }

    # 2) Detect if target needs fk_lookup (department_name, owning_department, etc.)
    if source_tables and join_edges:
        fk_result = _detect_fk_lookup(
            target_name, target_type, target_desc,
            all_candidates, source_tables, join_edges,
        )
        if fk_result and fk_result["score"] > 0.4:
            # Check if we already have a better direct match
            best_direct = None
            for c in all_candidates:
                if c.get("is_single") and c.get("confidence", 0) > 0.9:
                    # Very high confidence direct match trumps fk_lookup
                    best_direct = c
                    break
            if not best_direct or fk_result["score"] > best_direct.get("score", 0):
                return {
                    "source_columns": fk_result["source_columns"],
                    "transform_type": fk_result["transform_type"],
                    "confidence": fk_result["confidence"],
                    "join_path": fk_result.get("join_path", []),
                }

    # ===== General ranking =====

    def _rank_score(c: Dict) -> float:
        base = c.get("confidence", 0)
        src_cols = c["source_columns"]
        n = len(src_cols)
        transform = c.get("transform_type", "unknown")

        # Simplicity bonus
        if n == 1:
            base += 0.15
        elif n == 2:
            base += 0.05
        else:
            base -= 0.05 * (n - 2)

        # Name-match priority
        if c.get("_source") == "name_match":
            base += 0.10

        # Specific transform bonus
        if transform in ("direct_copy", "rename"):
            base += 0.10
        elif transform in ("date_part", "concat", "fk_lookup"):
            base += 0.08
        elif transform == "lookup_join":
            base -= 0.05

        # Source column relevance
        for ref in src_cols:
            parts = ref.split(".", 1)
            if len(parts) == 2:
                sim = _name_similarity(parts[1], target_name)
                base += sim * 0.15

        # Penalize irrelevant extra columns
        if n > 1:
            relevant = 0
            for ref in src_cols:
                parts = ref.split(".", 1)
                if len(parts) == 2:
                    col_tok = set(_tokenize(parts[1]))
                    if col_tok & (tgt_tok | desc_tok):
                        relevant += 1
            base -= (n - relevant) * 0.1

        return base

    all_candidates.sort(key=lambda c: -_rank_score(c))
    best = all_candidates[0]

    return {
        "source_columns": best["source_columns"],
        "transform_type": best["transform_type"],
        "confidence": best.get("confidence", 0),
        "join_path": best.get("join_path", []),
    }


# ---------------------------------------------------------------
# Converters: JSON -> domain objects
# ---------------------------------------------------------------

def _parse_constraints(col_json: dict) -> ColumnConstraints:
    return ColumnConstraints(
        nullable=col_json.get("nullable", True),
        is_primary_key=col_json.get("is_pk", False),
        is_unique=col_json.get("is_pk", False) or col_json.get("is_unique", False),
        is_foreign_key=col_json.get("is_fk", False),
    )


def json_to_tables(tables_json: List[dict]) -> Dict[str, Table]:
    """Convert JSON table definitions to Table objects."""
    tables: Dict[str, Table] = {}
    for tj in tables_json:
        tname = tj["name"]
        sample_rows = tj.get("sample_data", [])
        columns: Dict[str, Column] = {}
        for cj in tj.get("columns", []):
            cname = cj["name"]
            ctype = cj.get("type", "string")
            values = [r.get(cname) for r in sample_rows] if sample_rows else []
            columns[cname] = Column(
                name=cname,
                col_type=ColumnType(ctype),
                constraints=_parse_constraints(cj),
                stats=ColumnStats(values),
            )
        tables[tname] = Table(
            name=tname,
            columns=columns,
            row_count=len(sample_rows),
            rows=sample_rows if sample_rows else None,
        )
    return tables


# ---------------------------------------------------------------
# Stage 1: Join Graph Discovery
# ---------------------------------------------------------------

def discover_joins(tables: Dict[str, Table]) -> Tuple[List[JoinEdge], float]:
    """Run JoinGraphBuilderV2 and return edges + elapsed time."""
    t0 = time.time()
    builder = JoinGraphBuilderV2(
        tables=tables,
        value_sim_class=ValueSimilarity,
        constraint_engine=ConstraintCompatibilityJoin,
    )
    edges = builder.build()
    elapsed = time.time() - t0
    return edges, elapsed


def edges_to_json(edges: List[JoinEdge]) -> List[dict]:
    """Serialize JoinEdge list for the frontend."""
    out = []
    for e in edges:
        out.append({
            "left_table": e.left_table,
            "right_table": e.right_table,
            "left_cols": list(e.left_cols),
            "right_cols": list(e.right_cols),
            "cardinality": e.cardinality,
            "confidence": round(float(e.confidence), 3),
            "reasons": list(e.reasons) if e.reasons else [],
        })
    return out


def tables_to_summary(tables: Dict[str, Table]) -> List[dict]:
    """Build a frontend-friendly summary of tables."""
    out = []
    for tname, t in tables.items():
        cols = []
        for cname, c in t.columns.items():
            cols.append({
                "name": cname,
                "type": _base_type(c.col_type),
                "is_pk": c.constraints.is_primary_key if c.constraints else False,
                "is_fk": c.constraints.is_foreign_key if c.constraints else False,
                "nullable": c.constraints.nullable if c.constraints else True,
            })
        out.append({
            "name": tname,
            "columns": cols,
            "row_count": t.row_count,
            "col_count": len(t.columns),
        })
    return out


# ---------------------------------------------------------------
# Stage A: Candidate Generation (with post-processing)
# ---------------------------------------------------------------

def _load_embedder() -> NameEmbedder:
    """Try to load SentenceTransformer embedder, fall back to hashed."""
    try:
        emb = NameEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return emb
    except Exception:
        return NameEmbedder(model_name=None)


def _load_stage_b(artifacts_root: Path) -> Optional[Any]:
    """Load Stage B selector if available."""
    if not HAS_STAGE_B:
        return None
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

    try:
        return CandidateSelectorStage1(biencoder_path=bi_path, cross_encoder_path=cross_path)
    except Exception:
        return None


def _try_v3_mapping(
    source_tables: Dict[str, Table],
    join_edges: List[JoinEdge],
    target_tables_json: List[dict],
) -> Optional[Dict[str, Any]]:
    """Try the V3 LLM-based engine. Returns None if not available."""
    if not HAS_V3:
        return None

    model_dir = "schema_matcher_llm1"
    adapter_path = Path(model_dir) / "adapter" / "adapter_config.json"

    if not adapter_path.exists():
        print("[V3] No fine-tuned LLM found, skipping V3")
        return None

    try:
        engine = SchemaMatcherLLM(
            source_tables=source_tables,
            join_edges=join_edges,
            model_dir=model_dir,
            device="cpu",
        )
        return engine.rank_all_targets(target_tables_json, top_k=5)
    except Exception as exc:
        print(f"[V3] Failed: {exc}")
        return None


def _try_v2_mapping(
    source_tables: Dict[str, Table],
    join_edges: List[JoinEdge],
    target_tables_json: List[dict],
) -> Optional[Dict[str, Any]]:
    """Try the V2 ML-based engine. Returns None if not available."""
    if not HAS_V2:
        return None

    # Check for fine-tuned models, fall back to base models
    bi_path = "candidate_v2_models/bi_encoder"
    cross_path = "candidate_v2_models/cross_encoder"

    if not Path(bi_path).exists():
        bi_path = "sentence-transformers/all-MiniLM-L6-v2"
    if not Path(cross_path).exists():
        cross_path = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    try:
        engine = CandidateGeneratorV2(
            source_tables=source_tables,
            join_edges=join_edges,
            bi_encoder_path=bi_path,
            cross_encoder_path=cross_path,
        )
        return engine.rank_all_targets(target_tables_json, top_k=5)
    except Exception as exc:
        print(f"[V2] Failed: {exc}")
        return None


def run_full_mapping(
    source_tables: Dict[str, Table],
    join_edges: List[JoinEdge],
    target_tables_json: List[dict],
    artifacts_root: str = "artifacts/stage1_candidate_selector",
    use_v2: bool = False,  # Bi-encoder disabled: use schema_matcher_llm1 (V3)
    use_v3: bool = True,   # V3 LLM (schema_matcher_llm1) on CPU
) -> Dict[str, Any]:
    """
    Run candidate selection pipeline for every column in every target table.
    Uses schema_matcher_llm1 (V3 LLM) on CPU. Bi-encoder (V2) disabled.
    """
    # V3 LLM (schema_matcher_llm1) on CPU
    if use_v3:
        v3_result = _try_v3_mapping(source_tables, join_edges, target_tables_json)
        if v3_result is not None:
            return v3_result

    # V2 bi-encoder (optional fallback)
    if use_v2:
        v2_result = _try_v2_mapping(source_tables, join_edges, target_tables_json)
        if v2_result is not None:
            return v2_result
    results: Dict[str, Any] = {"tables": [], "stats": {}}
    total_cols = 0
    mapped_cols = 0
    t_start = time.time()

    # Init engines
    embedder = _load_embedder()
    engine = CandidateGenerationEngine(
        source_tables=source_tables,
        join_edges=join_edges,
        embedder=embedder,
    )

    for ttable in target_tables_json:
        tname = ttable["name"]
        table_result = {"name": tname, "columns": []}

        for tcol in ttable.get("columns", []):
            total_cols += 1
            col_name = tcol["name"]
            col_type = tcol.get("type", "string")
            col_desc = tcol.get("description", "")

            target_spec = TargetSpec(
                table=tname,
                name=col_name,
                col_type=ColumnType(col_type),
                stats=ColumnStats([]),
                constraints=ColumnConstraints(),
                description=col_desc,
            )

            col_result: Dict[str, Any] = {
                "name": col_name,
                "type": col_type,
                "description": col_desc,
            }

            try:
                # --- Stage A: Candidate Generation ---
                t0 = time.time()
                sa = engine.rank_candidates(
                    target=target_spec,
                    coarse_top_m=50, fine_top_m=25,
                    max_arity=4, max_hops=3,
                    top_k=10, abstain_threshold=0.30,
                )
                a_time = time.time() - t0

                stage_a_cands = sa.get("top_candidates", [])

                # --- Name-Match Fallback ---
                t1 = time.time()
                name_match_cands = _find_name_match_candidates(
                    target_name=col_name,
                    target_type=col_type,
                    target_desc=col_desc,
                    target_table=tname,
                    source_tables=source_tables,
                    join_edges=join_edges,
                )
                nm_time = time.time() - t1

                # --- Smart Selection ---
                best = _select_best_mapping(
                    target_name=col_name,
                    target_type=col_type,
                    target_desc=col_desc,
                    stage_a_candidates=stage_a_cands,
                    name_match_candidates=name_match_cands,
                    source_tables=source_tables,
                    join_edges=join_edges,
                )

                col_result["stage_a"] = {
                    "source_columns": best["source_columns"],
                    "join_path": best.get("join_path", []),
                    "transform_family": best["transform_type"],
                    "confidence": best["confidence"],
                    "time": round(a_time + nm_time, 2),
                    "abstain": sa.get("abstain", True),
                }

                # Build alternatives from remaining candidates
                all_alts = name_match_cands[1:4] if len(name_match_cands) > 1 else []
                if not all_alts and len(stage_a_cands) > 1:
                    for c in stage_a_cands[1:4]:
                        src = c.get("candidate_columns", [])
                        parsed = [(s.split(".", 1)[0], s.split(".", 1)[1], "string")
                                  for s in src if "." in s]
                        t = infer_transform_type(col_name, col_type, parsed, bool(c.get("join_path")))
                        all_alts.append({
                            "source_columns": src,
                            "transform_type": t,
                            "confidence": c.get("confidence", 0),
                        })

                col_result["alternatives"] = [
                    {
                        "source_columns": a.get("source_columns", []),
                        "transform_family": a.get("transform_type", "unknown"),
                        "confidence": round(a.get("confidence", 0), 3),
                    }
                    for a in all_alts[:3]
                ]

                if best["source_columns"]:
                    mapped_cols += 1

                # Final result
                col_result["final_source"] = best["source_columns"]
                col_result["final_transform"] = best["transform_type"]
                col_result["final_confidence"] = best["confidence"]

            except Exception as exc:
                import traceback
                traceback.print_exc()
                col_result["error"] = str(exc)
                col_result["final_source"] = []
                col_result["final_transform"] = "unmapped"
                col_result["final_confidence"] = 0

            table_result["columns"].append(col_result)

        results["tables"].append(table_result)

    results["stats"] = {
        "total_columns": total_cols,
        "mapped_columns": mapped_cols,
        "mapping_rate": round(mapped_cols / max(1, total_cols) * 100, 1),
        "total_time": round(time.time() - t_start, 2),
        "has_stage_b": False,  # We use smart selection instead
    }
    return results
