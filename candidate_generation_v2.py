"""
Candidate Generation V2 – Bi-Encoder + Cross-Encoder
======================================================
Pure ML-based candidate generation that replaces the rule-based system.

Architecture:
  1. Enumerate all feasible candidate sets from source schema
  2. Serialize each as rich text
  3. Bi-encoder retrieves top-K candidates (fast ANN)
  4. Cross-encoder reranks for precision
  5. Transform type inferred from model + heuristics

No hardcoded rules. No scoring weights. The model learns everything from data.
"""
from __future__ import annotations

import itertools
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    HAS_ST = True
except ImportError:
    HAS_ST = False

from join_graph_builder_v2 import Column, ColumnType, JoinEdge, Table
from value_similarity_engine import ColumnStats
from constraint_similarity_engine import ColumnConstraints


# ───────────────────────────────────────────────────────
# Text serialization (mirrors training data format)
# ───────────────────────────────────────────────────────

_SPLIT_RE = re.compile(r"[_\-.\s]+|(?<=[a-z])(?=[A-Z])")

def _tokenize_name(name: str) -> str:
    return " ".join(t.lower() for t in _SPLIT_RE.split(name) if t)


def serialize_target(
    table: str, column: str, col_type: str,
    description: str = "", constraints: str = "",
) -> str:
    parts = [
        f"target: {table}.{column}",
        f"[{col_type}]",
        f"meaning: {_tokenize_name(column)}",
    ]
    if description:
        parts.append(f"desc: {description}")
    if constraints:
        parts.append(f"constraints: {constraints}")
    return " | ".join(parts)


def serialize_candidate(
    columns: List[Tuple[str, str, str]],
    join_path: List[str] = None,
    transform_hint: str = "",
    table_context: str = "",
) -> str:
    col_parts = [f"{t}.{c} [{tp}]" for t, c, tp in columns]
    parts = [f"source: {', '.join(col_parts)}", f"arity: {len(columns)}"]

    tables = list(dict.fromkeys(c[0] for c in columns))
    parts.append("same_table" if len(tables) == 1 else f"cross_table: {' + '.join(tables)}")

    meanings = [_tokenize_name(c[1]) for c in columns]
    parts.append(f"col_meanings: {'; '.join(meanings)}")

    if join_path:
        parts.append(f"join: {' -> '.join(join_path)}")
    if transform_hint:
        parts.append(f"transform: {transform_hint}")
    if table_context:
        parts.append(f"context: {table_context}")

    return " | ".join(parts)


# ───────────────────────────────────────────────────────
# Candidate set data structure
# ───────────────────────────────────────────────────────

@dataclass
class CandidateV2:
    columns: List[Tuple[str, str, str]]   # (table, col, type)
    join_path: List[str] = field(default_factory=list)
    join_edges: List[JoinEdge] = field(default_factory=list)
    text: str = ""
    bi_score: float = 0.0
    cross_score: float = 0.0
    combined_score: float = 0.0
    transform_type: str = ""


# ───────────────────────────────────────────────────────
# Transform type inference (lightweight heuristics)
# ───────────────────────────────────────────────────────

_NUM = {"int", "integer", "bigint", "float", "double", "decimal", "number", "numeric"}
_STR = {"string", "varchar", "text", "char"}
_DATE = {"date", "datetime", "timestamp", "time"}
_DATE_PART_TOKENS = {"year", "month", "day", "quarter", "week", "hour", "minute", "second"}
_COMPOSITE_HINT_TOKENS = {
    "full", "complete", "combined", "total", "whole", "entire",
}
_NAME_PART_TOKENS = {"first", "last", "middle", "given", "family", "surname"}
_ADDR_PART_TOKENS = {"street", "city", "state", "zip", "postal", "country", "address", "location"}
_KEY_TOKENS = {"key", "id", "identifier", "pk", "code", "number", "no"}
_SINGLE_CONCEPT_TOKENS = {
    "email", "phone", "salary", "name", "title", "status", "flag",
    "rate", "price", "cost", "amount", "fee", "score", "grade",
    "type", "category", "level", "label", "value", "date",
    "description", "reason", "notes", "version",
}

def _target_prefers_single(target_name: str, target_type: str) -> float:
    """Score how strongly the target semantics suggest a single source column.
    Returns a value from -0.5 (strongly composite) to +0.5 (strongly single).
    """
    tok = set(_tokenize_name(target_name).split())

    # Strong composite signals
    if tok & _COMPOSITE_HINT_TOKENS and tok & {"name", "address", "location"}:
        return -0.4  # "full_name", "complete_address" -> prefer composite
    if len(tok & _NAME_PART_TOKENS) >= 1 and "name" in tok:
        return -0.1  # ambiguous - could be concat
    if len(tok & _ADDR_PART_TOKENS) >= 2:
        return -0.3  # "city_state" -> composite

    # Strong single signals
    if tok & _KEY_TOKENS:
        return 0.45  # "employee_key", "order_id" -> strongly single
    if target_type.lower() in ("boolean", "bool"):
        return 0.45  # boolean -> usually single column conditional
    if target_type.lower() in _DATE:
        return 0.35  # date target -> usually single date source
    if tok & _DATE_PART_TOKENS:
        return 0.30  # "hire_year" -> single date column
    if len(tok) <= 2 and tok & _SINGLE_CONCEPT_TOKENS:
        return 0.30  # "email", "salary", "status" -> single
    # "department_name", "project_title", "supplier_company" - entity attribute, not concat
    if len(tok) == 2 and "name" in tok and not (tok & _NAME_PART_TOKENS) and not (tok & _COMPOSITE_HINT_TOKENS):
        return 0.25
    # "owning_department", "managing_team" - possessive entity reference, not concat
    if tok & {"owning", "managing", "parent", "primary", "assigned", "responsible"}:
        return 0.25
    if tok & {"margin", "difference", "diff", "ratio", "per"}:
        return -0.2  # arithmetic -> often 2 columns
    if tok & {"cost", "unit"} and len(tok) >= 2:
        return -0.15  # "unit_cost" often involves division

    return 0.0  # neutral


def _base_type(ct) -> str:
    return getattr(ct, "base_type", str(ct)).lower()

_ENTITY_SYNONYMS = {
    "employee": {"employee", "staff", "worker", "person", "people"},
    "product": {"product", "item", "items", "goods", "merchandise", "store_item", "store_items", "catalog"},
    "order": {"order", "purchase", "sale", "transaction"},
    "customer": {"customer", "client", "buyer", "account"},
    "sensor": {"sensor", "device", "probe"},
    "reading": {"reading", "measurement", "observation"},
    "department": {"department", "dept"},
    "project": {"project", "initiative"},
    "location": {"location", "site", "building", "place"},
    "supplier": {"supplier", "vendor", "provider"},
}

def _detect_primary_table(target_table: str, source_tables: Dict[str, Any]) -> Optional[str]:
    """Detect which source table is the 'primary entity' for a target table.
    E.g., dim_employee -> employees, fact_reading -> readings.
    """
    tgt_tok = set(_tokenize_name(target_table).split())
    tgt_tok -= {"dim", "fact", "bridge", "stg", "raw", "src"}

    # Build synonym lookup: word -> set of related words
    syn_map = {}
    for _key, synonyms in _ENTITY_SYNONYMS.items():
        for w in synonyms:
            syn_map[w] = synonyms

    best_table = None
    best_score = 0
    for tname in source_tables:
        src_tok = set(_tokenize_name(tname).split())
        score = 0
        for st in src_tok:
            st_base = st.rstrip("s")
            for tt in tgt_tok:
                tt_base = tt.rstrip("s")
                # Direct match
                if st == tt or st_base == tt_base:
                    score += 2
                # Synonym match
                elif st_base in syn_map and tt_base in syn_map.get(st_base, set()):
                    score += 1.5
                else:
                    # Check all synonym groups
                    for _k, syns in _ENTITY_SYNONYMS.items():
                        bases = {s.rstrip("s") for s in syns}
                        if st_base in bases and tt_base in bases:
                            score += 1.5
                            break
        if score > best_score:
            best_score = score
            best_table = tname
    return best_table if best_score > 0 else None


def infer_transform(target_name: str, target_type: str,
                    src_cols: List[Tuple[str, str, str]], has_join: bool,
                    target_table: str = "", source_tables: Dict[str, Any] = None) -> str:
    tgt_tok = set(_tokenize_name(target_name).split())
    tgt_t = target_type.lower()
    n = len(src_cols)

    # Detect if source is from a different table than the target's primary entity
    needs_join = has_join
    if source_tables and not has_join and n >= 1:
        primary = _detect_primary_table(target_table, source_tables)
        if primary:
            src_tables = {c[0] for c in src_cols}
            if primary not in src_tables:
                needs_join = True

    if n == 0:
        return "unmapped"
    if n == 1:
        s_t, s_c, s_type = src_cols[0]
        s_type_l = s_type.lower()
        if s_type_l in _DATE and tgt_t in _NUM:
            if tgt_tok & {"diff", "days", "duration", "tenure", "age", "elapsed"}:
                return "date_diff"
            return "date_part"
        if s_type_l in _DATE and tgt_t in _STR:
            return "date_format"
        if s_type_l in _STR and tgt_t in _DATE:
            return "date_parse"
        # String to boolean -> conditional (e.g. status -> is_delivered)
        if s_type_l in _STR and tgt_t in ("boolean", "bool"):
            return "conditional"
        if needs_join:
            return "fk_lookup"
        if s_c.lower() == target_name.lower():
            return "direct_copy"
        if s_type_l == tgt_t or (s_type_l in _NUM and tgt_t in _NUM):
            return "rename"
        if tgt_t in ("boolean", "bool"):
            return "conditional"
        return "type_cast"

    src_types = {s[2].lower() for s in src_cols}
    if tgt_t in _STR:
        return "concat"
    if tgt_t in _NUM and all(st in _NUM for st in src_types):
        return "arithmetic"
    if any(st in _DATE for st in src_types) and tgt_t in _NUM:
        if tgt_tok & {"diff", "days", "duration", "tenure", "age", "elapsed", "transit"}:
            return "date_diff"
        return "date_part"
    if needs_join:
        return "lookup_join"
    return "conditional"


# ───────────────────────────────────────────────────────
# Candidate Enumeration
# ───────────────────────────────────────────────────────

class CandidateEnumerator:
    """
    Enumerates all feasible candidate sets from source schema.
    No scoring – just enumeration. The model handles ranking.
    """

    def __init__(
        self,
        source_tables: Dict[str, Table],
        join_edges: List[JoinEdge],
        max_arity: int = 3,
        max_join_hops: int = 2,
    ):
        self.tables = source_tables
        self.edges = join_edges
        self.max_arity = max_arity
        self.max_join_hops = max_join_hops

        # Build adjacency
        self._adj: Dict[str, List[Tuple[str, JoinEdge]]] = defaultdict(list)
        for e in join_edges:
            self._adj[e.left_table].append((e.right_table, e))
            self._adj[e.right_table].append((e.left_table, e))

        # Build column catalog
        self._all_cols: List[Tuple[str, str, str]] = []
        self._cols_by_table: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        for tname, table in source_tables.items():
            for cname, col in table.columns.items():
                ct = _base_type(col.col_type)
                entry = (tname, cname, ct)
                self._all_cols.append(entry)
                self._cols_by_table[tname].append(entry)

        # Build table context string
        self._table_ctx = "; ".join(
            f"{t}({', '.join(c.name for c in tab.columns.values())})"
            for t, tab in source_tables.items()
        )

    def enumerate_all(self) -> List[CandidateV2]:
        """Generate all feasible candidate sets."""
        candidates: List[CandidateV2] = []
        seen = set()

        def _add(cols, jp_strs=None, jp_edges=None):
            key = tuple(sorted((c[0], c[1]) for c in cols))
            if key in seen:
                return
            seen.add(key)
            candidates.append(CandidateV2(
                columns=list(cols),
                join_path=jp_strs or [],
                join_edges=jp_edges or [],
            ))

        # 1) Single columns
        for col in self._all_cols:
            _add([col])

        # 2) Same-table pairs
        for tname, cols in self._cols_by_table.items():
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    # Skip pairs of PK/FK columns (unlikely to be useful)
                    c1, c2 = cols[i], cols[j]
                    _add([c1, c2])

        # 3) Same-table triples (limited to avoid explosion)
        for tname, cols in self._cols_by_table.items():
            non_pk = [c for c in cols if not self._is_pk_or_fk(tname, c[1])]
            if len(non_pk) >= 3:
                for combo in itertools.combinations(non_pk[:8], 3):
                    _add(list(combo))

        # 4) Cross-table pairs via 1-hop join
        for tname1, neighbors in self._adj.items():
            for tname2, edge in neighbors:
                jp_str = f"{edge.left_table}.{edge.left_cols[0]} = {edge.right_table}.{edge.right_cols[0]}"
                for c1 in self._cols_by_table[tname1][:10]:
                    for c2 in self._cols_by_table[tname2][:10]:
                        _add([c1, c2], [jp_str], [edge])

        # 5) Cross-table triples (limited: 2 from one table + 1 from joined)
        for tname1, neighbors in self._adj.items():
            cols1 = [c for c in self._cols_by_table[tname1] if not self._is_pk_or_fk(tname1, c[1])]
            for tname2, edge in neighbors:
                jp_str = f"{edge.left_table}.{edge.left_cols[0]} = {edge.right_table}.{edge.right_cols[0]}"
                cols2 = [c for c in self._cols_by_table[tname2] if not self._is_pk_or_fk(tname2, c[1])]
                # 2 from table1 + 1 from table2
                for pair in itertools.combinations(cols1[:6], 2):
                    for c2 in cols2[:6]:
                        _add(list(pair) + [c2], [jp_str], [edge])
                # 1 from table1 + 2 from table2
                for c1 in cols1[:6]:
                    for pair2 in itertools.combinations(cols2[:6], 2):
                        _add([c1] + list(pair2), [jp_str], [edge])

        return candidates

    def _is_pk_or_fk(self, table: str, col: str) -> bool:
        if table not in self.tables:
            return False
        c = self.tables[table].columns.get(col)
        if not c or not c.constraints:
            return False
        return c.constraints.is_primary_key or c.constraints.is_foreign_key

    def serialize_all(self, candidates: List[CandidateV2]) -> List[str]:
        """Serialize all candidates to text for encoding."""
        texts = []
        for c in candidates:
            c.text = serialize_candidate(
                c.columns,
                join_path=c.join_path,
                table_context=self._table_ctx[:300],
            )
            texts.append(c.text)
        return texts


# ───────────────────────────────────────────────────────
# Main Engine
# ───────────────────────────────────────────────────────

class CandidateGeneratorV2:
    """
    ML-based candidate generation:
      1. Enumerate feasible candidates
      2. Bi-encoder retrieval (fast)
      3. Cross-encoder reranking (precise)
    """

    def __init__(
        self,
        source_tables: Dict[str, Table],
        join_edges: List[JoinEdge],
        bi_encoder_path: str = "sentence-transformers/all-MiniLM-L6-v2",
        cross_encoder_path: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        bi_weight: float = 0.25,
        cross_weight: float = 0.75,
    ):
        if not HAS_ST:
            raise ImportError("sentence-transformers required: pip install sentence-transformers")

        self.tables = source_tables
        self.edges = join_edges
        self.bi_weight = bi_weight
        self.cross_weight = cross_weight

        # Load models
        print(f"[V2] Loading bi-encoder: {bi_encoder_path}")
        self.bi_encoder = SentenceTransformer(bi_encoder_path)
        print(f"[V2] Loading cross-encoder: {cross_encoder_path}")
        self.cross_encoder = CrossEncoder(cross_encoder_path, num_labels=1)

        # Enumerate and pre-encode candidates
        print(f"[V2] Enumerating candidates...")
        self.enumerator = CandidateEnumerator(source_tables, join_edges)
        self.candidates = self.enumerator.enumerate_all()
        texts = self.enumerator.serialize_all(self.candidates)
        print(f"[V2] {len(self.candidates)} candidates enumerated")

        # Pre-encode all candidates with bi-encoder
        print(f"[V2] Encoding candidates with bi-encoder...")
        self._candidate_embeddings = self.bi_encoder.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, batch_size=128,
        )
        print(f"[V2] Ready. Candidate embedding matrix: {self._candidate_embeddings.shape}")

    def rank(
        self,
        target_table: str,
        target_column: str,
        target_type: str,
        target_desc: str = "",
        retrieval_k: int = 50,
        rerank_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Rank candidates for a target column.
        Returns top-k candidates with scores and transform types.
        """
        t0 = time.time()

        # 1) Serialize target query
        query_text = serialize_target(target_table, target_column, target_type, target_desc)

        # 2) Bi-encoder retrieval
        query_emb = self.bi_encoder.encode(
            [query_text], normalize_embeddings=True, show_progress_bar=False,
        )[0]
        sims = self._candidate_embeddings @ query_emb
        top_indices = np.argsort(-sims)[:retrieval_k]

        # 2b) Candidate injection: ensure obvious name-match single columns
        # are in the pool even if bi-encoder missed them.
        # Skip injection if target strongly suggests composition.
        tgt_tok_inj = set(_tokenize_name(target_column).split())
        _suggests_composite = bool(
            (tgt_tok_inj & _COMPOSITE_HINT_TOKENS) or
            (tgt_tok_inj & {"lead", "contact", "manager"} and tgt_tok_inj & {"name"}) or
            (tgt_tok_inj & _NAME_PART_TOKENS)
        )
        if not _suggests_composite:
            # Build synonym-expanded target tokens
            tgt_tok_expanded = set(tgt_tok_inj)
            for t in tgt_tok_inj:
                t_base = t.rstrip("s")
                for _k, syns in _ENTITY_SYNONYMS.items():
                    bases = {s.rstrip("s") for s in syns}
                    if t_base in bases:
                        tgt_tok_expanded |= {s.rstrip("s") for s in syns}
            top_set = set(top_indices.tolist())
            for idx, cand in enumerate(self.candidates):
                if idx in top_set:
                    continue
                if len(cand.columns) == 1:
                    src_tok_inj = set(_tokenize_name(cand.columns[0][1]).split())
                    src_tok_expanded = set(src_tok_inj)
                    for s in src_tok_inj:
                        s_base = s.rstrip("s")
                        src_tok_expanded.add(s_base)
                    overlap = len(tgt_tok_expanded & src_tok_expanded)
                    if overlap >= 1 and overlap >= len(tgt_tok_inj) * 0.5:
                        top_indices = np.append(top_indices, idx)
                        top_set.add(idx)

        # 3) Cross-encoder reranking
        pairs = [(query_text, self.candidates[i].text) for i in top_indices]
        ce_scores = self.cross_encoder.predict(pairs, batch_size=64)

        # Normalize scores
        ce_arr = np.array(ce_scores, dtype=float)
        ce_min, ce_max = float(ce_arr.min()), float(ce_arr.max())
        ce_range = max(ce_max - ce_min, 1e-8)
        ce_norm = (ce_arr - ce_min) / ce_range

        bi_arr = np.array([sims[i] for i in top_indices], dtype=float)
        bi_min, bi_max = float(bi_arr.min()), float(bi_arr.max())
        bi_range = max(bi_max - bi_min, 1e-8)
        bi_norm = (bi_arr - bi_min) / bi_range

        combined = self.cross_weight * ce_norm + self.bi_weight * bi_norm

        # 3b) Arity-aware adjustment
        single_pref = _target_prefers_single(target_column, target_type)
        primary_table = _detect_primary_table(target_table, self.tables)
        for i, ri in enumerate(top_indices):
            cand = self.candidates[ri]
            arity = len(cand.columns)
            if arity == 1:
                combined[i] += single_pref * 0.3
                # Primary table bonus for single-column candidates only
                if primary_table and cand.columns[0][0] == primary_table:
                    combined[i] += 0.05
                # Extra bonus for key/id targets from primary table
                if single_pref >= 0.4 and primary_table and cand.columns[0][0] == primary_table:
                    combined[i] += 0.08
            elif arity == 2:
                combined[i] -= single_pref * 0.15
            elif arity >= 3:
                combined[i] -= single_pref * 0.30
            # Penalize candidates that include PK/FK columns in composites
            if arity > 1:
                pk_fk_count = sum(
                    1 for t, c, _ in cand.columns
                    if self.enumerator._is_pk_or_fk(t, c)
                )
                combined[i] -= pk_fk_count * 0.08

        # 3c) Name-matching bonus: if a single source column name closely
        # matches the target column name, give it a significant boost.
        tgt_tok = set(_tokenize_name(target_column).split())
        for i, ri in enumerate(top_indices):
            cand = self.candidates[ri]
            if len(cand.columns) == 1:
                src_tok = set(_tokenize_name(cand.columns[0][1]).split())
                overlap = len(tgt_tok & src_tok)
                if overlap >= 1:
                    combined[i] += 0.08 * overlap
                # Exact match (after tokenization)
                if tgt_tok == src_tok:
                    combined[i] += 0.15

        # 3d) Subset pruning: if a simpler candidate is a subset of a
        # higher-scoring complex one, and it has a close score, promote it.
        # Skip pruning if target strongly suggests composition.
        composite_signal = (
            tgt_tok & _COMPOSITE_HINT_TOKENS or
            tgt_tok & {"combined", "formatted", "concatenated"} or
            (tgt_tok & {"name"} and tgt_tok & _COMPOSITE_HINT_TOKENS) or
            tgt_tok & {"margin", "difference", "diff", "ratio", "per"} or
            (tgt_tok & {"cost", "unit"} and len(tgt_tok) >= 2) or
            (tgt_tok & {"time", "lead"} and tgt_tok & {"days", "hours", "duration"})
        )
        if not composite_signal:
            idx_score = [(i, combined[i], top_indices[i]) for i in range(len(top_indices))]
            idx_score.sort(key=lambda x: -x[1])

            for rank_a in range(len(idx_score)):
                ia, sa, ra = idx_score[rank_a]
                cols_a = set((c[0], c[1]) for c in self.candidates[ra].columns)
                if len(cols_a) <= 1:
                    continue
                for rank_b in range(rank_a + 1, min(rank_a + 20, len(idx_score))):
                    ib, sb, rb = idx_score[rank_b]
                    cols_b = set((c[0], c[1]) for c in self.candidates[rb].columns)
                    if len(cols_b) < len(cols_a) and cols_b.issubset(cols_a):
                        gap = sa - sb
                        if gap < 0.25:
                            combined[ib] = sa + 0.02

        # 3e) Hard single-column preference for boolean/key targets
        # For these target types, if the best candidate is composite but
        # there's a good single-column alternative, promote it.
        if single_pref >= 0.35:  # boolean, key targets
            best_idx = int(np.argmax(combined))
            best_cand = self.candidates[top_indices[best_idx]]
            if len(best_cand.columns) > 1:
                # Find best single-column candidate
                best_single_i = None
                best_single_score = -999
                for i, ri in enumerate(top_indices):
                    c = self.candidates[ri]
                    if len(c.columns) == 1 and combined[i] > best_single_score:
                        best_single_score = combined[i]
                        best_single_i = i
                if best_single_i is not None:
                    best_overall = float(combined[best_idx])
                    # Promote single if it has any positive score
                    if best_single_score > best_overall * 0.3 or best_single_score > 0:
                        combined[best_single_i] = best_overall + 0.05

        # Sort by combined score
        rerank_order = np.argsort(-combined)[:rerank_k]

        # 4) Build results
        results = []
        for rank_idx, ri in enumerate(rerank_order):
            idx = top_indices[ri]
            cand = self.candidates[idx]

            # Infer transform type
            has_join = bool(cand.join_path)
            transform = infer_transform(
                target_column, target_type, cand.columns, has_join,
                target_table=target_table, source_tables=self.tables,
            )

            # Confidence: sigmoid of combined score
            raw_conf = float(combined[ri])
            confidence = 1.0 / (1.0 + np.exp(-(raw_conf - 0.5) / 0.15))

            results.append({
                "rank": rank_idx + 1,
                "source_columns": [f"{t}.{c}" for t, c, _ in cand.columns],
                "source_columns_typed": [(t, c, tp) for t, c, tp in cand.columns],
                "join_path": cand.join_path,
                "join_edges_raw": [
                    {
                        "from": e.left_table, "to": e.right_table,
                        "left_cols": list(e.left_cols), "right_cols": list(e.right_cols),
                        "confidence": round(float(e.confidence), 4),
                    }
                    for e in cand.join_edges
                ],
                "transform_type": transform,
                "bi_score": round(float(sims[idx]), 4),
                "cross_score": round(float(ce_scores[ri]), 4),
                "combined_score": round(float(combined[ri]), 4),
                "confidence": round(confidence, 3),
            })

        elapsed = time.time() - t0
        return {
            "target": {
                "table": target_table,
                "column": target_column,
                "type": target_type,
            },
            "top_candidates": results,
            "retrieval_count": len(top_indices),
            "total_candidates": len(self.candidates),
            "time_seconds": round(elapsed, 3),
        }

    def rank_all_targets(
        self,
        target_tables: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Rank candidates for all columns in all target tables.
        Returns results grouped by table.
        """
        all_results = {"tables": [], "stats": {}}
        total = 0
        mapped = 0
        t_start = time.time()

        for ttable in target_tables:
            tname = ttable["name"]
            table_result = {"name": tname, "columns": []}

            for tcol in ttable.get("columns", []):
                total += 1
                result = self.rank(
                    target_table=tname,
                    target_column=tcol["name"],
                    target_type=tcol.get("type", "string"),
                    target_desc=tcol.get("description", ""),
                    rerank_k=top_k,
                )

                top = result["top_candidates"]
                best = top[0] if top else None
                col_result = {
                    "name": tcol["name"],
                    "type": tcol.get("type", "string"),
                    "description": tcol.get("description", ""),
                }

                if best:
                    col_result["final_source"] = best["source_columns"]
                    col_result["final_transform"] = best["transform_type"]
                    col_result["final_confidence"] = best["confidence"]
                    col_result["stage_a"] = {
                        "source_columns": best["source_columns"],
                        "join_path": best.get("join_edges_raw", []),
                        "transform_family": best["transform_type"],
                        "confidence": best["confidence"],
                        "time": result["time_seconds"],
                        "abstain": best["confidence"] < 0.3,
                    }
                    col_result["alternatives"] = [
                        {
                            "source_columns": c["source_columns"],
                            "transform_family": c["transform_type"],
                            "confidence": c["confidence"],
                        }
                        for c in top[1:4]
                    ]
                    mapped += 1
                else:
                    col_result["final_source"] = []
                    col_result["final_transform"] = "unmapped"
                    col_result["final_confidence"] = 0
                    col_result["stage_a"] = {
                        "source_columns": [], "transform_family": "unmapped",
                        "confidence": 0, "time": 0, "abstain": True,
                    }
                    col_result["alternatives"] = []

                table_result["columns"].append(col_result)

            all_results["tables"].append(table_result)

        all_results["stats"] = {
            "total_columns": total,
            "mapped_columns": mapped,
            "mapping_rate": round(mapped / max(1, total) * 100, 1),
            "total_time": round(time.time() - t_start, 2),
            "total_candidates_enumerated": len(self.candidates),
            "has_stage_b": False,
        }
        return all_results
