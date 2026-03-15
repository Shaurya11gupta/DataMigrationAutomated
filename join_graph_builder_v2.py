from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


# =========================================================
# Optional engine imports (kept compatible with your setup)
# =========================================================

try:
    from seg_classf_abbrev_test import name_similarity as default_name_similarity  # type: ignore
except Exception:
    default_name_similarity = None

try:
    from value_similarity_engine import ColumnStats, ValueSimilarity
except Exception as exc:
    raise ImportError("value_similarity_engine.ColumnStats and ValueSimilarity are required") from exc

try:
    from constraint_similarity_engine import ColumnConstraints, ConstraintCompatibilityJoin
except Exception as exc:
    raise ImportError(
        "constraint_similarity_engine.ColumnConstraints and ConstraintCompatibilityJoin are required"
    ) from exc


# =========================================================
# Schema Data Model
# =========================================================


class ColumnType:
    def __init__(self, base_type: str, max_length: Optional[int] = None):
        self.base_type = base_type
        self.max_length = max_length


@dataclass
class Column:
    name: str
    col_type: ColumnType
    constraints: ColumnConstraints
    stats: ColumnStats


class Table:
    """
    Optional rows support (recommended for robust composite joins):
      rows: list[dict[column_name -> value]]
    If rows are not provided, builder falls back to stats.sample alignment.
    """

    def __init__(
        self,
        name: str,
        columns: Dict[str, Column],
        row_count: int,
        rows: Optional[List[Dict[str, Any]]] = None,
    ):
        self.name = name
        self.columns = columns
        self.row_count = row_count
        self.rows = rows or []


@dataclass
class JoinEdge:
    left_table: str
    right_table: str
    left_cols: List[str]
    right_cols: List[str]
    cardinality: str
    confidence: float
    reasons: List[str]
    score_details: Dict[str, float] = field(default_factory=dict)
    direction: str = "undirected"


# =========================================================
# Type Compatibility
# =========================================================


class DefaultTypeCompatibility:
    NUM = {"int", "integer", "bigint", "float", "double", "decimal", "number", "numeric"}
    STR = {"string", "varchar", "text", "char", "clob"}
    DATE = {"date", "datetime", "timestamp", "time"}

    @staticmethod
    def score(a: Any, b: Any) -> float:
        ta = getattr(a, "base_type", str(a)).lower()
        tb = getattr(b, "base_type", str(b)).lower()

        if ta == tb:
            return 1.0
        if ta in DefaultTypeCompatibility.NUM and tb in DefaultTypeCompatibility.NUM:
            return 0.95
        if ta in DefaultTypeCompatibility.STR and tb in DefaultTypeCompatibility.STR:
            return 0.90
        if ta in DefaultTypeCompatibility.DATE and tb in DefaultTypeCompatibility.DATE:
            return 0.93
        return 0.0


# =========================================================
# Join Graph Builder V2
# =========================================================


class JoinGraphBuilderV2:
    """
    Enhancements over V1:
      1) No accidental self-joins by default
      2) Softer structural gate (no hard reject on cs == 0)
      3) Stronger ID lexical guardrail for name score
      4) Composite candidate generation from top pairings (not only >0.80 singles)
      5) Composite value score is row-order independent (set/frequency overlap)
      6) Composite cardinality inferred from tuple uniqueness ratios
      7) Better direction inference for composite joins
      8) Optional sibling-FK penalty
    """

    def __init__(
        self,
        tables: Dict[str, Table],
        name_engine: Optional[Callable[[str, str], Any]] = None,
        value_sim_class: Any = ValueSimilarity,
        constraint_engine: Any = ConstraintCompatibilityJoin,
        type_engine: Any = None,
        include_self_joins: bool = False,
        max_composite_arity: int = 3,
        min_single_conf: float = 0.58,
        composite_min_conf: float = 0.72,
        top_k: int = 5,
        top_k_composites: int = 3,
        max_single_pairs_per_table_pair: int = 80,
        max_composite_candidates: int = 2000,
        enable_sibling_fk_penalty: bool = True,
    ):
        self.tables = tables
        self.name_engine = name_engine or default_name_similarity
        self.ValueSim = value_sim_class
        self.ConstraintEngine = constraint_engine
        self.TypeEngine = type_engine or DefaultTypeCompatibility

        self.include_self_joins = include_self_joins
        self.max_composite_arity = max_composite_arity
        self.min_single_conf = min_single_conf
        self.composite_min_conf = composite_min_conf
        self.top_k = top_k
        self.top_k_composites = top_k_composites
        self.max_single_pairs_per_table_pair = max_single_pairs_per_table_pair
        self.max_composite_candidates = max_composite_candidates
        self.enable_sibling_fk_penalty = enable_sibling_fk_penalty

        self._name_cache: Dict[Tuple[str, str, str, str], float] = {}
        self._value_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._tuple_cache: Dict[Tuple[str, Tuple[str, ...]], List[Tuple[Any, ...]]] = {}
        self._all_edges_cache: List[JoinEdge] = []

    # =====================================================
    # Main Build
    # =====================================================

    def build(self) -> List[JoinEdge]:
        edges: List[JoinEdge] = []
        names = sorted(self.tables.keys())

        for i in range(len(names)):
            start_j = i if self.include_self_joins else i + 1
            for j in range(start_j, len(names)):
                if i == j and not self.include_self_joins:
                    continue
                t1 = self.tables[names[i]]
                t2 = self.tables[names[j]]
                edges.extend(self._discover_between(t1, t2))

        self._all_edges_cache = list(edges)
        if self.enable_sibling_fk_penalty:
            edges = self._apply_sibling_fk_penalty(edges)

        edges = self._dedup(edges)
        return self._topk(edges)

    # =====================================================
    # Discovery
    # =====================================================

    def _discover_between(self, t1: Table, t2: Table) -> List[JoinEdge]:
        singles = self._discover_singles(t1, t2)
        edges: List[JoinEdge] = [
            self._make_single_edge(t1, a, t2, b, score, details)
            for a, b, score, details in singles
        ]

        if len(singles) >= 2:
            edges.extend(self._discover_composites(t1, t2, singles))

        return edges

    # =====================================================
    # Single-Column Join Discovery
    # =====================================================

    def _discover_singles(
        self,
        t1: Table,
        t2: Table,
    ) -> List[Tuple[str, str, float, Dict[str, float]]]:
        out: List[Tuple[str, str, float, Dict[str, float]]] = []

        for c1 in t1.columns.values():
            if not self._joinable(c1):
                continue

            for c2 in t2.columns.values():
                if not self._joinable(c2):
                    continue

                if t1.name == t2.name and c1.name == c2.name:
                    continue

                type_score = self.TypeEngine.score(c1.col_type, c2.col_type)
                if type_score < 0.20:
                    continue

                value_obj = self._value_score_obj(c1.stats, c2.stats)
                value_score = float(value_obj.get("final", 0.0))
                if value_score < 0.08:
                    # Low but non-zero allowance to keep hard-but-true PK/FK candidates alive.
                    if not (
                        c1.constraints.is_primary_key
                        or c2.constraints.is_primary_key
                        or c1.constraints.is_foreign_key
                        or c2.constraints.is_foreign_key
                    ):
                        continue

                name_score = self._name_score(t1.name, c1.name, t2.name, c2.name)
                cscore = self._constraint_score(c1.constraints, c2.constraints)

                # Avoid hard dropping on unknown constraints; only disjoint constraints should force zero.
                if cscore <= 0.0:
                    continue

                raw = (
                    0.30 * name_score
                    + 0.36 * value_score
                    + 0.14 * type_score
                    + 0.20 * cscore
                )

                # Boost high-confidence PK/FK hints with supporting value signal.
                if c1.constraints.is_primary_key and c2.constraints.is_foreign_key and value_score > 0.20:
                    raw = max(raw, 0.90)
                if c2.constraints.is_primary_key and c1.constraints.is_foreign_key and value_score > 0.20:
                    raw = max(raw, 0.90)

                # Size-aware confidence damping to reduce tiny-table overfitting.
                vol = min(math.log(max(t1.row_count, 2)), math.log(max(t2.row_count, 2))) / 10.0
                score = raw * min(1.0, 0.72 + vol)

                if score >= self.min_single_conf:
                    details = {
                        "name": round(name_score, 4),
                        "value": round(value_score, 4),
                        "type": round(type_score, 4),
                        "constraint": round(cscore, 4),
                    }
                    out.append((c1.name, c2.name, score, details))

        out.sort(key=lambda x: x[2], reverse=True)
        return out[: self.max_single_pairs_per_table_pair]

    # =====================================================
    # Composite Join Discovery
    # =====================================================

    def _discover_composites(
        self,
        t1: Table,
        t2: Table,
        singles: List[Tuple[str, str, float, Dict[str, float]]],
    ) -> List[JoinEdge]:
        # Use moderate threshold to avoid missing composites whose individual columns are weaker.
        candidate_pairs = [s for s in singles if s[2] >= max(self.min_single_conf - 0.12, 0.45)]
        candidate_pairs = candidate_pairs[: min(24, len(candidate_pairs))]

        if len(candidate_pairs) < 2:
            return []

        edges: List[JoinEdge] = []
        checked = 0

        for k in range(2, self.max_composite_arity + 1):
            for combo in combinations(candidate_pairs, k):
                checked += 1
                if checked > self.max_composite_candidates:
                    break

                left_cols = [x[0] for x in combo]
                right_cols = [x[1] for x in combo]

                if len(set(left_cols)) < k or len(set(right_cols)) < k:
                    continue

                # Canonicalize by left column order to reduce duplicates.
                paired = sorted(zip(left_cols, right_cols, combo), key=lambda z: z[0])
                left_cols = [p[0] for p in paired]
                right_cols = [p[1] for p in paired]
                pair_scores = [p[2][2] for p in paired]

                tuple_metrics = self._composite_tuple_metrics(t1, left_cols, t2, right_cols)
                contain = tuple_metrics["containment"]
                jacc = tuple_metrics["jaccard"]

                # Composite must show real overlap to be considered.
                if contain < 0.25 and jacc < 0.15:
                    continue

                pair_mean = sum(pair_scores) / k
                type_mean = self._mean_type_score(t1, left_cols, t2, right_cols)
                cons_mean = self._mean_constraint_score(t1, left_cols, t2, right_cols)

                conf = (
                    0.42 * pair_mean
                    + 0.26 * contain
                    + 0.10 * jacc
                    + 0.10 * tuple_metrics["freq_overlap"]
                    + 0.06 * type_mean
                    + 0.06 * cons_mean
                )

                # Small bonus when one side is highly unique and the other is not.
                u_left = tuple_metrics["uniq_left"]
                u_right = tuple_metrics["uniq_right"]
                if u_left > 0.98 and u_right < 0.98:
                    conf += 0.02
                elif u_right > 0.98 and u_left < 0.98:
                    conf += 0.02

                if conf >= self.composite_min_conf:
                    details = {
                        "pair_mean": round(pair_mean, 4),
                        "containment": round(contain, 4),
                        "jaccard": round(jacc, 4),
                        "freq_overlap": round(tuple_metrics["freq_overlap"], 4),
                        "uniq_left": round(u_left, 4),
                        "uniq_right": round(u_right, 4),
                        "type_mean": round(type_mean, 4),
                        "constraint_mean": round(cons_mean, 4),
                    }

                    edge = JoinEdge(
                        left_table=t1.name,
                        right_table=t2.name,
                        left_cols=list(left_cols),
                        right_cols=list(right_cols),
                        cardinality=self._cardinality_from_uniqueness(u_left, u_right),
                        confidence=round(min(1.0, conf), 4),
                        reasons=["composite"],
                        score_details=details,
                        direction=self._composite_direction(t1, left_cols, t2, right_cols, u_left, u_right),
                    )
                    edges.append(edge)

            if checked > self.max_composite_candidates:
                break

        # keep strongest composites per table pair
        edges.sort(key=lambda e: e.confidence, reverse=True)
        return edges[: max(self.top_k_composites * 3, self.top_k_composites)]

    # =====================================================
    # Scores and Features
    # =====================================================

    def _name_score(self, t1: str, c1: str, t2: str, c2: str) -> float:
        key = (t1, c1, t2, c2)
        if key in self._name_cache:
            return self._name_cache[key]

        a = c1.lower().strip()
        b = c2.lower().strip()

        if a == b:
            score = 1.0
        elif a.replace("_", "") == b.replace("_", ""):
            score = 0.95
        elif self._id_like(a) and self._id_like(b):
            # Guardrail: shared stem required for strong ID shortcut.
            sa = self._stem_without_id(a)
            sb = self._stem_without_id(b)
            if sa and sb and (sa == sb or sa in sb or sb in sa):
                score = 0.88
            else:
                score = 0.48
        else:
            score = self._name_engine_score(c1, c2)

        score = float(max(0.0, min(1.0, score)))
        self._name_cache[key] = score
        return score

    def _name_engine_score(self, c1: str, c2: str) -> float:
        if self.name_engine is None:
            return self._token_jaccard(c1, c2)
        try:
            res = self.name_engine(c1, c2)
            if isinstance(res, dict):
                return float(res.get("similarity", 0.0))
            return float(res)
        except Exception:
            return self._token_jaccard(c1, c2)

    def _value_score_obj(self, s1: ColumnStats, s2: ColumnStats) -> Dict[str, Any]:
        key = tuple(sorted((id(s1), id(s2))))
        if key not in self._value_cache:
            try:
                self._value_cache[key] = self.ValueSim(s1, s2).compute_score()
            except Exception:
                self._value_cache[key] = {"final": 0.0, "reason": "error"}
        return self._value_cache[key]

    def _constraint_score(self, c1: ColumnConstraints, c2: ColumnConstraints) -> float:
        try:
            return float(self.ConstraintEngine.score(c1, c2).get("final", 0.0))
        except Exception:
            return 0.0

    def _mean_type_score(
        self,
        t1: Table,
        left_cols: Sequence[str],
        t2: Table,
        right_cols: Sequence[str],
    ) -> float:
        vals = [
            self.TypeEngine.score(t1.columns[a].col_type, t2.columns[b].col_type)
            for a, b in zip(left_cols, right_cols)
        ]
        return float(sum(vals) / max(1, len(vals)))

    def _mean_constraint_score(
        self,
        t1: Table,
        left_cols: Sequence[str],
        t2: Table,
        right_cols: Sequence[str],
    ) -> float:
        vals = [
            self._constraint_score(t1.columns[a].constraints, t2.columns[b].constraints)
            for a, b in zip(left_cols, right_cols)
        ]
        return float(sum(vals) / max(1, len(vals)))

    # =====================================================
    # Composite Tuple Metrics
    # =====================================================

    def _composite_tuple_metrics(
        self,
        t1: Table,
        left_cols: Sequence[str],
        t2: Table,
        right_cols: Sequence[str],
    ) -> Dict[str, float]:
        tup1 = self._table_tuples(t1, tuple(left_cols))
        tup2 = self._table_tuples(t2, tuple(right_cols))

        if not tup1 or not tup2:
            return {
                "containment": 0.0,
                "jaccard": 0.0,
                "freq_overlap": 0.0,
                "uniq_left": 0.0,
                "uniq_right": 0.0,
            }

        c1 = Counter(tup1)
        c2 = Counter(tup2)
        set1 = set(c1.keys())
        set2 = set(c2.keys())
        inter = set1 & set2
        union = set1 | set2

        containment = len(inter) / max(1, min(len(set1), len(set2)))
        jaccard = len(inter) / max(1, len(union))

        # Frequency overlap (weighted Jaccard on tuple counts).
        num = sum(min(c1[k], c2[k]) for k in inter)
        den = sum(max(c1.get(k, 0), c2.get(k, 0)) for k in union)
        freq_overlap = num / max(1, den)

        uniq_left = len(set1) / max(1, len(tup1))
        uniq_right = len(set2) / max(1, len(tup2))

        return {
            "containment": float(containment),
            "jaccard": float(jaccard),
            "freq_overlap": float(freq_overlap),
            "uniq_left": float(uniq_left),
            "uniq_right": float(uniq_right),
        }

    def _table_tuples(self, table: Table, cols: Tuple[str, ...]) -> List[Tuple[Any, ...]]:
        key = (table.name, cols)
        if key in self._tuple_cache:
            return self._tuple_cache[key]

        tuples: List[Tuple[Any, ...]] = []

        if table.rows:
            # Preferred path: guaranteed row alignment.
            for row in table.rows:
                vals = tuple(self._norm(row.get(c)) for c in cols)
                if any(v is None for v in vals):
                    continue
                tuples.append(vals)
        else:
            # Fallback path: uses ColumnStats.sample. Works best when samples preserve row alignment.
            samples = []
            for c in cols:
                samples.append(getattr(table.columns[c].stats, "sample", []))

            if samples and all(len(s) > 0 for s in samples):
                n = min(len(s) for s in samples)
                for i in range(n):
                    vals = tuple(self._norm(samples[j][i]) for j in range(len(cols)))
                    if any(v is None for v in vals):
                        continue
                    tuples.append(vals)

        self._tuple_cache[key] = tuples
        return tuples

    # =====================================================
    # Sibling FK penalty
    # =====================================================

    def _find_pk_targets(self, table_name: str, col_name: str) -> List[Tuple[str, str]]:
        targets = []
        for e in self._all_edges_cache:
            if len(e.left_cols) != 1 or len(e.right_cols) != 1:
                continue

            if (
                e.left_table == table_name
                and e.left_cols[0] == col_name
                and self.tables[e.right_table].columns[e.right_cols[0]].constraints.is_primary_key
            ):
                targets.append((e.right_table, e.right_cols[0]))

            if (
                e.right_table == table_name
                and e.right_cols[0] == col_name
                and self.tables[e.left_table].columns[e.left_cols[0]].constraints.is_primary_key
            ):
                targets.append((e.left_table, e.left_cols[0]))
        return targets

    def _apply_sibling_fk_penalty(self, edges: List[JoinEdge]) -> List[JoinEdge]:
        adjusted: List[JoinEdge] = []
        for e in edges:
            if len(e.left_cols) != 1 or len(e.right_cols) != 1:
                adjusted.append(e)
                continue

            ltab, lcol = e.left_table, e.left_cols[0]
            rtab, rcol = e.right_table, e.right_cols[0]

            l_targets = set(self._find_pk_targets(ltab, lcol))
            r_targets = set(self._find_pk_targets(rtab, rcol))
            common = l_targets & r_targets

            if common and "pk_fk_relation" not in e.reasons:
                e.confidence = round(e.confidence * 0.80, 4)
                e.reasons.append("sibling_fk_collision")
            adjusted.append(e)
        return adjusted

    # =====================================================
    # Edge construction and cardinality
    # =====================================================

    def _make_single_edge(
        self,
        t1: Table,
        left_col: str,
        t2: Table,
        right_col: str,
        conf: float,
        details: Dict[str, float],
    ) -> JoinEdge:
        u_left = self._single_uniqueness_ratio(t1.columns[left_col].stats)
        u_right = self._single_uniqueness_ratio(t2.columns[right_col].stats)
        cardinality = self._cardinality_from_uniqueness(u_left, u_right)

        reasons = ["single"]
        if t1.columns[left_col].constraints.is_primary_key and t2.columns[right_col].constraints.is_foreign_key:
            reasons.append("pk_fk_relation")
        if t2.columns[right_col].constraints.is_primary_key and t1.columns[left_col].constraints.is_foreign_key:
            reasons.append("pk_fk_relation")

        return JoinEdge(
            left_table=t1.name,
            right_table=t2.name,
            left_cols=[left_col],
            right_cols=[right_col],
            cardinality=cardinality,
            confidence=round(conf, 4),
            reasons=reasons,
            score_details={**details, "uniq_left": round(u_left, 4), "uniq_right": round(u_right, 4)},
            direction=self._single_direction(t1.columns[left_col], t2.columns[right_col], u_left, u_right),
        )

    def _single_uniqueness_ratio(self, stats: ColumnStats) -> float:
        n = max(1, getattr(stats, "valid_count", 0))
        d = getattr(stats, "n_distinct", 0)
        return float(d / n) if n > 0 else 0.0

    def _cardinality_from_uniqueness(self, u_left: float, u_right: float) -> str:
        l_unique = u_left >= 0.98
        r_unique = u_right >= 0.98
        if l_unique and r_unique:
            return "1:1"
        if l_unique and not r_unique:
            return "1:N"
        if r_unique and not l_unique:
            return "N:1"
        return "N:M"

    def _single_direction(self, c1: Column, c2: Column, u_left: float, u_right: float) -> str:
        if c1.constraints.is_primary_key and c2.constraints.is_foreign_key:
            return "left_parent"
        if c2.constraints.is_primary_key and c1.constraints.is_foreign_key:
            return "right_parent"
        if u_left >= 0.98 and u_right < 0.98:
            return "left_parent"
        if u_right >= 0.98 and u_left < 0.98:
            return "right_parent"
        return "undirected"

    def _composite_direction(
        self,
        t1: Table,
        left_cols: Sequence[str],
        t2: Table,
        right_cols: Sequence[str],
        u_left: float,
        u_right: float,
    ) -> str:
        left_pk_fk = all(
            t1.columns[c].constraints.is_primary_key or t1.columns[c].constraints.is_unique for c in left_cols
        ) and any(t2.columns[c].constraints.is_foreign_key for c in right_cols)
        right_pk_fk = all(
            t2.columns[c].constraints.is_primary_key or t2.columns[c].constraints.is_unique for c in right_cols
        ) and any(t1.columns[c].constraints.is_foreign_key for c in left_cols)

        if left_pk_fk and not right_pk_fk:
            return "left_parent"
        if right_pk_fk and not left_pk_fk:
            return "right_parent"
        if u_left >= 0.98 and u_right < 0.98:
            return "left_parent"
        if u_right >= 0.98 and u_left < 0.98:
            return "right_parent"
        return "undirected"

    # =====================================================
    # Utility
    # =====================================================

    def _joinable(self, col: Column) -> bool:
        s = col.stats
        if getattr(s, "valid_count", 0) == 0:
            return False
        if getattr(s, "null_frac", 1.0) > 0.95:
            return False
        if getattr(s, "n_distinct", 0) <= 1 and getattr(s, "valid_count", 0) > 20:
            return False
        return True

    def _id_like(self, name: str) -> bool:
        name = name.lower()
        return name.endswith("_id") or name == "id" or name.endswith("id")

    def _stem_without_id(self, name: str) -> str:
        s = name.lower()
        if s.endswith("_id"):
            s = s[:-3]
        elif s.endswith("id") and len(s) > 2:
            s = s[:-2]
        return s.strip("_")

    def _token_jaccard(self, a: str, b: str) -> float:
        ta = set(self._simple_tokens(a))
        tb = set(self._simple_tokens(b))
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    def _simple_tokens(self, s: str) -> List[str]:
        out = []
        cur = []
        for ch in s.lower():
            if ch.isalnum():
                cur.append(ch)
            else:
                if cur:
                    out.append("".join(cur))
                    cur = []
        if cur:
            out.append("".join(cur))
        return out

    def _norm(self, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, str):
            x = v.strip().lower()
            return x if x else None
        if isinstance(v, (tuple, list)):
            return tuple(self._norm(x) for x in v)
        return v

    def _dedup(self, edges: List[JoinEdge]) -> List[JoinEdge]:
        best: Dict[Tuple[str, str, Tuple[Tuple[str, str], ...]], JoinEdge] = {}
        for e in edges:
            pair_cols = tuple(sorted(zip(e.left_cols, e.right_cols), key=lambda x: x[0]))
            key = (e.left_table, e.right_table, pair_cols)
            if key not in best or best[key].confidence < e.confidence:
                best[key] = e
        return list(best.values())

    def _topk(self, edges: List[JoinEdge]) -> List[JoinEdge]:
        grouped: Dict[Tuple[str, str], List[JoinEdge]] = defaultdict(list)
        for e in edges:
            grouped[(e.left_table, e.right_table)].append(e)

        out: List[JoinEdge] = []
        for _, bucket in grouped.items():
            singles = [e for e in bucket if len(e.left_cols) == 1]
            comps = [e for e in bucket if len(e.left_cols) > 1]
            singles.sort(key=lambda x: x.confidence, reverse=True)
            comps.sort(key=lambda x: x.confidence, reverse=True)
            out.extend(singles[: self.top_k])
            out.extend(comps[: self.top_k_composites])
        out.sort(key=lambda x: x.confidence, reverse=True)
        return out


# =========================================================
# Demo test
# =========================================================


def _demo_make_stats(values: List[Any]) -> ColumnStats:
    return ColumnStats(values)


def _demo_make_col(name: str, values: List[Any], typ: str, cons: ColumnConstraints) -> Column:
    return Column(
        name=name,
        col_type=ColumnType(typ),
        constraints=cons,
        stats=_demo_make_stats(values),
    )


def demo_run() -> List[JoinEdge]:
    # Customers and orders (single key)
    customers_rows = [
        {"customer_id": i, "email": f"user{i}@mail.com", "country": ["us", "in", "uk"][i % 3]}
        for i in range(1, 101)
    ]
    orders_rows = [
        {"order_id": 1000 + i, "customer_id": i + 1, "amount": [100, 200, 300, 150][i % 4]}
        for i in range(100)
    ]

    # Composite key pair, intentionally different row order between tables.
    order_items_rows = []
    fulfillment_rows = []
    for oid in range(1000, 1030):
        for line_no in [1, 2, 3]:
            order_items_rows.append(
                {
                    "order_id": oid,
                    "line_no": line_no,
                    "sku": f"sku_{oid}_{line_no}",
                }
            )
            fulfillment_rows.append(
                {
                    "ord_id": oid,
                    "line_number": line_no,
                    "event_code": f"evt_{oid}_{line_no}",
                }
            )
    # Reverse to break any accidental cross-table positional assumptions.
    fulfillment_rows = list(reversed(fulfillment_rows))

    customers = Table(
        "customers",
        {
            "customer_id": _demo_make_col(
                "customer_id",
                [r["customer_id"] for r in customers_rows],
                "int",
                ColumnConstraints(nullable=False, is_primary_key=True, is_unique=True),
            ),
            "email": _demo_make_col(
                "email",
                [r["email"] for r in customers_rows],
                "string",
                ColumnConstraints(nullable=False, is_unique=True),
            ),
            "country": _demo_make_col(
                "country",
                [r["country"] for r in customers_rows],
                "string",
                ColumnConstraints(),
            ),
        },
        row_count=len(customers_rows),
        rows=customers_rows,
    )

    orders = Table(
        "orders",
        {
            "order_id": _demo_make_col(
                "order_id",
                [r["order_id"] for r in orders_rows],
                "int",
                ColumnConstraints(nullable=False, is_primary_key=True, is_unique=True),
            ),
            "customer_id": _demo_make_col(
                "customer_id",
                [r["customer_id"] for r in orders_rows],
                "int",
                ColumnConstraints(nullable=False, is_foreign_key=True),
            ),
            "amount": _demo_make_col(
                "amount",
                [r["amount"] for r in orders_rows],
                "float",
                ColumnConstraints(min_value=0, max_value=10000),
            ),
        },
        row_count=len(orders_rows),
        rows=orders_rows,
    )

    order_items = Table(
        "order_items",
        {
            "order_id": _demo_make_col(
                "order_id",
                [r["order_id"] for r in order_items_rows],
                "int",
                ColumnConstraints(is_foreign_key=True),
            ),
            "line_no": _demo_make_col(
                "line_no",
                [r["line_no"] for r in order_items_rows],
                "int",
                ColumnConstraints(),
            ),
            "sku": _demo_make_col(
                "sku",
                [r["sku"] for r in order_items_rows],
                "string",
                ColumnConstraints(),
            ),
        },
        row_count=len(order_items_rows),
        rows=order_items_rows,
    )

    fulfillment = Table(
        "fulfillment_events",
        {
            "ord_id": _demo_make_col(
                "ord_id",
                [r["ord_id"] for r in fulfillment_rows],
                "int",
                ColumnConstraints(is_foreign_key=True),
            ),
            "line_number": _demo_make_col(
                "line_number",
                [r["line_number"] for r in fulfillment_rows],
                "int",
                ColumnConstraints(),
            ),
            "event_code": _demo_make_col(
                "event_code",
                [r["event_code"] for r in fulfillment_rows],
                "string",
                ColumnConstraints(),
            ),
        },
        row_count=len(fulfillment_rows),
        rows=fulfillment_rows,
    )

    tables = {
        "customers": customers,
        "orders": orders,
        "order_items": order_items,
        "fulfillment_events": fulfillment,
    }

    builder = JoinGraphBuilderV2(
        tables=tables,
        name_engine=default_name_similarity,
        value_sim_class=ValueSimilarity,
        constraint_engine=ConstraintCompatibilityJoin,
        include_self_joins=False,
        max_composite_arity=2,
        min_single_conf=0.55,
        composite_min_conf=0.70,
        top_k=3,
        top_k_composites=2,
    )

    edges = builder.build()
    return edges


if __name__ == "__main__":
    print("\n==============================")
    print("🚀 JoinGraphBuilderV2 Demo Run")
    print("==============================\n")

    discovered = demo_run()
    for e in discovered:
        print(
            f"{e.left_table}.{e.left_cols} ↔ {e.right_table}.{e.right_cols} "
            f"| card={e.cardinality} conf={e.confidence} dir={e.direction} reasons={e.reasons}"
        )
