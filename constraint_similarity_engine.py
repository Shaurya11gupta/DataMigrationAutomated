from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Pattern, Set


@dataclass
class ColumnConstraints:
    nullable: bool = True
    is_unique: bool = False
    is_primary_key: bool = False
    is_foreign_key: bool = False
    allowed_values: Optional[Set[Any]] = None
    regex_pattern: Optional[Pattern] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    referenced_table: Optional[str] = None
    referenced_column: Optional[str] = None


class ConstraintCompatibilityJoin:
    """
    Constraint compatibility score for join candidacy.
    Returns:
      {
        "final": float in [0,1],
        "reasons": [...],
        "details": {...}
      }
    """

    @staticmethod
    def score(a: ColumnConstraints, b: ColumnConstraints) -> Dict[str, Any]:
        score = 0.52
        reasons: List[str] = []
        details: Dict[str, float] = {"base": 0.52}

        def add(delta: float, reason: Optional[str] = None, key: Optional[str] = None) -> None:
            nonlocal score
            score += delta
            if reason:
                reasons.append(reason)
            if key:
                details[key] = round(delta, 4)

        def clamp() -> None:
            nonlocal score
            score = max(0.0, min(1.0, score))

        a_unique = a.is_unique or a.is_primary_key
        b_unique = b.is_unique or b.is_primary_key
        a_nullable = a.nullable and not a.is_primary_key
        b_nullable = b.nullable and not b.is_primary_key

        # -----------------------------
        # 1) Key relationship
        # -----------------------------
        pk_fk = (a.is_primary_key and b.is_foreign_key) or (b.is_primary_key and a.is_foreign_key)
        pk_pk = a.is_primary_key and b.is_primary_key
        fk_fk = a.is_foreign_key and b.is_foreign_key

        if pk_fk:
            add(0.34, "pk_fk_relation", "pk_fk")
        elif pk_pk:
            add(0.17, "pk_pk", "pk_pk")
        elif fk_fk:
            add(0.09, "fk_fk", "fk_fk")

        # -----------------------------
        # 2) Uniqueness compatibility
        # -----------------------------
        if a_unique and b_unique:
            add(0.08, "unique_unique", "unique")
        elif a_unique != b_unique:
            add(-0.10, "unique_mismatch", "unique")
        else:
            add(-0.01, "both_non_unique", "unique")

        # -----------------------------
        # 3) Nullability compatibility
        # -----------------------------
        if a_nullable != b_nullable:
            add(-0.05, "nullable_mismatch", "nullable")
        else:
            add(0.02, "nullable_compatible", "nullable")

        # -----------------------------
        # 4) Enum/domain compatibility
        # -----------------------------
        if a.allowed_values and b.allowed_values:
            A = {_norm_text(x) for x in a.allowed_values}
            B = {_norm_text(x) for x in b.allowed_values}
            A.discard("")
            B.discard("")
            if A and B:
                inter = A & B
                if not inter:
                    return {"final": 0.0, "reasons": ["enum_disjoint"], "details": {"enum_jaccard": 0.0}}
                jacc = len(inter) / len(A | B)
                # Moderate effect: domain overlap is helpful but not absolute.
                enum_delta = -0.05 + 0.25 * jacc
                add(enum_delta, f"enum_overlap_{round(jacc, 2)}", "enum")
                details["enum_jaccard"] = round(jacc, 4)
        elif a.allowed_values or b.allowed_values:
            add(-0.04, "enum_one_sided", "enum")

        # -----------------------------
        # 5) Regex compatibility
        # -----------------------------
        pa = _regex_text(a.regex_pattern)
        pb = _regex_text(b.regex_pattern)
        if pa and pb:
            if pa == pb:
                add(0.06, "regex_equal", "regex")
            elif _regex_signature(pa) == _regex_signature(pb):
                add(0.02, "regex_compatible", "regex")
            else:
                add(-0.08, "regex_diff", "regex")
        elif pa or pb:
            add(-0.02, "regex_one_sided", "regex")

        # -----------------------------
        # 6) Numeric range compatibility
        # -----------------------------
        a_has_range = a.min_value is not None and a.max_value is not None
        b_has_range = b.min_value is not None and b.max_value is not None

        if a_has_range and b_has_range:
            overlap = _range_overlap_ratio(a.min_value, a.max_value, b.min_value, b.max_value)
            if overlap <= 0.0:
                return {"final": 0.0, "reasons": ["range_disjoint"], "details": {"range_overlap": 0.0}}
            # Slight penalty for very low overlap, bonus for strong overlap.
            range_delta = -0.04 + 0.22 * overlap
            add(range_delta, f"range_overlap_{round(overlap, 2)}", "range")
            details["range_overlap"] = round(overlap, 4)
        elif a_has_range or b_has_range:
            add(-0.03, "range_one_sided", "range")

        # -----------------------------
        # 7) Metadata quality adjustments
        # -----------------------------
        if a.is_primary_key and a.nullable:
            add(-0.03, "pk_nullable_inconsistent_left", "metadata_quality")
        if b.is_primary_key and b.nullable:
            add(-0.03, "pk_nullable_inconsistent_right", "metadata_quality")

        clamp()
        final = round(score, 4)
        if final < 1.0 and not reasons:
            reasons.append("weak_structural_match")

        return {"final": final, "reasons": reasons, "details": details}


def _norm_text(x: Any) -> str:
    return str(x).strip().lower()


def _regex_text(p: Any) -> str:
    if p is None:
        return ""
    if hasattr(p, "pattern"):
        return str(p.pattern)
    return str(p)


def _regex_signature(pattern: str) -> str:
    """
    Very light-weight regex family signature.
    Helps treat equivalent families as partially compatible.
    """
    p = pattern.lower()
    signature_bits = [
        "digit" if ("\\d" in p or "[0-9]" in p) else "",
        "alpha" if ("[a-z]" in p or "[a-zA-Z]" in p or "\\w" in p) else "",
        "anchor" if ("^" in p and "$" in p) else "",
        "plus" if "+" in p else "",
        "star" if "*" in p else "",
    ]
    return "|".join(bit for bit in signature_bits if bit)


def _range_overlap_ratio(a_min: float, a_max: float, b_min: float, b_max: float) -> float:
    if a_min > a_max:
        a_min, a_max = a_max, a_min
    if b_min > b_max:
        b_min, b_max = b_max, b_min

    lo = max(a_min, b_min)
    hi = min(a_max, b_max)
    if hi < lo:
        return 0.0

    span_a = a_max - a_min
    span_b = b_max - b_min

    # Point intervals.
    if span_a == 0 and span_b == 0:
        return 1.0 if a_min == b_min else 0.0
    if span_a == 0:
        return 1.0 if b_min <= a_min <= b_max else 0.0
    if span_b == 0:
        return 1.0 if a_min <= b_min <= a_max else 0.0

    denom = max(min(span_a, span_b), 1e-9)
    overlap = (hi - lo) / denom
    return max(0.0, min(1.0, overlap))
