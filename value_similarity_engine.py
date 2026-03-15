import math
import random
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np


NULL_STRINGS = {"", "null", "none", "nan", "na", "n/a", "nil"}


def is_null(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    if isinstance(x, str):
        return x.strip().lower() in NULL_STRINGS
    return False


def safe_float(x: Any) -> Optional[float]:
    if is_null(x):
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, str):
        s = x.strip().replace(",", "")
        if s.endswith("%") and len(s) > 1:
            s = s[:-1]
        try:
            v = float(s)
            return v if math.isfinite(v) else None
        except ValueError:
            return None
    return None


def normalize(x: Any) -> Any:
    if is_null(x):
        return None
    if isinstance(x, (list, tuple)):
        return tuple(normalize(i) for i in x)
    if isinstance(x, str):
        s = " ".join(x.strip().lower().split())
        return s if s else None
    return x


def detect_type_robust(values: List[Any]) -> str:
    non_null = [v for v in values if v is not None]
    if not non_null:
        return "unknown"

    if any(isinstance(v, (tuple, list)) for v in non_null):
        return "categorical"

    numeric = sum(1 for v in non_null if safe_float(v) is not None)
    ratio = numeric / max(1, len(non_null))

    # Keep threshold strict so alpha-numeric IDs are not treated as numeric.
    if ratio >= 0.97:
        return "numeric"
    return "categorical"


def entropy_from_counter(freq: Counter) -> float:
    total = sum(freq.values())
    if total <= 0:
        return 0.0
    return -sum((c / total) * math.log((c / total) + 1e-12) for c in freq.values())


def monotonic_score(arr: np.ndarray) -> float:
    if len(arr) < 5:
        return 0.0
    inc = np.mean(arr[:-1] <= arr[1:])
    dec = np.mean(arr[:-1] >= arr[1:])
    return float(max(inc, dec))


def outlier_ratio(arr: np.ndarray) -> float:
    if len(arr) < 10:
        return 0.0
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    if iqr <= 0:
        return 0.0
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    return float(np.mean((arr < lo) | (arr > hi)))


def length_profile(values: Iterable[Any]) -> Tuple[float, float]:
    lengths = [len(v) for v in values if isinstance(v, str)]
    if not lengths:
        return (0.0, 0.0)
    arr = np.array(lengths, dtype=float)
    return (float(np.mean(arr)), float(np.std(arr)))


class ColumnStats:
    """
    Column profiling object consumed by ValueSimilarity.
    Interface is kept compatible with your join builder:
      - sample
      - valid_count
      - null_frac
      - n_distinct
      - type
    """

    def __init__(
        self,
        values: List[Any],
        sample_size: int = 20000,
        mcv_k: int = 100,
        random_state: int = 7,
        set_cap: int = 10000,
    ):
        if len(values) > sample_size:
            rng = random.Random(random_state)
            self.sample = rng.sample(values, sample_size)
        else:
            self.sample = list(values)

        self.total_rows = len(self.sample)
        norm = [normalize(v) for v in self.sample]
        self.non_null = [v for v in norm if v is not None]
        self.valid_count = len(self.non_null)
        self.null_frac = 1.0 - (self.valid_count / max(1, self.total_rows))

        self.type = detect_type_robust(self.non_null)
        self.n_distinct = 0
        self.distinct_ratio = 0.0
        self.entropy = 0.0
        self.min = None
        self.max = None

        self.mcv: Dict[Any, float] = {}
        self.mcv_keys: Set[Any] = set()
        self.mcv_magnitude = 0.0
        self.clean_categorical: List[Any] = []
        self.value_set: Set[Any] = set()
        self.avg_len = 0.0
        self.std_len = 0.0

        self.clean_numeric: np.ndarray = np.array([], dtype=float)
        self.data_for_hist: Optional[np.ndarray] = None
        self.integer_ratio = 0.0
        self.monotonic = 0.0
        self.outlier_ratio = 0.0
        self.q05 = 0.0
        self.q50 = 0.0
        self.q95 = 0.0

        if self.type == "numeric":
            numeric_vals = []
            for v in self.non_null:
                f = safe_float(v)
                if f is not None and math.isfinite(f):
                    numeric_vals.append(f)

            if numeric_vals:
                arr = np.array(numeric_vals, dtype=float)
                self.clean_numeric = arr
                self.data_for_hist = arr
                self.min = float(np.min(arr))
                self.max = float(np.max(arr))
                self.n_distinct = len(set(arr.tolist()))
                self.distinct_ratio = self.n_distinct / max(1, len(arr))
                self.integer_ratio = float(np.mean(np.isclose(arr, np.round(arr))))
                self.monotonic = monotonic_score(arr)
                self.outlier_ratio = outlier_ratio(arr)
                self.q05, self.q50, self.q95 = [float(x) for x in np.quantile(arr, [0.05, 0.50, 0.95])]

                rounded = np.round(arr, 6).tolist()
                if len(rounded) > set_cap:
                    step = max(1, len(rounded) // set_cap)
                    rounded = rounded[::step][:set_cap]
                self.value_set = set(rounded)
            else:
                # Fallback: if numeric parsing fails after type detection, treat as categorical.
                self.type = "categorical"

        if self.type == "categorical":
            self.clean_categorical = self.non_null
            freq = Counter(self.clean_categorical)
            self.n_distinct = len(freq)
            self.distinct_ratio = self.n_distinct / max(1, self.valid_count)
            self.entropy = entropy_from_counter(freq)

            total = len(self.clean_categorical)
            self.mcv = {k: v / total for k, v in freq.most_common(mcv_k)} if total > 0 else {}
            self.mcv_keys = set(self.mcv.keys())
            self.mcv_magnitude = math.sqrt(sum(v * v for v in self.mcv.values()))

            values = list(freq.keys())
            if len(values) > set_cap:
                step = max(1, len(values) // set_cap)
                values = values[::step][:set_cap]
            self.value_set = set(values)
            self.avg_len, self.std_len = length_profile(self.clean_categorical)

        if self.type == "unknown":
            self.n_distinct = 0
            self.distinct_ratio = 0.0


class ValueSimilarity:
    def __init__(self, A: ColumnStats, B: ColumnStats):
        self.A = A
        self.B = B

    def compute_score(self) -> Dict[str, Any]:
        if self.A.valid_count == 0 or self.B.valid_count == 0:
            return {"final": 0.0, "reason": "empty"}

        if self.A.type == "unknown" or self.B.type == "unknown":
            return {"final": 0.0, "reason": "unknown_type"}

        if self.A.type != self.B.type:
            return {"final": 0.0, "reason": "type_mismatch"}

        if self.A.type == "categorical":
            return self._compute_categorical()

        if self.A.type == "numeric":
            return self._compute_numeric()

        return {"final": 0.0, "reason": "unsupported_type"}

    def _compute_categorical(self) -> Dict[str, Any]:
        s_cos = self._mcv_cosine()
        s_wj = self._mcv_weighted_jaccard()
        s_cont = self._set_containment()
        s_jacc = self._set_jaccard()
        s_nd = self._distinct_log_similarity()
        s_null = self._null_compatibility()
        s_ent = self._entropy_similarity()
        s_len = self._length_similarity()

        value_evidence = max(s_cos, s_wj, s_cont, s_jacc)
        if value_evidence < 0.03:
            # Guardrail: same cardinality/null profile but no value overlap should stay low.
            final = 0.08 * (0.5 * s_nd + 0.5 * s_null)
        else:
            final = (
                0.36 * value_evidence
                + 0.16 * s_cos
                + 0.14 * s_wj
                + 0.10 * s_cont
                + 0.08 * s_jacc
                + 0.07 * s_nd
                + 0.04 * s_null
                + 0.03 * s_ent
                + 0.02 * s_len
            )

        final = float(round(max(0.0, min(1.0, final)), 4))
        return {
            "final": final,
            "details": {
                "mcv_cos": round(s_cos, 3),
                "weighted_jacc": round(s_wj, 3),
                "contain": round(s_cont, 3),
                "jacc": round(s_jacc, 3),
                "n_distinct": round(s_nd, 3),
                "null": round(s_null, 3),
                "entropy": round(s_ent, 3),
                "len_profile": round(s_len, 3),
                "value_evidence": round(value_evidence, 3),
            },
        }

    def _compute_numeric(self) -> Dict[str, Any]:
        s_hist = self._shared_histogram_intersection()
        s_cdf = self._cdf_similarity()
        s_range = self._range_containment()
        s_exact = self._numeric_set_overlap()
        s_nd = self._distinct_log_similarity()
        s_null = self._null_compatibility()
        s_int = self._integer_ratio_similarity()
        s_mono = self._monotonic_similarity()
        s_out = self._outlier_similarity()
        s_scale = self._scale_similarity()

        value_evidence = max(s_hist, s_cdf, s_exact, 0.6 * s_hist + 0.4 * s_cdf)
        if value_evidence < 0.03:
            final = 0.08 * (0.5 * s_nd + 0.5 * s_null)
        else:
            final = (
                0.32 * value_evidence
                + 0.16 * s_hist
                + 0.12 * s_cdf
                + 0.10 * s_exact
                + 0.08 * s_range
                + 0.08 * s_nd
                + 0.05 * s_null
                + 0.03 * s_int
                + 0.02 * s_mono
                + 0.02 * s_out
                + 0.02 * s_scale
            )

        final = float(round(max(0.0, min(1.0, final)), 4))
        return {
            "final": final,
            "details": {
                "hist": round(s_hist, 3),
                "cdf": round(s_cdf, 3),
                "range": round(s_range, 3),
                "exact_overlap": round(s_exact, 3),
                "n_distinct": round(s_nd, 3),
                "null": round(s_null, 3),
                "int_ratio": round(s_int, 3),
                "monotonic": round(s_mono, 3),
                "outliers": round(s_out, 3),
                "scale": round(s_scale, 3),
                "value_evidence": round(value_evidence, 3),
            },
        }

    # -----------------------------
    # Categorical signals
    # -----------------------------

    def _mcv_cosine(self) -> float:
        if not self.A.mcv or not self.B.mcv:
            return 0.0
        keys = self.A.mcv_keys & self.B.mcv_keys
        if not keys:
            return 0.0
        dot = sum(self.A.mcv[k] * self.B.mcv[k] for k in keys)
        denom = self.A.mcv_magnitude * self.B.mcv_magnitude
        return float(dot / denom) if denom > 0 else 0.0

    def _mcv_weighted_jaccard(self) -> float:
        if not self.A.mcv or not self.B.mcv:
            return 0.0
        keys = self.A.mcv_keys | self.B.mcv_keys
        if not keys:
            return 0.0
        num = sum(min(self.A.mcv.get(k, 0.0), self.B.mcv.get(k, 0.0)) for k in keys)
        den = sum(max(self.A.mcv.get(k, 0.0), self.B.mcv.get(k, 0.0)) for k in keys)
        if den <= 0:
            return 0.0
        return float(num / den)

    def _set_containment(self) -> float:
        if not self.A.value_set or not self.B.value_set:
            return 0.0
        inter = len(self.A.value_set & self.B.value_set)
        denom = min(len(self.A.value_set), len(self.B.value_set))
        return float(inter / max(1, denom))

    def _set_jaccard(self) -> float:
        if not self.A.value_set or not self.B.value_set:
            return 0.0
        inter = len(self.A.value_set & self.B.value_set)
        union = len(self.A.value_set | self.B.value_set)
        return float(inter / max(1, union))

    def _length_similarity(self) -> float:
        if self.A.avg_len <= 0 and self.B.avg_len <= 0:
            return 0.5
        mean_gap = abs(self.A.avg_len - self.B.avg_len) / max(1.0, max(self.A.avg_len, self.B.avg_len))
        std_gap = abs(self.A.std_len - self.B.std_len) / max(1.0, max(self.A.std_len, self.B.std_len))
        score = 1.0 - 0.7 * mean_gap - 0.3 * std_gap
        return float(max(0.0, min(1.0, score)))

    # -----------------------------
    # Numeric signals
    # -----------------------------

    def _shared_histogram_intersection(self) -> float:
        if self.A.data_for_hist is None or self.B.data_for_hist is None:
            return 0.0
        if len(self.A.data_for_hist) < 2 or len(self.B.data_for_hist) < 2:
            return 0.0

        a = self.A.data_for_hist
        b = self.B.data_for_hist

        # Robust clipping to reduce outlier dominance.
        a_lo, a_hi = np.quantile(a, [0.01, 0.99])
        b_lo, b_hi = np.quantile(b, [0.01, 0.99])
        g_min = min(a_lo, b_lo)
        g_max = max(a_hi, b_hi)
        if g_max <= g_min:
            return 1.0

        bins = np.linspace(g_min, g_max, 41)
        ah = np.clip(a, g_min, g_max)
        bh = np.clip(b, g_min, g_max)
        hist_a, _ = np.histogram(ah, bins=bins)
        hist_b, _ = np.histogram(bh, bins=bins)
        hist_a = hist_a / max(1, np.sum(hist_a))
        hist_b = hist_b / max(1, np.sum(hist_b))
        return float(np.sum(np.minimum(hist_a, hist_b)))

    def _cdf_similarity(self) -> float:
        if self.A.data_for_hist is None or self.B.data_for_hist is None:
            return 0.0
        if len(self.A.data_for_hist) < 5 or len(self.B.data_for_hist) < 5:
            return 0.0

        q = np.linspace(0.05, 0.95, 19)
        aq = np.quantile(self.A.data_for_hist, q)
        bq = np.quantile(self.B.data_for_hist, q)

        diff = float(np.mean(np.abs(aq - bq)))
        span = max(
            self.A.q95 - self.A.q05,
            self.B.q95 - self.B.q05,
            abs(self.A.q50 - self.B.q50),
            1e-9,
        )
        return float(max(0.0, 1.0 - diff / span))

    def _numeric_set_overlap(self) -> float:
        if not self.A.value_set or not self.B.value_set:
            return 0.0
        inter = len(self.A.value_set & self.B.value_set)
        denom = min(len(self.A.value_set), len(self.B.value_set))
        return float(inter / max(1, denom))

    def _range_containment(self) -> float:
        if self.A.min is None or self.B.min is None:
            return 0.0
        o_min = max(self.A.min, self.B.min)
        o_max = min(self.A.max, self.B.max)
        if o_max < o_min:
            return 0.0
        overlap = o_max - o_min
        span_a = self.A.max - self.A.min
        span_b = self.B.max - self.B.min
        denom = min(span_a, span_b)
        if denom <= 0:
            return 1.0
        return float(max(0.0, min(1.0, overlap / denom)))

    # -----------------------------
    # Shared structural signals
    # -----------------------------

    def _distinct_log_similarity(self) -> float:
        da = max(2, self.A.n_distinct)
        db = max(2, self.B.n_distinct)
        la = math.log(da)
        lb = math.log(db)
        return float(max(0.0, 1.0 - abs(la - lb) / max(la, lb)))

    def _null_compatibility(self) -> float:
        return float(max(0.0, 1.0 - abs(self.A.null_frac - self.B.null_frac)))

    def _entropy_similarity(self) -> float:
        ea = getattr(self.A, "entropy", None)
        eb = getattr(self.B, "entropy", None)
        if ea is None or eb is None:
            return 0.5
        den = max(abs(ea), abs(eb), 1e-9)
        return float(max(0.0, 1.0 - abs(ea - eb) / den))

    def _integer_ratio_similarity(self) -> float:
        return float(max(0.0, 1.0 - abs(self.A.integer_ratio - self.B.integer_ratio)))

    def _monotonic_similarity(self) -> float:
        return float(max(0.0, 1.0 - abs(self.A.monotonic - self.B.monotonic)))

    def _outlier_similarity(self) -> float:
        return float(max(0.0, 1.0 - abs(self.A.outlier_ratio - self.B.outlier_ratio)))

    def _scale_similarity(self) -> float:
        if self.A.min is None or self.B.min is None:
            return 0.5
        span_a = self.A.max - self.A.min
        span_b = self.B.max - self.B.min
        if span_a <= 0 or span_b <= 0:
            return 1.0
        ratio = max(span_a, span_b) / max(1e-9, min(span_a, span_b))
        return float(max(0.0, 1.0 - math.log10(ratio) / 3.0))
