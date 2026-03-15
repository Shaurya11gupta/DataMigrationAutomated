#!/usr/bin/env python3
"""
Practical candidate-generation pipeline for target-column mapping.

Implements:
1) Single-column retrieval (coarse):
   - name embedding ANN (or fast exact fallback)
   - type compatibility filtering
2) Single-column scoring (fine):
   - name similarity
   - value similarity
   - constraint compatibility
   - type compatibility
   - null/distinct compatibility
3) Composite candidate generation:
   - join-graph expansion with beam-like seeded search
   - max arity and max hops constraints
   - cycle-safe path expansion
4) Transformation feasibility filter:
   - infers feasible transformation families for each candidate
5) Execution-guided quick scoring:
   - lightweight template checks on samples
   - exact/fuzzy/MAE/token overlap metrics
6) Final ranking:
   - weighted score + calibrated confidence
   - top-k + alternatives + abstain flag
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from constraint_similarity_engine import ColumnConstraints, ConstraintCompatibilityJoin
from join_graph_builder_v2 import Column, ColumnType, DefaultTypeCompatibility, JoinEdge, Table
from value_similarity_engine import ColumnStats, ValueSimilarity

try:  # optional ANN backend
    import hnswlib  # type: ignore
except Exception:  # pragma: no cover
    hnswlib = None

try:  # optional stronger embeddings
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None


# ---------------------------------------------------------------------------
# Legacy token sets (kept ONLY for minimal fallback when no embedding model
# is available).  All primary logic now goes through SemanticEngine.
# ---------------------------------------------------------------------------
AMOUNT_TOKENS = {
    "amount", "amt", "gross", "net", "revenue", "cost", "price", "balance",
    "total", "payment", "charge",
}
RATE_TOKENS = {"rate", "fx", "exchange", "conversion", "multiplier"}
CURRENCY_TOKENS = {"usd", "eur", "inr", "gbp", "jpy", "currency", "forex"}
EMAIL_TOKENS = {"email", "mail"}
USERNAME_TOKENS = {"username", "user", "login", "localpart"}
FULLNAME_TOKENS = {"full", "fullname", "name", "display"}
FIRST_LAST_TOKENS = {"first", "last", "middle", "name"}


# =========================================================
# Dataclasses
# =========================================================


@dataclass(frozen=True)
class ColumnRef:
    table: str
    column: str


@dataclass
class CandidateSet:
    refs: List[ColumnRef]
    join_path: List[JoinEdge]
    base_score: float
    component_scores: Dict[str, float]
    feasible_families: List[str] = field(default_factory=list)
    family_scores: Dict[str, float] = field(default_factory=dict)
    best_family: str = "unknown"
    quick_score: float = 0.0
    final_score: float = 0.0
    confidence: float = 0.0


@dataclass
class TargetSpec:
    table: str
    name: str
    col_type: ColumnType
    constraints: ColumnConstraints
    stats: Any
    description: str = ""
    sample_values: Optional[List[Any]] = None

    @classmethod
    def from_column(cls, table: str, column: Column, description: str = "") -> "TargetSpec":
        return cls(
            table=table,
            name=column.name,
            col_type=column.col_type,
            constraints=column.constraints,
            stats=column.stats,
            description=description,
            sample_values=getattr(column.stats, "sample", None),
        )


# =========================================================
# Name embeddings + ANN
# =========================================================


class NameEmbedder:
    """
    Optional sentence-transformer embedder with deterministic hashed fallback.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        hashed_dim: int = 384,
    ):
        self.model_name = model_name
        self.hashed_dim = hashed_dim
        self.model = None
        if model_name and SentenceTransformer is not None:
            self.model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.hashed_dim), dtype=np.float32)

        if self.model is not None:
            arr = self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(arr, dtype=np.float32)

        # Hashed fallback: token n-gram hashing + L2 normalization.
        mats = [self._hash_embed(t) for t in texts]
        return np.vstack(mats).astype(np.float32)

    def _hash_embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.hashed_dim, dtype=np.float32)
        toks = _name_tokens(text)
        if not toks:
            return v
        for tok in toks:
            for n in (2, 3, 4):
                for i in range(max(0, len(tok) - n + 1)):
                    gram = tok[i : i + n]
                    idx = (hash(gram) % self.hashed_dim + self.hashed_dim) % self.hashed_dim
                    v[idx] += 1.0
        nrm = np.linalg.norm(v)
        return v / nrm if nrm > 0 else v


# ---------------------------------------------------------------------------
# Lexical similarity (pure token/string-based, no hardcoded semantic groups)
# ---------------------------------------------------------------------------

def _lexical_name_similarity(target_name: str, source_name: str) -> float:
    """Pure lexical similarity between column names.
    Uses token overlap, containment, and substring matching.
    No hardcoded semantic groups.
    """
    ta = set(_name_tokens(target_name))
    tb = set(_name_tokens(source_name))
    if not ta or not tb:
        return 0.0

    # 1. Token Jaccard
    jaccard = len(ta & tb) / len(ta | tb)

    # 2. Directional containment (what fraction of target tokens appear in source)
    containment = len(ta & tb) / len(ta) if ta else 0.0

    # 3. Substring containment bonus
    tn = target_name.lower().replace("_", "").replace("-", "").replace(".", "")
    sn = source_name.lower().replace("_", "").replace("-", "").replace(".", "")
    substr_bonus = 0.15 if (tn in sn or sn in tn) else 0.0

    # 4. Character n-gram overlap (captures sub-word similarity)
    t_ngrams = _char_ngrams(target_name, 3)
    s_ngrams = _char_ngrams(source_name, 3)
    ngram_sim = len(t_ngrams & s_ngrams) / max(1, len(t_ngrams | s_ngrams)) if (t_ngrams and s_ngrams) else 0.0

    combined = max(
        jaccard,
        0.40 * jaccard + 0.25 * containment + 0.20 * ngram_sim + 0.15 * substr_bonus,
    )
    return min(1.0, combined)


def _char_ngrams(name: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from a lowered, cleaned name."""
    s = re.sub(r"[^a-z0-9]", "", name.lower())
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i+n] for i in range(len(s) - n + 1)}


def _tokenize_name_to_str(name: str) -> str:
    """Convert column_name/camelCase to 'column name camel case'."""
    return " ".join(_name_tokens(name))


# ---------------------------------------------------------------------------
# SemanticEngine: embedding-based understanding (replaces hardcoded tokens)
# ---------------------------------------------------------------------------

class SemanticEngine:
    """
    Embedding-based semantic understanding engine.

    Replaces ALL hardcoded token sets (AMOUNT_TOKENS, _DATE_PART_TOKENS,
    _SEMANTIC_GROUPS, etc.) with learned concept vectors.  At init time we
    encode a set of natural-language *concept descriptions* and cache the
    resulting vectors.  At query time we compute cosine similarity between
    column-name embeddings and concept vectors – the model's trained
    knowledge of language does the heavy lifting, not a hand-curated word
    list.

    When no SentenceTransformer model is available the engine degrades
    gracefully to pure-lexical similarity.
    """

    # Concept descriptions – phrased as natural-language descriptions so the
    # model can leverage its full vocabulary.
    _CONCEPT_DESCRIPTIONS: Dict[str, str] = {
        "monetary":
            "monetary amount payment cost price revenue salary wage balance "
            "total charge fee income expense profit earning",
        "exchange_rate":
            "exchange rate conversion factor multiplier forex currency rate "
            "FX cross rate spot rate",
        "currency":
            "currency denomination dollar euro rupee pound yen USD EUR INR "
            "GBP JPY money unit",
        "email":
            "email address mail electronic message inbox e-mail",
        "username":
            "username user login account handle identifier profile screen name",
        "full_name":
            "full name complete name display name person name whole name",
        "name_component":
            "first name last name middle name given name family name surname "
            "forename",
        "date_temporal":
            "date time datetime timestamp when moment period created updated "
            "modified occurred happened event scheduled registered signed up "
            "born birthday anniversary",
        "date_extraction":
            "year month day hour minute second quarter week weekday epoch "
            "date part extraction component day of week day of year",
        "duration_span":
            "duration tenure age span elapsed time difference interval length "
            "period gap days between months since years of service seniority",
        "identifier":
            "identifier id key primary key foreign key reference code number "
            "unique serial index",
        "location_geo":
            "country state city region province zip postal address location "
            "place area coordinates geography",
        "status_flag":
            "status state active inactive enabled disabled flag boolean "
            "indicator yes no true false",
        "text_description":
            "text description comment note remark message content body "
            "narrative memo",
        "file_resource":
            "file path document attachment url uri link resource endpoint "
            "filename extension",
        "aggregation":
            "sum count average mean total aggregate minimum maximum "
            "statistics number of",
        "percentage_ratio":
            "percentage ratio proportion fraction rate share percent",
        "phone_contact":
            "phone telephone mobile cell number contact call fax",
        "category_class":
            "category type class group classification label tag kind code "
            "mapping lookup dimension",
        "concatenation":
            "concatenation combination merge join combine full complete "
            "composite assembled format template",
        "parsing_extraction":
            "parse extract split separate tokenize decompose derive pull "
            "out substring regex",
    }

    def __init__(self, embedder: NameEmbedder):
        self.embedder = embedder
        self._concept_vectors: Dict[str, np.ndarray] = {}
        self._col_emb_cache: Dict[str, np.ndarray] = {}
        self._concept_score_cache: Dict[str, Dict[str, float]] = {}
        self._pair_sim_cache: Dict[str, float] = {}

        if self.has_model:
            self._precompute_concept_vectors()

    # ------------------------------------------------------------------
    @property
    def has_model(self) -> bool:
        return self.embedder.model is not None

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _precompute_concept_vectors(self) -> None:
        keys = list(self._CONCEPT_DESCRIPTIONS.keys())
        texts = list(self._CONCEPT_DESCRIPTIONS.values())
        vecs = self.embedder.encode(texts)
        for k, v in zip(keys, vecs):
            self._concept_vectors[k] = v

    # ------------------------------------------------------------------
    # Column embedding (cached)
    # ------------------------------------------------------------------

    def _get_col_embedding(
        self, name: str, col_type: str = "", table: str = "",
    ) -> np.ndarray:
        key = f"{table}|{name}|{col_type}"
        if key not in self._col_emb_cache:
            tokens = _name_tokens(name)
            text = " ".join(tokens)
            if col_type:
                text += f" {col_type} column"
            if table:
                text = " ".join(_name_tokens(table)) + " " + text
            self._col_emb_cache[key] = self.embedder.encode([text])[0]
        return self._col_emb_cache[key]

    # ------------------------------------------------------------------
    # Core: name similarity (replaces _enhanced_name_similarity)
    # ------------------------------------------------------------------

    def name_similarity(
        self,
        target_name: str, source_name: str,
        target_type: str = "", source_type: str = "",
        target_table: str = "", source_table: str = "",
    ) -> float:
        """
        Semantic + lexical similarity between two column names.
        Combines embedding cosine similarity with pure-lexical signals.
        """
        cache_key = (
            f"{target_table}.{target_name}<{target_type}>"
            f"|{source_table}.{source_name}<{source_type}>"
        )
        if cache_key in self._pair_sim_cache:
            return self._pair_sim_cache[cache_key]

        # Pure lexical (always available)
        lexical = _lexical_name_similarity(target_name, source_name)

        if not self.has_model:
            self._pair_sim_cache[cache_key] = lexical
            return lexical

        # Semantic cosine similarity (L2-normalized → dot = cosine)
        t_vec = self._get_col_embedding(target_name, target_type, target_table)
        s_vec = self._get_col_embedding(source_name, source_type, source_table)
        semantic = float(np.dot(t_vec, s_vec))

        # Cross-type affinity (e.g. "signup_year" ↔ "signup_date")
        cross = self.cross_type_affinity(
            target_name, source_name, target_type, source_type,
        )

        combined = max(
            0.55 * semantic + 0.30 * lexical + 0.15 * cross,
            0.40 * lexical + 0.35 * semantic + 0.25 * cross,
            lexical,  # never go below pure lexical
        )
        result = float(min(1.0, max(0.0, combined)))
        self._pair_sim_cache[cache_key] = result
        return result

    # ------------------------------------------------------------------
    # Concept matching
    # ------------------------------------------------------------------

    def concept_scores(self, name: str, col_type: str = "") -> Dict[str, float]:
        """How well does *name* match each semantic concept?"""
        if not self._concept_vectors:
            return {}
        cache_key = f"{name}|{col_type}"
        if cache_key not in self._concept_score_cache:
            vec = self._get_col_embedding(name, col_type)
            scores = {
                c: float(np.dot(vec, cv))
                for c, cv in self._concept_vectors.items()
            }
            self._concept_score_cache[cache_key] = scores
        return self._concept_score_cache[cache_key]

    def matches_concept(
        self, name: str, concept: str,
        threshold: float = 0.45, col_type: str = "",
    ) -> bool:
        scores = self.concept_scores(name, col_type)
        return scores.get(concept, 0.0) >= threshold

    def best_concepts(
        self, name: str, col_type: str = "", top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        scores = self.concept_scores(name, col_type)
        if not scores:
            return []
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # ------------------------------------------------------------------
    # Cross-type affinity (detects transformations across types)
    # ------------------------------------------------------------------

    def cross_type_affinity(
        self,
        target_name: str, source_name: str,
        target_type: str = "", source_type: str = "",
    ) -> float:
        """
        Detect cross-type transformation potential via concept matching.
        E.g.  target='signup_year' (int)  ←  source='signup_date' (date).
        """
        if not self._concept_vectors:
            return 0.0

        t_sc = self.concept_scores(target_name, target_type)
        s_sc = self.concept_scores(source_name, source_type)

        aff = 0.0

        # date-part extraction: target ∼ date_extraction, source ∼ date_temporal
        t_dp = t_sc.get("date_extraction", 0)
        s_dt = s_sc.get("date_temporal", 0)
        if t_dp > 0.38 and s_dt > 0.38:
            aff = max(aff, 0.35 * min(t_dp, s_dt) / 0.50)

        # duration / tenure: target ∼ duration_span, source ∼ date_temporal
        t_dur = t_sc.get("duration_span", 0)
        if t_dur > 0.38 and s_dt > 0.38:
            aff = max(aff, 0.30 * min(t_dur, s_dt) / 0.50)

        # email → username
        t_user = t_sc.get("username", 0)
        s_email = s_sc.get("email", 0)
        if t_user > 0.38 and s_email > 0.38:
            aff = max(aff, 0.35 * min(t_user, s_email) / 0.50)

        # currency conversion: target ∼ monetary, source ∼ exchange_rate
        t_mon = t_sc.get("monetary", 0)
        s_rate = s_sc.get("exchange_rate", 0)
        if t_mon > 0.38 and s_rate > 0.35:
            aff = max(aff, 0.30)

        # full-name concatenation: target ∼ full_name, source ∼ name_component
        t_fn = t_sc.get("full_name", 0)
        s_nc = s_sc.get("name_component", 0)
        if t_fn > 0.38 and s_nc > 0.38:
            aff = max(aff, 0.30)

        # parsing / extraction: target ∼ parsing, source ∼ text/email/file
        t_parse = t_sc.get("parsing_extraction", 0)
        s_text = max(
            s_sc.get("text_description", 0),
            s_sc.get("email", 0),
            s_sc.get("file_resource", 0),
        )
        if t_parse > 0.35 and s_text > 0.35:
            aff = max(aff, 0.20)

        # Also check direct embedding similarity as a soft cross-type signal
        t_vec = self._get_col_embedding(target_name, target_type)
        s_vec = self._get_col_embedding(source_name, source_type)
        direct_sim = float(np.dot(t_vec, s_vec))
        if direct_sim > 0.55:
            aff = max(aff, 0.10 * direct_sim)

        return min(1.0, aff)

    # ------------------------------------------------------------------
    # Semantic role detection (replaces _target_flags / hardcoded tokens)
    # ------------------------------------------------------------------

    def detect_target_role(
        self, target_name: str, target_type: str = "",
        description: str = "",
    ) -> Dict[str, bool]:
        """
        Detect what semantic role a target column plays.
        Replaces _target_flags() that used hardcoded AMOUNT_TOKENS etc.
        """
        name_for_scores = target_name
        if description:
            name_for_scores = f"{target_name} {description}"

        if not self._concept_vectors:
            # Minimal lexical fallback (kept small)
            toks = set(_name_tokens(name_for_scores))
            return {
                "currency_like": "currency" in toks or "usd" in toks,
                "amount_like": "amount" in toks or "total" in toks or "price" in toks,
                "username_like": "username" in toks,
                "email_like": "email" in toks or "mail" in toks,
                "full_name_like": "fullname" in target_name.lower().replace("_", "")
                                  or ("full" in toks and "name" in toks),
                "date_part_like": False,
                "duration_like": False,
                "identifier_like": "id" in toks or "key" in toks,
            }

        sc = self.concept_scores(name_for_scores, target_type)
        return {
            "currency_like": (
                sc.get("currency", 0) > 0.42
                or (sc.get("monetary", 0) > 0.45
                    and sc.get("exchange_rate", 0) > 0.30)
            ),
            "amount_like": sc.get("monetary", 0) > 0.42,
            "username_like": sc.get("username", 0) > 0.42,
            "email_like": sc.get("email", 0) > 0.42,
            "full_name_like": sc.get("full_name", 0) > 0.42,
            "date_part_like": sc.get("date_extraction", 0) > 0.40,
            "duration_like": sc.get("duration_span", 0) > 0.40,
            "identifier_like": sc.get("identifier", 0) > 0.42,
        }

    # ------------------------------------------------------------------
    # Semantic role compatibility (replaces _semantic_role_score)
    # ------------------------------------------------------------------

    def role_compatibility(
        self,
        target_name: str,
        source_names: List[str],
        target_type: str = "",
        source_types: Optional[List[str]] = None,
        family_hint: Optional[str] = None,
    ) -> float:
        """
        How semantically compatible are source columns with this target?
        Replaces _semantic_role_score() that used AMOUNT_TOKENS, etc.
        """
        if not source_names:
            return 0.0

        source_types = source_types or [""] * len(source_names)

        if not self._concept_vectors:
            # Lexical fallback
            t_toks = set(_name_tokens(target_name))
            overlaps = []
            for sn in source_names:
                s_toks = set(_name_tokens(sn))
                overlaps.append(
                    len(t_toks & s_toks) / max(1, len(t_toks | s_toks))
                )
            return 0.20 * float(np.mean(overlaps))

        t_sc = self.concept_scores(target_name, target_type)
        score = 0.0

        # Pairwise cross-type affinity
        max_aff = 0.0
        for sn, st in zip(source_names, source_types):
            aff = self.cross_type_affinity(
                target_name, sn, target_type, st,
            )
            max_aff = max(max_aff, aff)
        if max_aff > 0.1:
            score += max_aff

        # Collective source analysis
        all_src_sc = [
            self.concept_scores(sn, st)
            for sn, st in zip(source_names, source_types)
        ]

        # Monetary target + rate/amount sources
        if t_sc.get("monetary", 0) > 0.40 or t_sc.get("currency", 0) > 0.40:
            if any(sc.get("monetary", 0) > 0.40 for sc in all_src_sc):
                score += 0.25
            if any(sc.get("exchange_rate", 0) > 0.35 for sc in all_src_sc):
                score += 0.30

        # Full name target + name parts sources
        if t_sc.get("full_name", 0) > 0.40 or t_sc.get("concatenation", 0) > 0.35:
            name_parts = sum(
                1 for sc in all_src_sc if sc.get("name_component", 0) > 0.40
            )
            if name_parts >= 2:
                score += 0.60
            elif name_parts == 1:
                score += 0.25

        # Username target + email source
        if t_sc.get("username", 0) > 0.40:
            if any(sc.get("email", 0) > 0.40 for sc in all_src_sc):
                score += 0.50

        # Date extraction target + temporal source
        if t_sc.get("date_extraction", 0) > 0.40 or t_sc.get("duration_span", 0) > 0.40:
            if any(sc.get("date_temporal", 0) > 0.40 for sc in all_src_sc):
                score += 0.40

        # Fallback: direct embedding similarity
        if score < 0.1:
            t_vec = self._get_col_embedding(target_name, target_type)
            sims = [
                float(np.dot(t_vec, self._get_col_embedding(sn, st)))
                for sn, st in zip(source_names, source_types)
            ]
            score = 0.20 * max(sims) if sims else 0.0

        return float(max(0.0, min(1.0, score)))

    # ------------------------------------------------------------------
    # Semantic feasibility hints (replaces _target_date_part_flags etc.)
    # ------------------------------------------------------------------

    def infer_semantic_families(
        self, target_name: str, source_names: List[str],
        target_type: str = "", source_types: Optional[List[str]] = None,
    ) -> Set[str]:
        """
        Infer additional feasible transformation families based on semantic
        understanding, beyond what type-matching alone would produce.
        """
        extra: Set[str] = set()
        if not self._concept_vectors:
            return extra

        source_types = source_types or [""] * len(source_names)

        t_sc = self.concept_scores(target_name, target_type)

        # Date part extraction
        if t_sc.get("date_extraction", 0) > 0.38:
            for sn, st in zip(source_names, source_types):
                s_sc = self.concept_scores(sn, st)
                if s_sc.get("date_temporal", 0) > 0.38:
                    extra.update(["date_part", "date_diff"])

        # Duration / tenure
        if t_sc.get("duration_span", 0) > 0.38:
            date_src_count = sum(
                1 for sn, st in zip(source_names, source_types)
                if self.concept_scores(sn, st).get("date_temporal", 0) > 0.38
            )
            if date_src_count >= 1:
                extra.update(["date_diff", "date_part"])
            if date_src_count >= 2:
                extra.add("arithmetic")

        # Username extraction
        if t_sc.get("username", 0) > 0.38:
            if any(
                self.concept_scores(sn, st).get("email", 0) > 0.38
                for sn, st in zip(source_names, source_types)
            ):
                extra.update(["email_username_extract", "split", "regex_extract"])

        # Full name concatenation
        if t_sc.get("full_name", 0) > 0.38 or t_sc.get("concatenation", 0) > 0.35:
            name_parts = sum(
                1 for sn, st in zip(source_names, source_types)
                if self.concept_scores(sn, st).get("name_component", 0) > 0.38
            )
            if name_parts >= 2:
                extra.update(["concat", "format"])

        # Currency conversion
        if t_sc.get("monetary", 0) > 0.40 or t_sc.get("currency", 0) > 0.40:
            has_amount = any(
                self.concept_scores(sn, st).get("monetary", 0) > 0.38
                for sn, st in zip(source_names, source_types)
            )
            has_rate = any(
                self.concept_scores(sn, st).get("exchange_rate", 0) > 0.35
                for sn, st in zip(source_names, source_types)
            )
            if has_amount and has_rate:
                extra.add("currency_convert")
            if has_amount:
                extra.update(["arithmetic", "ratio", "scale"])

        return extra

    # ------------------------------------------------------------------
    # Semantic description for richer text representations
    # ------------------------------------------------------------------

    def describe_column(self, name: str, col_type: str = "") -> str:
        """
        Produce a short semantic description of a column, useful for
        enriching text fed to embedding models (ANN / bi-encoder / cross-encoder).
        """
        concepts = self.best_concepts(name, col_type, top_k=3)
        if not concepts:
            return _tokenize_name_to_str(name)
        desc_parts = [c.replace("_", " ") for c, s in concepts if s > 0.35]
        base = _tokenize_name_to_str(name)
        if desc_parts:
            return f"{base} ({', '.join(desc_parts[:2])})"
        return base


class ANNIndex:
    """
    ANN wrapper:
      - Uses hnswlib if available
      - Falls back to exact dot-product top-k
    """

    def __init__(self, vectors: np.ndarray, ids: List[str], use_hnsw: bool = True):
        self.vectors = vectors.astype(np.float32)
        self.ids = list(ids)
        self.use_hnsw = bool(use_hnsw and hnswlib is not None and len(ids) > 0)
        self.index = None
        self.id_to_pos = {ids[i]: i for i in range(len(ids))}

        if self.use_hnsw:
            dim = self.vectors.shape[1]
            self.index = hnswlib.Index(space="cosine", dim=dim)
            self.index.init_index(max_elements=len(ids), ef_construction=200, M=16)
            self.index.add_items(self.vectors, np.arange(len(ids)))
            self.index.set_ef(min(200, max(20, len(ids))))

    def query(self, qvec: np.ndarray, k: int) -> Tuple[List[str], List[float]]:
        if len(self.ids) == 0 or k <= 0:
            return [], []
        k = min(k, len(self.ids))

        if self.use_hnsw and self.index is not None:
            labels, dists = self.index.knn_query(qvec.reshape(1, -1), k=k)
            pos = labels[0].tolist()
            # cosine distance -> cosine similarity
            sims = [float(1.0 - d) for d in dists[0].tolist()]
            return [self.ids[p] for p in pos], sims

        sims = self.vectors @ qvec
        idx = np.argsort(-sims)[:k]
        return [self.ids[i] for i in idx.tolist()], [float(sims[i]) for i in idx.tolist()]


# =========================================================
# Candidate generation engine
# =========================================================


class CandidateGenerationEngine:
    def __init__(
        self,
        source_tables: Dict[str, Table],
        join_edges: Sequence[JoinEdge],
        name_similarity_fn: Optional[Callable[[str, str], Any]] = None,
        embedder: Optional[NameEmbedder] = None,
        type_engine: Any = None,
        value_sim_class: Any = ValueSimilarity,
        constraint_engine: Any = ConstraintCompatibilityJoin,
        use_hnsw_ann: bool = True,
        ann_type_filter: bool = True,
    ):
        self.source_tables = source_tables
        self.join_edges = list(join_edges)
        self.name_similarity_fn = name_similarity_fn
        self.embedder = embedder or NameEmbedder(model_name=None)
        self.type_engine = type_engine or DefaultTypeCompatibility
        self.value_sim_class = value_sim_class
        self.constraint_engine = constraint_engine
        self.use_hnsw_ann = use_hnsw_ann
        self.ann_type_filter = ann_type_filter

        # Semantic engine (embedding-based concept matching)
        self.semantic = SemanticEngine(self.embedder)

        self._col_id_to_ref: Dict[str, ColumnRef] = {}
        self._ref_to_col: Dict[Tuple[str, str], Column] = {}
        self._table_to_ids: Dict[str, List[str]] = defaultdict(list)

        self._index_by_bucket: Dict[str, ANNIndex] = {}
        self._id_to_bucket: Dict[str, str] = {}

        self._table_graph: Dict[str, List[Tuple[str, JoinEdge]]] = defaultdict(list)
        self._build_catalog()
        self._build_ann_indexes()
        self._build_table_graph()

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------

    def rank_candidates(
        self,
        target: TargetSpec,
        coarse_top_m: int = 50,
        fine_top_m: int = 20,
        max_arity: int = 4,
        max_hops: int = 3,
        seed_count: int = 8,
        per_table_candidates: int = 3,
        top_k: int = 10,
        abstain_threshold: float = 0.45,
    ) -> Dict[str, Any]:
        # 1) Coarse retrieval
        coarse = self._coarse_retrieve(target, coarse_top_m)
        # 2) Fine scoring
        fine = self._fine_score(target, coarse, keep_top=fine_top_m)

        # always include single-column candidates
        candidate_sets = self._single_candidates_from_fine(fine)

        # 3) Composite expansion
        composites = self._generate_composites(
            target=target,
            fine_rows=fine,
            max_arity=max_arity,
            max_hops=max_hops,
            seed_count=seed_count,
            per_table_candidates=per_table_candidates,
        )
        candidate_sets.extend(composites)

        # de-duplicate candidates by canonical refs
        candidate_sets = self._dedup_candidates(candidate_sets)

        # 4) Feasibility filter + 5) execution-guided quick scoring + 6) final ranking
        for c in candidate_sets:
            c.feasible_families = self._infer_feasible_families(target, c)
            c.family_scores = self._quick_family_scores(target, c, c.feasible_families)
            if c.family_scores:
                c.best_family = max(c.family_scores, key=lambda k: c.family_scores[k])
                c.quick_score = c.family_scores[c.best_family]
            else:
                c.best_family = "unknown"
                c.quick_score = 0.0

            fam_prior = self._family_prior(c.best_family)
            semantic_fit = self._semantic_role_score(target, c.refs, family_hint=c.best_family)
            complexity_penalty = 0.025 * max(0, len(c.refs) - 1) + 0.015 * len(c.join_path)
            logical_penalty = self._logical_mismatch_penalty(target, c, c.best_family)

            # Give more weight to quick_score (execution evidence) when available
            if c.quick_score > 0.5:
                # Strong execution evidence: trust it more
                c.final_score = (
                    0.35 * c.base_score +
                    0.42 * c.quick_score +
                    0.12 * fam_prior +
                    0.11 * semantic_fit
                ) - complexity_penalty - logical_penalty
            else:
                c.final_score = (
                    0.48 * c.base_score +
                    0.28 * c.quick_score +
                    0.13 * fam_prior +
                    0.11 * semantic_fit
                ) - complexity_penalty - logical_penalty

            c.final_score = max(0.0, min(1.0, c.final_score))
            # Recalibrated sigmoid: center at 0.42, spread 0.20 -> higher confidence output
            c.confidence = _sigmoid((c.final_score - 0.42) / 0.20)

        candidate_sets.sort(key=lambda x: x.confidence, reverse=True)
        primary = candidate_sets[:top_k]
        alternatives = candidate_sets[top_k : top_k + top_k]
        abstain = True if not primary else (primary[0].confidence < abstain_threshold)

        return {
            "target": {
                "table": target.table,
                "column": target.name,
                "type": getattr(target.col_type, "base_type", str(target.col_type)),
            },
            "abstain": abstain,
            "top_candidates": [self._serialize_candidate(c) for c in primary],
            "alternatives": [self._serialize_candidate(c) for c in alternatives],
            "debug": {
                "coarse_count": len(coarse),
                "fine_count": len(fine),
                "candidate_count_total": len(candidate_sets),
            },
        }

    # -----------------------------------------------------
    # Build catalog + indexes
    # -----------------------------------------------------

    def _build_catalog(self) -> None:
        for tname, table in self.source_tables.items():
            for cname, col in table.columns.items():
                cid = f"{tname}.{cname}"
                ref = ColumnRef(tname, cname)
                self._col_id_to_ref[cid] = ref
                self._ref_to_col[(tname, cname)] = col
                self._table_to_ids[tname].append(cid)
                self._id_to_bucket[cid] = _type_bucket(col.col_type)

    def _build_ann_indexes(self) -> None:
        texts_by_bucket: Dict[str, List[str]] = defaultdict(list)
        ids_by_bucket: Dict[str, List[str]] = defaultdict(list)

        for cid, ref in self._col_id_to_ref.items():
            col = self._ref_to_col[(ref.table, ref.column)]
            text = self._column_text(ref, col)
            bucket = self._id_to_bucket[cid]
            texts_by_bucket[bucket].append(text)
            ids_by_bucket[bucket].append(cid)

        for bucket, texts in texts_by_bucket.items():
            vecs = self.embedder.encode(texts)
            self._index_by_bucket[bucket] = ANNIndex(
                vectors=vecs,
                ids=ids_by_bucket[bucket],
                use_hnsw=self.use_hnsw_ann,
            )

    def _build_table_graph(self) -> None:
        for e in self.join_edges:
            self._table_graph[e.left_table].append((e.right_table, e))
            self._table_graph[e.right_table].append((e.left_table, e))

        # deterministic ordering
        for t in self._table_graph:
            self._table_graph[t].sort(
                key=lambda x: (x[0], -float(getattr(x[1], "confidence", 0.0)))
            )

    # -----------------------------------------------------
    # Stage-1 coarse retrieval
    # -----------------------------------------------------

    def _coarse_retrieve(self, target: TargetSpec, top_m: int) -> List[Tuple[str, float]]:
        q_text = self._target_text(target)
        q_vec = self.embedder.encode([q_text])[0]

        candidate_scores: Dict[str, float] = {}
        buckets = _compatible_buckets(target.col_type) if self.ann_type_filter else list(self._index_by_bucket.keys())

        if not buckets:
            buckets = list(self._index_by_bucket.keys())

        # Query each compatible bucket and merge.
        per_bucket_k = max(5, top_m)
        for b in buckets:
            idx = self._index_by_bucket.get(b)
            if idx is None:
                continue
            ids, sims = idx.query(q_vec, per_bucket_k)
            for cid, sim in zip(ids, sims):
                if cid not in candidate_scores or candidate_scores[cid] < sim:
                    candidate_scores[cid] = sim

        rows = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return rows[:top_m]

    # -----------------------------------------------------
    # Stage-2 fine scoring
    # -----------------------------------------------------

    def _fine_score(
        self,
        target: TargetSpec,
        coarse_rows: Sequence[Tuple[str, float]],
        keep_top: int,
    ) -> List[Dict[str, Any]]:
        tgt_type_str = getattr(target.col_type, "base_type", str(target.col_type))
        out = []
        for cid, coarse_sim in coarse_rows:
            ref = self._col_id_to_ref[cid]
            src = self._ref_to_col[(ref.table, ref.column)]
            src_type_str = getattr(src.col_type, "base_type", str(src.col_type))

            # Name similarity now uses SemanticEngine (embeddings + lexical)
            nscore = self._name_similarity(
                target.name, src.name,
                target_type=tgt_type_str, source_type=src_type_str,
                target_table=target.table, source_table=ref.table,
            )
            vscore = self._value_similarity(target.stats, src.stats)
            cscore = self._constraint_similarity(target.constraints, src.constraints)
            tscore = self.type_engine.score(target.col_type, src.col_type)
            ndscore = self._null_distinct_compatibility(target.stats, src.stats)
            semscore = self._semantic_role_score(target, [ref], family_hint=None)

            # Cross-type bonus via SemanticEngine (no hardcoded token sets)
            cross_type_bonus = 0.0
            src_bucket = _type_bucket(src.col_type)
            tgt_bucket = _type_bucket(target.col_type)
            if src_bucket != tgt_bucket:
                # Use semantic engine to detect cross-type affinity
                sem_cross = self.semantic.cross_type_affinity(
                    target.name, src.name, tgt_type_str, src_type_str,
                )
                cross_type_bonus = max(sem_cross, 0.12 * nscore if nscore > 0.30 else 0.0)

            score = (
                0.12 * coarse_sim +
                0.26 * nscore +
                0.18 * vscore +
                0.08 * cscore +
                0.06 * tscore +
                0.06 * ndscore +
                0.12 * semscore +
                0.12 * cross_type_bonus
            )
            out.append(
                {
                    "ref": ref,
                    "score": float(max(0.0, min(1.0, score))),
                    "parts": {
                        "coarse": round(float(coarse_sim), 4),
                        "name": round(float(nscore), 4),
                        "value": round(float(vscore), 4),
                        "constraint": round(float(cscore), 4),
                        "type": round(float(tscore), 4),
                        "null_distinct": round(float(ndscore), 4),
                        "semantic": round(float(semscore), 4),
                        "cross_type": round(float(cross_type_bonus), 4),
                    },
                }
            )

        out.sort(key=lambda x: x["score"], reverse=True)
        return out[:keep_top]

    def _single_candidates_from_fine(self, fine_rows: Sequence[Dict[str, Any]]) -> List[CandidateSet]:
        out = []
        for row in fine_rows:
            ref: ColumnRef = row["ref"]
            out.append(
                CandidateSet(
                    refs=[ref],
                    join_path=[],
                    base_score=float(row["score"]),
                    component_scores=row["parts"],
                )
            )
        return out

    # -----------------------------------------------------
    # Stage-3 composite generation
    # -----------------------------------------------------

    def _generate_composites(
        self,
        target: TargetSpec,
        fine_rows: Sequence[Dict[str, Any]],
        max_arity: int,
        max_hops: int,
        seed_count: int,
        per_table_candidates: int,
    ) -> List[CandidateSet]:
        if max_arity <= 1 or not fine_rows:
            return []

        seed_rows = list(fine_rows[:seed_count])
        table_best: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in fine_rows:
            ref: ColumnRef = r["ref"]
            table_best[ref.table].append(r)

        # keep top per table
        for t in table_best:
            table_best[t].sort(key=lambda x: x["score"], reverse=True)
            table_best[t] = table_best[t][:per_table_candidates]

        candidates: List[CandidateSet] = []
        seen: Set[Tuple[Tuple[str, str], ...]] = set()

        for seed in seed_rows:
            seed_ref: ColumnRef = seed["ref"]
            best_paths = self._best_paths_from(seed_ref.table, max_hops=max_hops)

            pool_rows: List[Dict[str, Any]] = [seed]
            # Include same-table companions first (critical for transforms like concat).
            for item in table_best.get(seed_ref.table, []):
                pool_rows.append(item)
            for table_name, path_info in best_paths.items():
                if table_name == seed_ref.table:
                    continue
                for item in table_best.get(table_name, []):
                    pool_rows.append(item)

            # unique by ref
            unique_pool = {}
            for r in pool_rows:
                ref = r["ref"]
                unique_pool[(ref.table, ref.column)] = r
            pool = list(unique_pool.values())

            # generate combinations containing seed
            for arity in range(2, max_arity + 1):
                for comb in combinations(pool, arity):
                    refs = [x["ref"] for x in comb]
                    if not any(r.table == seed_ref.table and r.column == seed_ref.column for r in refs):
                        continue

                    # type/constraint plausibility prune
                    if not self._composite_possible(target, refs):
                        continue

                    tables = {r.table for r in refs}
                    all_path_edges = []
                    feasible = True
                    for t in tables:
                        if t == seed_ref.table:
                            continue
                        if t not in best_paths:
                            feasible = False
                            break
                        all_path_edges.extend(best_paths[t]["path_edges"])
                    if not feasible:
                        continue

                    # dedupe join path edges by object identity-ish key
                    path_unique = _dedup_join_edges(all_path_edges)
                    path_conf = _path_confidence(path_unique)

                    single_mean = float(np.mean([x["score"] for x in comb]))
                    arity_pen = 0.95 ** (arity - 1)
                    path_pen = 0.93 ** len(path_unique)
                    sem_bonus = self._semantic_role_score(target, refs, family_hint=None)
                    score = (((0.78 * single_mean) + (0.18 * path_conf) + (0.04 * sem_bonus)) * arity_pen * path_pen)

                    refs_key = tuple(sorted((r.table, r.column) for r in refs))
                    if refs_key in seen:
                        continue
                    seen.add(refs_key)

                    candidates.append(
                        CandidateSet(
                            refs=sorted(refs, key=lambda r: (r.table, r.column)),
                            join_path=path_unique,
                            base_score=float(max(0.0, min(1.0, score))),
                            component_scores={
                                "single_mean": round(single_mean, 4),
                                "join_path_conf": round(path_conf, 4),
                                "semantic": round(sem_bonus, 4),
                            },
                        )
                    )

        candidates.sort(key=lambda c: c.base_score, reverse=True)
        return candidates

    def _best_paths_from(self, source_table: str, max_hops: int) -> Dict[str, Dict[str, Any]]:
        """
        Best paths by (fewest hops, then highest confidence) up to max_hops.
        """
        best: Dict[str, Dict[str, Any]] = {
            source_table: {"hops": 0, "cost": 0.0, "path_edges": []}
        }
        q = deque([(source_table, 0, 0.0, [], {source_table})])

        while q:
            node, hops, cost, path, visited = q.popleft()
            if hops >= max_hops:
                continue
            for nxt, edge in self._table_graph.get(node, []):
                if nxt in visited:
                    continue
                e_conf = max(1e-6, float(getattr(edge, "confidence", 0.0)))
                new_cost = cost - math.log(e_conf)
                new_hops = hops + 1
                new_path = path + [edge]

                cur = best.get(nxt)
                better = (
                    cur is None or
                    new_hops < cur["hops"] or
                    (new_hops == cur["hops"] and new_cost < cur["cost"])
                )
                if better:
                    best[nxt] = {"hops": new_hops, "cost": new_cost, "path_edges": new_path}
                    q.append((nxt, new_hops, new_cost, new_path, visited | {nxt}))

        return best

    def _composite_possible(self, target: TargetSpec, refs: Sequence[ColumnRef]) -> bool:
        if not refs:
            return False
        src_cols = [self._ref_to_col[(r.table, r.column)] for r in refs]
        src_buckets = [_type_bucket(c.col_type) for c in src_cols]
        tgt_bucket = _type_bucket(target.col_type)
        flags = self._target_flags(target)

        # Fast impossibility rules.
        if tgt_bucket == "numeric":
            if sum(1 for b in src_buckets if b in {"numeric", "date"}) == 0:
                return False
            if flags.get("currency_like", False) and len(refs) >= 2:
                # Use semantic concept matching instead of hardcoded tokens
                src_concepts = [
                    self.semantic.concept_scores(r.column) for r in refs
                ]
                has_monetary = any(sc.get("monetary", 0) > 0.38 for sc in src_concepts)
                has_rate = any(sc.get("exchange_rate", 0) > 0.35 for sc in src_concepts)
                has_currency = any(sc.get("currency", 0) > 0.35 for sc in src_concepts)
                if not (has_monetary and (has_rate or has_currency)):
                    return False
        if tgt_bucket == "date":
            if not any(b in {"date", "string"} for b in src_buckets):
                return False
        if tgt_bucket == "boolean":
            if all(getattr(c.stats, "valid_count", 0) == 0 for c in src_cols):
                return False
        return True

    # -----------------------------------------------------
    # Stage-4 feasibility filter
    # -----------------------------------------------------

    def _infer_feasible_families(self, target: TargetSpec, cand: CandidateSet) -> List[str]:
        refs = cand.refs
        src_cols = [self._ref_to_col[(r.table, r.column)] for r in refs]
        src_types = [_type_bucket(c.col_type) for c in src_cols]
        tgt_type = _type_bucket(target.col_type)
        has_join = len(cand.join_path) > 0
        flags = self._target_flags(target)

        fam: Set[str] = set()

        # --- Type-based feasibility (core logic, still needed) ---
        if len(refs) == 1:
            st = src_types[0]
            if tgt_type == "string":
                if st == "string":
                    fam.update(["identity", "substring", "split", "regex_extract",
                                "lower", "upper", "trim", "format"])
                elif st in {"numeric", "date"}:
                    fam.update(["cast_to_string", "format"])
                if st == "date":
                    fam.update(["format_date"])
            elif tgt_type == "numeric":
                if st == "numeric":
                    fam.update(["identity", "round", "scale", "bucket"])
                elif st == "string":
                    fam.update(["parse_numeric"])
                elif st == "date":
                    fam.update(["date_part", "date_diff"])
            elif tgt_type == "date":
                if st == "date":
                    fam.update(["identity", "date_shift", "date_trunc", "date_part"])
                elif st == "string":
                    fam.update(["parse_date"])
                elif st == "numeric":
                    fam.update(["epoch_to_date"])
            elif tgt_type == "boolean":
                fam.update(["conditional", "regex_match", "in_list"])
                if st == "date":
                    fam.update(["date_range_check"])
                if st == "numeric":
                    fam.update(["threshold_flag"])
            else:
                fam.update(["identity"])
        else:
            if tgt_type == "string" and all(t in {"string", "numeric", "date"} for t in src_types):
                fam.update(["concat", "format"])
            if tgt_type == "numeric":
                if sum(1 for t in src_types if t == "numeric") >= 2:
                    fam.update(["arithmetic", "ratio"])
                if any(t == "date" for t in src_types):
                    fam.update(["date_diff", "date_part"])
            if tgt_type == "date" and any(t in {"date", "string"} for t in src_types):
                fam.update(["date_construct"])
            fam.update(["conditional"])

        if has_join:
            fam.update(["lookup_join"])

        # --- Semantic-based feasibility expansion (via SemanticEngine) ---
        # This replaces all hardcoded token checks (EMAIL_TOKENS, AMOUNT_TOKENS
        # etc.) with embedding-based concept matching.
        source_names = [r.column for r in refs]
        src_type_strs = [
            getattr(c.col_type, "base_type", str(c.col_type)) for c in src_cols
        ]
        tgt_type_str = getattr(target.col_type, "base_type", str(target.col_type))
        semantic_extra = self.semantic.infer_semantic_families(
            target.name, source_names,
            target_type=tgt_type_str, source_types=src_type_strs,
        )
        fam.update(semantic_extra)

        # Semantic role flags for additional patterns
        if flags.get("username_like", False):
            fam.update(["email_username_extract", "split", "regex_extract"])
        if flags.get("currency_like", False) and len(refs) >= 2:
            fam.add("currency_convert")

        # Remove families impossible by very sparse data
        valid_counts = [getattr(c.stats, "valid_count", 0) for c in src_cols]
        if min(valid_counts or [0]) == 0:
            fam.discard("identity")
            fam.discard("concat")
            fam.discard("arithmetic")

        return sorted(fam)

    def _target_date_part_flags(self, target: TargetSpec) -> Dict[str, bool]:
        """Detect if target name implies a date-part extraction (semantic)."""
        flags = self._target_flags(target)
        is_dp = flags.get("date_part_like", False) or flags.get("duration_like", False)

        # Also check lexically for common date-part tokens as a safety net
        toks = set(_name_tokens(target.name))
        desc_toks = set(_name_tokens(target.description or ""))
        all_toks = toks | desc_toks
        _DP = {"year", "month", "day", "hour", "minute", "quarter", "week",
               "dow", "doy", "epoch", "weekday", "yr", "mo", "dy", "hr"}
        lexical_dp = bool(all_toks & _DP)

        return {
            "is_date_part": is_dp or lexical_dp,
            "date_part_tokens": all_toks & _DP,
        }

    # -----------------------------------------------------
    # Stage-5 execution-guided quick scoring
    # -----------------------------------------------------

    def _quick_family_scores(
        self,
        target: TargetSpec,
        cand: CandidateSet,
        families: Sequence[str],
        sample_limit: int = 300,
    ) -> Dict[str, float]:
        if not families:
            return {}

        target_values = self._target_samples(target, sample_limit)
        if not target_values and "lookup_join" not in families:
            return {}

        rows = self._extract_candidate_rows(cand, sample_limit)
        scores: Dict[str, float] = {}

        for fam in families:
            sem_fit = self._semantic_role_score(target, cand.refs, family_hint=fam)
            if fam == "lookup_join":
                # lightweight evidence based on path confidence
                # lookup-only evidence is weaker than direct value checks.
                lookup_score = 0.22 + 0.38 * _path_confidence(cand.join_path) + 0.20 * sem_fit
                scores[fam] = min(0.82, lookup_score)
                continue

            preds = self._apply_template_family(fam, rows, cand=cand)
            if not preds:
                # Allow semantic prior to keep logically plausible families alive.
                if sem_fit > 0.75:
                    scores[fam] = round(0.22 + 0.25 * sem_fit, 4)
                continue
            score = self._match_score(preds, target_values)
            score = (0.86 * score) + (0.14 * sem_fit)
            scores[fam] = round(float(score), 4)

        return scores

    def _extract_candidate_rows(self, cand: CandidateSet, limit: int) -> List[Tuple[Any, ...]]:
        # 1-table direct extraction path
        tables = {r.table for r in cand.refs}
        if len(tables) == 1:
            table_name = next(iter(tables))
            table = self.source_tables.get(table_name)
            if table is None:
                return []

            col_names = [r.column for r in cand.refs]
            rows: List[Tuple[Any, ...]] = []

            if table.rows:
                for row in table.rows[:limit]:
                    vals = tuple(_normalize_value(row.get(c)) for c in col_names)
                    if any(v is None for v in vals):
                        continue
                    rows.append(vals)
                return rows

            # fallback from aligned stats.sample
            samples = [getattr(table.columns[c].stats, "sample", []) for c in col_names]
            if not samples or any(len(s) == 0 for s in samples):
                return []
            n = min(min(len(s) for s in samples), limit)
            for i in range(n):
                vals = tuple(_normalize_value(samples[j][i]) for j in range(len(col_names)))
                if any(v is None for v in vals):
                    continue
                rows.append(vals)
            return rows

        # Multi-table extraction via join_path edges (best effort).
        if not cand.join_path:
            return []
        return self._extract_joined_rows(cand, limit)

    def _extract_joined_rows(self, cand: CandidateSet, limit: int) -> List[Tuple[Any, ...]]:
        needed_tables = {r.table for r in cand.refs}
        if any(self.source_tables.get(t) is None for t in needed_tables):
            return []
        if any(not self.source_tables[t].rows for t in needed_tables):
            return []

        edges = list(cand.join_path)
        if not edges:
            return []

        anchor = cand.refs[0].table
        anchor_rows = list(self.source_tables[anchor].rows)
        if not anchor_rows:
            return []

        # Pre-build edge indexes for efficient matching.
        edge_index_lr: Dict[int, Dict[Tuple[Any, ...], List[Dict[str, Any]]]] = {}
        edge_index_rl: Dict[int, Dict[Tuple[Any, ...], List[Dict[str, Any]]]] = {}
        for i, e in enumerate(edges):
            left_rows = self.source_tables.get(e.left_table).rows if self.source_tables.get(e.left_table) else []
            right_rows = self.source_tables.get(e.right_table).rows if self.source_tables.get(e.right_table) else []
            idx_lr: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
            idx_rl: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)

            for rr in right_rows:
                key = tuple(_normalize_value(rr.get(c)) for c in e.right_cols)
                if any(v is None for v in key):
                    continue
                idx_lr[key].append(rr)
            for lr in left_rows:
                key = tuple(_normalize_value(lr.get(c)) for c in e.left_cols)
                if any(v is None for v in key):
                    continue
                idx_rl[key].append(lr)
            edge_index_lr[i] = idx_lr
            edge_index_rl[i] = idx_rl

        # Partial joined records: dict(table_name -> row_dict)
        partials: List[Dict[str, Dict[str, Any]]] = [{anchor: r} for r in anchor_rows[: max(limit * 4, 100)]]

        for _ in range(len(edges) + 2):
            progressed = False
            next_partials: List[Dict[str, Dict[str, Any]]] = []
            for part in partials:
                expanded_any = False
                for i, e in enumerate(edges):
                    ltab, rtab = e.left_table, e.right_table
                    if ltab in part and rtab not in part:
                        key = tuple(_normalize_value(part[ltab].get(c)) for c in e.left_cols)
                        if any(v is None for v in key):
                            continue
                        matches = edge_index_lr[i].get(key, [])
                        if matches:
                            expanded_any = True
                            progressed = True
                            for m in matches:
                                npart = dict(part)
                                npart[rtab] = m
                                next_partials.append(npart)
                    elif rtab in part and ltab not in part:
                        key = tuple(_normalize_value(part[rtab].get(c)) for c in e.right_cols)
                        if any(v is None for v in key):
                            continue
                        matches = edge_index_rl[i].get(key, [])
                        if matches:
                            expanded_any = True
                            progressed = True
                            for m in matches:
                                npart = dict(part)
                                npart[ltab] = m
                                next_partials.append(npart)
                if not expanded_any:
                    next_partials.append(part)

            # de-duplicate partial states to control combinatorial growth
            dedup = {}
            for p in next_partials:
                key = tuple(sorted((t, id(r)) for t, r in p.items()))
                dedup[key] = p
            partials = list(dedup.values())[: max(limit * 8, 200)]
            if not progressed:
                break

        # Materialize candidate tuples
        tuples: List[Tuple[Any, ...]] = []
        for p in partials:
            if not needed_tables.issubset(set(p.keys())):
                continue
            vals = []
            ok = True
            for ref in cand.refs:
                v = _normalize_value(p[ref.table].get(ref.column))
                if v is None:
                    ok = False
                    break
                vals.append(v)
            if ok:
                tuples.append(tuple(vals))
            if len(tuples) >= limit:
                break

        return tuples

    def _apply_template_family(
        self,
        family: str,
        rows: Sequence[Tuple[Any, ...]],
        cand: Optional[CandidateSet] = None,
    ) -> List[Any]:
        if not rows:
            return []

        arity = len(rows[0])
        out: List[Any] = []

        if family in {"identity", "cast_to_string", "parse_numeric", "parse_date"} and arity == 1:
            for r in rows:
                v = r[0]
                if family == "cast_to_string":
                    out.append(None if v is None else str(v))
                elif family == "parse_numeric":
                    out.append(_safe_float(v))
                elif family == "parse_date":
                    out.append(_simple_date_parse(v))
                else:
                    out.append(v)
            return [x for x in out if x is not None]

        if family == "lower" and arity == 1:
            return [str(r[0]).lower() for r in rows if r[0] is not None]
        if family == "upper" and arity == 1:
            return [str(r[0]).upper() for r in rows if r[0] is not None]
        if family == "trim" and arity == 1:
            return [str(r[0]).strip() for r in rows if r[0] is not None]
        if family == "substring" and arity == 1:
            # best-effort generic: last 4 chars
            return [str(r[0])[-4:] for r in rows if r[0] is not None and len(str(r[0])) >= 4]
        if family == "email_username_extract" and arity == 1:
            out = []
            for r in rows:
                s = str(r[0]).strip()
                if "@" in s:
                    out.append(s.split("@", 1)[0])
            return out
        if family == "split" and arity == 1:
            out = []
            for r in rows:
                s = str(r[0])
                if "@" in s:
                    left = s.split("@", 1)[0]
                    if left:
                        out.append(left)
                        continue
                toks = re.split(r"[_\-\s/]+", s)
                if toks and toks[0]:
                    out.append(toks[0].strip("."))
            return out
        if family == "regex_extract" and arity == 1:
            out = []
            for r in rows:
                s = str(r[0])
                m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", s)
                if m:
                    out.append(m.group(0))
                    continue
                m2 = re.search(r"\.([A-Za-z0-9]{1,8})$", s.strip())
                if m2:
                    out.append(m2.group(1).lower())
            return out
        if family == "format" and arity >= 2:
            return ["-".join(str(x) for x in r if x is not None) for r in rows]
        if family == "concat" and arity >= 2:
            return [" ".join(str(x) for x in r if x is not None) for r in rows]
        if family == "round" and arity == 1:
            out = []
            for r in rows:
                f = _safe_float(r[0])
                if f is not None:
                    out.append(round(f, 2))
            return out
        if family == "scale" and arity == 1:
            out = []
            for r in rows:
                f = _safe_float(r[0])
                if f is not None:
                    out.append(f / 100.0)
            return out
        if family == "arithmetic" and arity >= 2:
            out = []
            for r in rows:
                vals = [_safe_float(x) for x in r]
                if any(v is None for v in vals):
                    continue
                out.append(float(sum(vals)))
            return out
        if family == "ratio" and arity >= 2:
            out = []
            for r in rows:
                a = _safe_float(r[0])
                b = _safe_float(r[1])
                if a is None or b is None or b == 0:
                    continue
                out.append(a / b)
            return out
        if family == "currency_convert" and arity >= 2:
            idx_amount = 0
            idx_rate = 1
            if cand is not None:
                # pick amount/rate columns by semantic concept matching
                amount_candidates = []
                rate_candidates = []
                for i, ref in enumerate(cand.refs):
                    sc = self.semantic.concept_scores(ref.column)
                    if sc.get("monetary", 0) > 0.38:
                        amount_candidates.append(i)
                    if sc.get("exchange_rate", 0) > 0.35 or sc.get("currency", 0) > 0.35:
                        rate_candidates.append(i)
                # Fallback to legacy token matching if semantic engine has no model
                if not amount_candidates and not rate_candidates:
                    ref_tokens = [set(_name_tokens(ref.column)) for ref in cand.refs]
                    amount_candidates = [i for i, t in enumerate(ref_tokens) if t & AMOUNT_TOKENS]
                    rate_candidates = [i for i, t in enumerate(ref_tokens) if t & (RATE_TOKENS | CURRENCY_TOKENS)]
                if amount_candidates:
                    idx_amount = amount_candidates[0]
                if rate_candidates:
                    # avoid selecting same column as amount
                    idx_rate = next((i for i in rate_candidates if i != idx_amount), rate_candidates[0])
            out = []
            for r in rows:
                if idx_amount >= len(r) or idx_rate >= len(r):
                    continue
                a = _safe_float(r[idx_amount])
                rate = _safe_float(r[idx_rate])
                if a is None or rate is None:
                    continue
                out.append(a * rate)
            return out
        if family == "date_part" and arity == 1:
            out = []
            for r in rows:
                d = _simple_date_parse(r[0])
                if d is None:
                    continue
                # Return multiple date parts as integers for robust matching.
                # The _match_score will pick the best alignment.
                try:
                    parts = d.split("-")
                    year = int(parts[0])
                    month = int(parts[1]) if len(parts) > 1 else 1
                    day = int(parts[2]) if len(parts) > 2 else 1
                    out.append(year)
                except (ValueError, IndexError):
                    out.append(d[:4])
            return out
        if family == "date_diff" and arity >= 2:
            # difference between two date columns in days
            out = []
            date_indices = [i for i, r_tuple in enumerate(rows[0] if rows else ())
                           for _ in [_simple_date_parse(r_tuple)] if _ is not None]
            # fallback: try first two columns
            for r in rows:
                d1 = _simple_date_parse(r[0])
                d2 = _simple_date_parse(r[1]) if len(r) > 1 else None
                if d1 and d2:
                    try:
                        from datetime import datetime as _dt
                        dt1 = _dt.strptime(d1, "%Y-%m-%d")
                        dt2 = _dt.strptime(d2, "%Y-%m-%d")
                        out.append(abs((dt2 - dt1).days))
                    except Exception:
                        pass
            return out
        if family == "conditional":
            out = []
            if arity == 1:
                vals = [_safe_float(r[0]) for r in rows]
                num_vals = [v for v in vals if v is not None]
                if len(num_vals) >= max(3, int(0.6 * len(rows))):
                    thr = float(np.median(np.array(num_vals, dtype=float)))
                    for r in rows:
                        v = _safe_float(r[0])
                        if v is None:
                            continue
                        out.append(v >= thr)
                    return out
                for r in rows:
                    s = str(r[0]).strip().lower()
                    if not s:
                        continue
                    out.append(s in {"active", "open", "yes", "true", "1"})
                return out
            if arity >= 2:
                for r in rows:
                    a = _safe_float(r[0])
                    b = _safe_float(r[1])
                    if a is not None and b is not None:
                        out.append(a > b)
                return out

        return []

    def _target_samples(self, target: TargetSpec, limit: int) -> List[Any]:
        if target.sample_values:
            vals = [_normalize_value(v) for v in target.sample_values[:limit]]
            return [v for v in vals if v is not None]
        vals = getattr(target.stats, "sample", [])
        vals = [_normalize_value(v) for v in vals[:limit]]
        return [v for v in vals if v is not None]

    def _match_score(self, preds: Sequence[Any], target_values: Sequence[Any]) -> float:
        if not preds or not target_values:
            return 0.0

        preds = [x for x in preds if x is not None]
        target_values = [x for x in target_values if x is not None]
        if not preds or not target_values:
            return 0.0

        # Try numeric MAE score when numeric-like.
        p_num = [_safe_float(x) for x in preds]
        t_num = [_safe_float(x) for x in target_values]
        if sum(x is not None for x in p_num) >= len(p_num) * 0.8 and sum(x is not None for x in t_num) >= len(t_num) * 0.8:
            p = np.array([x for x in p_num if x is not None], dtype=float)
            t = np.array([x for x in t_num if x is not None], dtype=float)
            n = min(len(p), len(t))
            if n == 0:
                return 0.0
            p = np.sort(p[:n])
            t = np.sort(t[:n])
            mae = float(np.mean(np.abs(p - t)))
            span = float(max(np.max(t) - np.min(t), np.max(p) - np.min(p), 1e-9))
            mae_sim = max(0.0, 1.0 - (mae / span))
            exact = _bag_overlap_ratio([round(x, 6) for x in p], [round(x, 6) for x in t])
            return 0.75 * mae_sim + 0.25 * exact

        # string-like scoring
        p_txt = [str(x).strip().lower() for x in preds]
        t_txt = [str(x).strip().lower() for x in target_values]
        exact = _bag_overlap_ratio(p_txt, t_txt)
        token = _token_overlap_ratio(p_txt, t_txt)
        return 0.6 * exact + 0.4 * token

    # -----------------------------------------------------
    # Stage-6 scoring helpers
    # -----------------------------------------------------

    def _family_prior(self, family: str) -> float:
        priors = {
            "identity": 0.78,
            "lookup_join": 0.60,
            "currency_convert": 0.78,
            "email_username_extract": 0.76,
            "concat": 0.74,
            "format": 0.70,
            "format_date": 0.68,
            "substring": 0.64,
            "split": 0.64,
            "regex_extract": 0.62,
            "lower": 0.60,
            "upper": 0.60,
            "trim": 0.60,
            "arithmetic": 0.70,
            "ratio": 0.66,
            "round": 0.62,
            "scale": 0.60,
            "bucket": 0.58,
            "parse_date": 0.66,
            "parse_numeric": 0.60,
            "date_part": 0.72,
            "date_shift": 0.62,
            "date_trunc": 0.62,
            "date_diff": 0.68,
            "date_construct": 0.60,
            "date_range_check": 0.58,
            "epoch_to_date": 0.56,
            "conditional": 0.58,
            "threshold_flag": 0.60,
            "regex_match": 0.56,
            "in_list": 0.56,
            "cast_to_string": 0.55,
        }
        return priors.get(family, 0.52)

    def _logical_mismatch_penalty(self, target: TargetSpec, cand: CandidateSet, family: str) -> float:
        """
        Penalize logically implausible family/column semantics.
        Uses SemanticEngine concept matching instead of hardcoded token sets.
        """
        flags = self._target_flags(target)
        refs = cand.refs

        # Semantic concept presence on source columns (replaces hardcoded token checks)
        src_concepts = [
            self.semantic.concept_scores(
                r.column,
                getattr(self._ref_to_col.get((r.table, r.column), None), "col_type", "")
                if self._ref_to_col.get((r.table, r.column)) else "",
            )
            for r in refs
        ]
        has_monetary_src = any(sc.get("monetary", 0) > 0.40 for sc in src_concepts) if src_concepts else False
        has_rate_src = any(sc.get("exchange_rate", 0) > 0.35 for sc in src_concepts) if src_concepts else False
        has_email_src = any(sc.get("email", 0) > 0.40 for sc in src_concepts) if src_concepts else False
        has_name_parts_src = sum(1 for sc in src_concepts if sc.get("name_component", 0) > 0.40)

        pen = 0.0
        if flags.get("currency_like", False):
            if family == "identity" and not has_rate_src:
                pen += 0.12
            if family in {"ratio", "arithmetic"} and not has_rate_src:
                pen += 0.08
            if family == "currency_convert" and not (has_monetary_src and has_rate_src):
                pen += 0.15

        if flags.get("username_like", False):
            if family == "identity" and has_email_src:
                pen += 0.08
            if family in {"split", "email_username_extract"} and not has_email_src:
                pen += 0.12

        if flags.get("full_name_like", False):
            if family in {"concat", "format"} and has_name_parts_src < 2:
                pen += 0.09

        return float(max(0.0, min(0.25, pen)))

    # -----------------------------------------------------
    # Similarity sub-functions
    # -----------------------------------------------------

    def _name_similarity(self, target_name: str, source_name: str,
                         target_type: str = "", source_type: str = "",
                         target_table: str = "", source_table: str = "") -> float:
        if self.name_similarity_fn is not None:
            try:
                out = self.name_similarity_fn(target_name, source_name)
                if isinstance(out, dict):
                    return float(out.get("similarity", 0.0))
                return float(out)
            except Exception:
                pass
        # Embedding-based semantic + lexical similarity (via SemanticEngine)
        return self.semantic.name_similarity(
            target_name, source_name,
            target_type=target_type, source_type=source_type,
            target_table=target_table, source_table=source_table,
        )

    def _target_flags(self, target: TargetSpec) -> Dict[str, bool]:
        # Delegate to SemanticEngine (embedding-based concept matching)
        tgt_type = getattr(target.col_type, "base_type", str(target.col_type))
        return self.semantic.detect_target_role(
            target.name, target_type=tgt_type,
            description=target.description or "",
        )

    def _col_tokens(self, ref: ColumnRef) -> Set[str]:
        return set(_name_tokens(ref.column))

    def _refs_token_presence(self, refs: Sequence[ColumnRef], token_group: Set[str]) -> bool:
        for r in refs:
            if self._col_tokens(r) & token_group:
                return True
        return False

    def _semantic_role_score(
        self,
        target: TargetSpec,
        refs: Sequence[ColumnRef],
        family_hint: Optional[str] = None,
    ) -> float:
        """
        Semantic compatibility via embedding-based concept matching.
        Delegates to SemanticEngine.role_compatibility() which replaces
        all hardcoded token-group checks with learned concept vectors.
        """
        if not refs:
            return 0.0
        source_names = [r.column for r in refs]
        source_types = []
        for r in refs:
            col = self._ref_to_col.get((r.table, r.column))
            if col:
                source_types.append(
                    getattr(col.col_type, "base_type", str(col.col_type))
                )
            else:
                source_types.append("")
        tgt_type = getattr(target.col_type, "base_type", str(target.col_type))
        return self.semantic.role_compatibility(
            target.name, source_names,
            target_type=tgt_type, source_types=source_types,
            family_hint=family_hint,
        )

    def _value_similarity(self, target_stats: Any, source_stats: Any) -> float:
        try:
            return float(self.value_sim_class(target_stats, source_stats).compute_score().get("final", 0.0))
        except Exception:
            return 0.0

    def _constraint_similarity(self, target_cons: ColumnConstraints, source_cons: ColumnConstraints) -> float:
        try:
            return float(self.constraint_engine.score(target_cons, source_cons).get("final", 0.0))
        except Exception:
            return 0.0

    def _null_distinct_compatibility(self, s1: Any, s2: Any) -> float:
        n1 = float(getattr(s1, "null_frac", 1.0))
        n2 = float(getattr(s2, "null_frac", 1.0))
        null_sim = max(0.0, 1.0 - abs(n1 - n2))

        d1 = float(getattr(s1, "n_distinct", 0)) / max(1, float(getattr(s1, "valid_count", 1)))
        d2 = float(getattr(s2, "n_distinct", 0)) / max(1, float(getattr(s2, "valid_count", 1)))
        distinct_sim = max(0.0, 1.0 - abs(d1 - d2))
        return 0.55 * null_sim + 0.45 * distinct_sim

    # -----------------------------------------------------
    # Text utilities
    # -----------------------------------------------------

    def _column_text(self, ref: ColumnRef, col: Column) -> str:
        t = getattr(col.col_type, "base_type", str(col.col_type))
        # Richer text for ANN indexing: include tokenized name + semantic description
        tokens = _tokenize_name_to_str(ref.column)
        sem_desc = self.semantic.describe_column(ref.column, t) if self.semantic.has_model else ""
        base = f"{ref.table}.{ref.column}<{t}> {tokens}"
        if sem_desc and sem_desc != tokens:
            base += f" [{sem_desc}]"
        return base

    def _target_text(self, target: TargetSpec) -> str:
        t = getattr(target.col_type, "base_type", str(target.col_type))
        desc = f" {target.description}" if target.description else ""
        tokens = _tokenize_name_to_str(target.name)
        sem_desc = self.semantic.describe_column(target.name, t) if self.semantic.has_model else ""
        base = f"{target.table}.{target.name}<{t}> {tokens}{desc}"
        if sem_desc and sem_desc != tokens:
            base += f" [{sem_desc}]"
        return base

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

    def _serialize_candidate(self, c: CandidateSet) -> Dict[str, Any]:
        return {
            "candidate_columns": [f"{r.table}.{r.column}" for r in c.refs],
            "join_path": [
                {
                    "from": e.left_table,
                    "to": e.right_table,
                    "left_cols": list(e.left_cols),
                    "right_cols": list(e.right_cols),
                    "confidence": round(float(e.confidence), 4),
                }
                for e in c.join_path
            ],
            "best_transform_family": c.best_family,
            "feasible_families": c.feasible_families,
            "component_scores": {k: round(float(v), 4) for k, v in c.component_scores.items()},
            "family_scores": {k: round(float(v), 4) for k, v in c.family_scores.items()},
            "base_score": round(float(c.base_score), 4),
            "quick_score": round(float(c.quick_score), 4),
            "final_score": round(float(c.final_score), 4),
            "confidence": round(float(c.confidence), 4),
        }

    def _dedup_candidates(self, rows: Sequence[CandidateSet]) -> List[CandidateSet]:
        best: Dict[Tuple[Tuple[str, str], ...], CandidateSet] = {}
        for c in rows:
            key = tuple(sorted((r.table, r.column) for r in c.refs))
            if key not in best or best[key].base_score < c.base_score:
                best[key] = c
        return list(best.values())


# =========================================================
# Helper functions
# =========================================================


def _type_bucket(col_type: Any) -> str:
    t = getattr(col_type, "base_type", str(col_type)).lower()
    if t in {"string", "varchar", "char", "text", "clob"}:
        return "string"
    if t in {"int", "integer", "bigint", "smallint", "float", "double", "decimal", "numeric", "number"}:
        return "numeric"
    if t in {"date", "datetime", "timestamp", "time"}:
        return "date"
    if t in {"bool", "boolean"}:
        return "boolean"
    return "other"


def _compatible_buckets(target_col_type: Any) -> List[str]:
    """Return all type buckets that could plausibly feed this target type.

    Key insight: cross-type transformations are common:
      - date -> int   (extract_year, date_diff)
      - string -> date (parse_date)
      - date -> string (format_date)
      - numeric -> boolean (threshold flag)
      - string -> numeric (parse_numeric)
    So we must search broadly to avoid missing valid candidates.
    """
    b = _type_bucket(target_col_type)
    if b == "string":
        return ["string", "numeric", "date", "boolean", "other"]
    if b == "numeric":
        return ["numeric", "string", "date"]  # date for date_part / extract_year
    if b == "date":
        return ["date", "string", "numeric"]  # string for parse_date, numeric for epoch
    if b == "boolean":
        return ["boolean", "string", "numeric", "date"]  # broad: any type can produce flag
    return [b, "string", "numeric", "date"]


def _name_tokens(s: str) -> List[str]:
    s = s or ""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    s = re.sub(r"[^A-Za-z0-9]+", " ", s.lower())
    toks = [t for t in s.split() if t]
    return toks


def _token_jaccard(a: str, b: str) -> float:
    ta = set(_name_tokens(a))
    tb = set(_name_tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# Semantic equivalence groups: tokens within the same group are considered related.
_SEMANTIC_GROUPS: List[Set[str]] = [
    # date-part tokens that imply source is a date column
    {"year", "yr", "month", "mo", "day", "dy", "hour", "hr", "minute", "min",
     "quarter", "qtr", "week", "wk", "weekday", "dow", "doy", "epoch"},
    # date-like source tokens
    {"date", "datetime", "timestamp", "time", "dt", "ts", "created", "updated",
     "modified", "signup", "register", "event", "order", "birth", "start", "end",
     "expire", "expiry"},
    # name tokens
    {"name", "first", "last", "full", "middle", "given", "family", "surname"},
    # identifier tokens
    {"id", "key", "pk", "fk", "code", "identifier", "ref", "num", "number"},
    # amount / money tokens
    {"amount", "total", "sum", "price", "cost", "fee", "charge", "revenue",
     "salary", "wage", "balance", "payment", "gross", "net"},
    # email / contact tokens
    {"email", "mail", "username", "user", "login", "contact", "phone", "tel"},
    # location tokens
    {"country", "state", "city", "region", "province", "zip", "postal", "address"},
    # status tokens
    {"status", "state", "active", "inactive", "flag", "is", "has", "enabled"},
    # file / document tokens
    {"file", "path", "document", "doc", "attachment", "url", "uri", "link"},
    # text / description tokens
    {"text", "description", "desc", "comment", "note", "remark", "message"},
]

# Build a fast lookup: token -> group index
_TOKEN_TO_GROUPS: Dict[str, List[int]] = defaultdict(list)
for _gi, _grp in enumerate(_SEMANTIC_GROUPS):
    for _tok in _grp:
        _TOKEN_TO_GROUPS[_tok].append(_gi)

# Cross-type semantic pairs: (target_token_group, source_token_group, bonus)
# These capture the insight that e.g. "year" in target strongly implies a date source
_DATE_PART_TOKENS = {"year", "yr", "month", "mo", "day", "dy", "hour", "hr",
                     "minute", "min", "quarter", "qtr", "week", "wk", "weekday",
                     "dow", "doy", "epoch"}
_DATE_SOURCE_TOKENS = {"date", "datetime", "timestamp", "time", "dt", "ts",
                       "created", "updated", "modified", "signup", "register",
                       "event", "order", "birth", "start", "end", "expire", "expiry"}


def _enhanced_name_similarity(target_name: str, source_name: str) -> float:
    """
    Enhanced name similarity that understands:
    1. Token Jaccard (lexical overlap)
    2. Prefix/suffix overlap (e.g., 'signup_year' shares 'signup' with 'signup_date')
    3. Semantic group relationships (e.g., 'year' implies date-source compatibility)
    4. Substring containment
    """
    ta = set(_name_tokens(target_name))
    tb = set(_name_tokens(source_name))
    if not ta or not tb:
        return 0.0

    # 1. Base token Jaccard
    jaccard = len(ta & tb) / len(ta | tb)

    # 2. Directional containment: what fraction of target tokens appear in source
    # (important when target has qualifying tokens like "signup" that match source)
    containment = len(ta & tb) / len(ta) if ta else 0.0

    # 3. Semantic group bonus: tokens from related groups
    semantic_bonus = 0.0
    for t_tok in ta - tb:
        for s_tok in tb - ta:
            # Check if they're in the same semantic group
            t_groups = _TOKEN_TO_GROUPS.get(t_tok, [])
            s_groups = _TOKEN_TO_GROUPS.get(s_tok, [])
            if t_groups and s_groups and set(t_groups) & set(s_groups):
                semantic_bonus = max(semantic_bonus, 0.25)

    # 4. Cross-type semantic bonus: date-part target tokens + date source tokens
    tgt_has_date_part = bool(ta & _DATE_PART_TOKENS)
    src_has_date_source = bool(tb & _DATE_SOURCE_TOKENS)
    src_has_date_part = bool(tb & _DATE_PART_TOKENS)
    tgt_has_date_source = bool(ta & _DATE_SOURCE_TOKENS)

    cross_type_bonus = 0.0
    if tgt_has_date_part and src_has_date_source:
        # "signup_year" target + "signup_date" source -> strong match
        cross_type_bonus = 0.35
        # Extra bonus if they share a qualifying prefix (e.g., "signup")
        shared_non_type = ta & tb - _DATE_PART_TOKENS - _DATE_SOURCE_TOKENS
        if shared_non_type:
            cross_type_bonus = 0.50
    elif tgt_has_date_source and src_has_date_source:
        # "created_date" target + "created_at_text" source -> strong
        shared = ta & tb
        if shared - _DATE_SOURCE_TOKENS:
            cross_type_bonus = 0.15

    # 5. Edit distance bonus for very similar names
    edit_bonus = 0.0
    tn = target_name.lower().replace("_", "").replace("-", "")
    sn = source_name.lower().replace("_", "").replace("-", "")
    if tn in sn or sn in tn:
        edit_bonus = 0.15

    combined = max(
        jaccard,
        0.40 * jaccard + 0.25 * containment + 0.20 * semantic_bonus + 0.15 * cross_type_bonus,
        0.35 * containment + 0.30 * cross_type_bonus + 0.20 * semantic_bonus + 0.15 * edit_bonus,
    )
    return min(1.0, combined)


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        try:
            v = float(x)
            return v if math.isfinite(v) else None
        except Exception:
            return None
    if isinstance(x, str):
        s = x.strip().replace(",", "")
        if s.endswith("%") and len(s) > 1:
            s = s[:-1]
        try:
            v = float(s)
            return v if math.isfinite(v) else None
        except Exception:
            return None
    return None


def _simple_date_parse(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    m = re.match(r"^(\d{4})[-/](\d{2})[-/](\d{2})", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.match(r"^(\d{2})[-/](\d{2})[-/](\d{4})", s)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    return None


def _normalize_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    return v


def _bag_overlap_ratio(a_vals: Sequence[Any], b_vals: Sequence[Any]) -> float:
    if not a_vals or not b_vals:
        return 0.0
    ca = Counter(a_vals)
    cb = Counter(b_vals)
    inter = set(ca.keys()) & set(cb.keys())
    shared = sum(min(ca[k], cb[k]) for k in inter)
    denom = max(1, min(sum(ca.values()), sum(cb.values())))
    return shared / denom


def _token_overlap_ratio(a_vals: Sequence[str], b_vals: Sequence[str]) -> float:
    if not a_vals or not b_vals:
        return 0.0
    ta = Counter()
    tb = Counter()
    for s in a_vals:
        for t in re.split(r"[^a-z0-9]+", s.lower()):
            if t:
                ta[t] += 1
    for s in b_vals:
        for t in re.split(r"[^a-z0-9]+", s.lower()):
            if t:
                tb[t] += 1
    inter = set(ta.keys()) & set(tb.keys())
    shared = sum(min(ta[k], tb[k]) for k in inter)
    denom = max(1, min(sum(ta.values()), sum(tb.values())))
    return shared / denom


def _path_confidence(path_edges: Sequence[JoinEdge]) -> float:
    if not path_edges:
        return 1.0
    p = 1.0
    for e in path_edges:
        p *= max(1e-6, float(getattr(e, "confidence", 0.0)))
    return max(0.0, min(1.0, p ** (1 / max(1, len(path_edges)))))


def _dedup_join_edges(edges: Sequence[JoinEdge]) -> List[JoinEdge]:
    best: Dict[Tuple[Any, ...], JoinEdge] = {}
    for e in edges:
        if e.left_table <= e.right_table:
            pairs = tuple(sorted(zip(e.left_cols, e.right_cols), key=lambda x: (x[0], x[1])))
            key = (e.left_table, e.right_table, pairs)
        else:
            pairs = tuple(sorted(zip(e.right_cols, e.left_cols), key=lambda x: (x[0], x[1])))
            key = (e.right_table, e.left_table, pairs)
        if key not in best or best[key].confidence < e.confidence:
            best[key] = e
    vals = list(best.values())
    vals.sort(key=lambda x: (x.left_table, x.right_table, -x.confidence))
    return vals


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# =========================================================
# Demo
# =========================================================


def _demo_make_col(
    name: str,
    rows: Sequence[Dict[str, Any]],
    typ: str,
    constraints: Optional[ColumnConstraints] = None,
) -> Column:
    values = [r.get(name) for r in rows]
    return Column(
        name=name,
        col_type=ColumnType(typ),
        constraints=constraints or ColumnConstraints(),
        stats=ColumnStats(values),
    )


def _demo_print_result(title: str, result: Dict[str, Any], top_n: int = 5) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    print("Target:", result["target"])
    print("Abstain:", result["abstain"])
    print("Top candidates:")
    for i, row in enumerate(result["top_candidates"][:top_n], 1):
        print(
            f"{i:02d}. cols={row['candidate_columns']} "
            f"| family={row['best_transform_family']} "
            f"| conf={row['confidence']} "
            f"| base={row['base_score']} "
            f"| quick={row['quick_score']}"
        )
    print("Debug:", result["debug"])


def run_demo_candidate_generation() -> None:
    """
    End-to-end local test data for candidate generation.

    Run:
      python3 candidate_generation_algorithm.py
    """
    # ---------------------------
    # Source table rows
    # ---------------------------
    customers_rows = [
        {
            "customer_id": 1,
            "first_name": "Alice",
            "last_name": "Smith",
            "email": "alice.smith@acme.com",
            "country_code": "US",
            "city": "Boston",
            "state_code": "MA",
            "postal_code": "02108",
            "phone_number": "6175551234",
            "signup_date": "2021-01-15",
            "risk_score": 82.3,
            "status_code": "ACTIVE",
        },
        {
            "customer_id": 2,
            "first_name": "Bob",
            "last_name": "Jones",
            "email": "bob.jones@acme.com",
            "country_code": "IN",
            "city": "Bengaluru",
            "state_code": "KA",
            "postal_code": "560001",
            "phone_number": "8044412345",
            "signup_date": "2020-07-02",
            "risk_score": 45.1,
            "status_code": "ACTIVE",
        },
        {
            "customer_id": 3,
            "first_name": "Carol",
            "last_name": "Brown",
            "email": "carol.brown@acme.com",
            "country_code": "UK",
            "city": "London",
            "state_code": "LN",
            "postal_code": "EC1A1BB",
            "phone_number": "2071234567",
            "signup_date": "2019-03-21",
            "risk_score": 67.7,
            "status_code": "INACTIVE",
        },
        {
            "customer_id": 4,
            "first_name": "David",
            "last_name": "Miller",
            "email": "david.miller@acme.com",
            "country_code": "US",
            "city": "Seattle",
            "state_code": "WA",
            "postal_code": "98101",
            "phone_number": "4253339988",
            "signup_date": "2022-10-18",
            "risk_score": 91.4,
            "status_code": "ACTIVE",
        },
        {
            "customer_id": 5,
            "first_name": "Emma",
            "last_name": "Wilson",
            "email": "emma.wilson@acme.com",
            "country_code": "IN",
            "city": "Pune",
            "state_code": "MH",
            "postal_code": "411001",
            "phone_number": "2044008877",
            "signup_date": "2023-02-01",
            "risk_score": 38.9,
            "status_code": "ACTIVE",
        },
        {
            "customer_id": 6,
            "first_name": "Frank",
            "last_name": "Taylor",
            "email": "frank.taylor@acme.com",
            "country_code": "UK",
            "city": "Manchester",
            "state_code": "MN",
            "postal_code": "M11AE",
            "phone_number": "1610098765",
            "signup_date": "2021-08-29",
            "risk_score": 72.0,
            "status_code": "INACTIVE",
        },
    ]

    orders_rows = [
        {"order_id": 1001, "customer_id": 1, "amount_local": 120.50, "currency_code": "USD", "event_date": "2024-01-05", "service_code": "SVC_A", "quantity": 2, "discount_amount": 10.5, "tax_amount": 5.2},
        {"order_id": 1002, "customer_id": 2, "amount_local": 9500.00, "currency_code": "INR", "event_date": "2024-01-08", "service_code": "SVC_B", "quantity": 4, "discount_amount": 150.0, "tax_amount": 180.0},
        {"order_id": 1003, "customer_id": 3, "amount_local": 88.30, "currency_code": "GBP", "event_date": "2024-01-11", "service_code": "SVC_A", "quantity": 1, "discount_amount": 4.0, "tax_amount": 3.5},
        {"order_id": 1004, "customer_id": 4, "amount_local": 230.00, "currency_code": "USD", "event_date": "2024-01-13", "service_code": "SVC_C", "quantity": 5, "discount_amount": 15.0, "tax_amount": 9.3},
        {"order_id": 1005, "customer_id": 5, "amount_local": 4100.00, "currency_code": "INR", "event_date": "2024-01-18", "service_code": "SVC_B", "quantity": 8, "discount_amount": 125.0, "tax_amount": 80.0},
        {"order_id": 1006, "customer_id": 6, "amount_local": 67.20, "currency_code": "GBP", "event_date": "2024-01-21", "service_code": "SVC_C", "quantity": 2, "discount_amount": 2.2, "tax_amount": 2.0},
    ]

    country_lookup_rows = [
        {"country_code": "US", "country_name": "United States", "fx_rate": 1.0, "lookup_name": "USA Lookup"},
        {"country_code": "IN", "country_name": "India", "fx_rate": 0.012, "lookup_name": "India Lookup"},
        {"country_code": "UK", "country_name": "United Kingdom", "fx_rate": 1.27, "lookup_name": "UK Lookup"},
    ]

    service_lookup_rows = [
        {"service_code": "SVC_A", "service_category": "Consulting"},
        {"service_code": "SVC_B", "service_category": "Support"},
        {"service_code": "SVC_C", "service_category": "Platform"},
    ]

    order_items_rows = [
        {"order_id": 1001, "line_no": 1, "sku": "SKU_A1", "line_amount": 70.0},
        {"order_id": 1001, "line_no": 2, "sku": "SKU_A2", "line_amount": 50.5},
        {"order_id": 1002, "line_no": 1, "sku": "SKU_B1", "line_amount": 5500.0},
        {"order_id": 1002, "line_no": 2, "sku": "SKU_B2", "line_amount": 4000.0},
        {"order_id": 1003, "line_no": 1, "sku": "SKU_C1", "line_amount": 88.3},
        {"order_id": 1004, "line_no": 1, "sku": "SKU_D1", "line_amount": 120.0},
        {"order_id": 1004, "line_no": 2, "sku": "SKU_D2", "line_amount": 110.0},
        {"order_id": 1005, "line_no": 1, "sku": "SKU_E1", "line_amount": 2100.0},
        {"order_id": 1005, "line_no": 2, "sku": "SKU_E2", "line_amount": 2000.0},
        {"order_id": 1006, "line_no": 1, "sku": "SKU_F1", "line_amount": 67.2},
    ]

    documents_rows = [
        {"doc_id": 501, "customer_id": 1, "file_path": "/docs/us/alice_profile.pdf", "created_at_text": "2024-01-05 10:11:00"},
        {"doc_id": 502, "customer_id": 2, "file_path": "/docs/in/bob_statement.csv", "created_at_text": "2024-01-08 09:00:12"},
        {"doc_id": 503, "customer_id": 3, "file_path": "/docs/uk/carol_report.xlsx", "created_at_text": "2024-01-11 15:31:22"},
        {"doc_id": 504, "customer_id": 4, "file_path": "/docs/us/david_id.png", "created_at_text": "2024-01-13 07:22:44"},
        {"doc_id": 505, "customer_id": 5, "file_path": "/docs/in/emma_contract.docx", "created_at_text": "2024-01-18 14:07:09"},
        {"doc_id": 506, "customer_id": 6, "file_path": "/docs/uk/frank_note.txt", "created_at_text": "2024-01-21 19:20:02"},
    ]

    # ---------------------------
    # Table objects
    # ---------------------------
    customers = Table(
        name="src_customers",
        columns={
            "customer_id": _demo_make_col(
                "customer_id",
                customers_rows,
                "int",
                ColumnConstraints(nullable=False, is_primary_key=True, is_unique=True),
            ),
            "first_name": _demo_make_col("first_name", customers_rows, "string"),
            "last_name": _demo_make_col("last_name", customers_rows, "string"),
            "email": _demo_make_col("email", customers_rows, "string", ColumnConstraints(is_unique=True)),
            "country_code": _demo_make_col("country_code", customers_rows, "string"),
            "city": _demo_make_col("city", customers_rows, "string"),
            "state_code": _demo_make_col("state_code", customers_rows, "string"),
            "postal_code": _demo_make_col("postal_code", customers_rows, "string"),
            "phone_number": _demo_make_col("phone_number", customers_rows, "string"),
            "signup_date": _demo_make_col("signup_date", customers_rows, "date"),
            "risk_score": _demo_make_col("risk_score", customers_rows, "decimal"),
            "status_code": _demo_make_col("status_code", customers_rows, "string"),
        },
        row_count=len(customers_rows),
        rows=customers_rows,
    )

    orders = Table(
        name="src_orders",
        columns={
            "order_id": _demo_make_col(
                "order_id",
                orders_rows,
                "int",
                ColumnConstraints(nullable=False, is_primary_key=True, is_unique=True),
            ),
            "customer_id": _demo_make_col(
                "customer_id",
                orders_rows,
                "int",
                ColumnConstraints(is_foreign_key=True),
            ),
            "amount_local": _demo_make_col("amount_local", orders_rows, "decimal"),
            "currency_code": _demo_make_col("currency_code", orders_rows, "string"),
            "event_date": _demo_make_col("event_date", orders_rows, "date"),
            "service_code": _demo_make_col("service_code", orders_rows, "string"),
            "quantity": _demo_make_col("quantity", orders_rows, "int"),
            "discount_amount": _demo_make_col("discount_amount", orders_rows, "decimal"),
            "tax_amount": _demo_make_col("tax_amount", orders_rows, "decimal"),
        },
        row_count=len(orders_rows),
        rows=orders_rows,
    )

    country_lookup = Table(
        name="dim_country",
        columns={
            "country_code": _demo_make_col(
                "country_code",
                country_lookup_rows,
                "string",
                ColumnConstraints(nullable=False, is_primary_key=True, is_unique=True),
            ),
            "country_name": _demo_make_col("country_name", country_lookup_rows, "string"),
            "fx_rate": _demo_make_col("fx_rate", country_lookup_rows, "decimal"),
            "lookup_name": _demo_make_col("lookup_name", country_lookup_rows, "string"),
        },
        row_count=len(country_lookup_rows),
        rows=country_lookup_rows,
    )

    service_lookup = Table(
        name="dim_service",
        columns={
            "service_code": _demo_make_col(
                "service_code",
                service_lookup_rows,
                "string",
                ColumnConstraints(nullable=False, is_primary_key=True, is_unique=True),
            ),
            "service_category": _demo_make_col("service_category", service_lookup_rows, "string"),
        },
        row_count=len(service_lookup_rows),
        rows=service_lookup_rows,
    )

    order_items = Table(
        name="src_order_items",
        columns={
            "order_id": _demo_make_col("order_id", order_items_rows, "int", ColumnConstraints(is_foreign_key=True)),
            "line_no": _demo_make_col("line_no", order_items_rows, "int"),
            "sku": _demo_make_col("sku", order_items_rows, "string"),
            "line_amount": _demo_make_col("line_amount", order_items_rows, "decimal"),
        },
        row_count=len(order_items_rows),
        rows=order_items_rows,
    )

    documents = Table(
        name="src_documents",
        columns={
            "doc_id": _demo_make_col("doc_id", documents_rows, "int", ColumnConstraints(nullable=False, is_primary_key=True, is_unique=True)),
            "customer_id": _demo_make_col("customer_id", documents_rows, "int", ColumnConstraints(is_foreign_key=True)),
            "file_path": _demo_make_col("file_path", documents_rows, "string"),
            "created_at_text": _demo_make_col("created_at_text", documents_rows, "string"),
        },
        row_count=len(documents_rows),
        rows=documents_rows,
    )

    source_tables = {
        customers.name: customers,
        orders.name: orders,
        country_lookup.name: country_lookup,
        service_lookup.name: service_lookup,
        order_items.name: order_items,
        documents.name: documents,
    }

    # ---------------------------
    # Join edges
    # ---------------------------
    join_edges = [
        JoinEdge(
            left_table="src_customers",
            right_table="src_orders",
            left_cols=["customer_id"],
            right_cols=["customer_id"],
            cardinality="1:N",
            confidence=0.96,
            reasons=["single", "pk_fk_relation"],
            score_details={"name": 1.0, "value": 1.0, "type": 1.0, "constraint": 0.95},
            direction="left_parent",
        ),
        JoinEdge(
            left_table="src_customers",
            right_table="dim_country",
            left_cols=["country_code"],
            right_cols=["country_code"],
            cardinality="N:1",
            confidence=0.93,
            reasons=["single", "lookup"],
            score_details={"name": 0.93, "value": 1.0, "type": 0.9, "constraint": 0.9},
            direction="right_parent",
        ),
        JoinEdge(
            left_table="src_orders",
            right_table="dim_service",
            left_cols=["service_code"],
            right_cols=["service_code"],
            cardinality="N:1",
            confidence=0.91,
            reasons=["single", "lookup"],
            score_details={"name": 0.88, "value": 1.0, "type": 0.9, "constraint": 0.9},
            direction="right_parent",
        ),
        JoinEdge(
            left_table="src_orders",
            right_table="src_order_items",
            left_cols=["order_id"],
            right_cols=["order_id"],
            cardinality="1:N",
            confidence=0.94,
            reasons=["single", "pk_fk_relation"],
            score_details={"name": 0.96, "value": 1.0, "type": 0.95, "constraint": 0.95},
            direction="left_parent",
        ),
        JoinEdge(
            left_table="src_customers",
            right_table="src_documents",
            left_cols=["customer_id"],
            right_cols=["customer_id"],
            cardinality="1:N",
            confidence=0.92,
            reasons=["single", "pk_fk_relation"],
            score_details={"name": 0.95, "value": 1.0, "type": 0.95, "constraint": 0.95},
            direction="left_parent",
        ),
    ]

    engine = CandidateGenerationEngine(
        source_tables=source_tables,
        join_edges=join_edges,
    )

    # ---------------------------
    # Target tests
    # ---------------------------
    target_tests = [
        (
            "TEST 1: first_name + last_name -> full_name",
            TargetSpec(
                table="dim_customer",
                name="full_name",
                col_type=ColumnType("string"),
                constraints=ColumnConstraints(),
                stats=ColumnStats(
                    [
                        f"{r['first_name']} {r['last_name']}"
                        for r in customers_rows
                    ]
                ),
                description="Customer full name",
                sample_values=[f"{r['first_name']} {r['last_name']}" for r in customers_rows],
            ),
        ),
        (
            "TEST 2: email -> email_username",
            TargetSpec(
                table="dim_customer",
                name="email_username",
                col_type=ColumnType("string"),
                constraints=ColumnConstraints(),
                stats=ColumnStats([r["email"].split("@")[0] for r in customers_rows]),
                description="Local part from email",
                sample_values=[r["email"].split("@")[0] for r in customers_rows],
            ),
        ),
        (
            "TEST 3: amount_local + fx_rate -> amount_usd",
            TargetSpec(
                table="fct_order",
                name="amount_usd",
                col_type=ColumnType("decimal"),
                constraints=ColumnConstraints(),
                stats=ColumnStats(
                    [
                        round(
                            r["amount_local"]
                            * next(
                                x["fx_rate"] for x in country_lookup_rows
                                if x["country_code"]
                                == next(c["country_code"] for c in customers_rows if c["customer_id"] == r["customer_id"])
                            ),
                            4,
                        )
                        for r in orders_rows
                    ]
                ),
                description="Order amount converted to USD-equivalent",
            ),
        ),
        (
            "TEST 4: country_code -> country_name (lookup join)",
            TargetSpec(
                table="dim_customer",
                name="country_name",
                col_type=ColumnType("string"),
                constraints=ColumnConstraints(),
                stats=ColumnStats(
                    [
                        next(x["country_name"] for x in country_lookup_rows if x["country_code"] == c["country_code"])
                        for c in customers_rows
                    ]
                ),
                description="Country full name lookup",
            ),
        ),
        (
            "TEST 5: city + state_code -> city_state_display",
            TargetSpec(
                table="dim_customer",
                name="city_state_display",
                col_type=ColumnType("string"),
                constraints=ColumnConstraints(),
                stats=ColumnStats([f"{c['city']} {c['state_code']}" for c in customers_rows]),
                description="City and state display string",
                sample_values=[f"{c['city']} {c['state_code']}" for c in customers_rows],
            ),
        ),
        (
            "TEST 6: country_code + phone_number -> e164_phone",
            TargetSpec(
                table="dim_customer",
                name="e164_phone",
                col_type=ColumnType("string"),
                constraints=ColumnConstraints(),
                stats=ColumnStats([f"{c['country_code']} {c['phone_number']}" for c in customers_rows]),
                description="International phone text",
                sample_values=[f"{c['country_code']} {c['phone_number']}" for c in customers_rows],
            ),
        ),
        (
            "TEST 7: event_date -> event_year",
            TargetSpec(
                table="fct_order",
                name="event_year",
                col_type=ColumnType("int"),
                constraints=ColumnConstraints(),
                stats=ColumnStats([int(r["event_date"][:4]) for r in orders_rows]),
                description="Extract event year",
                sample_values=[int(r["event_date"][:4]) for r in orders_rows],
            ),
        ),
        (
            "TEST 8: amount_local / quantity -> unit_price",
            TargetSpec(
                table="fct_order",
                name="unit_price",
                col_type=ColumnType("decimal"),
                constraints=ColumnConstraints(),
                stats=ColumnStats([round(r["amount_local"] / max(1, r["quantity"]), 6) for r in orders_rows]),
                description="Per-unit price",
            ),
        ),
        (
            "TEST 9: amount_local + tax_amount -> gross_with_tax",
            TargetSpec(
                table="fct_order",
                name="gross_with_tax",
                col_type=ColumnType("decimal"),
                constraints=ColumnConstraints(),
                stats=ColumnStats([round(r["amount_local"] + r["tax_amount"], 6) for r in orders_rows]),
                description="Tax-inclusive amount",
            ),
        ),
        (
            "TEST 10: service_code -> service_category (lookup)",
            TargetSpec(
                table="fct_order",
                name="service_category",
                col_type=ColumnType("string"),
                constraints=ColumnConstraints(),
                stats=ColumnStats(
                    [
                        next(x["service_category"] for x in service_lookup_rows if x["service_code"] == r["service_code"])
                        for r in orders_rows
                    ]
                ),
                description="Lookup service category from service code",
            ),
        ),
        (
            "TEST 11: order_id + line_no -> order_line_key",
            TargetSpec(
                table="fct_order_item",
                name="order_line_key",
                col_type=ColumnType("string"),
                constraints=ColumnConstraints(),
                stats=ColumnStats([f"{r['order_id']} {r['line_no']}" for r in order_items_rows]),
                description="Order line composite key text",
                sample_values=[f"{r['order_id']} {r['line_no']}" for r in order_items_rows],
            ),
        ),
        (
            "TEST 12: file_path -> file_extension",
            TargetSpec(
                table="dim_document",
                name="file_extension",
                col_type=ColumnType("string"),
                constraints=ColumnConstraints(),
                stats=ColumnStats([r["file_path"].split(".")[-1].lower() for r in documents_rows]),
                description="Extract file extension from path",
                sample_values=[r["file_path"].split(".")[-1].lower() for r in documents_rows],
            ),
        ),
        (
            "TEST 13: created_at_text -> created_date",
            TargetSpec(
                table="dim_document",
                name="created_date",
                col_type=ColumnType("date"),
                constraints=ColumnConstraints(),
                stats=ColumnStats([r["created_at_text"][:10] for r in documents_rows]),
                description="Parse created timestamp text",
                sample_values=[r["created_at_text"][:10] for r in documents_rows],
            ),
        ),
        (
            "TEST 14: risk_score -> is_high_risk",
            TargetSpec(
                table="dim_customer",
                name="is_high_risk",
                col_type=ColumnType("boolean"),
                constraints=ColumnConstraints(),
                stats=ColumnStats([r["risk_score"] >= 70.0 for r in customers_rows]),
                description="High risk threshold flag",
                sample_values=[r["risk_score"] >= 70.0 for r in customers_rows],
            ),
        ),
    ]

    for title, target in target_tests:
        result = engine.rank_candidates(
            target=target,
            coarse_top_m=50,
            fine_top_m=25,
            max_arity=4,
            max_hops=3,
            top_k=8,
            abstain_threshold=0.58,
        )
        _demo_print_result(title, result, top_n=6)


if __name__ == "__main__":
    run_demo_candidate_generation()
