"""Evidence pool, k-means clustering, and trust-label aggregation.

The Planner emits a PACEPlan from trusted input only. The Executor
needs to fill that plan from *evidence* — tool outputs, retrieved
documents, memory entries — every byte of which is potentially
attacker-controlled. PACE does not let a single evidence span decide a
tool call. Instead, the evidence pool is partitioned into ``K``
semantic clusters and ``K`` parallel Executor calls each see only one
cluster. The Quorum voter then requires ``≥ q`` clusters to agree.

This module owns:

* :class:`EvidenceSpan` — a typed evidence record (text + provenance +
  trust labels attached by AIS-L1/L2/L5 feature extractors).
* :class:`EvidencePool` — accumulator with deterministic span IDs.
* :func:`kmeans_cluster` — scikit-learn k-means over sentence-transformer
  embeddings, with explicit fallbacks when the optional deps are
  missing (so PACE runs in CI containers without torch).

The clustering implementation is an adaptation of the document
partition idea from TrustRAG (Zhou et al. 2025) and the Self-
Consistency sampling-then-vote paradigm (Wang et al. 2022,
arXiv:2203.11171). Generalising from "validate retrieved documents"
to "validate every evidence-derived tool call" is the PACE contribution.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from aaps.attacks._core.logging_config import get_logger

log = get_logger("pace.clusters")


@dataclass
class EvidenceSpan:
    """One unit of evidence the Executor may consume.

    Provenance carries the source channel (user_input, tool_output,
    memory, retrieval); the capability shim uses it to gate
    high-privilege calls. The CFI + Quorum gates do enforcement.
    """

    text: str
    provenance: str  # "user_input" | "tool_output:<tool_name>" | "memory" | "retrieval"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def span_id(self) -> str:
        h = hashlib.sha256(
            (self.provenance + "::" + self.text).encode("utf-8")
        ).hexdigest()
        return h[:10]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "provenance": self.provenance,
            "text_preview": self.text[:120],
            "metadata": dict(self.metadata),
        }


class EvidencePool:
    """Accumulator for evidence spans the Executor will see."""

    def __init__(self) -> None:
        self.spans: List[EvidenceSpan] = []

    def add(self, span: EvidenceSpan) -> None:
        self.spans.append(span)

    def extend(self, spans: Sequence[EvidenceSpan]) -> None:
        self.spans.extend(spans)

    def __len__(self) -> int:
        return len(self.spans)


def _seed_from(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _hash_embedding(text: str, dim: int = 64) -> List[float]:
    """Deterministic, dependency-free embedding fallback.

    Uses a per-dim hash bucket; not as good as MiniLM but stable in CI
    containers without torch. The k-means routine still works.
    """
    vec = [0.0] * dim
    for token in text.lower().split():
        h = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _embed(spans: Sequence[EvidenceSpan], embedder_id: str) -> List[List[float]]:
    if not spans:
        return []
    if os.environ.get("PACE_FORCE_HASH_EMBED") == "1":
        log.debug("embedding: PACE_FORCE_HASH_EMBED=1 — using hash fallback for %d spans", len(spans))
        return [_hash_embedding(s.text) for s in spans]
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        model = SentenceTransformer(embedder_id)
        out = model.encode([s.text for s in spans], normalize_embeddings=True)
        log.debug("embedding: sentence-transformers %r encoded %d spans", embedder_id, len(spans))
        return [list(map(float, row)) for row in out]
    except Exception as exc:
        log.warning(
            "embedding: sentence-transformers UNAVAILABLE (%s) — "
            "falling back to hash embedding for %d spans. "
            "Clustering quality will be degraded.",
            exc, len(spans),
        )
        return [_hash_embedding(s.text) for s in spans]


def _kmeans(
    vectors: List[List[float]],
    k: int,
    seed: int,
    max_iter: int = 50,
) -> List[int]:
    """Tiny pure-Python k-means; deterministic given ``seed``.

    We avoid sklearn so PACE stays importable in environments where
    only torch + sentence-transformers are present (or neither).
    """
    n = len(vectors)
    if n == 0 or k <= 1:
        return [0] * n
    k = min(k, n)
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    centroids = [list(vectors[i]) for i in indices[:k]]
    assignment = [0] * n
    dim = len(vectors[0])
    for _ in range(max_iter):
        changed = False
        for i, v in enumerate(vectors):
            best, best_d = 0, float("inf")
            for c, cen in enumerate(centroids):
                d = sum((v[j] - cen[j]) ** 2 for j in range(dim))
                if d < best_d:
                    best_d, best = d, c
            if assignment[i] != best:
                assignment[i] = best
                changed = True
        new_centroids = [[0.0] * dim for _ in range(k)]
        counts = [0] * k
        for i, v in enumerate(vectors):
            c = assignment[i]
            counts[c] += 1
            for j in range(dim):
                new_centroids[c][j] += v[j]
        for c in range(k):
            if counts[c] > 0:
                centroids[c] = [x / counts[c] for x in new_centroids[c]]
        if not changed:
            break
    return assignment


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


# Module-level NLI pipeline cache; lazily loaded on first call.
_NLI_PIPE: Optional[Any] = None
_NLI_AVAILABLE: Optional[bool] = None


def _get_nli_pipeline(model_name: str) -> Optional[Any]:
    """Lazy-load HuggingFace cross-encoder NLI pipeline.

    Returns None if transformers unavailable or model load fails.
    Pipeline returns labels in {entailment, neutral, contradiction}.
    """
    global _NLI_PIPE, _NLI_AVAILABLE
    if _NLI_AVAILABLE is False:
        return None
    if _NLI_PIPE is not None:
        return _NLI_PIPE
    try:
        from transformers import pipeline as hf_pipeline  # type: ignore
        _NLI_PIPE = hf_pipeline(
            "text-classification",
            model=model_name,
            device=-1,  # CPU; override via env if needed
            top_k=None,  # return scores for all labels
        )
        _NLI_AVAILABLE = True
        log.info("nli_filter: loaded cross-encoder NLI model %r", model_name)
        return _NLI_PIPE
    except Exception as exc:
        log.warning(
            "nli_filter: transformers/NLI model UNAVAILABLE (%s) — "
            "falling back to cosine-similarity heuristic. "
            "Install transformers and pull %r for true entailment filtering.",
            exc, model_name,
        )
        _NLI_AVAILABLE = False
        return None


def _is_entailed_nli(
    pipe: Any,
    premise: str,
    hypothesis: str,
    entailment_threshold: float,
) -> bool:
    """True if NLI model labels (premise → hypothesis) as entailment with score ≥ threshold."""
    try:
        result = pipe({"text": premise, "text_pair": hypothesis}, truncation=True)
        # pipeline with top_k=None returns list of {label, score} dicts
        if isinstance(result, list) and result and isinstance(result[0], list):
            result = result[0]
        if isinstance(result, list):
            for r in result:
                lbl = str(r.get("label", "")).lower()
                if lbl == "entailment":
                    return float(r.get("score", 0.0)) >= entailment_threshold
        elif isinstance(result, dict):
            lbl = str(result.get("label", "")).lower()
            if lbl == "entailment":
                return float(result.get("score", 0.0)) >= entailment_threshold
    except Exception as exc:
        log.debug("nli_filter: pairwise NLI call failed (%s); treating as not-entailed", exc)
    return False


def nli_independence_filter(
    spans: List[EvidenceSpan],
    embedder_id: str = "all-MiniLM-L6-v2",
    entailment_threshold: float = 0.70,
    nli_model: str = "cross-encoder/nli-deberta-v3-small",
    cosine_fallback_threshold: float = 0.92,
) -> List[EvidenceSpan]:
    """Drop spans that are entailed by an earlier kept span (NLI Independence Filter).

    Primary path: cross-encoder NLI model (DeBERTa-v3 small) classifies
    pairs as ``{entailment, neutral, contradiction}``. A candidate span
    is dropped if any kept span entails it with score ≥ ``entailment_threshold``.

    Fallback path (when transformers is unavailable): cosine similarity
    on sentence-transformer embeddings, with the conservative threshold
    ``cosine_fallback_threshold=0.92``. The fallback is a strict superset
    filter (lexical near-duplicates only); true paraphrase entailment
    requires the NLI model.

    user_input spans are anchors and never filtered.

    Cluster-collusion countermeasure: forces an adaptive attacker to
    craft pairwise NLI-neutral injections that all still steer toward
    the same malicious action. Per `nasr2025attacker` (cluster-collusion
    analysis) and `li2025cparag` (covert poisoning), thresholds are
    chosen conservatively to favour false negatives over false positives.

    Returns ``anchors + kept_candidates`` preserving relative order.
    """
    if len(spans) <= 1:
        return list(spans)

    anchors = [s for s in spans if s.provenance == "user_input"]
    candidates = [s for s in spans if s.provenance != "user_input"]
    if not candidates:
        return list(spans)

    pipe = _get_nli_pipeline(nli_model) if os.environ.get("PACE_DISABLE_NLI") != "1" else None

    if pipe is not None:
        # Primary path: pairwise NLI entailment.
        kept: List[EvidenceSpan] = []
        for cand in candidates:
            entailed = False
            for prev in kept:
                if _is_entailed_nli(pipe, prev.text, cand.text, entailment_threshold):
                    entailed = True
                    log.debug(
                        "nli_filter[NLI]: dropped span entailed by earlier kept span "
                        "(threshold=%.2f) provenance=%r preview=%r",
                        entailment_threshold, cand.provenance, cand.text[:60],
                    )
                    break
            if not entailed:
                kept.append(cand)
        n_dropped = len(candidates) - len(kept)
        if n_dropped > 0:
            log.info(
                "nli_independence_filter[NLI]: dropped %d/%d entailed spans "
                "(threshold=%.2f, model=%r); %d remain",
                n_dropped, len(candidates), entailment_threshold, nli_model, len(kept),
            )
        return anchors + kept

    # Fallback: cosine similarity (lexical near-duplicate only).
    vectors = _embed(candidates, embedder_id)
    if not vectors:
        return list(spans)
    kept_indices: List[int] = []
    kept_vectors: List[List[float]] = []
    for i, (span, vec) in enumerate(zip(candidates, vectors)):
        entailed = False
        for kv in kept_vectors:
            if _cosine(vec, kv) >= cosine_fallback_threshold:
                entailed = True
                log.debug(
                    "nli_filter[cosine-fallback]: dropped span (cosine=%.3f >= %.3f) provenance=%r preview=%r",
                    _cosine(vec, kv), cosine_fallback_threshold,
                    span.provenance, span.text[:60],
                )
                break
        if not entailed:
            kept_indices.append(i)
            kept_vectors.append(vec)

    kept_candidates = [candidates[i] for i in kept_indices]
    n_dropped = len(candidates) - len(kept_candidates)
    if n_dropped > 0:
        log.info(
            "nli_independence_filter[cosine-fallback]: dropped %d/%d near-duplicate spans "
            "(threshold=%.2f); %d remain",
            n_dropped, len(candidates), cosine_fallback_threshold, len(kept_candidates),
        )
    return anchors + kept_candidates


def kmeans_cluster(
    pool: EvidencePool,
    K: int,
    *,
    embedder_id: str = "all-MiniLM-L6-v2",
    seed: int = 0,
    method: str = "kmeans",
    nli_filter: bool = True,
    nli_threshold: float = 0.70,
    nli_model: str = "cross-encoder/nli-deberta-v3-small",
    nli_cosine_fallback_threshold: float = 0.92,
) -> List[List[EvidenceSpan]]:
    """Partition ``pool`` into ``K`` clusters of evidence spans.

    Returns a list of ``K`` lists; empty clusters get an empty list
    (the Executor for that cluster will see no evidence and abstain).

    ``method='random'`` is the ablation control: same K, no clustering,
    spans assigned uniformly at random. Used in §6.2 of the design
    doc to answer "is the clustering doing the work?".
    """
    if K <= 0:
        K = 1
    if not pool.spans:
        log.warning(
            "kmeans_cluster called with empty evidence pool (K=%d). "
            "All %d executors will see no evidence and abstain.",
            K, K,
        )
        return [[] for _ in range(K)]

    # NLI Independence Filter: drop entailed spans before clustering.
    # Disabled when nli_filter=False (ablation: PACE_baseline without filter).
    spans_to_cluster = pool.spans
    if nli_filter and len(pool.spans) > 1:
        spans_to_cluster = nli_independence_filter(
            pool.spans,
            embedder_id=embedder_id,
            entailment_threshold=nli_threshold,
            nli_model=nli_model,
            cosine_fallback_threshold=nli_cosine_fallback_threshold,
        )

    log.debug(
        "kmeans_cluster K=%d n_spans_before_filter=%d n_spans_after=%d method=%s",
        K, len(pool.spans), len(spans_to_cluster), method,
    )
    if method == "random":
        rng = random.Random(seed)
        clusters: List[List[EvidenceSpan]] = [[] for _ in range(K)]
        for s in spans_to_cluster:
            clusters[rng.randrange(K)].append(s)
        return clusters
    vectors = _embed(spans_to_cluster, embedder_id)
    assignment = _kmeans(vectors, K, seed=seed or _seed_from("spq-clusters"))
    clusters: List[List[EvidenceSpan]] = [[] for _ in range(K)]
    for span, cid in zip(spans_to_cluster, assignment):
        clusters[cid].append(span)
    sizes = [len(c) for c in clusters]
    empty = sum(1 for s in sizes if s == 0)
    log.debug("kmeans_cluster result: cluster_sizes=%s empty_clusters=%d", sizes, empty)
    if empty > 0:
        log.warning(
            "kmeans_cluster: %d of %d cluster(s) are empty — "
            "those executors will abstain (possible cluster collapse with K=%d > n_spans=%d)",
            empty, K, K, len(pool.spans),
        )
    return clusters


def assignment_vector(clusters: List[List[EvidenceSpan]]) -> List[int]:
    """Inverse of ``kmeans_cluster``: span -> cluster id, in pool order.

    Convenience for the trace logger; expects a stable insertion order.
    """
    out: List[Tuple[str, int]] = []
    for cid, members in enumerate(clusters):
        for span in members:
            out.append((span.span_id, cid))
    return [cid for _, cid in out]
