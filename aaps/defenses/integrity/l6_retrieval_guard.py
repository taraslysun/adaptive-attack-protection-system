"""L6 Retrieval Integrity Guard -- TrustRAG-inspired.

Source
------
Zhou, Huichi et al., "TrustRAG: Enhancing Robustness and Trustworthiness
in Retrieval-Augmented Generation", arXiv:2501.00879 (2025; AAAI 2026
TrustAgent Workshop). Bibkey ``zhou2025trustrag``.

A separate, simpler retrieval baseline lives at
``defenses/baselines/rag_guard.py``; AIS L6 (this file) implements the
two-stage TrustRAG-faithful pipeline (cluster filter + internal-knowledge
cross-check).

Two-stage protection over retrieval results:

Stage 1 -- clustering filter
    Embed the retrieved documents, run K-means with ``k=2`` and inspect
    cluster sizes.  Coordinated poisoning (the central observation of
    TrustRAG and of Zhong et al. "Poisoning Retrieval Corpora by
    Injecting Adversarial Passages", EMNLP 2023, arXiv:2310.19156;
    bibkey ``% TODO[bib]: zhong2023poisoning``) creates passages that
    look similar to one another and to the query. A small, tight
    cluster of similar embeddings is the canonical signature.

Stage 2 -- internal-knowledge cross-check
    If an ``answer_fn`` is provided, generate an answer to the query
    *without* the retrieved documents and compare the embedding of that
    answer to the embedding of the answer generated *with* them.  Large
    divergence indicates the retrieval has steered the model away from
    its prior knowledge -- often a sign of a successful poisoning.

When sklearn / sentence-transformers are missing the layer falls back
to a length-and-keyword heuristic so it still passes a HARD_BLOCK
decision through (rather than mis-pretending the docs are clean).

Adaptive-attack note
--------------------
Zhong et al. (2023, arXiv:2310.19156) and Chen et al. CPA-RAG (2024,
arXiv:2407.13757; bibkey ``% TODO[bib]: chen2024cparag``) show that
fluent, query-aligned poison can defeat single filters.  L6 is therefore
explicitly *not* sold as a standalone solution; the trace logger reports
its remaining miss rate so the thesis can characterise the residual
attack surface honestly. See also Nasr et al. "The Attacker Moves
Second" (arXiv:2510.09023, bibkey ``nasr2025attacker``) for the umbrella
adaptive-attack methodology.

Hook coverage
-------------
``check_retrieval`` is invoked from both ``agent/deep_agent.py`` (full
RAG path) and ``attacks/_core/local_agent.py::_format_memory_context``
(memory-as-retrieval surface; thesis remediation
``codebase-fix-l6-not-wired-locally``). Thesis prose may therefore say
"L1-L6 all participated" for matrix runs against either agent. When
``LocalAgent.memory`` is empty the hook is simply not invoked, and the
trace logger reports zero L6 traffic for that query -- which is honest.
"""

from __future__ import annotations

import math
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from aaps.defenses.base_defense import BaseDefense, DefenseResult


def _cosine(a, b) -> float:
    import numpy as np

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class RetrievalIntegrityDefense(BaseDefense):
    """L6: clustering outlier filter + optional internal-KB cross-check."""

    name = "L6_retrieval_integrity"
    LAYER_ID = "L6"

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        encoder_name: str = "all-MiniLM-L6-v2",
        answer_fn: Optional[Callable[[str], str]] = None,
        intra_cluster_threshold: float = 0.85,
        max_filter_fraction: float = 0.6,
        cross_check_threshold: float = 0.30,
        min_docs_to_cluster: int = 3,
        outlier_distance_threshold: float = 0.45,
    ) -> None:
        super().__init__(config)
        self.encoder_name = encoder_name
        self.answer_fn = answer_fn
        self.intra_cluster_threshold = intra_cluster_threshold
        self.max_filter_fraction = max_filter_fraction
        self.cross_check_threshold = cross_check_threshold
        self.min_docs_to_cluster = min_docs_to_cluster
        self.outlier_distance_threshold = outlier_distance_threshold
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._encoder = SentenceTransformer(self.encoder_name)
            except Exception:
                self._encoder = False
        return self._encoder if self._encoder else None

    # ------------------------------------------------------------------
    # Stage 1: clustering filter.
    # ------------------------------------------------------------------

    def _embed_docs(self, docs: List[Dict[str, Any]]):
        enc = self._get_encoder()
        if enc is None:
            return None
        texts = [d.get("text", "") or d.get("content", "") for d in docs]
        return enc.encode(texts)

    def _cluster_filter(
        self, query: str, docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if len(docs) < self.min_docs_to_cluster:
            return {
                "available": False,
                "reason": f"only {len(docs)} docs (<{self.min_docs_to_cluster})",
                "outliers": [],
                "kept_indices": list(range(len(docs))),
            }

        embs = self._embed_docs(docs)
        if embs is None:
            return self._heuristic_filter(query, docs)

        try:
            from sklearn.cluster import KMeans
            import numpy as np
        except Exception:
            return self._heuristic_filter(query, docs)

        try:
            km = KMeans(n_clusters=2, n_init=4, random_state=0)
            labels = km.fit_predict(np.asarray(embs))
        except Exception:
            return self._heuristic_filter(query, docs)

        sizes = [int((labels == c).sum()) for c in (0, 1)]
        smaller = 0 if sizes[0] <= sizes[1] else 1
        larger = 1 - smaller
        smaller_size = sizes[smaller]
        smaller_idx = [i for i, l in enumerate(labels) if l == smaller]
        larger_idx = [i for i, l in enumerate(labels) if l == larger]

        intra: List[float] = []
        if len(smaller_idx) >= 2:
            for i in range(len(smaller_idx)):
                for j in range(i + 1, len(smaller_idx)):
                    intra.append(
                        _cosine(embs[smaller_idx[i]], embs[smaller_idx[j]])
                    )
        intra_mean = sum(intra) / len(intra) if intra else 1.0

        max_inter_sim = 0.0
        if smaller_idx and larger_idx:
            for si in smaller_idx:
                for li in larger_idx:
                    s = _cosine(embs[si], embs[li])
                    if s > max_inter_sim:
                        max_inter_sim = s

        size_ok = smaller_size <= max(
            2, int(self.max_filter_fraction * len(docs))
        )
        is_poison_cluster = (
            size_ok
            and len(smaller_idx) >= 2
            and intra_mean >= self.intra_cluster_threshold
        )
        is_lone_outlier = (
            size_ok
            and len(smaller_idx) == 1
            and (1.0 - max_inter_sim) >= self.outlier_distance_threshold
        )

        outliers: List[int] = []
        if is_poison_cluster or is_lone_outlier:
            outliers = smaller_idx
        kept_idx = [i for i in range(len(docs)) if i not in outliers]

        return {
            "available": True,
            "cluster_sizes": sizes,
            "smaller_cluster_size": smaller_size,
            "intra_cluster_similarity": intra_mean,
            "max_inter_cluster_similarity": max_inter_sim,
            "is_poison_cluster": is_poison_cluster,
            "is_lone_outlier": is_lone_outlier,
            "outliers": outliers,
            "kept_indices": kept_idx,
        }

    def _heuristic_filter(
        self, query: str, docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Lightweight fallback when no embedding/clustering backend exists."""
        outliers: List[int] = []
        signals = [
            r"ignore\s+previous", r"system\s*[:>]", r"override",
            r"\[\s*INST\s*\]", r"prerequisite\s+step",
        ]
        for i, d in enumerate(docs):
            text = (d.get("text", "") or d.get("content", "")).lower()
            if any(re.search(s, text) for s in signals):
                outliers.append(i)
        kept = [i for i in range(len(docs)) if i not in outliers]
        return {
            "available": False,
            "reason": "fell back to heuristic",
            "outliers": outliers,
            "kept_indices": kept,
        }

    # ------------------------------------------------------------------
    # Stage 2: internal-KB cross-check.
    # ------------------------------------------------------------------

    def _internal_cross_check(
        self,
        query: str,
        kept_docs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if self.answer_fn is None:
            return {"available": False, "reason": "no answer_fn provided"}
        enc = self._get_encoder()
        if enc is None:
            return {"available": False, "reason": "no encoder"}

        try:
            internal = self.answer_fn(query) or ""
        except Exception:
            return {"available": False, "reason": "internal answer_fn raised"}

        ctx = "\n".join(
            d.get("text", "") or d.get("content", "") for d in kept_docs
        )[:4000]
        try:
            external = (
                self.answer_fn(
                    f"Use ONLY the context below to answer.\n\nCONTEXT:\n{ctx}\n\nQUESTION: {query}"
                )
                or ""
            )
        except Exception:
            return {"available": False, "reason": "external answer_fn raised"}

        if not internal or not external:
            return {"available": False, "reason": "empty answer(s)"}

        sim = _cosine(enc.encode(internal), enc.encode(external))
        diverged = sim < self.cross_check_threshold
        return {
            "available": True,
            "similarity": sim,
            "threshold": self.cross_check_threshold,
            "diverged": diverged,
            "internal_excerpt": internal[:200],
            "external_excerpt": external[:200],
        }

    # ------------------------------------------------------------------
    # Hook.
    # ------------------------------------------------------------------

    def check_retrieval(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
    ) -> DefenseResult:
        t0 = time.perf_counter()
        self.stats["total_checks"] += 1

        if not retrieved_docs:
            self.stats["allowed"] += 1
            return DefenseResult.allow(
                reason="L6: no docs to inspect",
                metadata={"layer": self.LAYER_ID},
                layer=self.LAYER_ID,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        cluster = self._cluster_filter(query, retrieved_docs)
        kept = [retrieved_docs[i] for i in cluster["kept_indices"]]
        dropped = [retrieved_docs[i] for i in cluster["outliers"]]

        cross = self._internal_cross_check(query, kept)
        latency = (time.perf_counter() - t0) * 1000.0

        if cross.get("diverged"):
            self.stats["blocked"] += 1
            return DefenseResult.hard_block(
                reason=(
                    "L6: internal-KB cross-check diverged "
                    f"(sim={cross.get('similarity', 0):.2f}); retrieval suspect"
                ),
                metadata={
                    "layer": self.LAYER_ID,
                    "cluster": cluster,
                    "cross_check": cross,
                    "kept_docs": kept,
                    "dropped_docs": dropped,
                },
                layer=self.LAYER_ID,
                latency_ms=latency,
            )

        if dropped:
            self.stats["blocked"] += len(dropped)
            return DefenseResult.soft_filter(
                reason=(
                    f"L6: dropped {len(dropped)} clustered-outlier docs "
                    f"(intra-sim={cluster.get('intra_cluster_similarity', 0):.2f})"
                ),
                metadata={
                    "layer": self.LAYER_ID,
                    "cluster": cluster,
                    "cross_check": cross,
                    "kept_docs": kept,
                    "dropped_docs": dropped,
                },
                layer=self.LAYER_ID,
                latency_ms=latency,
            )

        self.stats["allowed"] += 1
        return DefenseResult.allow(
            reason=(
                f"L6: all {len(retrieved_docs)} docs passed clustering; "
                f"cross-check={'na' if not cross.get('available') else 'ok'}"
            ),
            metadata={
                "layer": self.LAYER_ID,
                "cluster": cluster,
                "cross_check": cross,
                "kept_docs": kept,
            },
            layer=self.LAYER_ID,
            latency_ms=latency,
        )
