"""Hybrid retriever: Reciprocal Rank Fusion of dense + BM25 + method_match, optional CE rerank.

Fusion rationale: BM25 scores and dense cosines live on incompatible scales (BM25 unbounded,
cosine in [-1, 1]); a weighted sum on raw scores is brittle. RRF fuses by rank: each signal
contributes `weight_s / (RRF_K + rank_s(pid))` and signals that didn't surface the paper
contribute 0. `signal_breakdown` returned to the caller carries the RAW per-signal scores
(not the RRF terms) — the UI's "reason tag" wants to show "Dense thought 0.95; method_match
thought 0.81", not normalized arithmetic.
"""

from __future__ import annotations

from typing import Literal

from schemas import MethodCard

from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .method_match import MethodCardMatcher
from .rerank import CrossEncoderReranker

RRF_K = 60

# TODO(C): tune via eval/run.py nDCG comparison on gold set
DEFAULT_WEIGHTS = {"dense": 0.4, "bm25": 0.2, "method_match": 0.4}


class HybridRetriever:
    """RRF-fused hybrid retriever over dense + BM25 + method_match, with optional CE rerank."""

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        use_rerank: bool = True,
        *,
        dense: DenseRetriever | None = None,
        bm25: BM25Retriever | None = None,
        method_match: MethodCardMatcher | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.weights = dict(weights) if weights else dict(DEFAULT_WEIGHTS)
        self.use_rerank = use_rerank
        self.dense = dense or DenseRetriever()
        self.bm25 = bm25 or BM25Retriever()
        self.method_match = method_match or MethodCardMatcher(self.dense)
        self.reranker = reranker or CrossEncoderReranker()

    def search(
        self,
        query: str,
        mode: Literal["short", "paper"] = "short",
        k: int = 10,
        top_n_for_rerank: int = 50,
        query_paper_id: str | None = None,
        query_card: MethodCard | None = None,
        use_rerank: bool | None = None,
    ) -> list[tuple[str, float, dict]]:
        """Top-k (paper_id, final_score, signal_breakdown).

        ``final_score`` is the CE score when reranked, else the RRF fused score.
        """
        do_rerank = self.use_rerank if use_rerank is None else use_rerank

        dense_results = self.dense.search(query, mode=mode, k=top_n_for_rerank)
        bm25_results = self.bm25.search(query, k=top_n_for_rerank)

        dense_score = dict(dense_results)
        bm25_score = dict(bm25_results)
        dense_rank = {pid: i + 1 for i, (pid, _) in enumerate(dense_results)}
        bm25_rank = {pid: i + 1 for i, (pid, _) in enumerate(bm25_results)}

        candidate_set: set[str] = set(dense_score) | set(bm25_score)

        mm_results: list[tuple[str, float]] = []
        if query_paper_id is not None or query_card is not None:
            mm_candidates = list(candidate_set)
            mm_results = self.method_match.match(
                query_paper_id, query_card, mm_candidates, k=top_n_for_rerank
            )
        mm_score = dict(mm_results)
        mm_rank = {pid: i + 1 for i, (pid, _) in enumerate(mm_results)}
        candidate_set |= set(mm_score)

        fused: list[tuple[str, float, dict]] = []
        signal_ranks = {
            "dense": dense_rank,
            "bm25": bm25_rank,
            "method_match": mm_rank,
        }
        for pid in candidate_set:
            score = 0.0
            for signal, ranks in signal_ranks.items():
                r = ranks.get(pid)
                if r is None:
                    continue
                score += self.weights.get(signal, 0.0) / (RRF_K + r)
            breakdown = {
                "dense": dense_score.get(pid),
                "bm25": bm25_score.get(pid),
                "method_match": mm_score.get(pid),
            }
            fused.append((pid, score, breakdown))
        fused.sort(key=lambda x: x[1], reverse=True)

        topk = fused[:k]
        if not do_rerank or not topk:
            return topk

        reranked = self.reranker.rerank(query, [(pid, s) for pid, s, _ in topk], k=k)
        breakdown_map = {pid: b for pid, _, b in topk}
        out: list[tuple[str, float, dict]] = []
        for pid, ce in reranked:
            b = dict(breakdown_map.get(pid, {}))
            b["cross_encoder"] = ce
            out.append((pid, float(ce), b))
        return out
