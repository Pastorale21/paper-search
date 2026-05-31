"""Retrieval metrics: nDCG@k, MRR, Recall@k. Pure functions, binary relevance, no deps."""

from __future__ import annotations

import math


def ndcg_at_k(predicted_ids: list[str], gold_ids: set[str], k: int = 5) -> float:
    """nDCG@k with binary relevance (1 if predicted_id in gold_ids else 0).

    Returns 0.0 when either ``gold_ids`` is empty or no gold paper appears in the top-k.
    Positions are 1-indexed inside the log: ``rel_i / log2(i + 1)``.
    """
    if not gold_ids or k <= 0:
        return 0.0
    top = predicted_ids[:k]
    dcg = 0.0
    for i, pid in enumerate(top, start=1):
        if pid in gold_ids:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(k, len(gold_ids))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def mrr(predicted_ids: list[str], gold_ids: set[str]) -> float:
    """Mean Reciprocal Rank: 1/(rank of first gold paper), or 0.0 if none found."""
    if not gold_ids:
        return 0.0
    for i, pid in enumerate(predicted_ids, start=1):
        if pid in gold_ids:
            return 1.0 / i
    return 0.0


def recall_at_k(predicted_ids: list[str], gold_ids: set[str], k: int = 10) -> float:
    """|predicted[:k] ∩ gold| / |gold|, or 0.0 if gold is empty."""
    if not gold_ids:
        return 0.0
    hits = sum(1 for pid in predicted_ids[:k] if pid in gold_ids)
    return hits / len(gold_ids)
