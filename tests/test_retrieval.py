"""Unit tests for the retrieval layer (no real SPECTER2 / CrossEncoder loads)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from schemas import MethodCard, Paper


def test_dense_search_returns_k_results():
    """DenseRetriever.search is a thin wrapper around spike.search.search — verify shape."""
    from retrieval.dense import DenseRetriever

    fake = [(Paper(paper_id=f"W{i}", title=f"T{i}"), 0.9 - i * 0.01) for i in range(5)]
    with patch("retrieval.dense.search.search", return_value=fake):
        r = DenseRetriever().search("q", mode="short", k=5)
    assert len(r) == 5
    assert r[0] == ("W0", pytest.approx(0.9))
    assert r[4] == ("W4", pytest.approx(0.86))


def test_bm25_search_handles_empty_query():
    """Empty query → [] without crashing; non-empty query still works."""
    from rank_bm25 import BM25Okapi

    from retrieval.bm25 import BM25Retriever

    r = BM25Retriever.__new__(BM25Retriever)
    r._bm25 = BM25Okapi([["graph", "neural", "network"], ["matrix", "factorization"]])
    r._ids = ["W1", "W2"]
    assert r.search("") == []
    assert r.search("   ") == []
    # Non-empty query returns results without crashing (small-corpus BM25 IDF can yield
    # zero scores; we only assert the contract here, not the ordering).
    hits = r.search("graph", k=2)
    assert len(hits) == 2


def _make_matcher_with(cards: dict[str, MethodCard], cache: dict[str, np.ndarray]):
    """Construct a MethodCardMatcher with mocked embedder + preloaded cache (no file IO)."""
    from retrieval.method_match import MethodCardMatcher

    m = MethodCardMatcher.__new__(MethodCardMatcher)
    m.embedder = MagicMock()
    m.cards = cards
    m._cache = cache
    m._embed_batch = lambda texts: np.array([[1.0, 0.0]] * len(texts), dtype="float32")
    m._get_or_embed_query_field = lambda v: np.array([1.0, 0.0], dtype="float32") if v else None
    return m


def test_method_match_skips_papers_without_card():
    """A candidate without a cached method card scores 0; doesn't crash the call."""
    cards = {
        "W1": MethodCard(paper_id="W1", task="topK", backbone="GCN"),
        "W2": MethodCard(paper_id="W2", task="topK", backbone="GCN"),
    }
    cache = {
        "W1::task": np.array([1.0, 0.0], dtype="float32"),
        "W1::backbone": np.array([1.0, 0.0], dtype="float32"),
        "W2::task": np.array([1.0, 0.0], dtype="float32"),
        "W2::backbone": np.array([1.0, 0.0], dtype="float32"),
    }
    matcher = _make_matcher_with(cards, cache)
    q = MethodCard(paper_id="Q", task="topK", backbone="GCN")
    result = matcher.match(None, q, ["W1", "W2", "W3"], k=10)
    pids = {pid for pid, _ in result}
    assert {"W1", "W2", "W3"} == pids  # W3 included but with score 0
    w3 = next(s for p, s in result if p == "W3")
    assert w3 == 0.0


def test_method_match_excludes_self_when_anchored_by_id():
    """When anchored by query_paper_id, that paper is excluded from candidates (no self-match)."""
    cards = {
        "W_anchor": MethodCard(paper_id="W_anchor", task="t", backbone="b"),
        "W_other": MethodCard(paper_id="W_other", task="t", backbone="b"),
    }
    cache = {
        "W_anchor::task": np.array([1.0, 0.0], dtype="float32"),
        "W_anchor::backbone": np.array([1.0, 0.0], dtype="float32"),
        "W_other::task": np.array([1.0, 0.0], dtype="float32"),
        "W_other::backbone": np.array([1.0, 0.0], dtype="float32"),
    }
    matcher = _make_matcher_with(cards, cache)
    # Anchor by id; even though W_anchor is in candidates, it must NOT appear in results.
    result = matcher.match("W_anchor", None, ["W_anchor", "W_other"], k=10)
    pids = {pid for pid, _ in result}
    assert "W_anchor" not in pids
    assert "W_other" in pids


def test_method_match_field_weights_applied(monkeypatch):
    """With FIELD_WEIGHTS={'task':1.0, others:0}, only the task field drives the ranking."""
    from retrieval import method_match

    monkeypatch.setattr(
        method_match,
        "FIELD_WEIGHTS",
        {"task": 1.0, "backbone": 0.0, "loss": 0.0, "key_idea": 0.0},
    )

    cards = {
        "W_good": MethodCard(paper_id="W_good", task="x", backbone="y"),
        "W_bad": MethodCard(paper_id="W_bad", task="y", backbone="x"),
    }
    cache = {
        # query (via _get_or_embed_query_field) is [1,0]; W_good task aligns, W_bad does not
        "W_good::task": np.array([1.0, 0.0], dtype="float32"),
        "W_good::backbone": np.array([0.0, 1.0], dtype="float32"),
        "W_bad::task": np.array([0.0, 1.0], dtype="float32"),
        "W_bad::backbone": np.array([1.0, 0.0], dtype="float32"),
    }
    matcher = _make_matcher_with(cards, cache)
    q = MethodCard(paper_id="Q", task="x", backbone="x")
    result = matcher.match(None, q, ["W_good", "W_bad"], k=10)
    assert result[0][0] == "W_good"
    assert result[0][1] > result[1][1]


def test_hybrid_signal_breakdown_present():
    """HybridRetriever output carries a signal_breakdown dict with all three signals."""
    from retrieval.hybrid import HybridRetriever

    fake_dense = MagicMock()
    fake_dense.search.return_value = [("W1", 0.95), ("W2", 0.93)]
    fake_bm25 = MagicMock()
    fake_bm25.search.return_value = [("W1", 12.3), ("W3", 8.1)]
    fake_mm = MagicMock()
    fake_mm.match.return_value = [("W1", 0.81), ("W2", 0.50)]

    h = HybridRetriever(
        dense=fake_dense,
        bm25=fake_bm25,
        method_match=fake_mm,
        reranker=MagicMock(),
        use_rerank=False,
    )
    results = h.search("q", mode="short", k=5, query_paper_id="Qpid")
    assert results
    for _pid, _score, br in results:
        assert isinstance(br, dict)
        assert set(br.keys()) == {"dense", "bm25", "method_match"}
