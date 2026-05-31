"""Unit tests for eval metrics + title resolver (no network, no model loads)."""

from __future__ import annotations

import math

from eval.gold_set import DEFAULT_ALIASES, TitleResolver
from eval.metrics import mrr, ndcg_at_k, recall_at_k


def test_ndcg_perfect():
    preds = ["a", "b", "c", "d", "e"]
    assert ndcg_at_k(preds, {"a", "b", "c", "d", "e"}, k=5) == 1.0


def test_ndcg_empty_intersection():
    preds = ["x", "y", "z"]
    assert ndcg_at_k(preds, {"a", "b"}, k=5) == 0.0


def test_ndcg_partial_known_value():
    # 2 gold in top-5 at positions 2 and 4: DCG = 1/log2(3) + 1/log2(5).
    preds = ["miss", "gold1", "miss", "gold2", "miss"]
    gold = {"gold1", "gold2"}
    dcg = 1.0 / math.log2(3) + 1.0 / math.log2(5)
    # IDCG for 2 gold: 1/log2(2) + 1/log2(3) = 1 + 1/log2(3).
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    expected = dcg / idcg
    assert math.isclose(ndcg_at_k(preds, gold, k=5), expected, rel_tol=1e-9)


def test_mrr_first_position():
    assert mrr(["gold", "x", "y"], {"gold"}) == 1.0


def test_mrr_third_position():
    assert math.isclose(mrr(["x", "y", "gold"], {"gold"}), 1.0 / 3.0)


def test_mrr_no_match():
    assert mrr(["x", "y", "z"], {"gold"}) == 0.0


def test_recall_at_10_partial():
    preds = ["a", "b", "c", "d", "x", "x", "x", "x", "x", "x"]
    assert recall_at_k(preds, {"a", "b", "c", "f", "g"}, k=10) == 0.6


def test_recall_at_k_empty_gold():
    assert recall_at_k(["a", "b"], set(), k=10) == 0.0


# --- TitleResolver -----------------------------------------------------------------------


def _papers(*pairs: tuple[str, str]) -> list[dict]:
    return [{"paper_id": pid, "title": title} for pid, title in pairs]


def test_title_resolver_exact_normalized_match():
    papers = _papers(
        ("W1", "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"),
        ("W2", "Some Unrelated Paper"),
    )
    resolver = TitleResolver(papers, aliases={})
    # Full title (lowercased, punctuation stripped) matches exactly.
    assert (
        resolver.resolve(
            "LightGCN Simplifying and Powering Graph Convolution Network for Recommendation"
        )
        == "W1"
    )


def test_title_resolver_colon_prefix_abbreviation():
    papers = _papers(
        ("W1", "LightGCN: Simplifying GCN for CF"),
    )
    resolver = TitleResolver(papers, aliases={})
    assert resolver.resolve("LightGCN") == "W1"


def test_title_resolver_strips_html_tags():
    papers = _papers(
        ("W1", "SiReN: Sign-Aware Recommendation via Networks"),
    )
    resolver = TitleResolver(papers, aliases={})
    assert resolver.resolve("<i>SiReN</i>") == "W1"


def test_title_resolver_alias_map_substring():
    # The alias maps "ngcf" to a substring of the canonical title.
    papers = _papers(
        ("W2945827670", "Neural Graph Collaborative Filtering"),
    )
    resolver = TitleResolver(papers)  # uses DEFAULT_ALIASES
    assert "ngcf" in DEFAULT_ALIASES
    assert resolver.resolve("NGCF") == "W2945827670"


def test_title_resolver_substring_match():
    # Gold "Graph Convolutional Matrix Completion" appears as a substring of the title.
    papers = _papers(
        ("W1", "Graph Convolutional Matrix Completion (a 2017 paper)"),
    )
    resolver = TitleResolver(papers, aliases={})
    assert resolver.resolve("Graph Convolutional Matrix Completion") == "W1"


def test_title_resolver_fuzzy_threshold():
    papers = _papers(
        ("W1", "Hypergraph Contrastive Collaborative Filtering"),
    )
    resolver = TitleResolver(papers, aliases={})
    # "Hypergraph Constrastive" — one typo close enough for difflib at 0.85 cutoff
    # AND the substring path may also catch it; either way it resolves.
    assert resolver.resolve("Hypergraph Contrastive Collaborativ Filtering") == "W1"


def test_title_resolver_unresolvable_returns_none():
    papers = _papers(
        ("W1", "Something Totally Unrelated"),
    )
    resolver = TitleResolver(papers, aliases={})
    assert resolver.resolve("UnknownNonExistentPaper") is None


def test_title_resolver_logs_non_exact_matches():
    papers = _papers(
        ("W1", "Neural Graph Collaborative Filtering"),
    )
    resolver = TitleResolver(papers)
    resolver.resolve("NGCF")
    assert any(strategy == "alias" for _, _, strategy in resolver.fuzzy_log)
