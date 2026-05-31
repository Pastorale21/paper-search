"""Unit tests for retrieval/graph_reason.py (no real SPECTER2 / FAISS / graph loads)."""

from __future__ import annotations

from unittest.mock import MagicMock

import networkx as nx
import pytest

from retrieval import graph_reason


@pytest.fixture(autouse=True)
def _clear_loader_caches():
    """Reset the module-level lru_cache loaders so each test sees its own monkeypatches."""
    graph_reason._load_graph.cache_clear()
    graph_reason._papers_by_id.cache_clear()
    graph_reason._matcher.cache_clear()
    yield
    graph_reason._load_graph.cache_clear()
    graph_reason._papers_by_id.cache_clear()
    graph_reason._matcher.cache_clear()


def _install(monkeypatch, graph: nx.DiGraph, papers: dict, matcher) -> None:
    """Install a tiny in-memory graph + papers + matcher for the queries to consume."""
    monkeypatch.setattr(graph_reason, "_load_graph", lambda: graph)
    monkeypatch.setattr(graph_reason, "_papers_by_id", lambda: papers)
    monkeypatch.setattr(graph_reason, "_matcher", lambda: matcher)


def _matcher_with(sims: dict[tuple[str, str], float], cards: dict | None = None):
    """Build a MagicMock matcher whose .similarity / .match honor a tiny lookup table."""
    m = MagicMock()
    m.cards = (
        cards
        if cards is not None
        else {pid: object() for pid in {p for pair in sims for p in pair}}
    )
    m.similarity.side_effect = lambda a, b: sims.get((a, b), sims.get((b, a), 0.0))

    def fake_match(anchor, query_card, candidates, k):
        scored = [
            (pid, sims.get((anchor, pid), sims.get((pid, anchor), 0.0)))
            for pid in candidates
            if pid != anchor
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    m.match.side_effect = fake_match
    return m


def test_find_ancestors_returns_paths_in_correct_direction(monkeypatch):
    """A->B->C: out-edges from A reach B (hop 1) and C (hop 2)."""
    g = nx.DiGraph()
    g.add_edges_from([("A", "B"), ("B", "C")])
    papers = {pid: {"paper_id": pid, "title": pid, "abstract": "", "year": 2020} for pid in "ABC"}
    matcher = _matcher_with({("A", "B"): 0.9, ("A", "C"): 0.7})
    _install(monkeypatch, g, papers, matcher)

    results = graph_reason.find_ancestors("A", max_hops=3, k=10)
    pids = [r.paper_id for r in results]
    assert "B" in pids and "C" in pids
    # B is reached in 1 hop, C in 2 hops; B should rank above C all else equal.
    assert pids.index("B") < pids.index("C")


def test_find_ancestors_respects_max_hops(monkeypatch):
    """max_hops=1 returns only direct successors."""
    g = nx.DiGraph()
    g.add_edges_from([("A", "B"), ("B", "C"), ("B", "D")])
    papers = {pid: {"paper_id": pid, "title": pid, "abstract": "", "year": 2020} for pid in "ABCD"}
    matcher = _matcher_with({("A", "B"): 0.5, ("A", "C"): 0.5, ("A", "D"): 0.5})
    _install(monkeypatch, g, papers, matcher)

    results = graph_reason.find_ancestors("A", max_hops=1, k=10)
    pids = {r.paper_id for r in results}
    assert pids == {"B"}


def test_find_opposing_filters_to_comparison_intent(monkeypatch):
    """With real intents on disk, only comparison-intent edges survive."""
    g = nx.DiGraph()
    g.add_edge("A", "B", intent="comparison")
    g.add_edge("A", "C", intent="method")
    g.add_edge("D", "A", intent="comparison")
    papers = {pid: {"paper_id": pid, "title": pid, "abstract": "", "year": 2020} for pid in "ABCD"}
    matcher = _matcher_with({("A", "B"): 0.2, ("A", "C"): 0.9, ("A", "D"): 0.3})
    _install(monkeypatch, g, papers, matcher)

    results = graph_reason.find_opposing("A", k=10)
    pids = {r.paper_id for r in results}
    # B (out, comparison) and D (in, comparison) are kept; C (out, method) is dropped.
    assert pids == {"B", "D"}


def test_find_opposing_fallback_ranks_by_mechanism_distance(monkeypatch):
    """LIVE PATH: no comparison-intent edges -> rank all 1-hop neighbors by mechanism distance."""
    g = nx.DiGraph()
    # Default intent ("background") everywhere — no comparison labels.
    g.add_edge("A", "B")
    g.add_edge("A", "C")
    g.add_edge("D", "A")
    papers = {pid: {"paper_id": pid, "title": pid, "abstract": "", "year": 2020} for pid in "ABCD"}
    # B is very mechanism-distant from A (sim=0.1), D is moderately distant (sim=0.5),
    # C is mechanism-similar (sim=0.95). Expected ordering: B (distance 0.9), D (0.5), C (0.05).
    matcher = _matcher_with({("A", "B"): 0.1, ("A", "C"): 0.95, ("A", "D"): 0.5})
    _install(monkeypatch, g, papers, matcher)

    results = graph_reason.find_opposing("A", k=10)
    pids = [r.paper_id for r in results]
    assert pids[0] == "B"  # most mechanism-distant
    assert pids.index("D") < pids.index("C")  # D more distant than C
    # The explanation should flag the fallback so the UI / caller knows.
    assert "fallback" in results[0].paths[0].explanation.lower()


def test_find_cross_domain_filters_by_sub_area(monkeypatch):
    """Cross-domain query from a KG paper returns only papers in OTHER sub-areas."""
    g = nx.DiGraph()
    # Three KG papers, two social papers; A is the anchor.
    papers = {
        "A": {
            "paper_id": "A",
            "title": "KGAT",
            "abstract": "knowledge graph attention network for recommendation",
            "year": 2019,
        },
        "K1": {
            "paper_id": "K1",
            "title": "KGIN",
            "abstract": "learning intents behind interactions knowledge-aware",
            "year": 2021,
        },
        "K2": {
            "paper_id": "K2",
            "title": "KGCN",
            "abstract": "knowledge graph convolutional networks",
            "year": 2019,
        },
        "S1": {
            "paper_id": "S1",
            "title": "DiffNet",
            "abstract": "social recommendation diffusion network",
            "year": 2019,
        },
        "S2": {
            "paper_id": "S2",
            "title": "GraphRec",
            "abstract": "graph neural networks for social recommendation",
            "year": 2019,
        },
    }
    sims = {("A", x): 0.9 for x in ("K1", "K2", "S1", "S2")}
    matcher = _matcher_with(sims, cards={pid: object() for pid in papers})
    _install(monkeypatch, g, papers, matcher)

    results = graph_reason.find_cross_domain_same_mechanism("A", k=10)
    pids = {r.paper_id for r in results}
    # KG papers (same sub-area as anchor) must be filtered out; only social papers remain.
    assert pids == {"S1", "S2"}


def test_path_explanation_includes_intent_label(monkeypatch):
    """GraphPath.explanation should carry a recognizable edge-label substring."""
    g = nx.DiGraph()
    g.add_edge("A", "B", intent="method")
    papers = {pid: {"paper_id": pid, "title": pid, "abstract": "", "year": 2020} for pid in "AB"}
    matcher = _matcher_with({("A", "B"): 0.9})
    _install(monkeypatch, g, papers, matcher)

    results = graph_reason.find_ancestors("A", max_hops=1, k=5)
    assert results
    assert "method-citation" in results[0].paths[0].explanation
