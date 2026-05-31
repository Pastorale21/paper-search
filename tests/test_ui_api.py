"""Unit tests for ``ui/api.py`` — mocked retrievers + mocked LLM client; no live calls."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from schemas import MethodCard


def _mock_papers():
    return {
        "W1": {
            "paper_id": "W1",
            "title": "Paper 1",
            "year": 2020,
            "citation_count": 100,
            "abstract": "abs1",
        },
        "W2": {
            "paper_id": "W2",
            "title": "Paper 2",
            "year": 2021,
            "citation_count": 50,
            "abstract": "abs2",
        },
    }


def _install_caches(
    monkeypatch,
    papers,
    *,
    hybrid=None,
    dense=None,
    bm25=None,
    matcher=None,
    reranker=None,
    cards=None,
):
    """Bypass Streamlit caches and replace each get_* loader with a thunk over a mock."""
    from ui import api

    cards = cards or {}
    monkeypatch.setattr(api, "get_papers_by_id", lambda: papers)
    if hybrid is not None:
        monkeypatch.setattr(api, "get_hybrid", lambda: hybrid)
    if dense is not None:
        monkeypatch.setattr(api, "get_dense", lambda: dense)
    if bm25 is not None:
        monkeypatch.setattr(api, "get_bm25", lambda: bm25)
    if matcher is not None:
        monkeypatch.setattr(api, "get_matcher", lambda: matcher)
    if reranker is not None:
        monkeypatch.setattr(api, "get_reranker", lambda: reranker)
    # Method-card loader returns dict-defined cards.
    monkeypatch.setattr(api, "load_method_card", lambda pid: cards.get(pid))
    monkeypatch.setattr(api, "has_method_card", lambda pid: pid in cards)


def test_api_search_returns_enriched_results(monkeypatch):
    """``search(method='hybrid')`` wraps every hit with paper, score, breakdown, method_card."""
    from ui import api

    papers = _mock_papers()
    card_w1 = MethodCard(paper_id="W1", task="topK rec", backbone="LightGCN")
    hybrid = MagicMock()
    hybrid.search.return_value = [
        ("W1", 0.95, {"dense": 0.95, "bm25": 0.12, "method_match": 0.81}),
        ("W2", 0.81, {"dense": 0.88, "bm25": None, "method_match": 0.74}),
    ]
    _install_caches(monkeypatch, papers, hybrid=hybrid, cards={"W1": card_w1})

    results = api.search("graph contrastive", mode="short", method="hybrid", k=2)
    assert len(results) == 2
    assert {r["paper_id"] for r in results} == {"W1", "W2"}
    first = next(r for r in results if r["paper_id"] == "W1")
    assert first["paper"]["title"] == "Paper 1"
    assert first["score"] == 0.95
    assert first["signal_breakdown"]["method_match"] == 0.81
    assert first["method_card"].task == "topK rec"


def test_api_handles_missing_method_card(monkeypatch):
    """Result for paper without a cached card has ``method_card=None``, not a crash."""
    from ui import api

    papers = _mock_papers()
    hybrid = MagicMock()
    hybrid.search.return_value = [("W2", 0.70, {"dense": 0.70, "bm25": None, "method_match": None})]
    _install_caches(monkeypatch, papers, hybrid=hybrid, cards={})  # no cards on disk

    results = api.search("any query", mode="short", method="hybrid", k=1)
    assert len(results) == 1
    assert results[0]["method_card"] is None
    assert results[0]["paper"]["title"] == "Paper 2"


def test_api_search_bm25_method_has_none_breakdown(monkeypatch):
    """Non-hybrid methods return signal_breakdown=None so the UI can branch on it."""
    from ui import api

    papers = _mock_papers()
    bm25 = MagicMock()
    bm25.search.return_value = [("W1", 12.3), ("W2", 8.1)]
    _install_caches(monkeypatch, papers, bm25=bm25)

    results = api.search("graph", mode="short", method="bm25", k=2)
    assert all(r["signal_breakdown"] is None for r in results)


def test_related_work_prompt_build_messages_and_parse():
    """build_messages() requires no network; parser handles fenced JSON + missing keys."""
    from ui.related_work_prompt import build_messages, extract_citation_markers, parse_llm_response

    retrieved = [
        {
            "paper": {"paper_id": "W1", "title": "LightGCN", "year": 2020},
            "method_card": MethodCard(
                paper_id="W1",
                task="topK rec",
                backbone="LightGCN",
                loss="BPR",
                key_idea="simplified GCN",
            ),
        },
        {
            "paper": {"paper_id": "W2", "title": "NGCF", "year": 2019},
            "method_card": None,
        },
    ]
    messages = build_messages("My idea is a hybrid GCN.", retrieved, target_words=200)
    assert messages[0]["role"] == "system"
    assert "JSON" in messages[0]["content"]  # DeepSeek json_object requirement
    assert "LightGCN" in messages[-1]["content"]
    assert "[2] NGCF" in messages[-1]["content"]
    assert "(no method card on disk" in messages[-1]["content"]  # graceful for W2

    # Parser handles ```json fences and missing fields.
    raw_fenced = (
        "```json\n"
        + json.dumps(
            {
                "paragraph": "Foo [1].",
                "references": [{"n": 1, "paper_id": "W1", "one_line_reason": "test"}],
            }
        )
        + "\n```"
    )
    parsed = parse_llm_response(raw_fenced)
    assert parsed["paragraph"] == "Foo [1]."
    assert parsed["references"][0]["paper_id"] == "W1"

    # Citation-marker extractor preserves order and dedupes.
    assert extract_citation_markers("Foo [1], bar [2], also [1] again.") == [1, 2]


def test_related_work_parse_fallback_on_malformed_json():
    """Malformed JSON falls back to raw paragraph + empty refs + _parse_error flag."""
    from ui.related_work_prompt import parse_llm_response

    parsed = parse_llm_response("not json at all [1] hi")
    assert parsed["_parse_error"] is True
    assert parsed["paragraph"] == "not json at all [1] hi"
    assert parsed["references"] == []


def test_api_get_llm_client_raises_without_key(monkeypatch):
    """``get_llm_client`` must NOT silently construct or call out when LLM_API_KEY is missing."""
    from nlp import config as nlp_config
    from ui import api

    monkeypatch.setattr(nlp_config, "LLM_API_KEY", None)
    # The cached resource may already have been built in another test; bust the cache.
    try:
        api.get_llm_client.clear()
    except AttributeError:
        pass

    import pytest

    with pytest.raises(RuntimeError, match="LLM_API_KEY not set"):
        api.get_llm_client()


def test_filter_survey_titles_drops_review_papers(monkeypatch):
    """The Tab 3 belt-and-suspenders filter drops papers with 'survey' / 'review' in the title."""
    from retrieval.graph_reason import ReasoningResult
    from ui import api

    papers = {
        "S1": {"paper_id": "S1", "title": "A Survey of Graph Neural Networks", "year": 2023},
        "R1": {"paper_id": "R1", "title": "Real Mechanism-Distant Paper", "year": 2021},
        "R2": {"paper_id": "R2", "title": "Systematic Literature Review of Rec", "year": 2024},
    }
    results = [
        ReasoningResult(paper_id="S1", score=1.0, paths=[]),
        ReasoningResult(paper_id="R1", score=0.6, paths=[]),
        ReasoningResult(paper_id="R2", score=0.9, paths=[]),
    ]
    kept = api.filter_survey_titles(results, papers)
    pids = {r.paper_id for r in kept}
    assert pids == {"R1"}
