"""Unit tests for the Semantic Scholar citation-context adapter (no network)."""

import pytest

from data.sources import s2_contexts
from data.sources.s2_contexts import normalize_doi, s2_paper_id
from schemas import Paper


class _Resp:
    """Minimal stand-in for a requests.Response."""

    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


_S2_ITEMS = [
    {
        "isInfluential": True,
        "intents": ["methodology"],
        "contexts": ["We adopt the BPR loss of [12] to train our model."],
        "citedPaper": {
            "paperId": "abc123",
            "title": "BPR",
            "externalIds": {"DOI": "10.5555/BPR", "MAG": 111},
        },
    },
    {
        "isInfluential": False,
        "intents": [],
        "contexts": [],
        "citedPaper": {"paperId": "def456", "title": "No DOI", "externalIds": {"MAG": 222}},
    },
]


def test_normalize_doi_strips_url_and_lowercases():
    assert normalize_doi("https://doi.org/10.1145/ABC") == "10.1145/abc"
    assert normalize_doi("10.1145/abc") == "10.1145/abc"
    assert normalize_doi(None) is None
    assert normalize_doi("") is None


def test_s2_paper_id_prefers_doi_then_mag():
    assert (
        s2_paper_id(Paper(paper_id="W1", title="t", source_ids={"doi": "10.1/x"})) == "DOI:10.1/x"
    )
    mag_only = Paper(paper_id="W2", title="t", external_ids={"mag": "999"})
    assert s2_paper_id(mag_only) == "MAG:999"
    assert s2_paper_id(Paper(paper_id="W3", title="t")) is None


def test_parse_references_shapes_records():
    records = s2_contexts._parse_references(_S2_ITEMS)
    assert records[0] == {
        "context": "We adopt the BPR loss of [12] to train our model.",
        "contexts": ["We adopt the BPR loss of [12] to train our model."],
        "intents": ["methodology"],
        "is_influential": True,
        "cited_doi": "10.5555/bpr",
        "cited_mag": "111",
        "cited_arxiv": None,
        "cited_corpus_id": None,
        "cited_s2_id": "abc123",
        "cited_title": "BPR",
    }
    # Reference with no DOI/context still parses (matchable by MAG, empty context text).
    assert records[1]["cited_doi"] is None
    assert records[1]["cited_mag"] == "222"
    assert records[1]["context"] == ""


def test_fetch_citation_contexts_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(s2_contexts, "CONTEXTS_DIR", tmp_path)
    calls = {"n": 0}

    def fake_fetch_raw(s2_id):
        calls["n"] += 1
        assert s2_id == "DOI:10.1/x"
        return _S2_ITEMS

    monkeypatch.setattr(s2_contexts, "_fetch_raw", fake_fetch_raw)

    first = s2_contexts.fetch_citation_contexts("W1", s2_id="DOI:10.1/x")
    assert [r["cited_mag"] for r in first] == ["111", "222"]
    assert (tmp_path / "W1.json").exists()

    # Second call hits the on-disk cache: _fetch_raw must not run again.
    second = s2_contexts.fetch_citation_contexts("W1", s2_id="DOI:10.1/x")
    assert second == first
    assert calls["n"] == 1


def test_fetch_citation_contexts_unresolvable_caches_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(s2_contexts, "CONTEXTS_DIR", tmp_path)
    monkeypatch.setattr(s2_contexts, "_corpus_s2_ids", lambda: {})

    records = s2_contexts.fetch_citation_contexts("W404")
    assert records == []
    assert (tmp_path / "W404.json").exists()


def test_fetch_raw_accumulates_pages_and_stops(monkeypatch):
    pages = {
        0: {"data": [{"citedPaper": {"externalIds": {"DOI": "10/a"}}}], "next": 1000},
        1000: {"data": [{"citedPaper": {"externalIds": {"DOI": "10/b"}}}], "next": None},
    }
    seen_offsets = []

    def fake_get(url, params):
        seen_offsets.append(params["offset"])
        return pages[params["offset"]]

    monkeypatch.setattr(s2_contexts, "_get", fake_get)
    monkeypatch.setattr(s2_contexts.time, "sleep", lambda *_: None)

    items = s2_contexts._fetch_raw("DOI:x")
    assert seen_offsets == [0, 1000]  # advanced via `next`, stopped on next=None
    assert len(items) == 2


def test_get_retries_then_succeeds_on_429(monkeypatch):
    calls = {"n": 0}

    def fake_requests_get(url, params, headers, timeout):
        calls["n"] += 1
        return _Resp(429) if calls["n"] == 1 else _Resp(200, {"ok": True})

    monkeypatch.setattr(s2_contexts.requests, "get", fake_requests_get)
    monkeypatch.setattr(s2_contexts.time, "sleep", lambda *_: None)

    assert s2_contexts._get("u", {}) == {"ok": True}
    assert calls["n"] == 2


def test_get_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(s2_contexts.requests, "get", lambda *a, **k: _Resp(429))
    monkeypatch.setattr(s2_contexts.time, "sleep", lambda *_: None)

    with pytest.raises(RuntimeError, match="rate limit"):
        s2_contexts._get("u", {}, max_retries=3)


def test_fetch_failure_leaves_no_cache(tmp_path, monkeypatch):
    # Resumability invariant: a failed crawl must NOT cache, so a later run retries.
    monkeypatch.setattr(s2_contexts, "CONTEXTS_DIR", tmp_path)

    def boom(_):
        raise RuntimeError("S2 down")

    monkeypatch.setattr(s2_contexts, "_fetch_raw", boom)

    with pytest.raises(RuntimeError):
        s2_contexts.fetch_citation_contexts("W1", s2_id="DOI:x")
    assert not (tmp_path / "W1.json").exists()
