"""Unit test for FAISS ranking in search.search (SPECTER2 stubbed out, no model load)."""

import faiss
import numpy as np

import spike.search as S
from schemas import Paper


def test_search_ranks_by_similarity(monkeypatch):
    corpus = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="float32")
    index = faiss.IndexFlatIP(3)
    index.add(corpus)
    ids = ["A", "B", "C"]
    papers = {i: Paper(paper_id=i, title=i) for i in ids}

    monkeypatch.setattr(S, "_index", index)
    monkeypatch.setattr(S, "_ids", ids)
    monkeypatch.setattr(S, "_papers_by_id", papers)
    monkeypatch.setattr(S, "_graph", object())
    monkeypatch.setattr(S, "_ensure_loaded", lambda: None)
    monkeypatch.setattr(
        S, "embed_short_query", lambda texts: np.array([[0.9, 0.1, 0.0]], dtype="float32")
    )

    res = S.search("q", mode="short", top_k=2)
    assert [p.paper_id for p, _ in res] == ["A", "B"]
    assert res[0][1] > res[1][1]
