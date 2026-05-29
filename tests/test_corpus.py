"""Unit tests for corpus dedup + abstract filtering (no network)."""

from data.corpus import dedupe, filter_with_abstract, normalize_title
from schemas import Paper


def _paper(pid: str, title: str, oa: str | None = None, abstract: str | None = "abs") -> Paper:
    return Paper(
        paper_id=pid,
        title=title,
        abstract=abstract,
        source_ids={"openalex": oa} if oa else {},
    )


def test_dedup_by_openalex_id():
    papers = [
        _paper("a", "LightGCN", oa="W1"),
        _paper("b", "Different Paper", oa="W2"),
        _paper("c", "LightGCN variant", oa="W1"),  # same OA id as the first
    ]
    kept, removed = dedupe(papers)
    assert len(kept) == 2
    assert removed == 1


def test_dedup_by_normalized_title():
    papers = [
        _paper("a", "Graph Neural Networks for Social Recommendation", oa="W1"),
        _paper("b", "graph neural networks for social recommendation!!", oa="W2"),
    ]
    assert normalize_title(papers[0].title) == normalize_title(papers[1].title)
    kept, removed = dedupe(papers)
    assert len(kept) == 1
    assert removed == 1


def test_filter_no_abstract():
    papers = [
        _paper("a", "Has abstract", abstract="non-empty"),
        _paper("b", "Empty abstract", abstract=""),
        _paper("c", "Also has abstract", abstract="present"),
    ]
    assert len(filter_with_abstract(papers)) == 2
