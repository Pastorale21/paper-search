"""Unit tests for corpus dedup + abstract filtering (no network)."""

from data.corpus import (
    _select_balanced,
    dedupe,
    filter_with_abstract,
    is_gnn_recsys,
    normalize_title,
    titles_are_duplicates,
)
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


def test_dedup_by_short_long_title_variant():
    papers = [
        _paper("a", "LightGCN", oa="W3004578093", abstract="short"),
        _paper(
            "b",
            "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation",
            oa="W3045200674",
            abstract="much richer abstract about graph collaborative filtering",
        ),
    ]
    kept, removed = dedupe(papers)
    assert len(kept) == 1
    assert removed == 1
    assert kept[0].paper_id == "b"


def test_title_variant_does_not_merge_generic_containment():
    left = normalize_title("Graph Neural Networks for Recommendation")
    right = normalize_title("Graph Neural Networks for Social Recommendation")
    assert not titles_are_duplicates(left, right)


def test_select_balanced_does_not_starve_late_subarea():
    early = [_paper(f"a{i}", f"Early {i}") for i in range(3)]
    late = [_paper("b1", "Late 1"), _paper("b2", "Late 2")]
    origin = {
        normalize_title(p.title): "graph neural network collaborative filtering" for p in early
    }
    origin.update(
        {normalize_title(p.title): "social recommendation with graph neural network" for p in late}
    )

    selected = _select_balanced([], early + late, target=4, origin=origin)
    assert any(p.title.startswith("Late") for p in selected)


def test_filter_no_abstract():
    papers = [
        _paper("a", "Has abstract", abstract="non-empty"),
        _paper("b", "Empty abstract", abstract=""),
        _paper("c", "Also has abstract", abstract="present"),
    ]
    assert len(filter_with_abstract(papers)) == 2


def test_gnn_recsys_filter_keeps_topical():
    p = _paper(
        "a",
        "LightGCN: Simplifying Graph Convolution Network for Recommendation",
        abstract=(
            "A graph neural network for collaborative filtering that propagates user-item "
            "embeddings over the interaction graph for top-K recommendation."
        ),
    )
    assert is_gnn_recsys(p)


def test_gnn_recsys_filter_drops_offtopic():
    p = _paper(
        "b",
        "ImageNet Classification with Deep Convolutional Neural Networks",
        abstract=(
            "We train a large deep convolutional neural network to classify 1.2 million "
            "high-resolution images into 1000 categories."
        ),
    )
    assert not is_gnn_recsys(p)
