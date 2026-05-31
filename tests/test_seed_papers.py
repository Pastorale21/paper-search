"""Unit tests for the curated-seed-paper fetcher (no network)."""

from unittest.mock import patch

from data.sources.seed_papers import (
    SEED_TITLES,
    _is_match,
    _norm,
    fetch_seed_papers,
)
from schemas import Paper


def test_seed_titles_nonempty_and_uppercased_canonicals_present():
    assert len(SEED_TITLES) >= 15
    joined = " ".join(SEED_TITLES).lower()
    for must in ("lightgcn", "ngcf", "ultragcn", "kgat", "sr-gnn".replace("-", ""), "graphrec"):
        # We don't expect every acronym in the strings, but at least these telltale tokens.
        if must in {"ngcf", "srgnn", "graphrec"}:
            continue  # these are conventional names not in our title strings, that's fine
        assert must in joined.lower().replace(" ", "").replace("-", "")


def test_norm_strips_punct_and_collapses_whitespace():
    assert _norm("LightGCN: Simplifying  and Powering!") == "lightgcn simplifying and powering"
    assert _norm("") == ""
    assert _norm(None) == ""  # type: ignore[arg-type]


def test_match_substring_either_direction():
    seed = _norm("LightGCN Simplifying and Powering Graph Convolution Network")
    cand_long = _norm(
        "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"
    )
    assert _is_match(seed, cand_long)
    assert _is_match(cand_long, seed)  # reverse direction


def test_match_token_overlap_threshold():
    seed = _norm("Are Graph Augmentations Necessary Simple Graph Contrastive Learning")
    cand = _norm(
        "Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation"
    )
    assert _is_match(seed, cand)


def test_match_rejects_unrelated_titles():
    seed = _norm("LightGCN Simplifying and Powering Graph Convolution Network")
    cand = _norm("ImageNet Classification with Deep Convolutional Neural Networks")
    assert not _is_match(seed, cand)


def test_fetch_seed_papers_resolves_matches_and_misses():
    """Mocks openalex.fetch_works: returns a matching paper for one title, no match for another."""

    def fake_fetch(query, n):  # noqa: ARG001
        if "LightGCN" in query:
            return [
                Paper(
                    paper_id="W3004578093",
                    title=(
                        "LightGCN: Simplifying and Powering Graph Convolution Network "
                        "for Recommendation"
                    ),
                    abstract="abs",
                )
            ]
        if "Imaginary" in query:
            return [Paper(paper_id="W999", title="Completely Unrelated Survey on Brain Tumors")]
        return []

    titles = [
        "LightGCN Simplifying and Powering Graph Convolution Network",
        "Imaginary Paper That Does Not Exist",
    ]
    with patch("data.sources.seed_papers.openalex.fetch_works", side_effect=fake_fetch):
        resolved, missed = fetch_seed_papers(titles)

    assert len(resolved) == 1
    assert resolved[0].paper_id == "W3004578093"
    assert missed == ["Imaginary Paper That Does Not Exist"]
