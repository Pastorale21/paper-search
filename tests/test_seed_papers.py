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
    assert len(SEED_TITLES) >= 30
    joined = " ".join(SEED_TITLES).lower().replace(" ", "").replace("-", "")
    for must in (
        "lightgcn",
        "ultragcn",
        "kgat",
        "disencdr",
        "ddtcdr",
        "s3-rec",
        "sociallgn",
        "xsimgcl",
        # canonical DiffNet title (regression: the old seed was "Diffusion Network …", wrong)
        "neuralinfluencediffusion",
        # gold-gap additions from docs/corpus_gap_request_for_a.md
        "featuregraphneural",  # FGNN
        "disenhan",  # DisenHAN
        "fairrec",  # FairRec
    ):
        # We don't expect every acronym in the strings, but at least these telltale tokens.
        assert must.replace(" ", "").replace("-", "") in joined


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
        "Collaborative Knowledge-aware Attentive Network for Recommender Systems",
    ]
    with patch("data.sources.seed_papers.openalex.fetch_works", side_effect=fake_fetch):
        resolved, missed = fetch_seed_papers(titles)

    assert len(resolved) == 2
    assert resolved[0].paper_id == "W3004578093"
    assert resolved[1].paper_id == "manual_CKAN"
    assert missed == ["Imaginary Paper That Does Not Exist"]
