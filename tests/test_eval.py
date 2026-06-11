"""Unit tests for eval metrics + title resolver (no network, no model loads)."""

from __future__ import annotations

import math
import sys

from eval.error_analysis import print_report
from eval.gold_set import DEFAULT_ALIASES, TitleResolver, _norm
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


def test_title_resolver_distinguishes_plus_suffix_alias():
    papers = _papers(
        ("W1", "A Neural Influence Diffusion Model for Social Recommendation"),
        ("W2", "DiffNet++: A Neural Influence and Interest Diffusion Network"),
    )
    resolver = TitleResolver(papers)
    assert resolver.resolve("DiffNet") == "W1"
    assert resolver.resolve("DiffNet++") == "W2"


def test_diffnet_plusplus_alias_key_is_reachable():
    # Regression: `_norm("DiffNet++")` collapses "++" to a single "plusplus" token, so the alias
    # key must be "diffnet plusplus" to ever fire. The old "diffnet plus plus" key was dead code.
    assert _norm("DiffNet++") in DEFAULT_ALIASES


def test_title_resolver_alias_miss_blocks_acronym_substring_fallback():
    papers = _papers(
        ("W1", "SelfGNN: Self-Supervised Graph Neural Networks for Sequential Recommendation"),
    )
    resolver = TitleResolver(
        papers,
        aliases={"fgnn": "feature graph neural networks for session based recommendation"},
    )
    assert resolver.resolve("FGNN") is None


def test_error_analysis_report_prints_low_score_context(capsys):
    payload = {
        "query_meta": {
            "Qx": {
                "text": "session graph query",
                "notes": "diagnostic case",
            }
        },
        "resolved_gold_per_query": {"Qx": ["G1"]},
        "unresolved_per_query": {"Qx": ["MissingGold"]},
        "per_query": [
            {
                "query_id": "Qx",
                "mode": "short",
                "method": "dense",
                "metrics": {"ndcg@5": 0.0, "mrr": 0.0, "recall@10": 0.0},
                "top10": ["P1", "G1"],
            }
        ],
    }
    titles = {"P1": "Wrong Paper", "G1": "Gold Paper"}

    print_report(payload, titles, threshold=0.30)

    out = capsys.readouterr().out
    assert "Low-score eval cases" in out
    assert "MissingGold" in out
    assert "  2. * G1 | Gold Paper" in out


def test_eval_registers_method_match_norm_variants(monkeypatch):
    """Variants must be in METHODS AND accepted by the separate --method CLI choices tuple."""
    import eval.run as run_mod

    assert "method_match_norm" in run_mod.METHODS
    assert "method_match_norm2" in run_mod.METHODS

    # The CLI choices is a second hardcoded list; verify it stays in sync by parsing an argv
    # that selects a variant (run() stubbed so no real eval executes).
    monkeypatch.setattr(run_mod, "run", lambda methods, output: 0)
    monkeypatch.setattr(sys, "argv", ["eval.run", "--method", "method_match_norm2"])
    assert run_mod.main() == 0  # raises SystemExit(2) if the choice were rejected
