"""Build the multi-sub-area GNN-recsys corpus: fetch per sub-area, dedupe, filter, persist.

CLI: `uv run python -m data.corpus --target 500 --per-query 100 [--force]`
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter

from schemas import Paper
from spike import config

from .sources import openalex

SUB_AREAS = [
    "graph neural network collaborative filtering",
    "graph contrastive learning for recommendation",
    "cross-domain recommendation with graph neural network",
    "knowledge graph enhanced recommendation",
    "session-based recommendation with graph neural network",
    "social recommendation with graph neural network",
]


def normalize_title(title: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — for fuzzy title dedup."""
    cleaned = re.sub(r"[^\w\s]", " ", (title or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _completeness(p: Paper) -> tuple:
    """Sort key for picking the richer record when two entries collide (higher is better)."""
    return (
        bool(p.abstract),
        len(p.abstract or ""),
        len(p.references),
        len(p.authors),
        len(p.source_ids),
    )


def dedupe(papers: list[Paper]) -> tuple[list[Paper], int]:
    """Union-dedupe by source_ids['openalex'] AND normalized title; keep the richer record.

    Returns (kept, n_removed). Kills the spike-results §2 rank-2/4 same-title duplicate.
    """
    kept: list[Paper] = []
    oa_index: dict[str, int] = {}
    title_index: dict[str, int] = {}
    removed = 0
    for p in papers:
        oa = p.source_ids.get("openalex")
        nt = normalize_title(p.title)
        pos = None
        if oa and oa in oa_index:
            pos = oa_index[oa]
        elif nt and nt in title_index:
            pos = title_index[nt]
        if pos is None:
            kept.append(p)
            pos = len(kept) - 1
        else:
            removed += 1
            if _completeness(p) <= _completeness(kept[pos]):
                continue
            kept[pos] = p
        if oa:
            oa_index[oa] = pos
        if nt:
            title_index[nt] = pos
    return kept, removed


def filter_with_abstract(papers: list[Paper]) -> list[Paper]:
    """Drop papers without an abstract (nothing to embed)."""
    return [p for p in papers if p.abstract]


def is_gnn_recsys(paper: Paper) -> bool:
    """True if the paper is topically GNN-based recommendation (title+abstract keyword gate).

    OpenAlex relevance ranking folds in citation_count, so a plain query surfaces high-cite
    off-topic papers (AlexNet, 6G surveys, etc.). This gate keeps only papers whose text shows
    BOTH a graph-method signal AND a recommendation signal.
    """
    text = ((paper.title or "") + " " + (paper.abstract or "")).lower()
    has_graph = any(
        k in text
        for k in [
            "graph neural",
            "graph convolution",
            "gnn",
            "gcn",
            "graph attention",
            "graph-based recommendation",
            "graph embedding",
            "graph learning",
        ]
    )
    has_rec = any(
        k in text
        for k in [
            "recommend",
            "collaborative filtering",
            "user-item",
            "rating prediction",
            "click-through",
            "next-item",
            "session-based",
            "click prediction",
        ]
    )
    return has_graph and has_rec


def filter_gnn_recsys(papers: list[Paper]) -> list[Paper]:
    """Keep only topically GNN-recsys papers (see is_gnn_recsys)."""
    return [p for p in papers if is_gnn_recsys(p)]


def _attribution_key(p: Paper) -> tuple[str | None, str | None]:
    """The (openalex_id, normalized_title) pair used to attribute a paper to a sub-area."""
    return p.source_ids.get("openalex"), normalize_title(p.title)


def build_corpus(sub_areas: list[str], per_query: int, target: int) -> list[Paper]:
    """Fetch each sub-area, dedupe + filter abstracts, stop at `target`, persist to cache."""
    accumulated: list[Paper] = []
    origin: dict[str, str] = {}  # attribution key -> first sub-area that surfaced it
    total_fetched = 0
    for sub_area in sub_areas:
        for p in openalex.fetch_works(sub_area, per_query):
            total_fetched += 1
            oa, nt = _attribution_key(p)
            for key in (oa, nt):
                if key and key not in origin:
                    origin[key] = sub_area
            accumulated.append(p)
        unique, _ = dedupe(accumulated)
        if len(filter_gnn_recsys(filter_with_abstract(unique))) >= target:
            break

    unique, removed = dedupe(accumulated)
    with_abstract = filter_with_abstract(unique)
    topical = filter_gnn_recsys(with_abstract)
    dropped = len(with_abstract) - len(topical)
    print(
        f"filtered {dropped} papers out of {len(with_abstract)} for off-topic "
        f"({len(with_abstract)} → {len(topical)} kept, {dropped} dropped)"
    )
    corpus = topical[:target]
    _save(corpus)
    _print_stats(corpus, total_fetched, len(unique), removed, origin)
    return corpus


def _save(papers: list[Paper]) -> None:
    """Overwrite the corpus cache JSON (shared with the spike pipeline)."""
    config.PAPERS_JSON.write_text(
        json.dumps([p.to_dict() for p in papers], ensure_ascii=False, indent=2)
    )


def _print_stats(
    corpus: list[Paper],
    total_fetched: int,
    n_unique: int,
    removed: int,
    origin: dict[str, str],
) -> None:
    """Print corpus-build summary: counts, per-sub-area distribution, DOI coverage."""
    with_doi = sum(1 for p in corpus if "doi" in p.source_ids)
    dist: Counter[str] = Counter()
    for p in corpus:
        oa, nt = _attribution_key(p)
        dist[origin.get(oa) or origin.get(nt) or "?"] += 1
    print("\n=== corpus build stats ===")
    print(f"total fetched (with dupes): {total_fetched}")
    print(f"unique after dedup:         {n_unique}")
    print(f"duplicates removed:         {removed}")
    print(f"with abstracts (final):     {len(corpus)}")
    print(f"with DOI in source_ids:     {with_doi}")
    print("per-sub-area distribution:")
    for sub_area in SUB_AREAS:
        print(f"  {dist.get(sub_area, 0):>4}  {sub_area}")
    other = sum(v for k, v in dist.items() if k not in SUB_AREAS)
    if other:
        print(f"  {other:>4}  (unattributed)")
    print(f"saved -> {config.PAPERS_JSON}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the GNN-recsys multi-sub-area corpus")
    ap.add_argument(
        "--target", type=int, default=500, help="stop after this many unique-with-abstract"
    )
    ap.add_argument("--per-query", type=int, default=100, help="works to fetch per sub-area")
    ap.add_argument("--force", action="store_true", help="rebuild even if papers.json exists")
    args = ap.parse_args()

    if config.PAPERS_JSON.exists() and not args.force:
        n = len(json.loads(config.PAPERS_JSON.read_text()))
        print(f"[corpus] cache exists ({n} papers) at {config.PAPERS_JSON}; use --force to rebuild")
        return
    build_corpus(SUB_AREAS, args.per_query, args.target)


if __name__ == "__main__":
    main()
