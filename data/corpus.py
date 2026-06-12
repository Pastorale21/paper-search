"""Build the multi-sub-area GNN-recsys corpus: fetch per sub-area, dedupe, filter, persist.

CLI:
  `uv run python -m data.corpus --target 800 --per-query 500 [--force]`  # full (re)build
  `uv run python -m data.corpus --merge-seeds`        # add seed titles, no re-crawl
  `uv run python -m data.corpus --merge-ids W1,W2`    # add specific papers by OpenAlex id (light)
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from schemas import Paper
from spike import config

from .sources import openalex
from .sources.seed_papers import SEED_TITLES, fetch_seed_papers

SUB_AREAS = [
    "graph neural network collaborative filtering",
    "light graph convolution collaborative filtering recommendation",
    "graph contrastive learning for recommendation",
    "self-supervised graph learning recommendation",
    "graph self-supervised recommendation collaborative filtering",
    "cross-domain recommendation with graph neural network",
    "transfer learning cross-domain recommendation graph",
    "cross-domain sequential recommendation graph",
    "knowledge graph enhanced recommendation",
    "knowledge graph neural network recommender systems",
    "knowledge graph collaborative filtering recommendation",
    "multi-behavior recommendation graph neural network",
    "session-based recommendation with graph neural network",
    "sequential recommendation graph neural network",
    "next item recommendation graph neural network",
    "social recommendation with graph neural network",
    "social graph neural network recommendation",
    "social recommendation graph convolution network",
    "federated recommendation graph neural network",
]

_GENERIC_TITLE_TOKENS = {
    "a",
    "an",
    "and",
    "are",
    "based",
    "collaborative",
    "contrastive",
    "convolution",
    "cross",
    "domain",
    "for",
    "from",
    "graph",
    "graphs",
    "in",
    "knowledge",
    "learning",
    "model",
    "network",
    "networks",
    "neural",
    "of",
    "on",
    "recommendation",
    "recommender",
    "recommenders",
    "system",
    "systems",
    "session",
    "sequential",
    "social",
    "the",
    "to",
    "towards",
    "via",
    "with",
}


def normalize_title(title: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — for fuzzy title dedup."""
    cleaned = re.sub(r"[^\w\s]", " ", (title or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _title_tokens(normalized_title: str) -> set[str]:
    """Return normalized, non-generic title tokens for fuzzy duplicate checks."""
    return {
        token for token in normalized_title.split() if token and token not in _GENERIC_TITLE_TOKENS
    }


def _is_distinctive_short_title(tokens: set[str]) -> bool:
    """True for acronym/model-name titles such as LightGCN, XSimGCL, KGAT, or SURGE."""
    if len(tokens) != 1:
        return False
    token = next(iter(tokens))
    return len(token) >= 4 and token not in _GENERIC_TITLE_TOKENS


def titles_are_duplicates(left: str, right: str) -> bool:
    """Return True for exact titles and safe prefix/acronym title variants.

    OpenAlex sometimes stores a canonical short work ("LightGCN") separately from a
    longer proceedings title ("LightGCN: Simplifying and Powering ..."). Exact normalized
    matching misses this, so we also accept containment when the shorter side is a
    distinctive model acronym/name. For longer variants, require high token containment
    and similar title lengths to avoid merging broad survey-ish titles.
    """
    if not left or not right:
        return False
    if left == right:
        return True

    left_tokens = _title_tokens(left)
    right_tokens = _title_tokens(right)
    if not left_tokens or not right_tokens:
        return False

    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    shorter_tokens, longer_tokens = (
        (left_tokens, right_tokens) if len(left) <= len(right) else (right_tokens, left_tokens)
    )

    if _is_distinctive_short_title(shorter_tokens) and shorter_tokens <= longer_tokens:
        return True

    overlap = len(shorter_tokens & longer_tokens) / len(shorter_tokens)
    length_ratio = len(shorter) / len(longer)
    return len(shorter_tokens) >= 3 and overlap >= 0.90 and length_ratio >= 0.75


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
    """Union-dedupe by source_ids['openalex'] AND fuzzy-normalized title.

    Returns (kept, n_removed). Kills both same-title duplicates and short/long title variants
    such as W3004578093 vs W3045200674 for LightGCN.
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
        elif nt:
            pos = next(
                (
                    known_pos
                    for known_title, known_pos in title_index.items()
                    if titles_are_duplicates(nt, known_title)
                ),
                None,
            )
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
    """Fetch each sub-area, dedupe + filter abstracts, balance to `target`, persist to cache."""
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

    unique, removed = dedupe(accumulated)
    with_abstract = filter_with_abstract(unique)
    topical = filter_gnn_recsys(with_abstract)
    dropped = len(with_abstract) - len(topical)
    print(
        f"filtered {dropped} papers out of {len(with_abstract)} for off-topic "
        f"({len(with_abstract)} -> {len(topical)} kept, {dropped} dropped)"
    )

    # Merge curated seed papers — they bypass the GNN-recsys filter (curated; canonical
    # comparators the gold set needs as candidates) and are guaranteed survival of the
    # target cap below.
    print("\n[seed] fetching curated seed papers from OpenAlex by title...")
    seeds_raw, missed = fetch_seed_papers(SEED_TITLES)
    seeds = filter_with_abstract(seeds_raw)
    abstract_misses = len(seeds_raw) - len(seeds)
    seed_unique = _dedupe_seed_papers(seeds)
    seed_removed = len(seeds) - len(seed_unique)
    topical_pids = {p.paper_id for p in topical}
    seed_pids = {p.paper_id for p in seed_unique}
    new_to_corpus = sum(1 for pid in seed_pids if pid not in topical_pids)
    already_in_corpus = len(seed_unique) - new_to_corpus
    print(
        f"[seed] resolved {len(seeds_raw)}/{len(SEED_TITLES)} titles; "
        f"{len(seed_unique)} usable ({new_to_corpus} NEW, {already_in_corpus} already in corpus); "
        f"{len(missed)} title-misses, {abstract_misses} abstract-misses"
    )
    if seed_removed:
        print(f"[seed] removed {seed_removed} duplicate seed hits")
    if missed:
        print("[seed] OpenAlex couldn't match these titles (tighten the SEED_TITLES string):")
        for t in missed:
            print(f"  - {t!r}")

    # Attribute seeds to a pseudo-sub-area for the per-sub-area print, then merge.
    for sp in seeds:
        oa, nt = _attribution_key(sp)
        for key in (oa, nt):
            if key and key not in origin:
                origin[key] = "seed"

    seed_titles = {normalize_title(p.title) for p in seed_unique}
    seed_oas = {p.source_ids.get("openalex") for p in seed_unique if p.source_ids.get("openalex")}
    topical_unique, _ = dedupe(topical)
    final_other = [
        p
        for p in topical_unique
        if p.paper_id not in seed_pids
        and p.source_ids.get("openalex") not in seed_oas
        and not any(titles_are_duplicates(normalize_title(p.title), st) for st in seed_titles)
    ]
    corpus = _select_balanced(seed_unique, final_other, target, origin)
    _save(corpus)
    _print_stats(corpus, total_fetched, len(unique), removed, origin)
    return corpus


def _select_balanced(
    seeds: list[Paper],
    papers: list[Paper],
    target: int,
    origin: dict[str, str],
) -> list[Paper]:
    """Select papers round-robin by source query so late sub-areas are not starved."""
    selected = list(seeds[:target])
    seen = {p.paper_id for p in selected}
    buckets: dict[str, list[Paper]] = {sub_area: [] for sub_area in SUB_AREAS}
    buckets["?"] = []

    for paper in papers:
        if paper.paper_id in seen:
            continue
        oa, nt = _attribution_key(paper)
        key = origin.get(oa) or origin.get(nt) or "?"
        buckets.setdefault(key, []).append(paper)

    while len(selected) < target:
        added = False
        for key in [*SUB_AREAS, "?"]:
            bucket = buckets.get(key) or []
            while bucket:
                paper = bucket.pop(0)
                if paper.paper_id in seen:
                    continue
                selected.append(paper)
                seen.add(paper.paper_id)
                added = True
                break
            if len(selected) >= target:
                break
        if not added:
            break
    return selected


def _dedupe_seed_papers(seeds: list[Paper]) -> list[Paper]:
    """Dedupe curated seeds by exact paper id only; do not fuzzy-merge canonical gold."""
    kept: list[Paper] = []
    seen: set[str] = set()
    for paper in seeds:
        if paper.paper_id in seen:
            continue
        kept.append(paper)
        seen.add(paper.paper_id)
    return kept


def merge_seeds_into_corpus(
    path: Path | None = None,
) -> tuple[list[Paper], list[Paper], list[str]]:
    """Add curated seed papers to the EXISTING corpus without re-crawling the sub-areas.

    A full ``--force`` rebuild re-rolls every sub-area crawl and can SHRINK the corpus (OpenAlex
    relevance ranking + the off-topic filter are not stable run-to-run). This adds only the
    genuinely-new seed papers on top of the committed corpus, so nothing already present is lost.

    Seeds are matched against the existing corpus by paper_id, OpenAlex id, and fuzzy title; only
    new ones are appended. Returns ``(merged, added, missed_seed_titles)``. Verify the printed
    ``added`` list (title + year) — OpenAlex can resolve an acronym to a look-alike paper.
    """
    if path is None:
        path = config.PAPERS_JSON
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; build the corpus first (data.corpus --force)")
    existing = [Paper.from_dict(d) for d in json.loads(path.read_text(encoding="utf-8"))]
    existing_pids = {p.paper_id for p in existing}
    existing_oas = {p.source_ids.get("openalex") for p in existing if p.source_ids.get("openalex")}
    existing_titles = {normalize_title(p.title) for p in existing}

    seeds_raw, missed = fetch_seed_papers(SEED_TITLES)
    seeds = filter_with_abstract(seeds_raw)

    added: list[Paper] = []
    for sp in seeds:
        nt = normalize_title(sp.title)
        oa = sp.source_ids.get("openalex")
        if sp.paper_id in existing_pids or (oa and oa in existing_oas):
            continue
        if any(titles_are_duplicates(nt, et) for et in existing_titles):
            continue
        added.append(sp)
        existing_pids.add(sp.paper_id)
        existing_titles.add(nt)
        if oa:
            existing_oas.add(oa)

    merged = existing + added
    path.write_text(
        json.dumps([p.to_dict() for p in merged], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[merge] {len(existing)} existing + {len(added)} new seeds = {len(merged)} papers")
    if added:
        print("[merge] added (verify these are the REAL papers, not acronym look-alikes):")
        for sp in added:
            print(f"  + {sp.paper_id} ({sp.year or '?'})  {sp.title}")
    if missed:
        print(f"[merge] OpenAlex couldn't match {len(missed)} seed titles:")
        for t in missed:
            print(f"  - {t!r}")
    return merged, added, missed


def merge_works_by_ids(
    ids: list[str], path: Path | None = None
) -> tuple[list[Paper], list[Paper], list[str]]:
    """Add specific papers by exact OpenAlex id to the EXISTING corpus (light, reliable path).

    Each id is a single direct lookup (``openalex.fetch_work_by_id``), so this stays usable when
    the search endpoint behind ``--merge-seeds`` is rate-limited. Skips ids already present
    (by id / OpenAlex id / fuzzy title). Returns ``(merged, added, failed_ids)``.
    """
    if path is None:
        path = config.PAPERS_JSON
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; build the corpus first (data.corpus --force)")
    existing = [Paper.from_dict(d) for d in json.loads(path.read_text(encoding="utf-8"))]
    existing_pids = {p.paper_id for p in existing}
    existing_oas = {p.source_ids.get("openalex") for p in existing if p.source_ids.get("openalex")}
    existing_titles = {normalize_title(p.title) for p in existing}

    added: list[Paper] = []
    failed: list[str] = []
    for oa_id in ids:
        p = openalex.fetch_work_by_id(oa_id)
        if p is None or not p.abstract:
            failed.append(oa_id)
            continue
        nt = normalize_title(p.title)
        oa = p.source_ids.get("openalex")
        if p.paper_id in existing_pids or (oa and oa in existing_oas):
            continue
        if any(titles_are_duplicates(nt, et) for et in existing_titles):
            continue
        added.append(p)
        existing_pids.add(p.paper_id)
        existing_titles.add(nt)
        if oa:
            existing_oas.add(oa)

    merged = existing + added
    path.write_text(
        json.dumps([p.to_dict() for p in merged], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[merge-ids] {len(existing)} existing + {len(added)} new = {len(merged)} papers")
    for p in added:
        print(f"  + {p.paper_id} ({p.year or '?'})  {p.title}")
    if failed:
        print(f"[merge-ids] could not fetch (skipped): {', '.join(failed)}")
    return merged, added, failed


def _save(papers: list[Paper]) -> None:
    """Overwrite the corpus cache JSON (shared with the spike pipeline)."""
    config.PAPERS_JSON.write_text(
        json.dumps([p.to_dict() for p in papers], ensure_ascii=False, indent=2),
        encoding="utf-8",
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
    seed_count = dist.get("seed", 0)
    if seed_count:
        print(f"  {seed_count:>4}  (curated seed papers)")
    other = sum(v for k, v in dist.items() if k not in SUB_AREAS and k != "seed")
    if other:
        print(f"  {other:>4}  (unattributed)")
    print(f"saved -> {config.PAPERS_JSON}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the GNN-recsys multi-sub-area corpus")
    ap.add_argument(
        "--target", type=int, default=800, help="stop after this many unique-with-abstract"
    )
    ap.add_argument("--per-query", type=int, default=500, help="works to fetch per sub-area")
    ap.add_argument("--force", action="store_true", help="rebuild even if papers.json exists")
    ap.add_argument(
        "--merge-seeds",
        action="store_true",
        help="add curated seed papers to the EXISTING corpus without re-crawling (no regression)",
    )
    ap.add_argument(
        "--merge-ids",
        type=str,
        help="comma-separated OpenAlex ids to add by direct lookup (light; use when the "
        "--merge-seeds title search is rate-limited)",
    )
    args = ap.parse_args()

    if args.merge_seeds:
        merge_seeds_into_corpus()
        return

    if args.merge_ids:
        merge_works_by_ids([s.strip() for s in args.merge_ids.split(",") if s.strip()])
        return

    if config.PAPERS_JSON.exists() and not args.force:
        n = len(json.loads(config.PAPERS_JSON.read_text()))
        print(f"[corpus] cache exists ({n} papers) at {config.PAPERS_JSON}; use --force to rebuild")
        return
    build_corpus(SUB_AREAS, args.per_query, args.target)


if __name__ == "__main__":
    main()
