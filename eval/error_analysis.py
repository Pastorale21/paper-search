"""Inspect low-scoring eval cases from a saved ``eval.run`` JSON payload.

CLI: ``uv run python -m eval.error_analysis data/cache/eval/<run>.json``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_THRESHOLD = 0.30
DEFAULT_PAPERS_JSON = Path(__file__).resolve().parents[1] / "data" / "cache" / "papers.json"


def _load_json(path: Path) -> dict | list:
    """Load UTF-8 JSON from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def _load_title_index(papers_path: Path | None) -> dict[str, str]:
    """Return ``paper_id -> title`` from a papers.json file, or an empty index."""
    if papers_path is None:
        path = DEFAULT_PAPERS_JSON
    else:
        path = papers_path
    if not path.exists():
        return {}
    raw = _load_json(path)
    papers = raw if isinstance(raw, list) else list(raw.values())
    return {p["paper_id"]: p.get("title") or "" for p in papers}


def _title(pid: str, titles: dict[str, str]) -> str:
    """Format a paper id with title when a title index is available."""
    title = titles.get(pid)
    return f"{pid} | {title}" if title else pid


def _rows_below_threshold(payload: dict, threshold: float) -> list[dict]:
    """Filter per-query rows with nDCG@5 below threshold."""
    rows = payload.get("per_query", [])
    return [
        row
        for row in rows
        if float(row.get("metrics", {}).get("ndcg@5", 0.0)) < threshold
    ]


def print_report(payload: dict, titles: dict[str, str], threshold: float) -> None:
    """Print a compact low-score diagnostic report."""
    rows = _rows_below_threshold(payload, threshold)
    if not rows:
        print(f"No eval rows below nDCG@5 < {threshold:.2f}.")
        return

    resolved_gold = payload.get("resolved_gold_per_query", {})
    unresolved = payload.get("unresolved_per_query", {})
    query_meta = payload.get("query_meta", {})

    print(f"Low-score eval cases: {len(rows)} rows with nDCG@5 < {threshold:.2f}")
    print()
    for row in rows:
        qid = row["query_id"]
        metrics = row.get("metrics", {})
        meta = query_meta.get(qid, {})
        print("=" * 88)
        print(
            f"{qid} [{row.get('mode', '?')}] via {row.get('method', '?')} | "
            f"nDCG@5={metrics.get('ndcg@5', 0):.3f} "
            f"MRR={metrics.get('mrr', 0):.3f} "
            f"Recall@10={metrics.get('recall@10', 0):.3f}"
        )
        if meta.get("text"):
            print(f"query: {meta['text']}")
        if meta.get("notes"):
            print(f"notes: {meta['notes']}")

        print("\nresolved gold:")
        for pid in resolved_gold.get(qid, []):
            print(f"  - {_title(pid, titles)}")

        misses = unresolved.get(qid, [])
        if misses:
            print("\nunresolved gold / corpus-gap candidates:")
            for title in misses:
                print(f"  - {title}")

        print("\npredicted top-10:")
        for rank, pid in enumerate(row.get("top10", []), start=1):
            marker = "*" if pid in set(resolved_gold.get(qid, [])) else " "
            print(f"  {rank:>2}. {marker} {_title(pid, titles)}")
        print()


def main() -> int:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser(description="Dump low-nDCG eval cases for error analysis.")
    ap.add_argument("eval_json", type=Path, help="path produced by `python -m eval.run`")
    ap.add_argument("--papers", type=Path, default=None, help="optional papers.json path")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = ap.parse_args()

    payload = _load_json(args.eval_json)
    if not isinstance(payload, dict):
        raise TypeError(f"{args.eval_json} must contain a JSON object")
    titles = _load_title_index(args.papers)
    print_report(payload, titles, args.threshold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
