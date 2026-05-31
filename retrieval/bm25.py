"""Sparse BM25 retriever over `title + " " + abstract`, with pickle cache.

Cache invalidation: persists `paper_id_order` alongside the BM25 object and rebuilds when
the cached order doesn't match the current `papers.json` (the same corpus-rebuild staleness
that bit the spike index — see data/HANDOFF.md known-issues).

CLI: `uv run python -m retrieval.bm25 --rebuild`
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from spike import config


def _tokenize(text: str) -> list[str]:
    """Whitespace + lowercase tokenization (sufficient for English abstracts)."""
    return (text or "").lower().split()


BM25_PKL: Path = config.CACHE_DIR / "bm25.pkl"


class BM25Retriever:
    """BM25Okapi over (title + abstract); cached to disk and rebuilt on corpus change."""

    def __init__(self, force_rebuild: bool = False) -> None:
        self._bm25: BM25Okapi
        self._ids: list[str]
        self._load_or_build(force_rebuild)

    def _current_papers(self) -> tuple[list[str], list[list[str]]]:
        papers = json.loads(config.PAPERS_JSON.read_text(encoding="utf-8"))
        ids = [p["paper_id"] for p in papers]
        tokens = [
            _tokenize((p.get("title") or "") + " " + (p.get("abstract") or "")) for p in papers
        ]
        return ids, tokens

    def _load_or_build(self, force_rebuild: bool) -> None:
        if BM25_PKL.exists() and not force_rebuild:
            with BM25_PKL.open("rb") as f:
                state = pickle.load(f)
            current_ids, _ = self._current_papers()
            if state["ids"] == current_ids:
                self._bm25 = state["bm25"]
                self._ids = state["ids"]
                print(f"[bm25] cache hit: {len(self._ids)} docs")
                return
            print("[bm25] cache stale (corpus changed); rebuilding")
        self._build()

    def _build(self) -> None:
        ids, tokens = self._current_papers()
        self._bm25 = BM25Okapi(tokens)
        self._ids = ids
        BM25_PKL.parent.mkdir(parents=True, exist_ok=True)
        with BM25_PKL.open("wb") as f:
            pickle.dump({"bm25": self._bm25, "ids": self._ids}, f)
        print(f"[bm25] built + cached: {len(self._ids)} docs -> {BM25_PKL}")

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Top-k (paper_id, BM25 score) for the query; empty query -> empty result."""
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        idx = scores.argsort()[::-1][:k]
        return [(self._ids[i], float(scores[i])) for i in idx]


def main() -> None:
    """CLI: force-rebuild the BM25 cache."""
    ap = argparse.ArgumentParser(description="Build / rebuild the BM25 cache.")
    ap.add_argument("--rebuild", action="store_true", help="ignore existing cache and rebuild")
    args = ap.parse_args()
    BM25Retriever(force_rebuild=args.rebuild)


if __name__ == "__main__":
    main()
