"""End-to-end spike smoke test: fetch -> embed -> index -> graph, then run fixed eval queries."""

from __future__ import annotations

import argparse

from . import build_graph, build_index, config, fetch, search


def run_pipeline(force: bool, query: str, n: int) -> None:
    """Run the cached build steps in order (embed is folded into the index step)."""
    fetch.main(force=force, query=query, n=n)
    build_index.build_index(force=force)
    build_graph.main(force=force)


def run_eval() -> None:
    """Run the fixed eval queries and print top-5 for manual mechanism-relevance check."""
    print("\n=== eval queries (top-5 each) ===")
    for mode, text in config.EVAL_QUERIES:
        label = text if len(text) < 70 else text[:67] + "..."
        print(f"\n[{mode}] {label}")
        for rank, (paper, score) in enumerate(search.search(text, mode=mode, top_k=5), 1):
            print(f"  {rank}. ({score:.3f}) {paper.title} [{paper.year}]")


def main() -> None:
    ap = argparse.ArgumentParser(description="GNN-recsys spike end-to-end smoke test")
    ap.add_argument("--force", action="store_true", help="rebuild all cached artifacts")
    ap.add_argument("--query", default=config.QUERY)
    ap.add_argument("--n", type=int, default=config.N_PAPERS)
    args = ap.parse_args()

    run_pipeline(args.force, args.query, args.n)
    run_eval()
    print("\n[spike] smoke test complete.")


if __name__ == "__main__":
    main()
