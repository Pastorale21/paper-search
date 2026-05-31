"""Side-by-side CLI for the 5 retrievers (dense / BM25 / CE rerank / method_match / hybrid).

Used to inspect whether method-card field matching visibly differentiates inside the
dense-saturated cluster (the project's core thesis). Method-match needs a paper anchor:
the caller passes `--query-paper-id` explicitly, or in `paper` mode we auto-anchor to the
dense top-1 result (i.e. "find papers similar to whichever paper this abstract is closest
to"). In `short` mode without an explicit anchor, the method_match section is skipped.

Examples:
    uv run python -m retrieval.compare --query "<abstract>" --mode paper --k 5
    uv run python -m retrieval.compare --eval-queries
"""

from __future__ import annotations

import argparse
import json

from spike import config as spike_config

from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .hybrid import HybridRetriever
from .method_match import MethodCardMatcher
from .rerank import CrossEncoderReranker

TOP_N = 50  # how many candidates each base retriever surfaces for downstream signals

EVAL_DUMP = spike_config.CACHE_DIR / "eval" / "retrieval_comparison.json"


def _papers_by_id() -> dict[str, dict]:
    return {p["paper_id"]: p for p in json.loads(spike_config.PAPERS_JSON.read_text())}


def _title(papers: dict[str, dict], pid: str) -> str:
    p = papers.get(pid)
    return (p.get("title") or "<unknown>") if p else f"<missing {pid}>"


def _fmt_row(rank: int, pid: str, score: float, title: str, width: int = 90) -> str:
    truncated = title if len(title) <= width else title[: width - 1] + "…"
    return f"  {rank:>2}. ({score:>7.3f}) {truncated}  [{pid}]"


def _spread(scored: list[tuple[str, float]]) -> float:
    if not scored:
        return 0.0
    scores = [s for _, s in scored]
    return max(scores) - min(scores)


def _print_section(label: str, results: list[tuple[str, float]], papers: dict[str, dict]) -> None:
    print(f"\n=== {label} ===")
    if not results:
        print("  (empty)")
        return
    for i, (pid, score) in enumerate(results, 1):
        print(_fmt_row(i, pid, score, _title(papers, pid)))


def _print_hybrid(
    label: str, results: list[tuple[str, float, dict]], papers: dict[str, dict]
) -> None:
    print(f"\n=== {label} ===")
    if not results:
        print("  (empty)")
        return
    for i, (pid, score, br) in enumerate(results, 1):
        print(_fmt_row(i, pid, score, _title(papers, pid)))
        sig = ", ".join(
            f"{k}={v:.3f}" if isinstance(v, (int, float)) else f"{k}=–" for k, v in br.items()
        )
        print(f"        signal_breakdown: {sig}")


def run_one(
    query: str,
    mode: str,
    k: int,
    query_paper_id: str | None,
    no_rerank: bool,
) -> dict:
    """Run all 5 retrievers on one query; print sections; return a dict suitable for JSON dump."""
    papers = _papers_by_id()
    dense = DenseRetriever()
    bm25 = BM25Retriever()
    matcher = MethodCardMatcher(embedder=dense)
    reranker = CrossEncoderReranker()
    hybrid = HybridRetriever(
        dense=dense, bm25=bm25, method_match=matcher, reranker=reranker, use_rerank=not no_rerank
    )

    dense_top = dense.search(query, mode=mode, k=TOP_N)
    bm25_top = bm25.search(query, k=TOP_N)

    anchor_pid = query_paper_id
    auto_anchor_note = None
    if mode == "paper" and anchor_pid is None and dense_top:
        anchor_pid = dense_top[0][0]
        auto_anchor_note = anchor_pid

    rerank_top = reranker.rerank(query, dense_top, k=k)

    mm_top: list[tuple[str, float]] = []
    if anchor_pid is not None:
        candidates = list(dict.fromkeys([p for p, _ in dense_top] + [p for p, _ in bm25_top]))
        mm_top = matcher.match(anchor_pid, None, candidates, k=k)

    hybrid_top = hybrid.search(
        query, mode=mode, k=k, top_n_for_rerank=TOP_N, query_paper_id=anchor_pid
    )

    if auto_anchor_note is not None:
        print(
            f"\n[method_match] auto-anchored on dense top-1: {auto_anchor_note} "
            f"({_title(papers, auto_anchor_note)[:80]})"
        )

    _print_section(f"DENSE top-{k}", dense_top[:k], papers)
    _print_section(f"BM25 top-{k}", bm25_top[:k], papers)
    _print_section(f"DENSE + CE rerank top-{k}", rerank_top, papers)
    if anchor_pid is not None:
        _print_section(f"METHOD_MATCH top-{k}  (anchor={anchor_pid})", mm_top, papers)
    else:
        print(
            "\n=== METHOD_MATCH ===\n"
            "  (skipped: no paper anchor; pass --query-paper-id or use --mode paper)"
        )
    _print_hybrid(f"HYBRID top-{k} (RRF + CE)", hybrid_top, papers)

    # Honest spread: exclude the anchor paper from both methods so the comparison is
    # apples-to-apples (method_match already excludes self after the fix; dense does not).
    print("\n=== TOP-5 SPREADS (anchor-excluded; apples-to-apples) ===")
    dense_excl = [(pid, s) for pid, s in dense_top if pid != anchor_pid][:5]
    mm_excl = [(pid, s) for pid, s in mm_top if pid != anchor_pid][:5]  # already excluded
    print(f"  dense        top-5 spread: {_spread(dense_excl):.4f}")
    if mm_top:
        print(f"  method_match top-5 spread: {_spread(mm_excl):.4f}")

    return {
        "query": query,
        "mode": mode,
        "k": k,
        "anchor_paper_id": anchor_pid,
        "dense": [(pid, float(s)) for pid, s in dense_top[:k]],
        "bm25": [(pid, float(s)) for pid, s in bm25_top[:k]],
        "dense_plus_rerank": [(pid, float(s)) for pid, s in rerank_top],
        "method_match": [(pid, float(s)) for pid, s in mm_top],
        "hybrid": [(pid, float(s), b) for pid, s, b in hybrid_top],
    }


def run_eval_queries(k: int) -> None:
    """Iterate spike.config.EVAL_QUERIES, dump JSON results for later inspection."""
    out = []
    for mode, text in spike_config.EVAL_QUERIES:
        label = text if len(text) < 70 else text[:67] + "…"
        print(f"\n{'#' * 80}\n# [{mode}] {label}\n{'#' * 80}")
        result = run_one(text, mode=mode, k=k, query_paper_id=None, no_rerank=False)
        out.append(result)
    EVAL_DUMP.parent.mkdir(parents=True, exist_ok=True)
    EVAL_DUMP.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[compare] dumped -> {EVAL_DUMP}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Side-by-side comparison of all 5 retrievers.")
    ap.add_argument("--query", type=str, help="query text (short phrase or pasted abstract)")
    ap.add_argument("--mode", choices=["short", "paper"], default="short")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument(
        "--query-paper-id", type=str, default=None, help="explicit anchor for method_match"
    )
    ap.add_argument(
        "--no-rerank", action="store_true", help="skip cross-encoder rerank inside hybrid"
    )
    ap.add_argument(
        "--eval-queries",
        action="store_true",
        help="run spike.config.EVAL_QUERIES, dump JSON to data/cache/eval/",
    )
    args = ap.parse_args()

    if args.eval_queries:
        run_eval_queries(k=args.k)
        return

    if not args.query:
        ap.error("--query is required unless --eval-queries is given")

    run_one(
        args.query,
        mode=args.mode,
        k=args.k,
        query_paper_id=args.query_paper_id,
        no_rerank=args.no_rerank,
    )


if __name__ == "__main__":
    main()
