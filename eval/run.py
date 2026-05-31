"""Eval runner: every retrieval method × every gold query → comparison + sub-set tables.

Two tables are printed:

1. **Full mixed table** — aggregates each method across whichever queries it can score.
   ``method_match`` only runs on paper-mode queries (where the gold-set's ``anchor_title``
   resolves), so its mean is over a different denominator from dense/bm25/dense_rerank/hybrid.
   This is fine for "which method is strongest overall" but is NOT directly comparable.

2. **Same-subset table (the thesis gate)** — restricts to the 5 paper-mode queries that
   method_match can actually score and reports dense / dense_rerank / method_match / hybrid
   on that identical subset. THIS table is what tells us whether method_match beats dense.

The standalone ``method_match`` retriever scores ALL 400 carded papers independently (not
just dense's top-50); that's the lesson from the retrieval reality check. ``hybrid`` keeps
its existing dense+bm25 candidate union for the method_match contribution inside RRF — that
behavior is by design (see retrieval/hybrid.py).

CLI: ``uv run python -m eval.run [--method {dense,bm25,dense_rerank,method_match,hybrid,all}]``
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from spike import config as spike_config

from .gold_set import GoldQuery, GoldSet, TitleResolver
from .history import record as record_history
from .metrics import mrr, ndcg_at_k, recall_at_k

METHODS = ("dense", "bm25", "dense_rerank", "method_match", "hybrid")
TOP_N_FOR_RERANK = 50  # cross-encoder reranks this many candidates
EVAL_OUT_DIR = spike_config.CACHE_DIR / "eval"
NDCG_FLAG_THRESHOLD = 0.30


def _resolve_anchor(q: GoldQuery, resolver: TitleResolver) -> str | None:
    """For paper-mode queries, resolve anchor_title to a paper_id; else None."""
    if q.mode != "paper" or not q.anchor_title:
        return None
    return resolver.resolve(q.anchor_title)


def _retrieve(
    method: str,
    q: GoldQuery,
    anchor_pid: str | None,
    *,
    dense,
    bm25,
    reranker,
    matcher,
    hybrid,
    all_ids: list[str],
) -> list[str]:
    """Top-10 paper_ids for one (method, query) — returns [] if the method can't run."""
    if method == "dense":
        return [pid for pid, _ in dense.search(q.text, mode=q.mode, k=10)]
    if method == "bm25":
        return [pid for pid, _ in bm25.search(q.text, k=10)]
    if method == "dense_rerank":
        candidates = dense.search(q.text, mode=q.mode, k=TOP_N_FOR_RERANK)
        return [pid for pid, _ in reranker.rerank(q.text, candidates, k=10)]
    if method == "method_match":
        if anchor_pid is None:
            return []
        # Independent retriever: score every paper, not just dense's top-50.
        top = matcher.match(anchor_pid, None, all_ids, k=10)
        return [pid for pid, _ in top]
    if method == "hybrid":
        triples = hybrid.search(
            q.text,
            mode=q.mode,
            k=10,
            top_n_for_rerank=TOP_N_FOR_RERANK,
            query_paper_id=anchor_pid,
        )
        return [pid for pid, *_ in triples]
    raise ValueError(f"unknown method: {method}")


def _eval_one(top10: list[str], gold_ids: set[str]) -> dict[str, float]:
    return {
        "ndcg@5": ndcg_at_k(top10, gold_ids, k=5),
        "mrr": mrr(top10, gold_ids),
        "recall@10": recall_at_k(top10, gold_ids, k=10),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _fmt_row(label: str, metrics: dict[str, float], n: int) -> str:
    return (
        f"  {label:<14}  {metrics['ndcg@5']:.3f}    {metrics['mrr']:.3f}    "
        f"{metrics['recall@10']:.3f}    (n={n})"
    )


def _print_table(title: str, header_n_label: str, by_method: dict[str, dict]) -> None:
    print()
    print(title)
    print(f"  {'method':<14}  nDCG@5    MRR      Recall@10    {header_n_label}")
    for m in METHODS:
        rec = by_method.get(m)
        if rec is None or rec["n_queries"] == 0:
            print(f"  {m:<14}  —         —         —            (n=0)")
            continue
        print(_fmt_row(m, rec["agg"], rec["n_queries"]))


def _print_per_query(rows: list[dict]) -> None:
    """Per-query breakdown with nDCG@5 < threshold flagged."""
    print()
    print("Per-query breakdown (cells where nDCG@5 < 0.30 are flagged with !):")
    print(f"  {'query':<6} {'mode':<6} {'method':<14}  nDCG@5    MRR      Recall@10  resolved-gold")
    for row in rows:
        flag = "!" if row["metrics"]["ndcg@5"] < NDCG_FLAG_THRESHOLD else " "
        m = row["metrics"]
        print(
            f"  {row['query_id']:<6} {row['mode']:<6} {row['method']:<14}  "
            f"{m['ndcg@5']:.3f} {flag}  {m['mrr']:.3f}    {m['recall@10']:.3f}      "
            f"{row['resolved_gold']}/{row['total_gold']}"
        )


def run(methods: list[str], output_path: Path | None) -> int:
    """End-to-end eval. Returns 0 on success, 2 on thesis-gate failure (same-subset)."""
    from retrieval.bm25 import BM25Retriever
    from retrieval.dense import DenseRetriever
    from retrieval.hybrid import HybridRetriever
    from retrieval.method_match import MethodCardMatcher
    from retrieval.rerank import CrossEncoderReranker

    gs = GoldSet.load()
    papers = json.loads(spike_config.PAPERS_JSON.read_text(encoding="utf-8"))
    all_ids = [p["paper_id"] for p in papers]
    resolver = TitleResolver(papers)

    # Resolve gold ids and anchor ids up front.
    resolved_gold: dict[str, set[str]] = {}
    unresolved_per_query: dict[str, list[str]] = {}
    anchor_for: dict[str, str | None] = {}
    for q in gs.queries:
        gold_pids: set[str] = set()
        misses: list[str] = []
        for t in q.gold_titles:
            pid = resolver.resolve(t)
            if pid:
                gold_pids.add(pid)
            else:
                misses.append(t)
        resolved_gold[q.id] = gold_pids
        unresolved_per_query[q.id] = misses
        anchor_for[q.id] = _resolve_anchor(q, resolver)

    total_gold = sum(len(q.gold_titles) for q in gs.queries)
    total_resolved = sum(len(s) for s in resolved_gold.values())
    print(
        f"[gold-set] {gs.version} — {total_resolved}/{total_gold} gold titles resolved "
        f"({total_resolved / total_gold:.1%})"
    )
    print()

    # Lazy-construct retrievers (only what's requested).
    dense = (
        DenseRetriever()
        if any(m in methods for m in ("dense", "dense_rerank", "method_match", "hybrid"))
        else None
    )
    bm25 = BM25Retriever() if any(m in methods for m in ("bm25", "hybrid")) else None
    reranker = (
        CrossEncoderReranker() if any(m in methods for m in ("dense_rerank", "hybrid")) else None
    )
    matcher = (
        MethodCardMatcher(embedder=dense)
        if any(m in methods for m in ("method_match", "hybrid"))
        else None
    )
    # No explicit use_rerank: pick up HybridRetriever's new False default. CE rerank is
    # default-disabled in hybrid per eval-v1 finding (see eval/HANDOFF.md).
    hybrid = (
        HybridRetriever(dense=dense, bm25=bm25, method_match=matcher, reranker=reranker)
        if "hybrid" in methods
        else None
    )

    # Run.
    per_query_rows: list[dict] = []
    by_method: dict[str, dict] = {
        m: {"n_queries": 0, "ndcg@5": [], "mrr": [], "recall@10": []} for m in methods
    }
    same_subset: dict[str, dict] = {
        m: {"n_queries": 0, "ndcg@5": [], "mrr": [], "recall@10": []}
        for m in methods
        if m != "bm25"
    }

    for q in gs.queries:
        gold_ids = resolved_gold[q.id]
        anchor_pid = anchor_for[q.id]
        for method in methods:
            top10 = _retrieve(
                method,
                q,
                anchor_pid,
                dense=dense,
                bm25=bm25,
                reranker=reranker,
                matcher=matcher,
                hybrid=hybrid,
                all_ids=all_ids,
            )
            # Skip queries where the method genuinely can't run (e.g. method_match w/o anchor).
            if method == "method_match" and anchor_pid is None:
                continue
            m = _eval_one(top10, gold_ids)
            per_query_rows.append(
                {
                    "query_id": q.id,
                    "mode": q.mode,
                    "method": method,
                    "metrics": m,
                    "top10": top10,
                    "resolved_gold": len(gold_ids),
                    "total_gold": len(q.gold_titles),
                }
            )
            by_method[method]["n_queries"] += 1
            by_method[method]["ndcg@5"].append(m["ndcg@5"])
            by_method[method]["mrr"].append(m["mrr"])
            by_method[method]["recall@10"].append(m["recall@10"])
            # Same-subset: paper queries with anchor resolvable.
            if q.mode == "paper" and anchor_pid is not None and method in same_subset:
                same_subset[method]["n_queries"] += 1
                same_subset[method]["ndcg@5"].append(m["ndcg@5"])
                same_subset[method]["mrr"].append(m["mrr"])
                same_subset[method]["recall@10"].append(m["recall@10"])

    # Aggregate.
    for tbl in (by_method, same_subset):
        for m in tbl:
            tbl[m]["agg"] = {k: _mean(tbl[m][k]) for k in ("ndcg@5", "mrr", "recall@10")}

    # Print: full mixed table.
    _print_table(
        "=== AGGREGATE (full mixed; method_match denominator differs from others) ===",
        "queries scored",
        by_method,
    )
    # Print: same-subset paper-queries-only table (the thesis gate).
    _print_table(
        "=== SAME-SUBSET (paper queries only — directly comparable; THE THESIS GATE) ===",
        "paper queries",
        same_subset,
    )

    # Thesis statement on the same-subset table.
    dense_agg = same_subset.get("dense", {}).get("agg")
    print()
    if dense_agg and dense_agg["ndcg@5"] > 0:
        for m in ("dense_rerank", "method_match", "hybrid"):
            agg = same_subset.get(m, {}).get("agg")
            if not agg:
                continue
            delta = agg["ndcg@5"] - dense_agg["ndcg@5"]
            verdict = "BEATS" if delta > 0 else ("matches" if abs(delta) < 1e-6 else "BELOW")
            print(f"  same-subset: {m:<14} {verdict} dense on nDCG@5 by {delta:+.3f}")
    else:
        print("  same-subset: dense has 0 nDCG@5 — gold resolution or anchor issue, see breakdown")

    _print_per_query(per_query_rows)

    # Unresolved gold for the human reviewer.
    print()
    print("Unresolved gold (corpus expansion candidates):")
    for qid, misses in unresolved_per_query.items():
        if misses:
            print(f"  {qid}: {', '.join(misses)}")

    # Save raw + history.
    EVAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_path or (EVAL_OUT_DIR / f"{stamp}.json")
    payload = {
        "version": "eval-v1",
        "created": stamp,
        "gold_set_version": gs.version,
        "resolved_gold_per_query": {qid: sorted(pids) for qid, pids in resolved_gold.items()},
        "unresolved_per_query": unresolved_per_query,
        "aggregates_full": {
            m: by_method[m]["agg"] | {"n_queries": by_method[m]["n_queries"]} for m in methods
        },
        "aggregates_same_subset_paper": {
            m: same_subset[m]["agg"] | {"n_queries": same_subset[m]["n_queries"]}
            for m in same_subset
        },
        "per_query": per_query_rows,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[eval] saved -> {out_path}")

    # Track in history.jsonl for change-over-time.
    for m, rec in by_method.items():
        if rec["n_queries"] > 0:
            record_history(
                m,
                rec["agg"] | {"n_queries": rec["n_queries"]},
                notes=f"gold={gs.version} subset=full",
            )
    for m, rec in same_subset.items():
        if rec["n_queries"] > 0:
            record_history(
                m,
                rec["agg"] | {"n_queries": rec["n_queries"]},
                notes=f"gold={gs.version} subset=paper-only",
            )

    # Thesis-gate exit code: same-subset method_match or hybrid below dense on nDCG@5 → 2.
    if dense_agg and dense_agg["ndcg@5"] > 0:
        for m in ("method_match", "hybrid"):
            agg = same_subset.get(m, {}).get("agg")
            if agg and agg["ndcg@5"] < dense_agg["ndcg@5"]:
                print(
                    f"\n[STOP] same-subset {m} ({agg['ndcg@5']:.3f}) BELOW dense "
                    f"({dense_agg['ndcg@5']:.3f}) on nDCG@5. See breakdown above."
                )
                return 2
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run all 5 retrievers on the gold set; report metrics."
    )
    ap.add_argument(
        "--method",
        choices=("dense", "bm25", "dense_rerank", "method_match", "hybrid", "all"),
        default="all",
    )
    ap.add_argument("--output", type=Path, default=None, help="raw JSON dump path override")
    args = ap.parse_args()
    methods = list(METHODS) if args.method == "all" else [args.method]
    return run(methods, args.output)


if __name__ == "__main__":
    sys.exit(main())
