"""Multi-hop citation-graph reasoning — the project's second differentiation pillar.

Three queries turn the raw citation DAG into evidence the UI can show as **why** a paper
was returned:

* ``find_ancestors``       — BFS over OUT-edges from the query paper, weighted by edge intent
                             and method-card similarity. Traces the mechanism's lineage.
* ``find_opposing``        — 1-hop neighbors (cites + cited-by) filtered to comparison-intent
                             edges, scored by method-card *distance*. **Today** (no intents on
                             disk) this falls back to mechanism-distance ranking over all 1-hop
                             neighbors; explanation says so.
* ``find_cross_domain_same_mechanism`` — primarily a method-card query: top-N by mechanism
                             similarity, then filter to a different inferred sub-area.

Each result is a :class:`ReasoningResult` carrying up to 3 :class:`GraphPath` traces — the
node sequence + per-edge intents + a human-readable explanation — so the UI can render
"why" without re-deriving it.

Integration hooks for future work:

* ``get_edge_intent(g, u, v)`` reads ``g.edges[u, v].get("intent", DEFAULT_INTENT)``. When B's
  SciCite classifier + A's S2 contexts land, populate the ``intent`` edge attribute and the
  queries pick up real intents automatically — no API change.
* ``infer_sub_area(paper)`` is keyword-heuristic ``TODO(C)`` — replace with corpus-recorded
  sub_area once ``data/corpus.py`` persists ``origin`` into papers.json.

CLI: ``uv run python -m retrieval.graph_reason --paper-id <pid> --query <name> --k 5``
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable

import networkx as nx

from spike import config as spike_config

from .method_match import MethodCardMatcher

# Edge-intent vocabulary the three queries reason over. None of these are populated on
# disk yet — see module docstring.
DEFAULT_INTENT = "background"
INTENT_WEIGHTS = {"method": 1.0, "background": 0.5, "comparison": 0.2}

# Sub-area keyword heuristic. First match wins; document with TODO(C) for the
# corpus-recorded replacement.
# TODO(C): replace with corpus-recorded sub_area once data/corpus.py persists origin
# into papers.json (today the corpus build's `origin` dict is internal-only).
SUB_AREA_KEYWORDS: dict[str, list[str]] = {
    "knowledge-graph": [
        "knowledge graph",
        "knowledge-aware",
        "kgat",
        "kgcn",
        "kgin",
        "ripplenet",
    ],
    "cross-domain": [
        "cross-domain",
        "cross domain",
        "transfer learning",
        "domain adaptation",
        "bitgcf",
    ],
    "session-based": [
        "session-based",
        "session based",
        "next-item",
        "next item",
        "sequential recommendation",
        "sr-gnn",
        "gc-san",
    ],
    "social": [
        "social recommendation",
        "social network",
        "trust-aware",
        "diffnet",
        "graphrec",
    ],
    "contrastive": [
        "contrastive",
        "self-supervised",
        "self supervised",
        "infonce",
    ],
    "collaborative-filtering": [
        "collaborative filtering",
        "matrix factorization",
        "implicit feedback",
        "ngcf",
        "lightgcn",
    ],
}


# --- module-level lazy loaders -------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_graph() -> nx.DiGraph:
    from spike.build_graph import load_graph

    return load_graph()


@lru_cache(maxsize=1)
def _papers_by_id() -> dict[str, dict]:
    raw = json.loads(spike_config.PAPERS_JSON.read_text(encoding="utf-8"))
    return {p["paper_id"]: p for p in raw}


@lru_cache(maxsize=1)
def _matcher() -> MethodCardMatcher:
    """Shared MethodCardMatcher — loads SPECTER2 lazily, reuses the .npz field cache."""
    return MethodCardMatcher()


# --- public types --------------------------------------------------------------------------


@dataclass
class GraphPath:
    """A trace through the citation graph explaining why a paper was returned.

    ``nodes[0]`` is the query paper; ``nodes[-1]`` is the result. ``edge_intents[i]`` is the
    intent label for the edge ``nodes[i] -> nodes[i+1]``. ``explanation`` is human-readable.
    """

    nodes: list[str]
    edge_intents: list[str]
    score: float
    explanation: str


@dataclass
class ReasoningResult:
    """One paper + its top paths back to the query."""

    paper_id: str
    score: float
    paths: list[GraphPath] = field(default_factory=list)


# --- helpers -------------------------------------------------------------------------------


def get_edge_intent(g: nx.DiGraph, src: str, dst: str) -> str:
    """Per-edge intent; defaults to ``DEFAULT_INTENT`` until intents are populated."""
    if not g.has_edge(src, dst):
        return DEFAULT_INTENT
    return g.edges[src, dst].get("intent", DEFAULT_INTENT)


def _title(papers: dict[str, dict], pid: str) -> str:
    p = papers.get(pid)
    return (p.get("title") or "<unknown>") if p else f"<missing {pid}>"


def _year(papers: dict[str, dict], pid: str) -> int | None:
    p = papers.get(pid)
    return p.get("year") if p else None


def infer_sub_area(paper: dict) -> str:
    """Return best-matching sub-area, or ``"other"`` if no keyword matched."""
    text = ((paper.get("title") or "") + " " + (paper.get("abstract") or "")).lower()
    for area, kws in SUB_AREA_KEYWORDS.items():
        for kw in kws:
            if kw in text:
                return area
    return "other"


def _short_title(papers: dict[str, dict], pid: str, max_len: int = 50) -> str:
    t = _title(papers, pid)
    return t if len(t) <= max_len else t[: max_len - 1] + "…"


def _format_edge_label(intent: str) -> str:
    """Render an intent as a parenthetical edge label for path explanations."""
    return {
        "method": "method-citation",
        "comparison": "comparison-intent",
        "background": "background-citation",
    }.get(intent, "background-citation")


# --- query 1: find_ancestors --------------------------------------------------------------


def find_ancestors(
    paper_id: str,
    max_hops: int = 3,
    k: int = 10,
) -> list[ReasoningResult]:
    """BFS over outgoing edges, scored by intent-weighted method-card-weighted hop discount.

    Returns up to ``k`` ancestors. Each ancestor carries up to 3 best paths. Foundational
    papers (out-degree 0) return ``[]`` rather than raising — see HANDOFF "OUT-sparsity".
    """
    g = _load_graph()
    if paper_id not in g:
        return []
    matcher = _matcher()
    papers = _papers_by_id()

    # BFS, capturing FULL paths (not just min-distance). We keep up to ~16 paths per node
    # to bound work; reasoning explanations only need 3 anyway.
    paths_to: dict[str, list[GraphPath]] = {}
    queue: list[tuple[str, list[str], list[str]]] = [(paper_id, [paper_id], [])]
    while queue:
        node, path_nodes, path_intents = queue.pop(0)
        if len(path_nodes) - 1 >= max_hops:
            continue
        for nxt in g.successors(node):
            if nxt in path_nodes:  # cycle guard
                continue
            intent = get_edge_intent(g, node, nxt)
            new_nodes = path_nodes + [nxt]
            new_intents = path_intents + [intent]
            hop = len(new_nodes) - 1
            mech_sim = matcher.similarity(paper_id, nxt)
            # Per-edge intent weights then averaged by hop count → smooth decay.
            intent_score = (
                sum(INTENT_WEIGHTS.get(i, INTENT_WEIGHTS["background"]) for i in new_intents) / hop
            )
            score = (intent_score / hop) * max(mech_sim, 1e-6)
            explanation = _ancestor_explanation(papers, new_nodes, new_intents, mech_sim)
            paths_to.setdefault(nxt, []).append(
                GraphPath(
                    nodes=new_nodes, edge_intents=new_intents, score=score, explanation=explanation
                )
            )
            if len(paths_to[nxt]) < 16:
                queue.append((nxt, new_nodes, new_intents))

    results: list[ReasoningResult] = []
    for pid, plist in paths_to.items():
        plist.sort(key=lambda p: p.score, reverse=True)
        best = plist[0].score
        results.append(ReasoningResult(paper_id=pid, score=best, paths=plist[:3]))
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:k]


def _ancestor_explanation(
    papers: dict[str, dict],
    nodes: list[str],
    intents: list[str],
    mech_sim: float,
) -> str:
    """Human-readable trace: ``Query → Hop1 (method-citation) → Hop2 (background-citation)``."""
    pieces: list[str] = [_short_title(papers, nodes[0], 28)]
    for i, intent in enumerate(intents, start=1):
        pieces.append(f"→ {_short_title(papers, nodes[i], 28)} ({_format_edge_label(intent)})")
    trail = " ".join(pieces)
    return f"{trail}; mechanism similarity {mech_sim:.2f}"


# --- query 2: find_opposing ---------------------------------------------------------------


def find_opposing(paper_id: str, k: int = 10) -> list[ReasoningResult]:
    """1-hop neighbors flagged as comparison-intent, OR (live fallback) mechanism-distant.

    With real intents on disk this returns papers explicitly compared against in citation
    contexts. With the default intent ("background") everywhere — the live state today —
    it falls back to ranking 1-hop neighbors by **mechanism distance** (``1 - similarity``).
    """
    g = _load_graph()
    if paper_id not in g:
        return []
    matcher = _matcher()
    papers = _papers_by_id()

    neighbors: list[tuple[str, str, str]] = []  # (neighbor_pid, direction, intent)
    for nxt in g.successors(paper_id):
        neighbors.append((nxt, "out", get_edge_intent(g, paper_id, nxt)))
    for prv in g.predecessors(paper_id):
        neighbors.append((prv, "in", get_edge_intent(g, prv, paper_id)))

    # Try comparison-intent path first.
    comparison = [n for n in neighbors if n[2] == "comparison"]
    using_fallback = not comparison

    if comparison:
        ranked: list[tuple[str, str, str, float]] = []
        for pid, direction, intent in comparison:
            distance = 1.0 - matcher.similarity(paper_id, pid)
            ranked.append((pid, direction, intent, distance))
        ranked.sort(key=lambda t: t[3], reverse=True)
    else:
        ranked = []
        for pid, direction, intent in neighbors:
            distance = 1.0 - matcher.similarity(paper_id, pid)
            ranked.append((pid, direction, intent, distance))
        ranked.sort(key=lambda t: t[3], reverse=True)

    results: list[ReasoningResult] = []
    for pid, direction, intent, distance in ranked[:k]:
        explanation = _opposing_explanation(
            papers, paper_id, pid, direction, intent, distance, using_fallback
        )
        if direction == "out":
            nodes = [paper_id, pid]
            intents = [intent]
        else:
            nodes = [pid, paper_id]
            intents = [intent]
        results.append(
            ReasoningResult(
                paper_id=pid,
                score=distance,
                paths=[
                    GraphPath(
                        nodes=nodes, edge_intents=intents, score=distance, explanation=explanation
                    )
                ],
            )
        )
    return results


def _opposing_explanation(
    papers: dict[str, dict],
    query: str,
    neighbor: str,
    direction: str,
    intent: str,
    distance: float,
    using_fallback: bool,
) -> str:
    arrow = "→" if direction == "out" else "←"
    trail = f"{_short_title(papers, query, 28)} {arrow} {_short_title(papers, neighbor, 28)}"
    fallback_tag = "[no intent metadata; mechanism-distance fallback] " if using_fallback else ""
    return (
        f"{fallback_tag}{trail} ({_format_edge_label(intent)}); "
        f"mechanism distance {distance:.2f}"
    )


# --- query 3: find_cross_domain_same_mechanism --------------------------------------------


def find_cross_domain_same_mechanism(
    paper_id: str,
    k: int = 10,
    candidate_pool: int = 30,
) -> list[ReasoningResult]:
    """Top-N by mechanism similarity, restricted to a DIFFERENT inferred sub-area.

    Optionally validates that returned candidates are not direct 1-hop citation neighbors
    (truly independent works); deprioritizes — does not drop — direct neighbors.
    """
    g = _load_graph()
    papers = _papers_by_id()
    matcher = _matcher()
    if paper_id not in papers:
        return []
    if paper_id not in matcher.cards:
        return []
    query_sub_area = infer_sub_area(papers[paper_id])

    all_ids = list(papers.keys())
    top_by_sim = matcher.match(paper_id, None, all_ids, k=candidate_pool)

    direct_neighbors = set()
    if paper_id in g:
        direct_neighbors.update(g.successors(paper_id))
        direct_neighbors.update(g.predecessors(paper_id))

    results: list[ReasoningResult] = []
    for pid, sim in top_by_sim:
        if pid == paper_id:
            continue
        cand_sub_area = infer_sub_area(papers.get(pid, {}))
        if cand_sub_area == query_sub_area:
            continue
        # Slight penalty for direct neighbors so independent works rank higher.
        adjusted = sim * (0.85 if pid in direct_neighbors else 1.0)
        explanation = (
            f"Same-mechanism candidate from sub-area '{cand_sub_area}' "
            f"(query is '{query_sub_area}'); field similarity {sim:.2f}"
            + (" — note: direct citation neighbor" if pid in direct_neighbors else "")
        )
        results.append(
            ReasoningResult(
                paper_id=pid,
                score=adjusted,
                paths=[
                    GraphPath(
                        nodes=[paper_id, pid],
                        edge_intents=["(no edge; mechanism-only)"],
                        score=adjusted,
                        explanation=explanation,
                    )
                ],
            )
        )
        if len(results) >= k:
            break
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:k]


# --- CLI ------------------------------------------------------------------------------------


def _print_results(label: str, results: Iterable[ReasoningResult], papers: dict[str, dict]) -> None:
    print(f"\n=== {label} ===")
    rs = list(results)
    if not rs:
        print("  (no results — see HANDOFF 'OUT-sparsity' or try a more recent paper)")
        return
    for i, r in enumerate(rs, 1):
        title = _title(papers, r.paper_id)
        year = _year(papers, r.paper_id)
        print(f"  {i:>2}. (score {r.score:.3f})  {title}  [{r.paper_id}, {year or '?'}]")
        for p in r.paths[:1]:
            print(f"        path: {p.explanation}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paper-id", required=True, help="OpenAlex paper_id of the query paper")
    ap.add_argument(
        "--query",
        required=True,
        choices=("ancestors", "opposing", "cross-domain"),
    )
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-hops", type=int, default=3, help="ancestors-only: max BFS depth")
    args = ap.parse_args()

    papers = _papers_by_id()
    if args.paper_id not in papers:
        print(
            f"[graph_reason] paper_id {args.paper_id!r} not in corpus; "
            f"see data/cache/papers.json",
            file=sys.stderr,
        )
        return 1
    print(
        f"[graph_reason] anchor: {args.paper_id}  "
        f"{_title(papers, args.paper_id)}  ({_year(papers, args.paper_id) or '?'})"
    )

    if args.query == "ancestors":
        results = find_ancestors(args.paper_id, max_hops=args.max_hops, k=args.k)
        _print_results(f"ANCESTORS top-{args.k}  (max_hops={args.max_hops})", results, papers)
    elif args.query == "opposing":
        results = find_opposing(args.paper_id, k=args.k)
        _print_results(f"OPPOSING top-{args.k}", results, papers)
    else:
        results = find_cross_domain_same_mechanism(args.paper_id, k=args.k)
        _print_results(f"CROSS-DOMAIN SAME-MECHANISM top-{args.k}", results, papers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
