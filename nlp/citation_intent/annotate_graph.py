"""Annotate citation-graph edges with intent labels — B's citation-intent P0 deliverable.

Loads the corpus citation graph, fetches Semantic Scholar reference contexts for each citing
paper (via ``data/sources/s2_contexts``), matches each context back to a corpus edge by the
cited paper's DOI/MAG, classifies it into ``background | method | comparison``, and writes the
label onto ``g.edges[u, v]["intent"]``. ``retrieval/graph_reason.get_edge_intent`` then reads
these automatically — no change to the graph or retrieval API.

Edge ``u -> v`` means *u cites v* (see ``spike/build_graph.py``), so the context is the sentence
where the citing paper ``u`` references ``v``. Edges with no matched context stay unlabeled and
fall back to ``DEFAULT_INTENT`` in graph_reason. Already-labeled edges are skipped on re-runs
(the persisted graph is the cache), so a restart never re-classifies — use --reclassify to override.

Classification modes (cost-aware):
  * default     — map Semantic Scholar's own ``intents`` (no model, no LLM; FREE).
  * --use-scicite — run B's local SciBERT/SciCite model on the context text (FREE, local).
  * --allow-llm — let the LLM fallback label edges S2 left unlabeled (PAID).

CLI (module: ``nlp.citation_intent.annotate_graph``):
  --dry-run       report coverage, write nothing
  (no flags)      full run; maps S2 intents and writes the graph in place
  --use-scicite   classify with B's local SciCite model
  --limit 5       smoke test over the first 5 citing papers -> citation_graph.limited.pkl sidecar
  --reclassify    re-label edges that already carry an intent
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from typing import Callable

import networkx as nx

from schemas import Paper

from .. import config
from .classifier import CitationIntentClassifier, map_s2_intent

GRAPH_PKL = config.CACHE_DIR / "citation_graph.pkl"
BACKUP_PKL = config.CACHE_DIR / "citation_graph.pre_intent.pkl"
EDGE_INTENTS_JSON = config.CACHE_DIR / "edge_intents.json"


def _normalize_doi(doi: str | None) -> str | None:
    """Strip the URL prefix and lowercase a DOI for cross-source matching (None-safe)."""
    if not doi:
        return None
    bare = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip().lower()
    return bare or None


def _corpus_index(papers: list[Paper]) -> tuple[dict[str, str], dict[str, str]]:
    """Build DOI->paper_id and MAG->paper_id reverse indexes for cited-paper matching."""
    doi_index: dict[str, str] = {}
    mag_index: dict[str, str] = {}
    for paper in papers:
        doi = _normalize_doi(paper.source_ids.get("doi") or paper.external_ids.get("doi"))
        if doi:
            doi_index[doi] = paper.paper_id
        mag = paper.external_ids.get("mag")
        if mag:
            mag_index[str(mag)] = paper.paper_id
    return doi_index, mag_index


def _resolve_cited(
    record: dict, doi_index: dict[str, str], mag_index: dict[str, str]
) -> str | None:
    """Map one S2 reference record to a corpus paper_id via DOI, then MAG."""
    doi = _normalize_doi(record.get("cited_doi"))
    if doi and doi in doi_index:
        return doi_index[doi]
    mag = record.get("cited_mag")
    if mag and str(mag) in mag_index:
        return mag_index[str(mag)]
    return None


def s2_intent_classify(record: dict) -> str | None:
    """FREE default: map S2's own intents, letting method/comparison win over background.

    S2 returns ``intents`` as a list; a single citation may carry several (e.g.
    ``["background", "methodology"]``). Prefer the stronger mechanism signal so those edges are
    not flattened to background. None when S2 gave no intent at all (leave the edge unlabeled).
    """
    labels = [map_s2_intent(i) for i in (record.get("intents") or [])]
    if not labels:
        return None
    for preferred in ("method", "comparison"):
        if preferred in labels:
            return preferred
    return labels[0]


def annotate_edges(
    g: nx.DiGraph,
    papers: list[Paper],
    fetch: Callable[[str], list[dict]],
    classify: Callable[[dict], str | None],
    *,
    limit: int | None = None,
    skip_labeled: bool = True,
) -> dict:
    """Write intent labels onto matched edges in place; return a coverage report.

    `fetch(u)` returns S2 reference records for citing paper `u`; `classify(record)` returns an
    intent label or None (None -> leave the edge unlabeled). Mutates `g`.

    `skip_labeled` (default) skips edges that already carry an ``intent`` — so re-running over a
    previously-annotated graph never re-classifies (and never re-pays the LLM) for settled edges.
    Pass `skip_labeled=False` to force re-labeling, e.g. when switching classification mode.
    """
    doi_index, mag_index = _corpus_index(papers)
    sources = sorted(u for u in g.nodes if g.out_degree(u) > 0)
    if limit is not None:
        sources = sources[:limit]

    counts: Counter = Counter()
    edge_intents: dict[str, str] = {}
    for u in sources:
        try:
            records = fetch(u)
        except Exception:  # noqa: BLE001 - one paper's failure must not abort the crawl
            counts["fetch_error"] += 1
            continue
        for record in records:
            v = _resolve_cited(record, doi_index, mag_index)
            if v is None or v == u or not g.has_edge(u, v):
                continue  # unmatched, self-citation, or not an in-corpus edge
            if skip_labeled and g.edges[u, v].get("intent"):
                counts["cached"] += 1
                continue
            label = classify(record)
            if not label:
                counts["unlabeled"] += 1
                continue
            g.edges[u, v]["intent"] = label
            edge_intents[f"{u}->{v}"] = label
            counts[label] += 1

    return {
        "sources_processed": len(sources),
        "edges_total": g.number_of_edges(),
        "edges_annotated": len(edge_intents),
        "counts": dict(counts),
        "edge_intents": edge_intents,
    }


def _build_classifier(use_scicite: bool, allow_llm: bool) -> Callable[[dict], str | None]:
    """Pick the classify callable for the requested cost-aware mode."""
    if use_scicite:
        clf = CitationIntentClassifier(use_scicite=True)
        return lambda record: clf.classify(record)
    if allow_llm:
        clf = CitationIntentClassifier(use_scicite=False)
        return lambda record: clf.classify(record)  # S2 intents if present, else paid LLM
    return s2_intent_classify


def _load_graph(path) -> nx.DiGraph:
    """Load the citation graph via plain pickle (no spike import; structure is unchanged)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Citation graph not found at {path}. Build it first: uv run python -m spike"
        )
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _report(result: dict) -> None:
    """Print a one-screen coverage summary."""
    print(
        f"[annotate] sources={result['sources_processed']} "
        f"edges_total={result['edges_total']} "
        f"edges_annotated={result['edges_annotated']}"
    )
    for label, n in sorted(result["counts"].items()):
        print(f"  {label}: {n}")


def main() -> None:
    """CLI entry point for citation-graph intent annotation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="only process the first N citing papers (smoke)")
    parser.add_argument("--use-scicite", action="store_true", help="classify with local SciCite")
    parser.add_argument(
        "--allow-llm", action="store_true", help="allow paid LLM for unlabeled edges"
    )
    parser.add_argument("--force", action="store_true", help="ignore the S2 context cache")
    parser.add_argument(
        "--reclassify", action="store_true", help="re-label edges that already have an intent"
    )
    parser.add_argument("--dry-run", action="store_true", help="report coverage; write nothing")
    parser.add_argument("--out", help="write the annotated graph here instead of in place")
    args = parser.parse_args()

    from data.sources.s2_contexts import fetch_citation_contexts

    papers = config.load_papers()
    g = _load_graph(GRAPH_PKL)
    classify = _build_classifier(args.use_scicite, args.allow_llm)

    def fetch(paper_id: str) -> list[dict]:
        return fetch_citation_contexts(paper_id, force=args.force)

    result = annotate_edges(
        g, papers, fetch, classify, limit=args.limit, skip_labeled=not args.reclassify
    )
    _report(result)

    if args.dry_run:
        print("[annotate] dry-run: graph not written.")
        return

    # A partial (--limit) run must never clobber the canonical graph: route it to a sidecar so
    # the live graph graph_reason reads is only ever written by a full run (or an explicit --out).
    if args.out:
        out = config.CACHE_DIR / args.out
    elif args.limit is not None:
        out = config.CACHE_DIR / "citation_graph.limited.pkl"
        print(f"[annotate] --limit: partial graph -> {out.name} (canonical graph untouched)")
    else:
        out = GRAPH_PKL

    if out == GRAPH_PKL and not BACKUP_PKL.exists():
        BACKUP_PKL.write_bytes(GRAPH_PKL.read_bytes())
        print(f"[annotate] backed up pre-intent graph to {BACKUP_PKL.name}")
    with open(out, "wb") as fh:
        pickle.dump(g, fh)
    intents_path = EDGE_INTENTS_JSON if out == GRAPH_PKL else out.with_suffix(".intents.json")
    intents_path.write_text(
        json.dumps(result["edge_intents"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[annotate] wrote {out.name} and {intents_path.name}")


if __name__ == "__main__":
    main()
