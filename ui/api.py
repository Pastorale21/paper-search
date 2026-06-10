"""Single backend surface for the Streamlit UI.

Other UI files import ONLY from ``ui.api`` — never from ``retrieval``, ``nlp``, ``data``,
or ``spike`` directly. If a downstream module signature changes, this is the one file
that has to follow.

Caching strategy:

* ``@st.cache_resource`` — long-lived heavy objects (SPECTER2, FAISS, BM25, MethodCard
  matcher, OpenAI client). Survive page navigations.
* ``@st.cache_data`` — JSON loads (papers.json, individual method cards). Survive
  navigations until the underlying file changes.

Where this module degrades gracefully:

* If a paper has no method card, ``load_method_card`` returns ``None`` rather than raise.
* Non-hybrid search methods return per-result ``signal_breakdown=None``; the Search tab
  branches on that instead of crashing.
* The LLM client constructs lazily and the build_messages helper has no network — the
  caller decides when (or whether) to actually invoke chat completion. **No paid LLM call
  is made anywhere in this module.**
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import streamlit as st

from nlp import config as nlp_config
from retrieval import graph_reason
from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.method_match import FIELD_WEIGHTS, FIELDS_TO_MATCH, MethodCardMatcher
from retrieval.rerank import CrossEncoderReranker
from schemas import MethodCard
from spike import config as spike_config

# --- Resources: cached once per session ---------------------------------------------------


@st.cache_data(show_spinner=False)
def get_papers_by_id() -> dict[str, dict]:
    """Load the corpus from ``data/cache/papers.json`` as a dict keyed by paper_id."""
    raw = json.loads(spike_config.PAPERS_JSON.read_text(encoding="utf-8"))
    return {p["paper_id"]: p for p in raw}


@st.cache_resource(show_spinner=False)
def get_dense() -> DenseRetriever:
    return DenseRetriever()


@st.cache_resource(show_spinner=False)
def get_bm25() -> BM25Retriever:
    return BM25Retriever()


@st.cache_resource(show_spinner=False)
def get_matcher() -> MethodCardMatcher:
    """Shared MethodCardMatcher — reuses the dense retriever's SPECTER2 instance."""
    return MethodCardMatcher(embedder=get_dense())


@st.cache_resource(show_spinner=False)
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()


@st.cache_resource(show_spinner=False)
def get_hybrid() -> HybridRetriever:
    """HybridRetriever with the eval-validated default (use_rerank=False)."""
    return HybridRetriever(
        dense=get_dense(),
        bm25=get_bm25(),
        method_match=get_matcher(),
        reranker=get_reranker(),
    )


@st.cache_resource(show_spinner=False)
def get_llm_client():
    """Construct an OpenAI-compatible client from ``nlp.config`` — does NOT call it."""
    from openai import OpenAI

    if not nlp_config.LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY not set. Configure .env per nlp/HANDOFF.md.")
    return OpenAI(api_key=nlp_config.LLM_API_KEY, base_url=nlp_config.LLM_BASE_URL)


def is_llm_configured() -> bool:
    """Return whether the paid LLM surface can be called by the UI."""
    return bool(nlp_config.LLM_API_KEY)


def llm_model_name() -> str:
    """Return the configured chat model name for display only."""
    return nlp_config.LLM_MODEL


# --- Method cards on disk -----------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_method_card(paper_id: str) -> MethodCard | None:
    """Read one cached method card. Returns ``None`` if not extracted yet."""
    path = nlp_config.METHOD_CARDS_DIR / f"{paper_id}.json"
    if not path.exists():
        return None
    return MethodCard.from_dict(json.loads(path.read_text(encoding="utf-8")))


def has_method_card(paper_id: str) -> bool:
    return (nlp_config.METHOD_CARDS_DIR / f"{paper_id}.json").exists()


# --- Search -------------------------------------------------------------------------------


def search(query: str, mode: str = "short", method: str = "hybrid", k: int = 10) -> list[dict]:
    """Run one of the four supported methods and enrich each hit with paper + method card.

    Returns a list of dicts: ``{paper, score, signal_breakdown, method_card}``.
    ``signal_breakdown`` is ``None`` unless the method is ``hybrid`` (the only method that
    natively produces a per-signal explanation).
    """
    papers = get_papers_by_id()
    if not query.strip():
        return []

    if method == "dense":
        hits = get_dense().search(query, mode=mode, k=k)  # type: ignore[arg-type]
        results = [
            {"paper_id": pid, "score": float(s), "signal_breakdown": None} for pid, s in hits
        ]
    elif method == "bm25":
        hits = get_bm25().search(query, k=k)
        results = [
            {"paper_id": pid, "score": float(s), "signal_breakdown": None} for pid, s in hits
        ]
    elif method == "dense_rerank":
        candidates = get_dense().search(query, mode=mode, k=50)  # type: ignore[arg-type]
        hits = get_reranker().rerank(query, candidates, k=k)
        results = [
            {"paper_id": pid, "score": float(s), "signal_breakdown": None} for pid, s in hits
        ]
    elif method == "hybrid":
        triples = get_hybrid().search(query, mode=mode, k=k)  # type: ignore[arg-type]
        results = [
            {"paper_id": pid, "score": float(s), "signal_breakdown": br} for pid, s, br in triples
        ]
    else:
        raise ValueError(f"unknown method: {method!r}")

    enriched: list[dict] = []
    for r in results:
        pid = r["paper_id"]
        enriched.append(
            {
                "paper": papers.get(pid),
                "paper_id": pid,
                "score": r["score"],
                "signal_breakdown": r["signal_breakdown"],
                "method_card": load_method_card(pid),
            }
        )
    return enriched


# --- Method-card matching (Tab 2's showcase) ----------------------------------------------


def match_similar_mechanism(paper_id: str, k: int = 10) -> list[dict]:
    """Top-k candidates ranked by weighted field-level cosine.

    Each result includes per-field cosines — the visible evidence of mechanism-level
    matching. Returns ``[]`` if the anchor has no cached method card.
    """
    matcher = get_matcher()
    if paper_id not in matcher.cards:
        return []

    all_ids = list(get_papers_by_id().keys())
    top = matcher.match(paper_id, None, all_ids, k=k)
    return [
        {
            "paper": get_papers_by_id().get(pid),
            "paper_id": pid,
            "score": float(score),
            "per_field": _per_field_cosines(matcher, paper_id, pid),
            "method_card": load_method_card(pid),
        }
        for pid, score in top
    ]


def _per_field_cosines(matcher: MethodCardMatcher, pid_a: str, pid_b: str) -> dict[str, float]:
    """Per-field unweighted cosine (``None`` when a field is missing on either side)."""
    import numpy as np

    out: dict[str, float | None] = {}
    for f in FIELDS_TO_MATCH:
        ea = matcher._candidate_field_emb(pid_a, f)
        eb = matcher._candidate_field_emb(pid_b, f)
        if ea is None or eb is None:
            out[f] = None
        else:
            out[f] = float(np.dot(ea, eb))
    return out  # type: ignore[return-value]


def field_weights() -> dict[str, float]:
    """Expose the matcher's per-field weights so the UI can label them honestly."""
    return dict(FIELD_WEIGHTS)


# --- Graph reasoning (Tab 3) --------------------------------------------------------------


def find_ancestors(
    paper_id: str, k: int = 5, max_hops: int = 3
) -> list[graph_reason.ReasoningResult]:
    return graph_reason.find_ancestors(paper_id, max_hops=max_hops, k=k)


def find_opposing(paper_id: str, k: int = 5) -> list[graph_reason.ReasoningResult]:
    return graph_reason.find_opposing(paper_id, k=k)


def find_cross_domain(paper_id: str, k: int = 5) -> list[graph_reason.ReasoningResult]:
    return graph_reason.find_cross_domain_same_mechanism(paper_id, k=k)


_SURVEY_TOKENS = ("survey", "review", "systematic literature")


def filter_survey_titles(results: list[Any], papers: dict[str, dict]) -> list[Any]:
    """Belt-and-suspenders: drop papers whose title screams 'survey' / 'review'.

    The matcher already excludes empty-card candidates, but a partially-filled survey can
    still slip through to Tab 3's Opposing view; this filter is intentionally cheap.
    """
    kept = []
    for r in results:
        title = (papers.get(r.paper_id, {}).get("title") or "").lower()
        if any(tok in title for tok in _SURVEY_TOKENS):
            continue
        kept.append(r)
    return kept


# --- Corpus stats (landing) ---------------------------------------------------------------


@st.cache_data(show_spinner=False)
def corpus_stats() -> dict[str, int]:
    papers = get_papers_by_id()
    with_abstract = sum(1 for p in papers.values() if p.get("abstract"))
    cards_dir = nlp_config.METHOD_CARDS_DIR
    n_cards = sum(1 for _ in cards_dir.glob("*.json")) if cards_dir.exists() else 0
    # Graph stats — load via spike's loader (no UI dep on networkx beyond reading numbers).
    from spike.build_graph import load_graph

    g = load_graph()
    isolated = sum(1 for n in g.nodes if g.in_degree(n) == 0 and g.out_degree(n) == 0)
    return {
        "papers": len(papers),
        "papers_with_abstract": with_abstract,
        "method_cards": n_cards,
        "graph_nodes": g.number_of_nodes(),
        "graph_edges": g.number_of_edges(),
        "graph_isolated": isolated,
    }


def cache_health() -> dict[str, bool]:
    """Report whether demo-critical cache artifacts are present on disk."""
    return {
        "papers": spike_config.PAPERS_JSON.exists(),
        "faiss_index": spike_config.FAISS_INDEX.exists(),
        "citation_graph": spike_config.GRAPH_PKL.exists(),
        "method_cards_dir": nlp_config.METHOD_CARDS_DIR.exists(),
    }


# --- Helpers for the UI ------------------------------------------------------------------


def method_card_to_dict(card: MethodCard | None) -> dict | None:
    if card is None:
        return None
    return asdict(card)


def project_root() -> Path:
    return spike_config.CACHE_DIR.parent.parent
