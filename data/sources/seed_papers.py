"""Curated seed-paper fetcher: reach into OpenAlex for canonical GNN-recsys papers by title.

The sub-area queries used by ``data/corpus.py`` are good at finding recent / derivative work
but miss the canonical older comparators the gold set requires (NGCF, GC-MC, UltraGCN, etc.)
because OpenAlex's relevance ranking buries them under more recent papers. This module
fetches each curated title directly and lets the caller merge them into the corpus, bypassing
the ``is_gnn_recsys`` filter (they're curated; we don't need to keyword-gate them).

Title matching: substring (either direction) wins immediately; otherwise we accept ≥70%
token-overlap between the seed and the candidate's normalized title. Unmatched titles are
returned to the caller so the SEED_TITLES list can be tightened.
"""

from __future__ import annotations

import re

from schemas import Paper

from . import openalex

# Curated canonical GNN-recsys papers our gold set expects as candidates. Grouped informally
# by sub-family: LightGCN family, contrastive learning, knowledge graph, cross-domain,
# session-based, social.
SEED_TITLES: list[str] = [
    "Neural Graph Collaborative Filtering",
    "Graph Convolutional Matrix Completion",
    "LightGCN Simplifying and Powering Graph Convolution Network",
    "UltraGCN",
    "Revisiting Graph based Collaborative Filtering",  # LR-GCCF
    "Disentangled Graph Collaborative Filtering",  # DGCF
    "Self-supervised Graph Learning for Recommendation",  # SGL
    "Are Graph Augmentations Necessary Simple Graph Contrastive Learning",  # SimGCL
    # NCL
    "Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning",
    "Hypergraph Contrastive Collaborative Filtering",  # HCCF
    "LightGCL Simple Yet Effective Graph Contrastive Learning",
    "KGAT Knowledge Graph Attention Network for Recommendation",
    "Learning Intents behind Interactions for Knowledge Graph",  # KGIN
    "RippleNet Propagating User Preferences on the Knowledge Graph",
    "Bi-directional Transfer Graph Collaborative Filtering",  # BiTGCF
    "Session-based Recommendation with Graph Neural Networks",  # SR-GNN
    "Graph Contextualized Self-Attention Network for Session-based Recommendation",  # GC-SAN
    "Diffusion Network for Social Recommendation",  # DiffNet
    "Graph Neural Networks for Social Recommendation",  # GraphRec
]

_TOKEN_OVERLAP_THRESHOLD = 0.70

# Candidates per title fetched from OpenAlex. Small (3-5) is enough because the search
# endpoint already ranks by relevance — but >1 so we can fall back if the top hit is wrong
# (e.g. a "Comments on …" / survey that mentions the canonical paper).
_CANDIDATES_PER_TITLE = 5


def _norm(title: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace (mirrors corpus.normalize_title)."""
    cleaned = re.sub(r"[^\w\s]", " ", (title or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _is_match(seed_norm: str, candidate_norm: str) -> bool:
    """True if the candidate's title plausibly is the seed (substring or ≥70% token overlap)."""
    if not seed_norm or not candidate_norm:
        return False
    if seed_norm in candidate_norm or candidate_norm in seed_norm:
        return True
    seed_tokens = set(seed_norm.split())
    cand_tokens = set(candidate_norm.split())
    if not seed_tokens:
        return False
    overlap = len(seed_tokens & cand_tokens) / len(seed_tokens)
    return overlap >= _TOKEN_OVERLAP_THRESHOLD


def fetch_seed_papers(titles: list[str]) -> tuple[list[Paper], list[str]]:
    """Fetch each title via OpenAlex search; return (resolved papers, list of missed titles)."""
    resolved: list[Paper] = []
    missed: list[str] = []
    for title in titles:
        seed_norm = _norm(title)
        candidates = openalex.fetch_works(title, n=_CANDIDATES_PER_TITLE)
        match = next(
            (c for c in candidates if _is_match(seed_norm, _norm(c.title or ""))),
            None,
        )
        if match is None:
            missed.append(title)
        else:
            resolved.append(match)
    return resolved, missed
