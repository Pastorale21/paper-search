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
import sys

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
    "Contrastive Cross-domain Recommendation in Matching",  # CCDR
    "DisenCDR",  # disentangled cross-domain recommendation
    "DDTCDR: Deep Dual Transfer Cross Domain Recommendation",
    "Cross-Domain Recommendation via Preference Propagation GraphNet",  # PPGN
    "Session-based Recommendation with Graph Neural Networks",  # SR-GNN
    "Graph Contextualized Self-Attention Network for Session-based Recommendation",  # GC-SAN
    "S3-Rec: Self-Supervised Learning for Sequential Recommendation",
    "Contrastive Learning for Sequential Recommendation",  # CL4SRec
    "Sequential Recommendation with Graph Neural Networks",  # SURGE
    # LESSR (this title is LESSR's paper, not FGNN — the real FGNN is added in the gap block below)
    "Handling Information Loss of Graph Neural Networks for Session-based Recommendation",
    "Global Context Enhanced Graph Neural Networks for Session-based Recommendation",  # GCE-GNN
    "TAGNN: Target Attentive Graph Neural Networks for Session-based Recommendation",
    "A Neural Influence Diffusion Model for Social Recommendation",  # DiffNet (canonical title)
    "DiffNet++: A Neural Influence and Interest Diffusion Network for Social Recommendation",
    # MHCN
    "Self-Supervised Multi-Channel Hypergraph Convolutional Network for Social Recommendation",
    "Graph Neural Networks for Social Recommendation",  # GraphRec
    "SocialLGN: Light graph convolution network for social recommendation",
    "Knowledge-aware Coupled Graph Neural Network for Social Recommendation",  # KCGN
    "Collaborative Knowledge Base Embedding for Recommender Systems",  # CKE
    "Collaborative Knowledge-aware Attentive Network for Recommender Systems",  # CKAN
    "XSimGCL: Towards Extremely Simple Graph Contrastive Learning for Recommendation",
    # --- Gold-set gaps requested in docs/corpus_gap_request_for_a.md (titles per that doc) ---
    # P0: directly affect the paper-query same-subset / unblock an anchor.
    "Knowledge Graph Convolutional Networks for Recommender Systems",  # KGCN (P0)
    # P1: repeated gold gaps across session / scalable-CF / sequential-contrastive queries.
    "Feature Graph Neural Networks for Session-based Recommendation",  # FGNN (P1)
    "Graph Filter Collaborative Filtering",  # GFCF (P1)
    # DuoRec (P1)
    "Contrastive Learning for Representation Degeneration Problem in Sequential Recommendation",
    "Contrastive Learning for Sequential Recommendation with Robust Augmentation",  # CoSeRec (P1)
    "Intent Contrastive Learning for Sequential Recommendation",  # ICLRec (P1)
    # P2 (add-if-time): hypergraph / disentanglement / cold-start / cross-domain breadth.
    "Dual Channel Hypergraph Collaborative Filtering",  # DHCF (P2)
    "Hypergraph Convolutional Network for Collaborative Filtering",  # HGCN (P2)
    "Next-item Recommendation with Sequential Hypergraphs",  # HyperRec (P2)
    "DisenHAN: Disentangled Heterogeneous Graph Attention Network for Recommendation",  # P2
    "Disentangled Contrastive Collaborative Filtering",  # DCCF (P2)
    # MetaHIN (P2)
    "Meta-learning on Heterogeneous Information Networks for Cold-start Recommendation",
    "Personalized Transfer of User Preferences for Cross-domain Recommendation",  # PTUPCDR (P2)
    # Fairness coverage (P2, Q13). NOTE: several fairness gold ALIASES in eval/gold_set.py are
    # bare acronyms that won't substring-match these real titles — D may need an alias follow-up
    # for Q13 to resolve even after these crawl in. Flagged in the PR, not fixed here (D's domain).
    "FairRec: Two-Sided Fairness for Personalized Recommendations",  # FairRec
    "Learning Fair Representations for Recommendation: A Graph-based Perspective",  # FairGo
    "Debiasing Career Recommendations with Neural Fair Collaborative Filtering",  # NFCF
    "Say No to the Discrimination: Learning Fair Graph Neural Networks",  # FairGNN
]

_TOKEN_OVERLAP_THRESHOLD = 0.70

# Candidates per title fetched from OpenAlex. Small (3-5) is enough because the search
# endpoint already ranks by relevance — but >1 so we can fall back if the top hit is wrong
# (e.g. a "Comments on …" / survey that mentions the canonical paper).
_CANDIDATES_PER_TITLE = 8

_MANUAL_SEEDS: dict[str, Paper] = {
    "collaborative knowledge aware attentive network for recommender systems": Paper(
        paper_id="manual_CKAN",
        title="CKAN: Collaborative Knowledge-aware Attentive Network for Recommender Systems",
        abstract=(
            "A knowledge-aware recommendation model that uses attentive propagation over "
            "collaborative signals and knowledge graph connections for top-N recommendation."
        ),
        source_ids={"seed": "CKAN"},
    ),
    "sociallgn light graph convolution network for social recommendation": Paper(
        paper_id="W4205132462",
        title="SocialLGN: Light graph convolution network for social recommendation",
        abstract=(
            "A social recommendation model based on light graph convolution that jointly "
            "uses user-item interactions and social relations for preference propagation."
        ),
        year=2022,
        citation_count=170,
        source_ids={
            "openalex": "W4205132462",
            "doi": "10.1016/j.ins.2022.01.001",
        },
    ),
}


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
        manual = _MANUAL_SEEDS.get(seed_norm)
        if manual is not None:
            resolved.append(manual)
            continue
        try:
            candidates = openalex.fetch_works(title, n=_CANDIDATES_PER_TITLE)
        except Exception as exc:  # noqa: BLE001 — one flaky OpenAlex call must not abort the crawl
            print(f"[seed] fetch failed for {title!r}: {exc}; skipping", file=sys.stderr)
            missed.append(title)
            continue
        match = next(
            (c for c in candidates if _is_match(seed_norm, _norm(c.title or ""))),
            None,
        )
        if match is None:
            missed.append(title)
        else:
            resolved.append(match)
    return resolved, missed
