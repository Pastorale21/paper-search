"""Spike-wide constants, cache paths, and SPECTER2 adapter names."""

from __future__ import annotations

import os
from pathlib import Path

# Guard against the torch + faiss double-libomp segfault on macOS. Must be set before
# torch / faiss are imported anywhere, so it lives at the top of the first-imported module.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Embedding device: CPU by default for spike robustness (MPS can crash on some ops); override
# with SPIKE_DEVICE=mps|cuda|cpu.
DEVICE = os.getenv("SPIKE_DEVICE", "cpu")

# --- corpus / query ---
QUERY = "graph neural network recommendation"
N_PAPERS = 100
TOP_K = 10

# --- data source backend: "openalex" (free, no key) or "s2" (needs key for reliability) ---
DATA_SOURCE = os.getenv("DATA_SOURCE", "openalex")

# --- Semantic Scholar Graph API ---
S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = (
    "title,abstract,year,authors,citationCount,externalIds," "references.paperId,citations.paperId"
)
S2_API_KEY = os.getenv("S2_API_KEY") or None

# --- OpenAlex API (free, no key; abstracts via inverted index, edges via referenced_works) ---
OPENALEX_URL = "https://api.openalex.org/works"
OPENALEX_SELECT = (
    "id,title,publication_year,authorships,cited_by_count,referenced_works,"
    "abstract_inverted_index,ids"
)
OPENALEX_MAILTO = os.getenv("OPENALEX_MAILTO") or None  # joins the faster "polite pool" if set

# --- SPECTER2 (adapters library) ---
MODEL_BASE = "allenai/specter2_base"
ADAPTER_PROX = "allenai/specter2"  # proximity: documents / paper-to-paper
ADAPTER_QUERY = "allenai/specter2_adhoc_query"  # ad-hoc short text queries
EMBED_DIM = 768

# --- LLM (OpenAI-compatible; DeepSeek default) ---
LLM_API_KEY = os.getenv("LLM_API_KEY") or None
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# --- cache paths (resolved against project root, robust to cwd) ---
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PAPERS_JSON = CACHE_DIR / "papers.json"
EMBEDDINGS_NPY = CACHE_DIR / "embeddings.npy"
IDS_JSON = CACHE_DIR / "ids.json"
FAISS_INDEX = CACHE_DIR / "faiss.index"
GRAPH_PKL = CACHE_DIR / "citation_graph.pkl"

# --- fixed eval queries for the smoke test: (mode, text) ---
_PAPER_AS_QUERY_ABSTRACT = (
    "We propose a graph neural network for collaborative filtering that propagates "
    "user and item embeddings over the user-item interaction graph. By stacking "
    "light-weight linear propagation layers without feature transformation or "
    "nonlinear activation, the model captures high-order connectivity and outperforms "
    "strong matrix-factorization baselines on top-K recommendation benchmarks."
)
EVAL_QUERIES: list[tuple[str, str]] = [
    ("short", "graph contrastive learning for recommendation"),
    ("short", "self-supervised collaborative filtering"),
    ("paper", _PAPER_AS_QUERY_ABSTRACT),
]
