"""Online retrieval: embed a query (short or paper-as-query), FAISS search, citation subgraph."""

from __future__ import annotations

import json

from schemas import Paper

from . import config
from .build_graph import load_graph
from .embed import embed_documents, embed_short_query
from .fetch import load_papers

_index = None
_ids: list[str] | None = None
_papers_by_id: dict[str, Paper] | None = None
_graph = None


def _ensure_loaded() -> None:
    """Lazily load index, id order, paper lookup, and citation graph (once)."""
    global _index, _ids, _papers_by_id, _graph
    if _index is None:
        import faiss

        _index = faiss.read_index(str(config.FAISS_INDEX))
        _ids = json.loads(config.IDS_JSON.read_text())
        _papers_by_id = {p.paper_id: p for p in load_papers()}
        _graph = load_graph()


def search(text: str, mode: str = "short", top_k: int = config.TOP_K) -> list[tuple[Paper, float]]:
    """Retrieve top-k papers for a short query (mode='short') or pasted abstract (mode='paper')."""
    _ensure_loaded()
    vec = embed_documents([text]) if mode == "paper" else embed_short_query([text])
    scores, idx = _index.search(vec.astype("float32"), min(top_k, len(_ids)))
    results: list[tuple[Paper, float]] = []
    for j, s in zip(idx[0], scores[0]):
        if j < 0:
            continue
        paper = _papers_by_id.get(_ids[j])
        if paper is not None:
            results.append((paper, float(s)))
    return results


def subgraph_for(paper_ids: list[str], hops: int = 1):
    """Return the directed subgraph around the given paper ids (plus 1-hop neighbors)."""
    _ensure_loaded()
    undirected = _graph.to_undirected()
    nodes: set[str] = set()
    for pid in paper_ids:
        if pid in undirected:
            nodes.add(pid)
            if hops >= 1:
                nodes.update(undirected.neighbors(pid))
    return _graph.subgraph(nodes).copy()
