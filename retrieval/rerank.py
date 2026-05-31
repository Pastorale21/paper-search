"""Cross-encoder reranker (sentence-transformers).

Performance: ~5-10s on CPU for 50 (query, doc) pairs with ms-marco-MiniLM-L-6-v2. The
UI layer should cache rerank results per (query, candidate-id-tuple); a fresh rerank on
every keystroke is unusable.

The model is cached at module scope via lru_cache so multiple `CrossEncoderReranker(...)`
instantiations share one loaded model — a per-instance lru_cache on a method would key on
`self` and defeat the cache.
"""

from __future__ import annotations

import json
from functools import lru_cache

from spike import config

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    """Lazily load a sentence-transformers CrossEncoder, cached across instances."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


@lru_cache(maxsize=1)
def _papers_by_id() -> dict[str, dict]:
    """Lazily build paper_id -> {title, abstract} from the corpus cache (cached)."""
    raw = json.loads(config.PAPERS_JSON.read_text(encoding="utf-8"))
    return {p["paper_id"]: p for p in raw}


class CrossEncoderReranker:
    """Cross-encoder reranker over (query, title+abstract) pairs."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name

    @property
    def model(self):
        """The shared CrossEncoder instance for this model name."""
        return _load_model(self.model_name)

    def rerank(
        self, query: str, candidates: list[tuple[str, float]], k: int
    ) -> list[tuple[str, float]]:
        """Re-rank candidates by cross-encoder score; returns top-k as (paper_id, ce_score)."""
        if not candidates:
            return []
        papers = _papers_by_id()
        pairs: list[tuple[str, str]] = []
        kept_ids: list[str] = []
        for pid, _ in candidates:
            p = papers.get(pid)
            if p is None:
                continue
            doc = (p.get("title") or "") + ". " + (p.get("abstract") or "")
            pairs.append((query, doc))
            kept_ids.append(pid)
        if not pairs:
            return []
        scores = self.model.predict(pairs)
        ranked = sorted(zip(kept_ids, scores), key=lambda x: float(x[1]), reverse=True)
        return [(pid, float(s)) for pid, s in ranked[:k]]
