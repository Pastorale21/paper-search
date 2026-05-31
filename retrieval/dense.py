"""Dense retriever: thin wrapper around spike's SPECTER2 + FAISS pipeline.

Reuses `spike.search.search` (which lazily loads the FAISS index, ids.json, and the SPECTER2
dual adapters) so we do NOT re-load SPECTER2 from disk for the retrieval layer. The spike
module is frozen reference per CLAUDE.md; we consume it as a library only.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from spike import embed, search


class DenseRetriever:
    """SPECTER2 + FAISS retriever returning (paper_id, cosine_score) pairs."""

    def search(
        self, query: str, mode: Literal["short", "paper"] = "short", k: int = 10
    ) -> list[tuple[str, float]]:
        """Top-k papers for a short query or pasted-abstract query."""
        return [(p.paper_id, score) for p, score in search.search(query, mode=mode, top_k=k)]

    def embed_field(self, text: str) -> np.ndarray:
        """Embed an arbitrary short field value with the proximity adapter (shape: (1, dim))."""
        return embed.embed_documents([text or ""])
