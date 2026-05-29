"""Build a FAISS inner-product index over corpus embeddings (cosine via normalized vectors)."""

from __future__ import annotations

from . import config
from .embed import build_corpus_embeddings


def build_index(force: bool = False):
    """Build + persist a FAISS IndexFlatIP from cached embeddings."""
    import faiss

    if config.FAISS_INDEX.exists() and not force:
        index = faiss.read_index(str(config.FAISS_INDEX))
        print(f"[index] cache hit: ntotal={index.ntotal}")
        return index
    emb, _ids = build_corpus_embeddings(force=force)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(config.FAISS_INDEX))
    print(f"[index] built FAISS IndexFlatIP, ntotal={index.ntotal}")
    return index
