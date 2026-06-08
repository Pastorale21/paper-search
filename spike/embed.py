"""SPECTER2 embeddings via the adapters library (proximity + ad-hoc query adapters)."""

from __future__ import annotations

import json
import os
import shutil

import numpy as np

from . import config
from .fetch import load_papers

_tokenizer = None
_model = None
_hf_symlink_patch_applied = False


def _device() -> str:
    return config.DEVICE


def _patch_hf_windows_symlink_fallback() -> None:
    """Copy files when HuggingFace symlink creation fails on non-admin Windows."""
    global _hf_symlink_patch_applied
    if _hf_symlink_patch_applied or os.name != "nt":
        return

    try:
        import huggingface_hub.file_download as file_download
    except ImportError:
        return

    original_create_symlink = file_download._create_symlink

    def _create_symlink_or_copy(src: str, dst: str, new_blob: bool = False) -> None:
        try:
            original_create_symlink(src, dst, new_blob=new_blob)
        except OSError as exc:
            if getattr(exc, "winerror", None) != 1314:
                raise
            abs_src = os.path.abspath(os.path.expanduser(src))
            abs_dst = os.path.abspath(os.path.expanduser(dst))
            os.makedirs(os.path.dirname(abs_dst), exist_ok=True)
            if new_blob:
                shutil.move(abs_src, abs_dst)
            else:
                shutil.copyfile(abs_src, abs_dst)

    file_download._create_symlink = _create_symlink_or_copy
    _hf_symlink_patch_applied = True


def _load_model():
    """Lazily load the SPECTER2 base model with both adapters registered."""
    global _tokenizer, _model
    if _model is not None:
        return _tokenizer, _model
    _patch_hf_windows_symlink_fallback()
    from adapters import AutoAdapterModel
    from transformers import AutoTokenizer

    _tokenizer = AutoTokenizer.from_pretrained(config.MODEL_BASE)
    _model = AutoAdapterModel.from_pretrained(config.MODEL_BASE)
    _model.load_adapter(config.ADAPTER_PROX, source="hf", load_as="proximity", set_active=True)
    _model.load_adapter(config.ADAPTER_QUERY, source="hf", load_as="adhoc_query")
    _model.to(_device())
    _model.eval()
    return _tokenizer, _model


def _encode(texts: list[str], adapter: str, batch_size: int = 16) -> np.ndarray:
    """Encode texts with the given active adapter; return L2-normalized CLS vectors."""
    import torch

    tok, model = _load_model()
    model.set_active_adapters(adapter)
    device = _device()
    out: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(
            device
        )
        with torch.no_grad():
            res = model(**inputs)
        cls = res.last_hidden_state[:, 0, :]
        cls = torch.nn.functional.normalize(cls, p=2, dim=1)
        out.append(cls.cpu().numpy())
    if not out:
        return np.zeros((0, config.EMBED_DIM), dtype="float32")
    return np.vstack(out).astype("float32")


def _paper_text(title: str, abstract: str | None, sep: str) -> str:
    return f"{title}{sep}{abstract or ''}"


def embed_documents(texts: list[str]) -> np.ndarray:
    """Embed full documents (corpus papers, paste-abstract queries) with the proximity adapter."""
    return _encode(texts, "proximity")


def embed_short_query(texts: list[str]) -> np.ndarray:
    """Embed short text queries with the ad-hoc query adapter."""
    return _encode(texts, "adhoc_query")


def build_corpus_embeddings(force: bool = False) -> tuple[np.ndarray, list[str]]:
    """Embed all cached papers (proximity adapter) and persist vectors + id order."""
    if config.EMBEDDINGS_NPY.exists() and config.IDS_JSON.exists() and not force:
        emb = np.load(config.EMBEDDINGS_NPY)
        ids = json.loads(config.IDS_JSON.read_text(encoding="utf-8"))
        print(f"[embed] cache hit: {emb.shape}")
        return emb, ids
    papers = load_papers()
    tok, _ = _load_model()
    sep = tok.sep_token or " "
    texts = [_paper_text(p.title, p.abstract, sep) for p in papers]
    emb = embed_documents(texts)
    ids = [p.paper_id for p in papers]
    np.save(config.EMBEDDINGS_NPY, emb)
    config.IDS_JSON.write_text(json.dumps(ids), encoding="utf-8")
    print(f"[embed] embedded {emb.shape[0]} papers, dim={emb.shape[1]}")
    return emb, ids
