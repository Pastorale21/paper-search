"""Method-card field-level matching — the project's differentiation core.

The thesis (per the spike write-up §3, §5): dense retrieval saturates inside topic clusters
(top-5 cosines bunched in ~0.95-0.96, spread ~0.006), so it can't pick the *right* GNN-recsys
paper for a query. Matching the *mechanism* — task / backbone / loss / key_idea — breaks
those ties because two LightGCN-family papers may share a topic but differ on backbone (GCN
vs hyperbolic GCN), loss (BPR vs InfoNCE), etc.

Scoring rule (per field): cosine of SPECTER2 embeddings of the two field values (same
adapter both sides → scores are comparable). Field cosines are combined as a weighted sum
via FIELD_WEIGHTS. **Empty-field behaviour ("weight unspent"): a field empty on either side
contributes 0** — a card with only task+backbone filled caps at sum(weights of present
fields) = 0.50 on a perfect match. This structurally penalizes sparse method cards and is
flagged as a tuning knob (see TODO above FIELD_WEIGHTS and retrieval/HANDOFF.md known-issues).

Field embeddings for the corpus are precomputed once to
`data/cache/method_card_field_embeddings.npz` (keyed `paper_id::field`) and invalidated when
any card file's mtime is newer than the cache. Query-side embeddings are computed on demand.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from nlp import config as nlp_config
from schemas import MethodCard
from spike import config as spike_config

from .dense import DenseRetriever

logger = logging.getLogger(__name__)

FIELDS_TO_MATCH = ["task", "backbone", "loss", "key_idea"]

# TODO(C): the current scoring uses "weight unspent" on empty fields, i.e. a card with
# only task+backbone filled caps at sum(weights of present fields) = 0.50 on a perfect
# match. Minor today (88-96% fill) but biases against sparse cards. Alternative to try
# in eval: renormalize by sum-of-weights-of-fields-present-on-BOTH-sides, so a card with
# 2/4 fields filled but perfectly matching scores 1.0. Ablate via nDCG@5 on the gold set.
# TODO(C): tune via eval/run.py nDCG comparison on gold set
FIELD_WEIGHTS = {
    "task": 0.20,
    "backbone": 0.30,
    "loss": 0.20,
    "key_idea": 0.30,
}

FIELD_EMB_NPZ: Path = spike_config.CACHE_DIR / "method_card_field_embeddings.npz"


def _load_method_cards() -> dict[str, MethodCard]:
    """Load every cached method card into a dict keyed by paper_id."""
    out: dict[str, MethodCard] = {}
    if not nlp_config.METHOD_CARDS_DIR.exists():
        return out
    for p in sorted(nlp_config.METHOD_CARDS_DIR.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        out[d["paper_id"]] = MethodCard.from_dict(d)
    return out


def _key(paper_id: str, field: str) -> str:
    return f"{paper_id}::{field}"


def _field_value(card: MethodCard | None, field: str) -> str:
    if card is None:
        return ""
    v = getattr(card, field, None)
    return v.strip() if isinstance(v, str) else ""


def _cards_newer_than(cache_path: Path) -> bool:
    """True if any cached method card has been modified after the field-embedding cache."""
    if not cache_path.exists():
        return True
    cache_mtime = cache_path.stat().st_mtime
    if not nlp_config.METHOD_CARDS_DIR.exists():
        return False
    return any(p.stat().st_mtime > cache_mtime for p in nlp_config.METHOD_CARDS_DIR.glob("*.json"))


class MethodCardMatcher:
    """Score candidate papers by weighted field-level cosine of their method cards."""

    def __init__(
        self,
        embedder: DenseRetriever | None = None,
        *,
        normalize: bool = False,
        min_comparable_fields: int = 1,
    ) -> None:
        # normalize=False / min_comparable_fields=1 reproduce the original "weight unspent"
        # scoring (the shipped default); the normalized variants are eval ablations only.
        self.embedder = embedder or DenseRetriever()
        self.cards: dict[str, MethodCard] = _load_method_cards()
        self._cache: dict[str, np.ndarray] = {}
        self.normalize = normalize
        self.min_comparable_fields = min_comparable_fields
        self._load_or_build_corpus_field_cache()

    def with_config(self, *, normalize: bool, min_comparable_fields: int) -> "MethodCardMatcher":
        """Return a reconfigured matcher SHARING this one's loaded cards + field cache (no IO)."""
        other = self.__class__.__new__(self.__class__)
        other.embedder = self.embedder
        other.cards = self.cards
        other._cache = self._cache
        other.normalize = normalize
        other.min_comparable_fields = min_comparable_fields
        return other

    def _combine(self, pairs: list[tuple[str, float]]) -> float:
        """Combine per-field (field, cosine) pairs into one score.

        ``pairs`` holds only fields non-empty on BOTH papers. Default (normalize=False) is the
        original weighted sum with weight unspent on missing fields. When normalize=True the sum
        is divided by the weights of the comparable fields, so a sparse-but-perfect card is not
        structurally capped. Returns 0.0 (never NaN) when fewer than ``min_comparable_fields``
        fields are comparable.
        """
        if len(pairs) < self.min_comparable_fields:
            return 0.0
        num = sum(FIELD_WEIGHTS[f] * cos for f, cos in pairs)
        if not self.normalize:
            return num
        denom = sum(FIELD_WEIGHTS[f] for f, _ in pairs)
        return num / denom if denom else 0.0

    def _load_or_build_corpus_field_cache(self) -> None:
        if not _cards_newer_than(FIELD_EMB_NPZ) and FIELD_EMB_NPZ.exists():
            with np.load(FIELD_EMB_NPZ) as npz:
                self._cache = {k: npz[k] for k in npz.files}
            return
        # (Re)build: embed every non-empty (paper_id, field) value.
        keys: list[str] = []
        texts: list[str] = []
        for pid, card in self.cards.items():
            for f in FIELDS_TO_MATCH:
                val = _field_value(card, f)
                if val:
                    keys.append(_key(pid, f))
                    texts.append(val)
        if not texts:
            logger.warning("no non-empty method-card fields to embed; cache is empty")
            self._cache = {}
            np.savez(FIELD_EMB_NPZ, **{})
            return
        embs = self._embed_batch(texts)
        self._cache = {k: embs[i] for i, k in enumerate(keys)}
        FIELD_EMB_NPZ.parent.mkdir(parents=True, exist_ok=True)
        np.savez(FIELD_EMB_NPZ, **self._cache)

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Batched proximity-adapter embedding via spike.embed (re-uses loaded SPECTER2)."""
        from spike import embed as spike_embed

        return spike_embed.embed_documents(texts)

    def _get_or_embed_query_field(self, value: str) -> np.ndarray | None:
        """Embed a query-side field value (no cache; queries are cheap)."""
        if not value:
            return None
        emb = self._embed_batch([value])
        return emb[0]

    def _candidate_field_emb(self, pid: str, field: str) -> np.ndarray | None:
        return self._cache.get(_key(pid, field))

    def similarity(self, pid_a: str, pid_b: str) -> float:
        """Weighted field-cosine similarity between two corpus papers (uses cached embeddings).

        Returns 0.0 if either paper has no embedded fields in common. Same "weight unspent"
        behaviour as ``match`` — fields empty on either side contribute 0; see the module
        docstring for the structural-cap caveat.
        """
        pairs: list[tuple[str, float]] = []
        for field in FIELDS_TO_MATCH:
            ea = self._candidate_field_emb(pid_a, field)
            eb = self._candidate_field_emb(pid_b, field)
            if ea is None or eb is None:
                continue
            pairs.append((field, float(np.dot(ea, eb))))
        return self._combine(pairs)

    def has_comparable_fields(self, pid_a: str, pid_b: str) -> bool:
        """True iff at least one ``FIELDS_TO_MATCH`` is non-empty on BOTH papers.

        When False, ``similarity(pid_a, pid_b)`` returns 0.0 not because the papers are
        mechanism-distant but because we have no data to compare — useful for filtering
        out papers (e.g. surveys with empty method cards) that would otherwise dominate
        mechanism-distance rankings with a spurious "distance 1.0".
        """
        for field in FIELDS_TO_MATCH:
            if (
                self._candidate_field_emb(pid_a, field) is not None
                and self._candidate_field_emb(pid_b, field) is not None
            ):
                return True
        return False

    def match(
        self,
        query_paper_id: str | None,
        query_card: MethodCard | None,
        candidates: list[str],
        k: int,
    ) -> list[tuple[str, float]]:
        """Score candidates by weighted field-cosine; return top-k (paper_id, score)."""
        if (query_paper_id is None) == (query_card is None):
            raise ValueError("Provide exactly one of query_paper_id or query_card.")
        if query_paper_id is not None:
            q_card = self.cards.get(query_paper_id)
            if q_card is None:
                logger.warning("no method card for query paper_id=%s", query_paper_id)
                return []
        else:
            q_card = query_card

        query_field_embs: dict[str, np.ndarray] = {}
        for f in FIELDS_TO_MATCH:
            val = _field_value(q_card, f)
            emb = self._get_or_embed_query_field(val)
            if emb is not None:
                query_field_embs[f] = emb

        # Exclude the query paper itself when an anchor id was given — otherwise the
        # anchor's card matches itself perfectly (cosine = 1) and inflates the top-5 spread
        # by a measurement artifact rather than real differentiation.
        scored: list[tuple[str, float]] = []
        any_card_found = False
        for pid in candidates:
            if query_paper_id is not None and pid == query_paper_id:
                continue
            if pid in self.cards:
                any_card_found = True
            pairs: list[tuple[str, float]] = []
            for f, q_emb in query_field_embs.items():
                c_emb = self._candidate_field_emb(pid, f)
                if c_emb is None:
                    continue  # "weight unspent" — see module docstring + _combine
                pairs.append((f, float(np.dot(q_emb, c_emb))))
            scored.append((pid, self._combine(pairs)))

        if not any_card_found:
            logger.warning("no method cards in candidate set; method_match contributes nothing")
            return []

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
