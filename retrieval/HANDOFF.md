# Retrieval Layer — Handoff

## You are
**C — retrieval + reranking owner.** This module stays with you; the doc exists so the
next-semester maintainer (or anyone debugging from the UI layer) has the load-bearing
context in one place.

## Current state
Implemented + runnable:
- `retrieval/dense.py` — `DenseRetriever`, thin wrapper around `spike.search.search` (shares
  the spike's lazy-loaded SPECTER2 dual adapters + FAISS index — no SPECTER2 re-load).
- `retrieval/bm25.py` — `BM25Retriever` over `title + abstract`, pickled to
  `data/cache/bm25.pkl`. Cache validates against the current `papers.json` paper-id order
  and rebuilds on mismatch. CLI: `uv run python -m retrieval.bm25 --rebuild`.
- `retrieval/rerank.py` — `CrossEncoderReranker` using `sentence-transformers`
  (`ms-marco-MiniLM-L-6-v2` by default). Model is cached at module scope via
  `@lru_cache(maxsize=2)` so re-instantiating the class does NOT re-load the model.
- `retrieval/method_match.py` ⭐ — `MethodCardMatcher`. Field-level cosine over
  `task / backbone / loss / key_idea`. Corpus-side field embeddings precomputed once to
  `data/cache/method_card_field_embeddings.npz` (keyed `paper_id::field`), invalidated when
  any card file's mtime is newer than the cache.
- `retrieval/hybrid.py` — `HybridRetriever`, **Reciprocal Rank Fusion** (RRF, k=60) of the
  three signals + optional cross-encoder rerank. Returns
  `(paper_id, final_score, signal_breakdown)`; `signal_breakdown` carries RAW per-signal
  scores (the UI's "reason tag" data).
- `retrieval/compare.py` — side-by-side comparison CLI:
  `uv run python -m retrieval.compare --query "<abstract>" --mode paper --k 5`
  or `--eval-queries` to run `spike.config.EVAL_QUERIES` and dump
  `data/cache/eval/retrieval_comparison.json`.

## Open tuning knobs

- **`method_match.FIELD_WEIGHTS`** (`{task:0.20, backbone:0.30, loss:0.20, key_idea:0.30}`).
  Ablate each, watch nDCG@5 on the gold set.
- **`hybrid.DEFAULT_WEIGHTS`** (`{dense:0.4, bm25:0.2, method_match:0.4}`). Initial guess
  favors dense + method_match. Sweep and pick the elbow.
- **`hybrid.RRF_K`** (60 by default). Smaller k → top ranks dominate; larger k → tail
  contributes more. Standard value is 60; rarely needs tuning, but worth a sanity sweep.
- **Cross-encoder model** — currently `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast).
  `BAAI/bge-reranker-v2-m3` is higher-quality but heavier; consider behind a flag if the
  UI can afford the latency.

## Interface contracts (consumers depend on these)

- **`HybridRetriever.search(...) -> list[tuple[paper_id, final_score, signal_breakdown]]`** —
  `signal_breakdown` is a `dict` with keys `{"dense", "bm25", "method_match"}` (values are
  raw per-signal scores or `None` if the signal didn't surface this paper) and additionally
  `"cross_encoder"` when reranked. The UI's "reason tag" feature reads this dict directly.
- **`DenseRetriever.search`, `BM25Retriever.search`** — both return
  `list[tuple[paper_id, float]]`, sorted descending by score.
- **`MethodCardMatcher.match(query_paper_id, query_card, candidates, k)`** — exactly one of
  `query_paper_id` / `query_card` is required.

## Known issues / gotchas

- **Cross-encoder is CPU-slow** (~5-10s for 50 pairs with the MiniLM model). Cache
  aggressively in the UI layer (key by `(query, sorted-tuple-of-candidate-ids)`); a fresh
  rerank on every keystroke is unusable.
- **Method-card scores are NOT magnitude-comparable to dense cosines.** Hybrid uses RRF
  precisely because of this — fusing on raw scores would let BM25 (unbounded) swamp dense
  (bounded in [-1, 1]) or vice-versa. RRF is rank-based.
- **`MethodCardMatcher` "weight unspent" on empty fields.** A card with only task+backbone
  filled caps at `sum(weights of present fields) = 0.50` on a perfect match — structurally
  biases against sparse cards. Minor today (88-96% field fill) but worth ablating. The
  alternative — renormalize by sum-of-weights-of-fields-present-on-BOTH-sides — is flagged
  as a TODO above `FIELD_WEIGHTS` and is the experimentation knob.
- **Field embedding cache is keyed on file mtime, not content hash.** Touching a method-card
  file (e.g. via `touch`) without changing content will trigger a rebuild. Acceptable trade
  for simplicity; switch to a content hash if it becomes painful.
- **BM25 cache invalidates on `paper_id` order change** (same staleness gotcha as the
  spike index — see data/HANDOFF.md). Rebuild after any corpus rebuild.

## Reality-check protocol (when tuning weights)

For any change to `FIELD_WEIGHTS` or `DEFAULT_WEIGHTS`, run:
```
uv run python -m retrieval.compare --query "<LightGCN-style abstract>" --mode paper --k 5
```
Check BOTH:
- **Quantitative**: `method_match` top-5 spread ≥ 0.05 (dense spread is ~0.006 in this corpus).
- **Qualitative**: method_match's top-5 should be MORE mechanism-relevant than dense's top-5
  (e.g. for a LightGCN-style query: NGCF, GC-MC, UltraGCN, LR-GCCF, DGCF climbing higher).
A large spread that ranks the WRONG papers high is a failure, not a pass.

## Graph reasoning (`retrieval/graph_reason.py`)

**Positioning — read this first.** Graph reasoning is the project's second
differentiation pillar but it is a **qualitative / demo feature** producing
*traceable path explanations* the UI can show as "why this paper was returned." It is
**NOT eval-validated** the way `method_match` is — no nDCG@5 gate, no headline number.
The deliverable is human-legible per-result trails; ship that, then the report can
quote the LightGCN/KGAT reality-check transcripts directly.

### What's implemented

Three public queries plus a CLI:

```bash
uv run python -m retrieval.graph_reason --paper-id W3045200674 --query ancestors    --k 5
uv run python -m retrieval.graph_reason --paper-id W3045200674 --query opposing     --k 5
uv run python -m retrieval.graph_reason --paper-id W3045200674 --query cross-domain --k 5
```

- **`find_ancestors(paper_id, max_hops=3, k=10)`** — BFS over OUT-edges
  (`g.add_edge(citing, cited)`, so out = papers this one cites). Per-path score is
  `sum(intent_weights) / hop_count × method_card_similarity`. Foundational papers (out=0)
  return `[]` with a clear CLI message, not a crash.
- **`find_opposing(paper_id, k=10)`** — 1-hop neighbors (cites ∪ cited-by) filtered to
  `comparison`-intent edges; otherwise ranked by **mechanism distance** (`1 - similarity`).
  See "Two structural limits" below.
- **`find_cross_domain_same_mechanism(paper_id, k=10)`** — top-30 by method-card similarity
  via `MethodCardMatcher.match()`, filtered to a different inferred sub-area. Direct citation
  neighbors are slightly penalized so genuinely independent works rank higher.

### Two structural limits (state these honestly in the report)

1. **Graph OUT-sparsity.** Canonical papers in the current 400-corpus have very few
   out-edges (LightGCN out=4, NGCF out=2, **GC-MC out=0**, **SR-GNN out=0**) because
   spike filters references to in-corpus targets only. `find_ancestors` therefore works
   downstream → upstream (recent papers trace back to canonical works) and legitimately
   returns `[]` for foundational anchors. **Fix is corpus-side**: when A scales 400 → 800
   (data/HANDOFF.md P0), out-density grows and multi-hop trails get richer. No code change
   needed here.

2. **No edge intents on disk.** `data/sources/s2_contexts.py::fetch_citation_contexts` is
   stubbed (NotImplementedError), so no per-edge context text exists for
   `nlp.citation_intent.CitationIntentClassifier` to label. Every edge defaults to
   `"background"` intent via `get_edge_intent(g, u, v)`. This means **`find_opposing` runs
   on the mechanism-distance fallback today** — explanations include the
   `[no intent metadata; mechanism-distance fallback]` banner so the user/UI knows. When B
   ships SciCite + A wires `s2_contexts`, populate `g.edges[u, v]["intent"]` during graph
   build and all three queries pick up real intents automatically — no API change.

### Tunable knobs

- **`graph_reason.INTENT_WEIGHTS`** (`{method: 1.0, background: 0.5, comparison: 0.2}`).
  TODO(C): tune by subjective evaluation on 5 known queries — no clean nDCG signal here
  (paths are evidence, not predictions).
- **`graph_reason.SUB_AREA_KEYWORDS`** — keyword heuristic for sub-area inference. TODO(C):
  replace with corpus-recorded sub_area when `data/corpus.py` persists `origin` into
  papers.json (today `origin` is internal to the build).

### Reusable building block

- **`MethodCardMatcher.similarity(pid_a, pid_b) -> float`** is the clean integration point
  for any cross-paper mechanism comparison (graph_reason uses it for ancestor weighting,
  opposing distance, and cross-domain filtering). Returns 0.0 when either paper has no
  cached field embeddings.

### Known issues / gotchas

- Sub-area inference is keyword-based and can misfire on titles that drop conventional
  vocabulary; eyeball cross-domain results when adding new queries.
- The mechanism-distance fallback for `find_opposing` is **not** the same as "papers that
  argue against this paper" — it just ranks neighbors by how different their *mechanism*
  is. Document this clearly in any UI surface; mislabeling it as "papers disagreeing with"
  would be misleading.

## Tools to learn (for whoever picks this up next)

- `rank_bm25`: https://github.com/dorianbrown/rank_bm25
- sentence-transformers CrossEncoder: https://sbert.net/docs/cross_encoder/usage/usage.html
- Reciprocal Rank Fusion (Cormack et al. 2009): the original 2-page paper is enough.
- NetworkX `DiGraph` BFS — `successors` is the load-bearing call in `find_ancestors`.

## Commands

```bash
uv run python -m retrieval.bm25 --rebuild
uv run python -m retrieval.compare --query "<text>" --mode paper --k 5
uv run python -m retrieval.compare --query "<text>" --mode paper --k 5 --query-paper-id W3105077084
uv run python -m retrieval.compare --eval-queries
uv run pytest tests/test_retrieval.py -q
```
