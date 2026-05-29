# /scaffold-retrieval

You are scaffolding the **retrieval + reranking layer** of the 
paper-search project. This module owns the project's central 
differentiation: method-card field-level matching that breaks ties 
within saturated dense-retrieval clusters (see docs/spike-results.md §3, §5).

Read CLAUDE.md, spike-results.md, data/HANDOFF.md, and nlp/HANDOFF.md 
first. The spike/ directory is frozen reference — DO NOT MODIFY. 
Consume papers.json (from data layer) and method_cards/*.json 
(from nlp layer).

## Goal
Stand up four retrieval methods (dense, BM25, cross-encoder rerank, 
hybrid) plus method-card field-level matching, with a comparison CLI 
that lets us measure each method's contribution on the gold set.

## Branch
`feat/scaffold-retrieval` (create from main).

## What to build

### 1. Module skeleton — `retrieval/`
Create:
- `retrieval/__init__.py`
- `retrieval/dense.py` — wrap spike's embed + FAISS into a clean class
- `retrieval/bm25.py` — sparse retrieval
- `retrieval/rerank.py` — cross-encoder reranker
- `retrieval/method_match.py` — **differentiation core**, field-level matching
- `retrieval/hybrid.py` — fuse multiple signals into one ranking
- `retrieval/compare.py` — CLI to run all methods side-by-side

### 2. Dense retriever — `dense.py`
Class `DenseRetriever`:
- `__init__(self)` — lazy-load SPECTER2 dual adapters (proximity for 
  docs/paper-query, adhoc_query for short query) and FAISS index 
  from spike's caches. Reuse spike code via import, don't copy-paste.
- `search(self, query: str, mode: Literal["short", "paper"], k: int) -> list[tuple[str, float]]`
  returns `[(paper_id, score), ...]`
- This is essentially spike.search.search() repackaged. Don't reinvent.

### 3. BM25 retriever — `bm25.py`
Class `BM25Retriever`:
- `__init__(self)` — load `data/cache/papers.json`, build BM25 over 
  `title + " " + abstract`. Simple whitespace tokenization is fine 
  (English abstracts). Persist BM25 state to `data/cache/bm25.pkl`.
- `search(self, query: str, k: int) -> list[tuple[str, float]]`
- CLI: `uv run python -m retrieval.bm25 --rebuild` to force rebuild

Use `rank_bm25` library (lightweight, no dependencies).

### 4. Cross-encoder reranker — `rerank.py`
Class `CrossEncoderReranker`:
- `__init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2")`
- `@lru_cache(maxsize=1)` on model loading (prevent Streamlit reloading 
  it every interaction)
- `rerank(self, query: str, candidates: list[tuple[str, float]], k: int) -> list[tuple[str, float]]`
  - Looks up paper title+abstract from papers.json for each candidate
  - Scores each `(query, doc)` pair via cross-encoder
  - Returns top-k re-sorted by cross-encoder score
- Performance note in docstring: cross-encoder runs ~5-10s on CPU 
  for 50 candidates; warn user to cache results in production.

### 5. Method-card field-level matching — `method_match.py` ⭐ DIFFERENTIATION CORE
Class `MethodCardMatcher`:
- `__init__(self, embedder: DenseRetriever | None = None)` — for embedding 
  fields. Reuse the DenseRetriever's SPECTER2 instance (don't load twice).
- `match(self, query_paper_id: str | None, query_card: MethodCard | None, candidates: list[str], k: int) -> list[tuple[str, float]]`
  - Caller provides EITHER a query_paper_id (look up its cached method card) 
    OR a query_card directly (for free-form mechanism queries)
  - For each candidate paper_id, load its method card from cache. If 
    missing, score 0 (signal: extend method card coverage)
  - **Field-level matching**: for each field in `FIELDS_TO_MATCH`, 
    embed the field value (query vs candidate), compute cosine, then 
    combine via `FIELD_WEIGHTS`
  - Module-level constants (clearly marked for tuning):
```python
    FIELDS_TO_MATCH = ["task", "backbone", "loss", "key_idea"]
    
    # TODO(C): tune via eval/run.py nDCG comparison on gold set
    FIELD_WEIGHTS = {
        "task": 0.20,
        "backbone": 0.30,
        "loss": 0.20,
        "key_idea": 0.30,
    }
```
  - Returns top-k sorted by weighted field score
- Edge cases (handle, don't crash):
  - Method card field is empty string → that field contributes 0 to score
  - All candidates have no method cards → return empty list with a log warning

### 6. Hybrid retriever — `hybrid.py`
Class `HybridRetriever`:
- Composes Dense + BM25 + MethodCard signals
- Constructor takes weight dict:
```python
  # TODO(C): tune via eval/run.py
  DEFAULT_WEIGHTS = {"dense": 0.4, "bm25": 0.2, "method_match": 0.4}
```
- `search(self, query, mode, k, top_n_for_rerank=50) -> list[tuple[str, float, dict]]`
  - Returns `[(paper_id, fused_score, signal_breakdown), ...]`
  - `signal_breakdown` is a dict like `{"dense": 0.95, "bm25": 0.12, "method_match": 0.81}` 
    — this IS the "reason tag" data the UI will surface
  - Recall: top_n_for_rerank candidates from each retriever, union, 
    score each via all signals, fuse with weights, optionally cross-
    encoder rerank top-k
  - Optional flag `use_rerank: bool = True` to toggle cross-encoder step

### 7. Comparison CLI — `compare.py`
`uv run python -m retrieval.compare --query "..." [--mode short|paper] [--k 10]`

Runs all methods side by side on the given query:
1. Dense baseline
2. BM25
3. Dense → cross-encoder rerank
4. Method card matching alone (paper mode only)
5. Hybrid (all signals fused)

Prints a 5-column comparison table to stdout: rank, dense, bm25, 
dense+rerank, method_match, hybrid. Each cell shows truncated title + score.

Also supports `--eval-queries` flag that runs the 3 EVAL_QUERIES from 
spike.config across all methods and dumps results to 
`data/cache/eval/retrieval_comparison.json` for later inspection.

### 8. Handoff document — `retrieval/HANDOFF.md`
This module stays with C (you), but the handoff doc still helps team 
context. Include:
- **You are**: C — you keep this module, but documenting state for 
  whoever picks it up next semester
- **Current state**:
  - All 5 methods implemented and runnable via `retrieval.compare`
  - FIELD_WEIGHTS and hybrid DEFAULT_WEIGHTS are placeholder defaults — 
    tuning happens via eval/run.py on the gold set in week 3
- **Open tuning knobs**:
  - `method_match.FIELD_WEIGHTS` — relative importance of task/backbone/
    loss/key_idea. Tune by ablating each, watching nDCG@5.
  - `hybrid.DEFAULT_WEIGHTS` — dense vs bm25 vs method_match contribution. 
    Initial guess favors dense + method_match.
  - Cross-encoder model — currently ms-marco; consider BAAI/bge-reranker-v2-m3 
    for higher quality (heavier).
- **Interface contracts (consumers depend on these)**:
  - `HybridRetriever.search` return signature including `signal_breakdown` 
    dict — UI's "reason tag" feature depends on this
  - All retrievers expose `search(query, k) -> list[tuple[str, float]]`
- **Known issues**:
  - Cross-encoder is CPU-slow; cache aggressively in UI layer
  - Method card matching scores are NOT directly comparable to dense 
    cosines (different scale); hybrid normalizes within rank, not magnitude
- **Tools to learn (if handed off)**:
  - rank_bm25: github.com/dorianbrown/rank_bm25
  - sentence-transformers CrossEncoder: sbert.net/docs/cross_encoder/usage/usage.html

### 9. Tests — `tests/test_retrieval.py`
- `test_dense_search_returns_k_results`: mock embed, FAISS, return shape
- `test_bm25_search_handles_empty_query`: empty string → no crash, empty result
- `test_method_match_skips_papers_without_card`: papers without cards 
  get score 0, not crash
- `test_method_match_field_weights_applied`: with FIELD_WEIGHTS = 
  {"task": 1.0, others: 0.0}, only task field matters
- `test_hybrid_signal_breakdown_present`: search result includes 
  `signal_breakdown` dict with all signal keys

**Do NOT load actual SPECTER2 / cross-encoder in tests.** Use mocks 
and tiny fake embedding matrices.

### 10. Reality check
After implementation, run:
uv run python -m retrieval.compare 
--query "Collaborative filtering benefits from graph convolution..." 
--mode paper --k 5
(use the LightGCN abstract from spike's EVAL_QUERIES)

Eyeball the 5 columns. Expect to see:
- Dense column: cosine 0.95-0.96 cluster (the saturation from spike)
- Method match column: scores spread out more (different fields differ)
- Hybrid: combines, ideally LightGCN-family papers (NGCF, GC-MC, 
  LR-GCCF) climb to top
- BM25: more lexical, may differ significantly from dense

Print this comparison and verbally describe to the human reviewer 
whether the method_match column **visibly differentiates** within the 
dense-saturated cluster. That's the project's whole thesis — if it 
doesn't, FLAG IT.

## Acceptance criteria
- All 5 retrievers callable; `retrieval.compare` CLI runs end-to-end
- For the LightGCN-abstract query, method_match column shows >= 0.05 
  spread between top-5 (vs dense's 0.006 spread) — or you flag it as 
  unexpected and we discuss
- `signal_breakdown` dict appears in HybridRetriever output
- `pytest tests/test_retrieval.py` passes
- `retrieval/HANDOFF.md` exists and lists tuning knobs
- `ruff check .` and `black --check .` clean

## Rules
1. **Plan mode FIRST.** Show plan, wait for "go".
2. **Do NOT modify spike/, data/, or nlp/.** Consume their outputs.
3. **Reuse, don't copy.** Import spike's embed/search logic; don't 
   re-implement SPECTER2 loading.
4. **One branch: feat/scaffold-retrieval.** Commit per major file.
5. **TODO(C) markers must remain.** FIELD_WEIGHTS and DEFAULT_WEIGHTS 
   are tuning placeholders — do NOT hide them inside a function or 
   "optimize" them away.
6. **If method_match doesn't visibly differentiate** in the reality 
   check, STOP and tell me what you observed. This affects the project's 
   core differentiation story — we may need to revise field choice or 
   weighting strategy before moving on.
