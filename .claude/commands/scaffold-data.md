# /scaffold-data

You are scaffolding the **data ingestion + corpus layer** of the 
paper-search project. Read CLAUDE.md and docs/spike-results.md first. 
The spike implementation in spike/ is frozen reference — DO NOT MODIFY IT.

## Goal
Expand corpus from 76 to ~500 papers across 6 GNN-recsys sub-areas, 
upgrade the Paper schema to support multi-source IDs, and leave clean 
extension points for GROBID full-text parsing and Semantic Scholar 
citation-context fetching (these are TODO for module owner A).

## Branch
`feat/scaffold-data` (create from main, not from spike branch).

## What to build

### 1. Schema upgrade — `schemas.py`
Add to Paper:
- `source_ids: dict[str, str]` (default `field(default_factory=dict)`)
  Populated like `{"openalex": "W...", "doi": "10.1145/...", "arxiv": "..."}`.
- Backwards compatible: existing papers.json must still load (missing 
  field → empty dict).
- Update `to_dict` / `from_dict` accordingly.
- Add a brief docstring noting which subsystem fills which key.

### 2. Multi-source ingestion — `data/sources/`
Create:
- `data/__init__.py`
- `data/sources/__init__.py`
- `data/sources/openalex.py` — port spike/fetch.py logic, generalize to 
  multi-query. Function: `fetch_works(query: str, n: int) -> list[Paper]`.
  Populate `source_ids` with openalex + doi (if available in OA response).
- `data/sources/s2_contexts.py` — **STUB ONLY**. Define the function 
  signature `fetch_citation_contexts(paper_id: str) -> list[dict]` that 
  raises `NotImplementedError("TODO(A): implement with S2 API key, see HANDOFF.md")`.
  This is the extension point for A in week 3.

### 3. Corpus orchestrator — `data/corpus.py`
Function `build_corpus(sub_areas: list[str], per_query: int, target: int) -> list[Paper]`:
- Fetch per_query papers for each sub-area query
- Dedupe by `source_ids["openalex"]` AND normalized title (lowercase, 
  strip punctuation, collapse whitespace) — the rank 2/4 duplicate in 
  spike-results.md §2 must not survive
- If duplicates found, keep the record with more populated fields 
  (longer abstract, more references, etc.)
- Filter out papers without abstracts
- Stop when unique-with-abstract count reaches target OR all sub-areas exhausted
- Persist to `data/cache/papers.json` (overwrite, not append)
- Print stats: total fetched / unique after dedup / with abstracts / 
  per-sub-area distribution / how many have DOI in source_ids / 
  how many duplicates removed

### 4. Parser stub — `data/parse/grobid_client.py`
**STUB ONLY**. Define class `GrobidClient` with method 
`parse_pdf(pdf_path: Path) -> dict` that raises 
`NotImplementedError("TODO(A): implement with Docker grobid service, see HANDOFF.md")`.
Add a docstring pointing to https://github.com/kermitt2/grobid-client-python.

### 5. CLI
`uv run python -m data.corpus --target 500 --per-query 100 [--force]`
- Sub-areas hard-coded in module constant:
```python
  SUB_AREAS = [
      "graph neural network collaborative filtering",
      "graph contrastive learning for recommendation",
      "cross-domain recommendation with graph neural network",
      "knowledge graph enhanced recommendation",
      "session-based recommendation with graph neural network",
      "social recommendation with graph neural network",
  ]
```
- `--force` rebuilds; default skips if papers.json exists.

### 6. Verify spike still works
After corpus rebuild:
- Run `uv run python -m spike --force` against the new papers.json
- Confirm |V|/|E| numbers, eval queries still produce sensible top-5
- If spike breaks, the schema migration is wrong — STOP and ASK.

### 7. Handoff document — `data/HANDOFF.md`
Write this file for module owner A. Include:
- **You are**: A (data + infra owner)
- **Current state**: what's working, what's stubbed
- **Your work, prioritized**:
  - P0: Implement `s2_contexts.fetch_citation_contexts` once S2 API key 
    arrives. B's intent classifier in week 3 depends on this.
  - P0: Scale corpus from 500 to 800-1000 if time permits.
  - P1: Implement `GrobidClient.parse_pdf` — Docker setup, parse top-100 
    high-citation PDFs to extract method sections for richer method cards.
  - P2: Multi-source verification — cross-check OpenAlex vs S2 results 
    on a sample to estimate corpus completeness.
- **Interface contracts (do NOT change without team review)**:
  - `Paper.source_ids` schema
  - `build_corpus` signature
  - `data/cache/papers.json` location and structure
- **Known issues / gotchas**:
  - OpenAlex `abstract_inverted_index` must be reconstructed to string 
    (see openalex.py for helper)
  - macOS users need `KMP_DUPLICATE_LIB_OK=TRUE` (already set in 
    spike/config.py, imported transitively)
  - GROBID Docker image is large (~3GB) and slow to start (~5-10 min 
    first time)
- **Tools to learn**:
  - OpenAlex API: https://docs.openalex.org/api-entities/works (~10 min)
  - GROBID: docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.8.0, 
    then http://localhost:8070 web UI to play with it (~20 min)
  - Semantic Scholar Graph API: https://api.semanticscholar.org/api-docs/

### 8. Tests
Add `tests/test_corpus.py`:
- `test_dedup_by_openalex_id`: 3 papers, 2 with same OA id → returns 2
- `test_dedup_by_normalized_title`: 2 papers, same title different case/
  punctuation → returns 1
- `test_filter_no_abstract`: 3 papers, 1 with empty abstract → returns 2
- Use small in-memory fixtures, do NOT hit OpenAlex API in tests.

## Acceptance criteria
- `uv run python -m data.corpus --target 500 --per-query 100` produces 
  ≥300 unique papers with abstracts
- Each sub-area contributes ≥30 papers after dedup
- Spike pipeline (`uv run python -m spike --force`) still runs against 
  the new papers.json without modification
- `pytest tests/test_corpus.py` passes
- `data/HANDOFF.md` exists and follows the structure above
- `ruff check .` and `black --check .` clean

## Rules
1. **Plan mode FIRST.** Show plan, wait for "go" before executing.
2. **Do NOT modify spike/** — it is the frozen reference baseline.
3. **One branch: feat/scaffold-data.** Commit per logical step, push 
   when phase complete.
4. **If blocked** (API error, schema disagreement, ambiguity), STOP 
   and ASK — do not guess or paper over.
5. **Stubs are stubs.** s2_contexts and grobid_client must raise 
   NotImplementedError with the TODO message. Do NOT half-implement them.
