# Data Layer — Handoff

## You are
**A — data + infra owner.** You own `data/` (ingestion, corpus, parsing) and the corpus cache
contract the rest of the team builds on.

## Current state
Working:
- `data/sources/openalex.py` — live OpenAlex adapter. `fetch_works(query, n)` returns parsed
  `Paper`s with `source_ids` (`openalex` short id + bare `doi`) populated.
- `data/corpus.py` — `build_corpus(sub_areas, per_query, target)` fetches widened sub-area
  queries, dedupes (OpenAlex id + exact/fuzzy normalized title, keeping the richer record),
  filters out abstract-less papers, and overwrites `data/cache/papers.json`. CLI:
  `python -m data.corpus --target 800 --per-query 500 --force`.
- `data/sources/seed_papers.py` — canonical seed-title fetcher. It now includes the previously
  missing gold papers called out in onboarding (CCDR / DisenCDR / DDTCDR / PPGN, S3-Rec /
  CL4SRec, MHCN / SocialLGN, CKE / CKAN, XSimGCL, SURGE / FGNN, plus nearby gold gaps).
- `schemas.py` — `Paper.source_ids` added; backward compatible (old caches load with `{}`).
- The spike pipeline (`uv run python -m spike`) consumes the new `papers.json` unchanged.

Stubbed (raise `NotImplementedError` — your job to implement):
- `data/sources/s2_contexts.py::fetch_citation_contexts`
- `data/parse/grobid_client.py::GrobidClient.parse_pdf`

## Your work, prioritized
- **P0 — `s2_contexts.fetch_citation_contexts`** once the S2 API key arrives. B's citation-intent
  classifier in week 3 depends on this (it needs the text *around* each citation, not just edges).
- **P0 — scale corpus 800 → 1000** if time permits (raise `--target`, maybe widen
  `SUB_AREAS` again if gold coverage stalls).
- **P1 — `GrobidClient.parse_pdf`** — stand up Docker GROBID, parse the top-~100 high-citation PDFs
  to extract method sections for richer method cards.
- **P2 — multi-source verification** — cross-check OpenAlex vs S2 on a sample to estimate corpus
  completeness (coverage / missing-paper rate).

## Interface contracts (do NOT change without team review)
- **`Paper.source_ids`** — `dict[str, str]`, canonical normalized IDs. Keys seen today:
  `openalex` (short `W…`), `doi` (bare, no `https://` prefix). Future: `arxiv`, `s2`.
- **`build_corpus(sub_areas: list[str], per_query: int, target: int) -> list[Paper]`** signature.
- **`data/cache/papers.json`** — location and structure: a JSON array of `Paper.to_dict()` objects.
  Gitignored; never committed. The spike and all downstream layers read from here.

## Known issues / gotchas
- **`external_ids` vs `source_ids`** (refactor candidate, week 4):
  - `external_ids` is **legacy** from the spike — stores the raw OpenAlex `ids` dict, i.e. full
    URLs like `"https://openalex.org/W..."` and `"https://doi.org/10...."`.
  - `source_ids` is the **new canonical normalized** ID map — short `"W..."` for openalex, bare
    DOI without the `https` prefix.
  - **Downstream consumers should read from `source_ids`.** `external_ids` is preserved only so
    spike code keeps loading old `papers.json`.
  - Consolidating the two into a single field is **deferred** — flag it for the week-4 cleanup.
- **OpenAlex abstracts** come as an `abstract_inverted_index` and must be reconstructed into a
  string — see `_reconstruct_abstract` in `data/sources/openalex.py`.
- **macOS libomp** — `torch` and `faiss-cpu` each ship `libomp` and double-loading segfaults. The
  guard `KMP_DUPLICATE_LIB_OK=TRUE` / `OMP_NUM_THREADS=1` lives at the top of `spike/config.py`
  and is imported transitively (`data` modules import `spike.config`). Don't remove it.
- **GROBID Docker image** is large (~3GB) and slow to start (~5–10 min the first time).
- **Spike cache invalidation**: after rebuilding the corpus via `data/corpus.py`, do **NOT** run
  `python -m spike --force` — its `fetch` step overwrites `papers.json` with its own 100-paper
  single-query result. The spike's index caches key on file existence, not corpus content, so a
  plain no-force run reuses the stale index. Correct procedure when the corpus changes:
  ```sh
  rm data/cache/embeddings.npy data/cache/ids.json \
     data/cache/faiss.index data/cache/citation_graph.pkl
  python -m spike   # no --force; fetch cache-hits the new papers.json,
                    # embed/index/graph rebuild against the new corpus
  ```
- **Near-duplicate dedup fixed locally**: `data.corpus.titles_are_duplicates` catches safe
  acronym/prefix variants such as "LightGCN" vs "LightGCN: Simplifying and Powering..." while
  avoiding generic containment such as "Graph Neural Networks for Recommendation" vs the social
  sub-area title.

## Tools to learn
- OpenAlex API (Works): https://docs.openalex.org/api-entities/works (~10 min)
- GROBID: `docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.8.0`, then play with the web UI at
  http://localhost:8070 (~20 min)
- Semantic Scholar Graph API: https://api.semanticscholar.org/api-docs/
