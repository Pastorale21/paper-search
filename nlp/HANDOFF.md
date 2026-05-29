# NLP Layer Handoff

**You are:** B — NLP extraction owner (`nlp/`).

This scaffold turns the spike's throwaway LLM probe into a real, cached, tested
extraction layer: mechanism-level **method cards** + a **citation-intent
classifier**. Read this top-to-bottom before changing anything.

---

## Current state

- **Method cards** — `nlp/method_card/`
  - `prompts.py`: `SYSTEM_PROMPT` + 3 hand-written few-shots (LightGCN, SimGCL,
    BiTGCF) + `build_prompt(abstract, title)`.
  - `extractor.py`: `MethodCardExtractor` (LLM call + robust JSON parse + per-paper
    cache) and a CLI.
  - Cards cache to `data/cache/method_cards/{paper_id}.json` (never committed).
  - Run status: _scaffold built and unit-tested; the paid `--top 50` run is done
    manually by the human (DeepSeek). Update these numbers after the first run:_
    - cards extracted: `__/50`
    - per-field non-empty rate: `task __%, backbone __%, key_idea __%` (target ≥80%).

- **Citation intent** — `nlp/citation_intent/classifier.py`
  - `CitationIntentClassifier.classify(context)` → `background | method | comparison`.
  - **Live path today is the LLM zero-shot fallback.** The data source is OpenAlex
    (see `.env.example: DATA_SOURCE=openalex`), which provides **no S2 intent
    labels**, so the S2-mapping branch is dormant until A adds S2 citation contexts.
  - SciBERT/SciCite path is a stub awaiting your work (raises `NotImplementedError`).

---

## Your work, prioritized

### P0 — Iterate method-card prompts for field reliability
Inspect `data/cache/method_cards/*.json`, find papers where fields are wrong or
empty, and improve `SYSTEM_PROMPT` / add few-shot examples in `prompts.py`.
Re-test a single paper or re-run with `--force`. Watch `backbone`, `loss`, and
`key_idea` — those are the mechanism-level fields the whole project differentiates on.

### P0 — Implement `classify_with_scicite`
Fine-tune SciBERT on SciCite and wire it behind the `USE_SCICITE_MODEL=true` flag.
This is a real NLP technique counting toward the project's "5+ NLP techniques"
requirement — **do not skip it** even though the LLM fallback works.
- dataset: https://huggingface.co/datasets/allenai/scicite
- base model: https://huggingface.co/allenai/scibert_scivocab_uncased
- Keep the `classify(context) -> {background, method, comparison}` contract; map
  SciCite's labels (`background`, `method`, `result`) the same way `map_s2_intent`
  does (`result → comparison`).

### P1 — Scientific NER
Extract method / dataset / metric names from abstracts (e.g. fine-tune or use
`scibert_scivocab_uncased`). Feeds richer method cards and the graph layer.

### P2 — Scale method cards 50 → ~500
All 381 cached papers have abstracts. Estimate cost first (DeepSeek is cheap, but
be deliberate): use `--dry-run` to print the token estimate before spending.

---

## Schema cleanup TODO (do during week 3, with team sign-off)

`schemas.py::MethodCard` currently carries **two overlapping generations of fields**:

| New (this scaffold, populated by extractor) | Legacy (spike-era, still present) |
| --- | --- |
| `task`, `input`, `output` | `problem` (≈ task) |
| `backbone`, `loss` | `method` (≈ backbone + loss, unstructured) |
| `key_idea`, `datasets`, `metrics` | `name`, `baselines` |

**Why both exist:** the new mechanism-level fields were added additively so
`spike/probe_llm.py` (frozen) and any teammate consumer kept working — nothing was
deleted. The legacy `problem`/`method`/`baselines` are now redundant with
`task`/`backbone`+`loss` and are **not populated** by the new extractor.

**Recommendation:** in week 3, once retrieval/UI consumers are settled, consolidate
to the new field set and drop `name`, `problem`, `method`, `baselines`. This is a
team-owned schema change (`CLAUDE.md`: don't touch `schemas.py` without notifying
the team) — coordinate before removing fields, and grep retrieval/ index/ ui/ for
any reads of the legacy names first. (`baselines` may be worth *keeping* if the
graph/eval layer wants explicit baseline lists — decide as a team.)

---

## Interface contracts — do NOT change without team review

- `MethodCardExtractor.extract_one(paper: Paper) -> MethodCard | None`
- `CitationIntentClassifier.classify(context) -> "background"|"method"|"comparison"`
  (context may be a plain string or an S2-style dict with an `intents` field)
- Cache location + JSON shape: `data/cache/method_cards/{paper_id}.json`, a
  serialized `MethodCard` (see `schemas.py`).

---

## Known issues / gotchas

- **DeepSeek `response_format={"type":"json_object"}` requires the literal word
  "JSON" in the prompt.** It's in `SYSTEM_PROMPT` — don't remove it.
- **LLMs sometimes wrap JSON in ```json fences** despite instructions. `_parse_json`
  in `extractor.py` strips fences and falls back to outermost-brace extraction with
  one retry — keep that logic if you refactor.
- **OpenAlex → LLM fallback is the live intent path.** Until A fetches S2 citation
  contexts (`intents`), every `classify()` call hits the LLM (slower, costs money).
  Cache results if you classify in bulk.
- `nlp/` is **self-contained** (`nlp/config.py` reads env + loads `papers.json`); it
  deliberately does **not** import from frozen `spike/`. Don't add a spike import.

---

## Debugging discipline

If LLM output format keeps breaking, **stop and inspect 2–3 raw replies** before
adding more parser branches. Don't pile up regex hacks — fix the prompt.

---

## Tools to learn

- DeepSeek API + OpenAI SDK — https://api-docs.deepseek.com/api/create-chat-completion (~10 min)
- HuggingFace transformers pipeline — https://huggingface.co/docs/transformers/quicktour#pipeline (~15 min)
- SciCite dataset — https://huggingface.co/datasets/allenai/scicite (~5 min)

---

## Commands

```bash
uv run python -m nlp.method_card.extractor --top 50 --dry-run   # token estimate, no API call
uv run python -m nlp.method_card.extractor --top 5              # smoke test: 5 cards (asks Proceed?)
uv run python -m nlp.method_card.extractor --top 50             # real run (asks Proceed?)
uv run python -m nlp.method_card.extractor --top 50 --force     # re-extract, ignore cache
uv run python -m nlp.method_card.extractor --sample 5           # print 5 cached cards, no API call
uv run pytest tests/test_method_card.py tests/test_citation_intent.py -q
```
