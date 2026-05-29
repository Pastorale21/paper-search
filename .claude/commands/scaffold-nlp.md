# /scaffold-nlp

You are scaffolding the **NLP extraction layer** (method cards + citation 
intent classification) of the paper-search project. Read CLAUDE.md, 
docs/spike-results.md, and data/HANDOFF.md first. The spike/ directory 
is frozen reference — DO NOT MODIFY IT. The data layer (Paper schema, 
data/cache/papers.json) is also frozen — consume it, don't change it.

## Goal
Extract structured method cards for the top-50 papers by citation_count, 
and stand up a citation intent classifier that works today via a 
fallback path, with the slot for B's SciCite-fine-tuned model in week 3.

## Branch
`feat/scaffold-nlp` (create from main).

## What to build

### 1. Method card schema reminder
The MethodCard dataclass already exists in schemas.py (defined during 
spike). If any required field is missing, add it. Required fields:
- `paper_id: str`
- `task: str`
- `input: str`
- `output: str`
- `backbone: str`
- `loss: str`
- `key_idea: str` (one-liner, the most important field)
- `datasets: list[str]`
- `metrics: list[str]`
- Plus `to_dict` / `from_dict`

### 2. Method card extraction — `nlp/method_card/`
Create:
- `nlp/__init__.py`
- `nlp/method_card/__init__.py`
- `nlp/method_card/prompts.py` — few-shot prompt templates
- `nlp/method_card/extractor.py` — LLM call + JSON parse + cache

**`prompts.py` content**:
- Define `SYSTEM_PROMPT` (instructions: extract structured method card 
  from a paper abstract; output STRICT JSON matching the MethodCard schema; 
  if a field cannot be determined from the abstract, use empty string or 
  empty list rather than hallucinating)
- Define `FEW_SHOT_EXAMPLES: list[dict]` with **3 hand-crafted examples**. 
  Use these three papers (you'll need to write the JSON for each — base 
  it on widely-known facts about these models):
  - **LightGCN** (graph CF, light linear propagation) — paper_id placeholder, 
    backbone="simplified GCN (no feature transformation, no nonlinearity)", 
    loss="BPR", key_idea="strip non-essential operations from GCN-based CF, 
    keep only neighborhood aggregation"
  - **SimGCL** (graph contrastive learning) — backbone="LightGCN encoder", 
    loss="BPR + InfoNCE", key_idea="replace graph augmentations with uniform 
    embedding noise for contrastive views"
  - **BiTGCF** (cross-domain GCF) — backbone="bidirectional transfer GCN", 
    loss="BPR per domain + transfer regularization", 
    key_idea="bidirectional information transfer between source and target 
    domains via shared user embeddings"
- Define `build_prompt(abstract: str, title: str) -> list[dict]` that 
  returns OpenAI-format messages: system + 3 few-shots (user/assistant 
  pairs) + final user message with the target paper.

**`extractor.py` content**:
- Class `MethodCardExtractor`:
  - `__init__(self, api_key: str, base_url: str, model: str)` — OpenAI-
    compatible client (default DeepSeek via env: LLM_API_KEY, 
    LLM_BASE_URL=https://api.deepseek.com, LLM_MODEL=deepseek-chat)
  - `extract_one(self, paper: Paper) -> MethodCard | None` — call LLM, 
    parse JSON. **Robust parsing**: strip ```json fences, handle markdown 
    wrapping, one retry on JSONDecodeError. Returns None on persistent failure.
  - `extract_batch(self, papers: list[Paper], force: bool = False) -> list[MethodCard]` 
    — iterate, cache each at `data/cache/method_cards/{paper_id}.json`, 
    skip cached unless force.
- CLI: `uv run python -m nlp.method_card.extractor --top 50 [--force] [--dry-run]`
  - `--top N`: extract for top-N by citation_count
  - `--dry-run`: print estimated input tokens × N, exit before any API call
  - On real run: print estimate, **ask "Proceed? [y/N]"**, abort if not y
  - At end print per-field non-empty rate: `task: 96%, backbone: 88%, ...`

### 3. Citation intent classification — `nlp/citation_intent/`
Create:
- `nlp/citation_intent/__init__.py`
- `nlp/citation_intent/classifier.py`

**`classifier.py` content**:
- Class `CitationIntentClassifier` with method 
  `classify(self, context: str) -> Literal["background", "method", "comparison"]`
- **Implementation strategy** (this is the key design decision):
  - **Primary path (works today)**: if the citation context dict from S2 
    already has an `intents` field (S2 auto-classifies), trust it. Map 
    S2's labels (`background`, `methodology`, `result`) to our 3-class 
    space (`background`, `method`, `comparison` respectively).
  - **Fallback path (when S2 data unavailable)**: zero-shot LLM 
    classification using the SAME LLM client as method_card extractor. 
    Simple 1-shot prompt: "Classify this citation context as one of: 
    background, method, comparison."
  - **TODO(B) path**: STUB function 
    `classify_with_scicite(context: str) -> str` that raises 
    `NotImplementedError("TODO(B): fine-tune SciBERT on SciCite, see HANDOFF.md")`.
    Hook it up so it's used IF a flag `USE_SCICITE_MODEL=true` is set, 
    otherwise falls through to S2 → LLM.

### 4. Handoff document — `nlp/HANDOFF.md`
Write this file for module owner B. Include:
- **You are**: B (NLP extraction owner)
- **Current state**:
  - Method cards: 50 extracted, ~80% field-level non-empty (numbers from 
    your actual run — update after running)
  - Citation intent: working today via S2's pre-classified labels with 
    LLM zero-shot fallback; SciBERT/SciCite stub awaiting your work
- **Your work, prioritized**:
  - P0: Iterate method card prompts to improve field reliability. Look 
    at `data/cache/method_cards/*.json`, find papers where fields are 
    wrong, add few-shot examples or refine SYSTEM_PROMPT. Re-run with 
    `--force` to test.
  - P0: Implement `classify_with_scicite`. Fine-tune SciBERT on 
    https://huggingface.co/datasets/allenai/scicite. This is a real 
    NLP technique for the project's "5+ NLP techniques" requirement — 
    don't skip it even though the S2/LLM fallback works.
  - P1: Add NER for scientific entities (method names, dataset names, 
    metric names). See https://huggingface.co/allenai/scibert_scivocab_uncased.
  - P2: Scale method card extraction from 50 to all papers with abstracts 
    (~500). Estimate cost first (DeepSeek is cheap, but be deliberate).
- **Interface contracts (do NOT change without team review)**:
  - `MethodCardExtractor.extract_one` signature
  - `CitationIntentClassifier.classify` signature
  - `data/cache/method_cards/{paper_id}.json` location + MethodCard JSON schema
- **Known issues / gotchas**:
  - DeepSeek's `response_format={"type": "json_object"}` REQUIRES the 
    word "JSON" to appear in the prompt — already handled in SYSTEM_PROMPT, 
    don't remove it.
  - LLMs sometimes wrap JSON in ```json fences despite instructions — 
    parser handles this, keep the strip logic if you refactor.
  - When citation contexts not yet fetched by A (s2_contexts is stubbed), 
    `classify` will fall through to LLM zero-shot — works but slower and 
    costs more.
- **Tools to learn**:
  - DeepSeek API + OpenAI SDK: 
    https://api-docs.deepseek.com/api/create-chat-completion (~10 min)
  - HuggingFace transformers pipeline: 
    https://huggingface.co/docs/transformers/quicktour#pipeline (~15 min)
  - SciCite dataset: 
    https://huggingface.co/datasets/allenai/scicite (~5 min)

### 5. Tests — `tests/test_method_card.py`, `tests/test_citation_intent.py`
**Do NOT hit the real LLM API in tests.** Use mock fixtures:
- `test_method_card.py`:
  - `test_prompt_includes_few_shots`: build_prompt returns ≥7 messages 
    (system + 3 user/assistant pairs + 1 user)
  - `test_robust_json_parsing`: parse handles `{...}`, ` ```json {...} ``` `, 
    `Here is the JSON: {...}`
  - `test_extract_one_handles_api_failure`: mock client raises → returns None
- `test_citation_intent.py`:
  - `test_s2_intent_mapping`: `methodology` → `method`, `background` → 
    `background`, `result` → `comparison`
  - `test_classify_dispatches_to_scicite_when_flag_set`: with flag, 
    raises NotImplementedError (stub working)

### 6. Reality check before declaring done
After extraction runs, **print 5 random method cards to stdout** so the 
human reviewer can eyeball-check. Add a CLI flag `--sample N` that prints 
N random cards without running new extraction (reads from cache).

## Acceptance criteria
- `uv run python -m nlp.method_card.extractor --top 50` produces 50 
  cards in `data/cache/method_cards/` (with confirmation prompt)
- Per-field non-empty rate ≥80% on task, backbone, key_idea
- `uv run python -m nlp.method_card.extractor --sample 5` prints 5 
  formatted cards for spot-check
- `CitationIntentClassifier().classify("...")` returns valid label 
  via S2 path OR LLM fallback
- `nlp/HANDOFF.md` exists and is concrete
- `pytest tests/test_method_card.py tests/test_citation_intent.py` passes
- `ruff check .` and `black --check .` clean

## Rules
1. **Plan mode FIRST.** Show plan, wait for "go".
2. **Do NOT modify spike/ or data/.** Consume data layer's outputs only.
3. **One branch: feat/scaffold-nlp.** Commit per logical step.
4. **Confirm before spending LLM money.** First real extraction run 
   MUST print cost estimate and ask "Proceed? [y/N]".
5. **Stubs are stubs.** classify_with_scicite raises NotImplementedError 
   with TODO message; do not half-fine-tune SciBERT here.
6. **If blocked, ASK.** Especially: if LLM output format keeps breaking, 
   stop and show me 2-3 raw outputs — don't keep adding parsers.
