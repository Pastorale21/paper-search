# /scaffold-eval

You are scaffolding the **evaluation framework** for the paper-search 
project. This module produces the *numbers* that justify every other 
module's design choices.

Read CLAUDE.md, docs/spike-results.md, and the HANDOFF.md files for 
data, nlp, retrieval. The spike/ directory is frozen — DO NOT MODIFY. 
Consume papers.json and the retrieval module's public interfaces.

## Goal
Stand up a reproducible evaluation pipeline that:
1. Loads a hand-curated gold set (queries with expected top-N papers)
2. Resolves gold paper titles against the actual corpus (papers.json)
3. Runs each retrieval method against gold queries
4. Computes nDCG@5, MRR, Recall@10 per method, per query
5. Outputs a comparison table and per-query breakdown for human review

This is the engine D uses to expand the gold set, C uses to tune 
hybrid weights, and the team uses to write the report's "Experiments" 
section.

## Branch
`feat/scaffold-eval` (create from main).

## What to build

### 1. Gold set seed — `eval/gold_set.json`
I will paste a markdown gold set seed at the END of this command. Parse 
it into JSON of the following structure:

```json
{
  "version": "v1-seed",
  "created": "2026-05-30",
  "queries": [
    {
      "id": "Q1",
      "mode": "short",
      "text": "graph contrastive learning for recommendation",
      "gold_titles": ["SGL", "SimGCL", "NCL", "HCCF", "LightGCN"],
      "notes": "第 5 是 baseline 噪声测试点"
    },
    {
      "id": "P1",
      "mode": "paper",
      "text": "<full abstract text>",
      "gold_titles": ["NGCF", "GC-MC", "LR-GCCF", "UltraGCN", "DGCF"],
      "notes": "..."
    }
  ]
}
```

### 2. Gold set loader + title resolver — `eval/gold_set.py`
- Class `GoldSet`:
  - `load(path: Path = Path("eval/gold_set.json")) -> GoldSet`
  - `.queries: list[GoldQuery]` (dataclass)
- Class `TitleResolver`:
  - `__init__(self, papers: list[Paper])` — build index over normalized titles
  - `resolve(title: str) -> str | None` — return paper_id or None
  - Normalization: lowercase, strip punctuation, collapse whitespace, 
    strip common prefixes like "<i>" / "</i>" / model-name colons
  - Fuzzy fallback: if exact normalized match fails, try fuzzy match 
    via `difflib.get_close_matches` with cutoff 0.85. Log all fuzzy 
    matches so human can verify.
- CLI: `uv run python -m eval.gold_set --check`
  - For each gold query, resolve all gold_titles against papers.json
  - Print: `Q1: 4/5 resolved (SGL ✓, SimGCL ✗ not in corpus, NCL ✓, HCCF ✓, LightGCN ✓)`
  - Summary: total gold papers / total resolved / per-query coverage
  - **This output is critical** — it tells A what to add to corpus.

### 3. Metrics — `eval/metrics.py`
Pure functions, no dependencies on retrieval/UI:
- `ndcg_at_k(predicted_ids: list[str], gold_ids: set[str], k: int = 5) -> float`
- `mrr(predicted_ids: list[str], gold_ids: set[str]) -> float`
- `recall_at_k(predicted_ids: list[str], gold_ids: set[str], k: int = 10) -> float`

Implementations:
- Use `sklearn.metrics.ndcg_score` if it cleanly fits, else hand-write 
  (relevance = 1 if gold else 0, no graded relevance for v1)
- MRR: 1/(rank of first gold paper), 0 if none in predictions
- Recall@k: |predicted ∩ gold| / |gold|

Test each metric with hand-computed cases (see tests).

### 4. Eval runner — `eval/run.py`
CLI: `uv run python -m eval.run [--method {dense,bm25,dense_rerank,method_match,hybrid,all}] [--output PATH]`

Behavior:
- Load GoldSet
- Build TitleResolver from papers.json, resolve all gold titles to paper_ids
- For each method (or all):
  - For each gold query: call the appropriate retriever, get top-10
  - Compute nDCG@5, MRR, Recall@10
- Aggregate: mean across queries per method
- Print comparison table to stdout:
Method            nDCG@5    MRR      Recall@10
dense             0.412     0.583    0.467
bm25              0.385     0.521    0.452
dense_rerank      0.547     0.712    0.534
method_match      0.621     0.745    0.589
hybrid            0.683     0.812    0.622
- Per-query breakdown printed below for inspection (especially queries 
  where any method scores < 0.3 — those are diagnostic).
- Save raw results to `data/cache/eval/{timestamp}.json` for tracking 
  changes over time.

### 5. Tracker — `eval/history.py`
A simple utility:
- `record(method: str, metrics: dict, notes: str = "")` appends to 
  `data/cache/eval/history.jsonl`
- CLI: `uv run python -m eval.history` prints history as table, latest 
  run highlighted
- Lets C see "after I changed FIELD_WEIGHTS, did nDCG go up?" without 
  re-running the full eval every comparison

### 6. Handoff document — `eval/HANDOFF.md`
Write this file for module owner D. Include:
- **You are**: D (eval + UI owner)
- **Current state**:
  - Gold set v1: 10 queries (5 short + 5 paper-as-query), ~5 gold papers 
    each = 50 gold paper instances
  - Title resolver works, X/50 gold papers resolve against current corpus 
    of ~500 papers (X = real number after first run)
  - All 5 methods can be evaluated end-to-end
- **Your work, prioritized**:
  - P0: Expand gold set from 10 to 30-50 queries. The seed covers main 
    sub-areas; expansion should cover:
    - Edge cases (queries where dense retrieval clearly fails)
    - Sub-areas not in seed (e.g., sequential rec, scalability) 
    - 2-3 "adversarial" queries (paste a paper from sub-area X, expect 
      sub-area X papers, ensure no sub-area Y leaks)
    - For paper-as-query mode: at least 5 more pasted abstracts from 
      diverse papers
  - P0: When you add new gold queries, ALWAYS run `eval/gold_set.py --check` 
    first — make sure all gold_titles resolve, else flag missing papers 
    to A for corpus expansion.
  - P1: Build error analysis tooling — for queries where nDCG@5 < 0.3, 
    dump top-10 predicted + gold + score, help diagnose if it's a 
    retriever bug or a corpus gap.
  - P2: Inter-annotator agreement — get another team member to 
    independently produce gold for 5 queries, measure overlap.
- **Interface contracts (do NOT change)**:
  - `gold_set.json` schema (queries[].id/mode/text/gold_titles/notes)
  - `metrics.py` function signatures (used by `run.py` and any future 
    eval extensions)
  - `data/cache/eval/{timestamp}.json` output structure (history tracker 
    reads this)
- **Known issues / gotchas**:
  - Title resolution is fuzzy — when adding new gold papers, run `--check` 
    and eyeball the fuzzy match log to catch wrong resolutions (e.g., 
    "Graph CF" matching "GraphCF" vs "Graph-CF" the wrong way)
  - Some gold papers WILL NOT BE IN CORPUS until A scales it — that's 
    expected, log them but don't fail the eval
  - Cross-encoder rerank is slow — full eval with all 5 methods on 30 
    queries takes ~5 minutes on CPU. Use `--method dense` for quick 
    iteration.
- **Tools to learn**:
  - sklearn.metrics: scikit-learn.org/stable/modules/model_evaluation.html
  - nDCG intuition: ~5 min read — it's "how many gold papers in top-k, 
    weighted by position, normalized so 1.0 = perfect ranking"

### 7. Tests — `tests/test_eval.py`
- `test_ndcg_perfect`: predicted == gold → nDCG@5 = 1.0
- `test_ndcg_empty`: predicted ∩ gold = ∅ → nDCG@5 = 0.0
- `test_ndcg_partial`: 2/5 gold in top-5 at known positions → known value
- `test_mrr_first_position`: gold at position 1 → MRR = 1.0
- `test_mrr_third_position`: first gold at position 3 → MRR = 1/3
- `test_recall_at_10`: 3/5 gold in top-10 → 0.6
- `test_title_resolver_exact_match`: "LightGCN" → resolves
- `test_title_resolver_normalization`: "<i>SiReN</i>" → resolves to "SiReN"
- `test_title_resolver_fuzzy_threshold`: typo within threshold → resolves

### 8. Reality check
After implementation:
1. Run `uv run python -m eval.gold_set --check` — observe resolution rate
2. Run `uv run python -m eval.run --method all`
3. Eyeball the comparison table. Report:
   - Which method has highest nDCG@5?
   - Does method_match or hybrid beat dense baseline? By how much?
   - Any query where ALL methods score < 0.3? (= corpus gap or hard query)

This first eval IS the headline number for the project. Report it 
clearly so we can decide whether to:
(a) Move forward to UI v2 confident in differentiation, OR
(b) Pause to investigate why a method underperformed

## Acceptance criteria
- `eval/gold_set.json` parsed from the markdown seed below, valid JSON
- `uv run python -m eval.gold_set --check` runs and reports per-query 
  resolution
- `uv run python -m eval.run --method all` produces comparison table
- `pytest tests/test_eval.py` all green
- `eval/HANDOFF.md` exists, lists at least 4 concrete tasks for D
- `ruff check .` and `black --check .` clean

## Rules
1. **Plan mode FIRST.** Show plan, wait for "go".
2. **Do NOT modify spike/, data/, nlp/, retrieval/.** Consume their 
   public interfaces only.
3. **One branch: feat/scaffold-eval.** Commit per file.
4. **If gold resolution rate < 60%**, STOP and report. We need to 
   decide whether to wait for A's corpus expansion or trim the gold 
   set first.
5. **If method_match or hybrid scores BELOW dense baseline**, STOP. 
   This contradicts the project's core thesis — we discuss before 
   declaring eval "done".

---

## Gold Set Seed (parse this into eval/gold_set.json)

[PASTE THE MARKDOWN GOLD SET HERE — see the full content I'll provide 
in a follow-up message]
