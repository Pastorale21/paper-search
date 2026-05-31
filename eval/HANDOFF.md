# Eval Framework — Handoff

## You are
**D — eval + UI owner.** This module is the engine that produces the *numbers* every
other module's design choices are justified by. The retrieval team consumes it to tune
weights (`FIELD_WEIGHTS`, `DEFAULT_WEIGHTS`); the report's "Experiments" section quotes
the comparison table directly.

## Current state
Implemented + runnable:
- `eval/gold_set.json` — 10 seed queries (5 short + 5 paper-as-query). Paper queries
  carry `anchor_title` so `method_match` anchors deterministically rather than guessing
  via dense top-1. Schema: `id, mode, text, gold_titles[], notes, anchor_title?, arxiv?`.
- `eval/gold_set.py` — `GoldSet` loader and `TitleResolver`. Five-tier resolution:
  exact normalized → colon-prefix abbreviation → **alias map** (`DEFAULT_ALIASES`) →
  substring → difflib fuzzy ≥ 0.85. CLI: `python -m eval.gold_set --check`.
- `eval/metrics.py` — hand-written `ndcg_at_k`, `mrr`, `recall_at_k`. Binary relevance.
- `eval/run.py` — runs every method × every gold query. Prints **two tables**:
  1. *Full mixed* — aggregates across whatever queries each method can score (denominator
     differs by method; `method_match` only runs on paper queries).
  2. *Same-subset paper-queries-only* — directly comparable across dense / dense_rerank /
     method_match / hybrid. **This is the thesis gate** — the "method_match beats dense"
     claim must hold here. Runner exits non-zero on Gate 2 failure.
- `eval/history.py` — append-only JSONL tracker (`data/cache/eval/history.jsonl`) plus
  table printer. Use after every weight change to see whether nDCG@5 moved.

After the first run (live, 400-paper corpus, full 400 method cards):
- **Gold-title resolution rate: 35 / 50 (70.0%)** — clears Gate 1 (≥ 60%).
- **Same-subset (5 paper queries) nDCG@5** — *after disabling CE rerank in hybrid
  (`HybridRetriever(use_rerank=False)` is now the class default)*:
  - dense **0.198** (baseline)
  - dense_rerank 0.092 (−0.106 vs dense) — CE genuinely hurts standalone, kept as
    an honest negative result in the comparison.
  - **method_match 0.310 (+0.112 vs dense)** — clears thesis Gate 2.
  - **hybrid 0.289 (+0.091 vs dense)** — clears Gate 2 once CE rerank is off.
- Per-query: hybrid wins P1 (0.384 vs 0.214), P2 (0.390 vs 0.246), and almost matches
  method_match on P4 (0.469 vs 0.478). P3 (KG) drops slightly below dense (0.202 vs
  0.296) — RRF dilutes the dense-only signal when BM25 + method_match both contribute
  0. P5 (DiffNet/social) still fails for hybrid (0.000) despite method_match standalone
  scoring 0.469 — non-gold BM25/method_match candidates push gold out of top-10 here.
  Both are `DEFAULT_WEIGHTS` tuning candidates (bump method_match's weight when its
  signal is strong and dense's isn't).

Unresolved gold titles flagged as **corpus-expansion candidates** for A:
- Cross-domain: CCDR, DisenCDR, DDTCDR, PPGN
- SSL/sequential: S3-Rec, CL4SRec
- Social: MHCN, SocialLGN
- KG: CKE, CKAN
- Contrastive: XSimGCL
- Session: SURGE, FGNN

## Your work, prioritized
- **P0 — expand the gold set to 30-50 queries.** The seed covers six sub-areas
  (CF, contrastive, cross-domain, KG, session, social). Expansion should cover:
  - Edge cases: queries where dense retrieval clearly fails on the seed.
  - Sub-areas not yet covered: sequential rec (S3-Rec / CL4SRec), scalability,
    cold-start, fairness.
  - 2-3 *adversarial* queries: paste a paper from sub-area X, expect sub-area X papers,
    confirm no sub-area Y leaks.
  - Paper-as-query mode: at least 5 more pasted abstracts from diverse sub-areas.
- **P0 — `--check` discipline.** Every time you add a gold query, run
  `python -m eval.gold_set --check` first. If a new gold title shows `✗`, decide:
  (a) the canonical paper isn't in the corpus → ping A to expand corpus, or
  (b) the title's just unresolvable via the current rules → **extend
  `DEFAULT_ALIASES`** (see Known issues below).
- **P1 — error-analysis tooling.** For any (method, query) where nDCG@5 < 0.30 (the
  runner already flags these with `!`), build a small command that dumps:
  predicted top-10 with titles, resolved gold titles, the gap. Distinguish "retriever
  bug" from "corpus gap" from "ambiguous query".
- **P2 — inter-annotator agreement.** Ask another teammate to independently produce
  `gold_titles` for 5 queries; measure overlap; flag queries with low IAA as ambiguous
  signal in the dataset.

## Interface contracts (do NOT change without team review)
- **`gold_set.json` schema**: `id, mode, text, gold_titles[], notes` are required;
  `anchor_title, arxiv` are optional. Adding new optional fields is fine; renaming or
  removing required fields breaks `GoldSet.load`.
- **`metrics.py` signatures**: consumed by `run.py` and any future eval extensions.
  `ndcg_at_k(predicted_ids, gold_ids, k)`, `mrr(predicted_ids, gold_ids)`,
  `recall_at_k(predicted_ids, gold_ids, k)`.
- **`data/cache/eval/{timestamp}.json` payload shape**: history tracker + the future
  comparison-over-time UI both read this. Keep the keys `version`, `created`,
  `gold_set_version`, `aggregates_full`, `aggregates_same_subset_paper`, `per_query`.

## Known issues / gotchas
- **Cross-encoder `ms-marco-MiniLM-L-6-v2` underperforms for academic mechanism
  matching — DEFAULT-DISABLED in hybrid.** First-run eval evidence:
  `dense_rerank` scored **−0.106** below dense on the same-subset nDCG@5 (CE
  hurts standalone too), and `hybrid` (with rerank on) crashed to **0.000** on
  P3/P4/P5 (KG / session / social) where `method_match` alone scored
  0.478 / 0.469 — the CE rerank reordered gold papers OUT of the top-10. Root
  cause is the model's web-search training distribution; cosine "is this a
  relevant web result for this query" is not the same as "is this paper
  mechanism-similar". `HybridRetriever(use_rerank=False)` is the default
  starting from this branch; the flag is kept toggleable. **Future C
  experiment**: swap in `BAAI/bge-reranker-v2-m3` (encoder-decoder, larger,
  scientific-text-friendlier), re-run the same eval, see whether
  `dense_rerank` and `hybrid+rerank=True` recover. `retrieval/rerank.py` reads
  the model name from the `CrossEncoderReranker(model_name=...)` constructor;
  no other change needed.
- **`DEFAULT_ALIASES` is hand-maintained.** Many gold titles are short acronyms
  (`NGCF`, `GC-MC`, `KCGN`) that do NOT appear verbatim in OpenAlex paper titles.
  The resolver only finds those via the alias map mapping
  `<normalized abbreviation> -> <substring of canonical paper title>`. When you add
  a new gold paper whose title is an abbreviation not already mapped, **you must
  extend `DEFAULT_ALIASES` in `eval/gold_set.py`** — otherwise `--check` will report
  `✗ not in corpus` even though the paper IS in the corpus. Verification protocol:
  1. Grep `data/cache/papers.json` for the canonical paper.
  2. Add the alias entry: `<acronym lower>` → `<lowercased substring of paper title>`.
  3. Re-run `--check`; confirm the new entry appears in the `[match log]` under
     `[alias]` and points at the right paper.
- **Title resolution is fuzzy by design.** The `[match log]` after `--check` lists
  every non-exact match (`alias`, `substring`, `fuzzy`). Eyeball it after adding gold
  queries — the substring path can occasionally catch the wrong paper when an
  abbreviation collides with a generic phrase.
- **Some gold papers will not be in the corpus** until A scales it (CCDR / DisenCDR /
  S3-Rec etc.). That's expected — the runner skips unresolved gold titles silently
  (they don't penalize the methods, but they also reduce per-query gold pool size).
- **Cross-encoder rerank is slow** — full eval with all 5 methods on 10 queries is
  ~2-5 min on CPU. For quick iteration on a single method, use `--method dense`.
- **`method_match` denominator**. In the *full* table, `method_match`'s mean is over
  the 5 paper queries only (it can't score short queries — no anchor). In the
  *same-subset* table, all four methods are scored on the same 5 paper queries.
  The same-subset table is the directly-comparable one.

## Reality-check protocol (when changing FIELD_WEIGHTS or DEFAULT_WEIGHTS)
1. `python -m eval.run --method all` → look at the same-subset table.
2. `python -m eval.history` → see whether the change moved nDCG@5.
3. If a (method, query) with nDCG@5 < 0.30 appears, run the dedicated error analysis
   (see P1) before adjusting weights.

## Tools to learn
- nDCG intuition — 5-min read; the metric is "how many gold papers in top-k, weighted
  by position, normalized so 1.0 = perfect ranking."
- sklearn.metrics — https://scikit-learn.org/stable/modules/model_evaluation.html
- difflib.get_close_matches — Python stdlib fuzzy match (we use cutoff 0.85).

## Commands
```bash
uv run python -m eval.gold_set --check                 # resolution check (corpus expansion signal)
uv run python -m eval.run --method all                 # full comparison + same-subset table
uv run python -m eval.run --method dense               # quick iteration on one method
uv run python -m eval.history                          # change-over-time tracker
uv run pytest tests/test_eval.py -q                    # unit tests
```
