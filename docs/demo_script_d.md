# D Demo Script

Owner: D (`ui/`, `tests/eval/`)  
Purpose: short, repeatable demo path for the paper-search UI and evaluation story.

## Before Demo

Start the app:

```powershell
uv run streamlit run ui/app.py
```

Open:

```text
http://localhost:8501
```

Useful search deep link:

```text
http://localhost:8501?query=graph%20contrastive%20learning%20for%20recommendation&mode=short&method=hybrid&k=10
```

## Demo Flow

### 1. Landing Page

Message:

> This is not just topic search. The system combines semantic retrieval with method-card
> matching and citation-graph reasoning, so we can inspect why a paper was retrieved.

Show corpus stats:

- papers
- papers with abstracts
- method cards
- citation graph nodes/edges

### 2. Search Tab

Use query:

```text
graph contrastive learning for recommendation
```

Settings:

- mode: `short`
- method: `hybrid`
- top-k: `10`

Show:

- result list
- score
- hybrid reason chips (`dense`, `bm25`, `method_match`)
- action buttons that jump to method card / citation graph

Talking point:

> The chips expose which retrieval signal contributed to each result, so hybrid retrieval
> is inspectable rather than a black box.

### 3. Method Card Tab

Open a high-confidence paper such as:

```text
LightGCN
```

Show:

- `task`
- `backbone`
- `loss`
- `key_idea`
- abstract
- "Find papers with similar mechanism"

Click:

```text
在全语料上运行机制匹配
```

Talking point:

> The per-field cosine badges are the visible evidence for mechanism-level matching. We
> can see whether the score comes from matching backbone, loss, key idea, or task.

### 4. Citation Graph Tab

Use the same selected paper if possible.

Click in this order:

1. Ancestors
2. Cross-domain same mechanism
3. Opposing method

Talking point:

> The graph tab turns retrieval into a reasoning trace. It returns not only papers but
> paths and explanations over the citation graph.

If "Opposing" is used, mention:

> Opposing is currently a mechanism-distance fallback because edge-level citation intents
> are not fully available yet. The UI labels this limitation explicitly.

### 5. Related Work Tab

Click:

```text
加载 demo 摘要
```

Then click generate only if `LLM_API_KEY` is configured.

Talking point:

> Paid LLM calls are user-initiated only. The UI first retrieves real candidate papers and
> method cards, then asks the LLM to write a citation-grounded paragraph with `[N]`
> markers. The fact-check expander shows the raw retrieved evidence and raw LLM response.

## Eval Story

Use latest documented run:

- `docs/eval_findings_d.md`
- eval artifact: `data/cache/eval/20260608T153517Z.json`
- gold-title resolution: `118 / 150 = 78.7%`

Same-subset paper-query result:

| method | nDCG@5 |
| --- | ---: |
| dense | 0.253 |
| method_match | 0.188 |
| hybrid | 0.266 |

Defensible claim:

> Hybrid retrieval beats dense on the directly comparable paper-query subset. Standalone
> method-card matching is useful but not robust enough alone, because it depends on corpus
> coverage, method-card completeness, and how gold relevance is defined.

Do not claim:

```text
standalone method_match beats dense
```

## If Asked Why Method Match Drops

Answer:

> The error analysis found three reasons. First, the corpus still misses canonical papers
> like original DiffNet and KGCN. Second, some gold labels are canonical comparators, while
> method matching retrieves mechanism-nearest newer papers. Third, classic papers often
> have sparse method cards, especially missing loss fields, and the current score does not
> renormalize over missing fields.

Point to:

- `docs/corpus_gap_request_for_a.md`
- `docs/method_match_followup_for_bc.md`

## Smoke Checks

Run before final presentation:

```powershell
uv run pytest tests/test_eval.py tests/test_ui_api.py -q
uv run python -m eval.gold_set --check
uv run python -m eval.run --method all
```

Expected:

- gold-title resolution around or above `118 / 150 = 78.7%`
- `hybrid` should remain above dense on same-subset nDCG@5
- `method_match` may remain below dense until A/B/C follow-ups are complete
