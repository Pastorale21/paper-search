# Method Match Follow-Up For B/C

Owner: D  
Target owners: B (`nlp/`) and C (`retrieval/`, `index/`)  
Reason: latest eval shows `hybrid` beats dense, but standalone `method_match` falls below
dense on the paper-query same subset.

## Current Evidence

Latest eval artifact: `data/cache/eval/20260608T153517Z.json`

Same-subset paper-query metrics:

| method | nDCG@5 | MRR | Recall@10 | n |
| --- | ---: | ---: | ---: | ---: |
| dense | 0.253 | 0.486 | 0.339 | 9 |
| method_match | 0.188 | 0.361 | 0.206 | 9 |
| hybrid | 0.266 | 0.476 | 0.283 | 9 |

Interpretation: method-card signals help as part of `hybrid`, but standalone field matching
is not yet robust enough to replace semantic retrieval.

## B: Method-Card Quality Request

Several classic gold papers have sparse method cards, especially missing `loss`. Because
`retrieval.method_match` currently does not renormalize scores over missing fields, sparse
cards have a lower maximum possible score.

Please inspect and improve method cards for:

| query | paper | issue |
| --- | --- | --- |
| `P3`, `P10` | `CKAN`, `KGIN`, `RippleNet`, `CKE` | KG classic comparators often have empty or vague `loss` / `backbone` fields. |
| `P7` | `GraphRec`, `SocialLGN`, `KCGN`, `DiffNet++` | Social comparator cards may be less detailed than newer hypergraph/self-supervised papers. |
| `P8` | `DDTCDR`, `CCDR`, `DisenCDR`, `PPGN` | Cross-domain gold cards should explicitly mention transfer, shared users, and domain-specific/shared representations. |

Suggested extraction rule: if the abstract does not name an exact loss, use a conservative
phrase such as `not specified in abstract` only when truly unknown; otherwise include the
training objective described by the paper.

## C: Retrieval Ablation Request

Please test a normalized field-score variant for `method_match`.

Current behavior:

- Fields: `task`, `backbone`, `loss`, `key_idea`
- Weights: `0.20`, `0.30`, `0.20`, `0.30`
- If either side lacks a field, that field contributes `0`.
- Total score is not renormalized.

Proposed ablation:

```text
score = sum(weight[field] * cosine(field)) / sum(weight[field] for comparable fields)
```

Only apply this when at least one comparable field exists. Keep the current behavior as the
default until eval confirms the change.

Suggested experiment matrix:

| variant | expected signal |
| --- | --- |
| current unnormalized field score | baseline from latest run |
| normalized comparable-field score | tests whether sparse classic cards stop being penalized |
| normalized score + minimum comparable fields >= 2 | avoids over-trusting a single generic field |

D will compare with:

```powershell
uv run python -m eval.run --method all
uv run python -m eval.history
```

Success criterion: `method_match` should improve without reducing `hybrid` below dense on
the same-subset paper-query table.
