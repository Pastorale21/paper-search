# D Eval Findings

## Latest Run (810 corpus)

- Corpus expanded to **810 papers** — 10 real gap papers added by id (`data.corpus --merge-ids`):
  DiffNet, DuoRec, CoSeRec, HyperRec, DisenHAN, MetaHIN, PTUPCDR, FairRec, FairGo, FairGNN.
- Gold set: `v2-expanded-d`
- Gold-title resolution: **133 / 150 = 88.7%** (was `118 / 150 = 78.7%`). `Q13` fairness recovered
  from 0/5 to **3/5**.
- `P5` re-enters the same-subset now that the real **DiffNet anchor resolves** → same-subset n: 9 → 10.
- Still absent (real paper not in corpus — only acronym **look-alikes** like "Double-End KGCN"):
  **KGCN, FGNN, GFCF, DHCF, DCCF, ICLRec, NFCF, GFair**. These need manual seeds (exact OpenAlex id)
  + alias fixes; do NOT point their aliases at look-alikes (false positives).

## Main Result To Report

Same-subset paper-query table (810 corpus, directly comparable over **10** paper queries):

| method | nDCG@5 | MRR | Recall@10 | n |
| --- | ---: | ---: | ---: | ---: |
| dense | 0.230 | 0.450 | 0.315 | 10 |
| dense_rerank | 0.089 | 0.198 | 0.240 | 10 |
| method_match | 0.189 | 0.425 | 0.180 | 10 |
| method_match_norm2 | 0.204 | 0.425 | 0.205 | 10 |
| hybrid | 0.227 | 0.448 | 0.270 | 10 |

(`method_match_norm` / `_norm2` = renormalized field-score ablations; `norm2` requires ≥2 comparable
fields. `norm2` is the best method_match variant but still below dense.)

Current defensible claim:

> On the aggregate, hybrid and dense are a **statistical tie** (0.227 vs 0.230, n=10; dense even
> leads Recall@10). The value of mechanism-level method-card matching is **per-query, not
> aggregate**: it rescues queries where dense saturates or fails — `P5` (dense 0.000 → cards 0.339)
> and `P2` (0.131 → 0.553) — at the cost of queries where dense's topical signal dominates
> (`P3`/`P9`/`P10`), netting a wash.

Do NOT claim hybrid beats dense on the aggregate, or that standalone `method_match` beats dense.
The earlier "hybrid +0.013" was on the **n=9 subset that was MISSING the P5 DiffNet anchor**; once
the real DiffNet is in the corpus and P5 re-enters, the edge disappears.

### Per-query differentiation (the story to lead with)

| query | dense | method_match | norm2 | hybrid | winner |
| --- | ---: | ---: | ---: | ---: | --- |
| `P2` | 0.131 | 0.553 | 0.553 | 0.214 | **cards** (dense saturates) |
| `P5` | 0.000 | 0.339 | 0.339 | 0.131 | **cards** (dense fails) |
| `P1` | 0.214 | 0.339 | 0.339 | 0.509 | cards / hybrid |
| `P4` | 0.195 | 0.246 | 0.246 | 0.390 | cards / hybrid |
| `P3` | 0.151 | 0.000 | 0.000 | 0.000 | dense |
| `P6` | 0.441 | 0.246 | 0.246 | 0.390 | dense |
| `P9` | 0.559 | 0.168 | 0.319 | 0.319 | dense |
| `P10` | 0.390 | 0.000 | 0.000 | 0.151 | dense |

The honest thesis: method cards are a **complementary mechanism-level signal** that wins exactly
where dense's topical/semantic signal is weakest (cross-cluster social/sequential queries), not a
blanket improvement.

## Why Method Match Drops

### Corpus Gaps

These missing gold titles reduce resolved-gold pools or remove paper-query anchors:

| missing title | count | affected areas |
| --- | ---: | --- |
| KGCN | 5 | KG recommendation: `Q4`, `Q14`, `Q20`, `P3`, `P10` |
| DiffNet | 4 | social recommendation: `Q8`, `Q12`, `Q17`, `P7`; also leaves `P5` anchor unresolved |
| GFCF | 3 | scalable/light graph CF: `Q11`, `Q18`, `P9` |
| FGNN | 2 | session recommendation: `Q7`, `P4` |
| DuoRec / CoSeRec / ICLRec | 2 each | sequential contrastive recommendation: `Q15`, `P6` |
| DHCF / HGCN / HyperRec | 1 each | hypergraph recommendation: `Q9` |
| FairGo / FairRec / NFCF / GFair / FairGNN | 1 each | fairness query `Q13` |
| DisenHAN / DCCF / MetaHIN / PTUPCDR | 1 each | disentanglement, cold-start, cross-domain |

Priority request for A: add `DiffNet` original and `KGCN` original first. They affect the
paper-query same subset most directly.

### Gold Definition Mismatch

For several paper queries, `method_match` retrieves mechanism-nearest newer papers, while
the gold set contains canonical related papers or classical baselines.

- `P7` (`MHCN`): top method-match papers are hypergraph + self-supervised / multi-channel
  models. This is mechanism-plausible, but gold labels are social-recommendation canonical
  comparators such as `GraphRec`, `DiffNet++`, `SocialLGN`, and `KCGN`.
- `P8` (`BiTGCF`): resolved gold papers rank just outside top 10 for standalone
  method_match, while newer cross-domain GNN papers occupy the top ranks.
- `P10` (`KGIN`): method_match favors newer KG-GNN / fine-grained / contrastive KG papers,
  while gold includes classic KG-aware comparators such as `KGAT`, `RippleNet`, `CKE`,
  and `CKAN`.

### Method-Card Sparsity

`method_match` uses weighted field cosine over `task`, `backbone`, `loss`, and `key_idea`.
If a field is empty on either side, that field contributes zero and the score is not
renormalized. Several classic gold papers have sparse method cards, especially missing
`loss`, so their maximum possible score is lower than papers with all four fields filled.

## Per-Query Notes

- Strong hybrid wins: `P1`, `P4`, `P6`, `P9`.
- Hybrid small win over dense overall: `0.266` vs `0.253` nDCG@5.
- Method-match strong cases: `P1`, `P2`, `P6`.
- Method-match failure cases: `P3`, `P7`, `P8`, `P10`.
- `P5` is not in same-subset because the `DiffNet` anchor is unresolved; add original
  DiffNet to recover this query for method_match.

## Report Wording

Suggested experiment paragraph:

> On the expanded 30-query gold set, hybrid retrieval achieves the best same-subset
> paper-query nDCG@5 among directly comparable methods (0.266 vs. dense 0.253). This
> supports the value of adding method-card and sparse lexical signals to semantic search.
> However, standalone method-card matching underperforms dense retrieval in this run
> (0.188 vs. 0.253), mainly because the current corpus misses several canonical papers
> and because method-card fields are incomplete for some classic baselines. Error analysis
> shows that method matching often retrieves mechanism-nearest newer papers, whereas the
> gold set sometimes encodes canonical comparator papers. We therefore treat method-card
> matching as a complementary signal rather than a standalone replacement for semantic
> retrieval.

## Next Actions

1. Ask A to expand the corpus with the priority gaps above.
2. Re-run `uv run python -m eval.gold_set --check` after corpus expansion.
3. Re-run `uv run python -m eval.run --method all`.
4. If method-card sparsity remains visible, ask B/C whether to either:
   - improve method-card extraction for missing `loss` fields, or
   - test a normalized field-score variant that divides by weights present on both papers.
