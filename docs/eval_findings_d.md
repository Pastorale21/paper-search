# D Eval Findings

## Latest Run

- Eval artifact: `data/cache/eval/20260608T153517Z.json`
- Gold set: `v2-expanded-d`
- Gold-title resolution: `118 / 150 = 78.7%`
- Skipped query: `Q13` because all fairness gold papers are absent from the current corpus.

## Main Result To Report

Same-subset paper-query table, directly comparable over 9 paper queries:

| method | nDCG@5 | MRR | Recall@10 | n |
| --- | ---: | ---: | ---: | ---: |
| dense | 0.253 | 0.486 | 0.339 | 9 |
| dense_rerank | 0.083 | 0.192 | 0.239 | 9 |
| method_match | 0.188 | 0.361 | 0.206 | 9 |
| hybrid | 0.266 | 0.476 | 0.283 | 9 |

Current defensible claim:

> Hybrid retrieval outperforms dense retrieval on the paper-query same subset, indicating
> that mechanism-level signals are useful when fused with semantic retrieval. Standalone
> method-card matching is less stable because it depends on corpus coverage, method-card
> completeness, and whether gold labels represent canonical comparators or mechanism-nearest
> papers.

Do not claim that standalone `method_match` beats dense on this run.

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
