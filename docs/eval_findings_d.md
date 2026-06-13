# D Eval Findings

## Latest Run (817 corpus, card-complete)

- Corpus expanded to **817 papers** — 17 real gap papers added by exact OpenAlex id
  (`data.corpus --merge-ids`), all with extracted method cards. Round 1 (10): DiffNet, DuoRec,
  CoSeRec, HyperRec, DisenHAN, MetaHIN, PTUPCDR, FairRec, FairGo, FairGNN. Round 2 (7): KGCN, FGNN,
  GFCF, DHCF, DCCF, ICLRec, NFCF.
- Gold set: `v2-expanded-d`
- Gold-title resolution: **148 / 150 = 98.7%** (was 78.7%). `Q13` fairness 4/5.
- Only **2 gold titles still unresolved**: `HGCN`, `GFair` (real papers not yet identified — do NOT
  point their aliases at look-alikes).
- Resolution robustness: several acronyms whose real title is a SUBSTRING of a corpus look-alike
  (e.g. real `KGCN` vs "Double-End KGCN") are pinned by exact id via `GOLD_ANCHORS` (a tier-0 in the
  resolver), not fuzzy alias substrings.

## Main Result To Report

Same-subset paper-query table (817 corpus, card-complete, directly comparable over **10** queries):

| method | nDCG@5 | MRR | Recall@10 | n |
| --- | ---: | ---: | ---: | ---: |
| dense | 0.221 | 0.450 | 0.320 | 10 |
| dense_rerank | 0.085 | 0.209 | 0.240 | 10 |
| method_match | 0.198 | 0.439 | 0.200 | 10 |
| method_match_norm2 | 0.196 | 0.434 | 0.220 | 10 |
| **hybrid** | **0.247** | 0.448 | 0.280 | 10 |

(`method_match_norm` / `_norm2` = renormalized field-score ablations; `norm2` requires ≥2 comparable
fields — best standalone variant, still just below dense.)

Current defensible claim:

> On the most-complete corpus (98.7% gold resolution), **hybrid 0.247 > dense 0.221 on nDCG@5
> (+0.026)**. The win is driven by **per-query mechanism-level differentiation**: method-cards
> rescue queries where dense saturates or fails — `P2` (dense 0.131 → cards 0.553), `P5` (0.000 →
> 0.339), `P1`/`P4` (hybrid 0.509 / 0.485 vs dense 0.214 / 0.170) — at the cost of KG-heavy queries
> (`P3`/`P9`/`P10`) where dense's topical signal dominates. Standalone `method_match` (0.198) stays
> just below dense; it is a complementary fusion signal, not a replacement.

Honest caveats: n=10, so deltas are small; dense still leads `Recall@10` (0.320 vs 0.280); MRR is a
tie. The edge tracks corpus completeness — hybrid was +0.013 (n=9, 800), −0.002 (n=10, 810,
pre-cards), **+0.026 (n=10, 817, card-complete)** — so report the 817 card-complete number.

### Per-query differentiation (the story to lead with)

| query | dense | method_match | norm2 | hybrid | winner |
| --- | ---: | ---: | ---: | ---: | --- |
| `P2` | 0.131 | 0.553 | 0.553 | 0.214 | **cards** (dense saturates) |
| `P5` | 0.000 | 0.339 | 0.339 | 0.131 | **cards** (dense fails) |
| `P1` | 0.214 | 0.339 | 0.339 | 0.509 | **cards / hybrid** |
| `P4` | 0.170 | 0.214 | 0.214 | 0.485 | **cards / hybrid** |
| `P6` | 0.530 | 0.384 | 0.384 | 0.553 | ~tie (hybrid edge) |
| `P3` | 0.131 | 0.000 | 0.000 | 0.000 | dense (KG) |
| `P9` | 0.485 | 0.146 | 0.131 | 0.277 | dense |
| `P10` | 0.339 | 0.000 | 0.000 | 0.131 | dense (KG) |

The honest thesis: method cards are a **complementary mechanism-level signal** that wins exactly
where dense's topical/semantic signal is weakest (social/sequential cross-cluster queries: P1/P2/
P4/P5), tipping the aggregate to hybrid on the complete corpus — not a blanket improvement.

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
