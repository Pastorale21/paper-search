# Corpus Gap Request For A

Owner: D  
Target owner: A (`data/`)  
Reason: latest D eval run (`data/cache/eval/20260608T153517Z.json`) shows that several
gold papers are absent from the current 800-paper corpus. These gaps lower gold-title
resolution and make paper-query evaluation less stable.

## Please Add First

These papers affect the paper-query same-subset table or remove a paper-query anchor.

| priority | gold title | exact title / search keyword | affected queries | why it matters |
| --- | --- | --- | --- | --- |
| P0 | DiffNet | `A Neural Influence Diffusion Model for Social Recommendation` | `Q8`, `Q12`, `Q17`, `P7`; anchor for `P5` | `P5` is currently excluded from method_match same-subset because the `DiffNet` anchor is unresolved. |
| P0 | KGCN | `Knowledge Graph Convolutional Networks for Recommender Systems` | `Q4`, `Q14`, `Q20`, `P3`, `P10` | Repeated KG gold gap; affects both short and paper KG queries. |
| P1 | FGNN | `Feature Graph Neural Networks for Session-based Recommendation` | `Q7`, `P4` | Session recommendation gold gap. |
| P1 | GFCF | `Graph Filter Collaborative Filtering` | `Q11`, `Q18`, `P9` | Scalability/light graph CF gold gap. |
| P1 | DuoRec | `DuoRec` / `Contrastive Learning for Representation Degeneration Problem in Sequential Recommendation` | `Q15`, `P6` | Sequential contrastive gold gap. |
| P1 | CoSeRec | `CoSeRec` / `Contrastive Learning for Sequential Recommendation with Robust Augmentation` | `Q15`, `P6` | Sequential contrastive gold gap. |
| P1 | ICLRec | `ICLRec` / `Intent Contrastive Learning for Sequential Recommendation` | `Q15`, `P6` | Sequential contrastive gold gap. |

## Add If Time Allows

These are useful for coverage but affect fewer current paper-query conclusions.

| priority | gold title | exact title / search keyword | affected query |
| --- | --- | --- | --- |
| P2 | DHCF | `Dual Channel Hypergraph Collaborative Filtering` | `Q9` |
| P2 | HGCN | `Hypergraph Convolutional Network for Collaborative Filtering` | `Q9` |
| P2 | HyperRec | `Hypergraph based recommendation` | `Q9` |
| P2 | DisenHAN | `DisenHAN` | `Q10` |
| P2 | DCCF | `Disentangled Contrastive Collaborative Filtering` | `Q10` |
| P2 | MetaHIN | `MetaHIN` / `Meta Learning on Heterogeneous Information Networks` | `Q12` |
| P2 | PTUPCDR | `Pre-train User Preference for Cross-Domain Recommendation` | `P8` |

## Fairness Coverage

`Q13` is skipped because none of its fairness gold papers resolve. Add this group if the
report needs fairness/cold-start breadth:

- `FairGo`
- `FairRec`
- `NFCF`
- `GFair`
- `FairGNN`

## Constraints

- Do not commit any `data/cache/` artifacts, PDFs, or model weights.
- Keep corpus additions compatible with `schemas.Paper`.
- After adding papers and rebuilding indexes, please tell D which exact titles resolved.

## D Verification After A Updates

D will run:

```powershell
# After A rebuilds the corpus (uv run python -m data.corpus --force):
uv run python -m eval.gold_set --check
# Rebuild indexes/graph from the updated papers.json (there is NO `index.build` module;
# building goes through spike). Do NOT pass --force to spike — it would overwrite papers.json.
rm data/cache/embeddings.npy data/cache/ids.json data/cache/faiss.index data/cache/citation_graph.pkl
uv run python -m spike
uv run python -m eval.run --method all
```

Success criteria for D:

- Gold-title resolution should improve beyond the current `118 / 150 = 78.7%`.
- `P5` should re-enter the method_match same-subset once `DiffNet` resolves as an anchor.
- KG paper-query reliability should improve once original `KGCN` resolves.
