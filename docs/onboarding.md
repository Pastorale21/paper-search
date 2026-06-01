# GNN-RecSys Paper Search — 团队 Onboarding

> 这份文档是项目总览 + 上手指南 + 任务分发。每个模块的细节在各自的 `HANDOFF.md` 里,这份是索引。
> 仓库:`github.com/Pastorale21/paper-search`(private)

---

## 0. 一句话项目

普通语义检索(dense embedding)在同一话题簇里**饱和**——一堆 GNN 推荐论文的余弦相似度挤在 0.95–0.96、彼此只差约 0.006,分不出哪篇是"对的"。本系统在**机制层**比对:用 LLM 把每篇摘要抽成结构化"方法卡"(task / backbone / loss / key_idea),再字段级匹配;并用**引文图多跳推理**给出可追溯的"为什么相关"路径。

**核心结果(同子集、未调参默认权重)**:method_match 较 dense **+0.112 nDCG@5**;hybrid **+0.091**。

---

## 1. 架构(5 层 + 数据流)

```
data/        语料(OpenAlex 抓取 + 去重 + 质量过滤)
   │         papers.json(400 篇)、citation_graph.pkl(1516 边)
   ▼
nlp/         方法卡抽取(DeepSeek LLM)、引文意图分类(待实现)
   │         method_cards/*.json(400 张)
   ▼
retrieval/   dense(SPECTER2)+ bm25 + method_match(字段级匹配)
   │         + rerank(cross-encoder)+ hybrid(RRF 融合)
   ├── graph_reason.py   引文图推理:ancestors / cross-domain / opposing
   ▼
eval/        gold set + nDCG@5 / MRR / Recall@10
   ▼
ui/          4-tab Streamlit(全部经 ui/api.py 单一后端入口)
```

- `spike/` 是**冻结的参考实现**(walking skeleton),**任何人不要改**。
- `ui/api.py` 是 UI 与后端的**唯一接口面**,所有 UI 调用只走它。
- 运行依赖顺序:data → (nlp ∥ retrieval) → eval / graph_reason → ui。

---

## 2. 上手(每个人 clone 后)

```bash
git clone https://github.com/Pastorale21/paper-search.git
cd paper-search
uv sync                                   # 装依赖
cp .env.example .env                      # 填各自的 key(.env 已 gitignore)

# 跑起来(macOS 需要 KMP fix)
KMP_DUPLICATE_LIB_OK=TRUE uv run streamlit run ui/app.py    # localhost:8501

# 其它常用命令
uv run pytest -q                          # 全套测试(应全绿)
uv run python -m eval.gold_set --check    # gold 解析率
uv run python -m eval.run --method all    # 五方法对比表(无付费)
ruff check . && black --check .           # 提交前必跑
```

`data/cache/`(papers.json + 400 张方法卡 + 索引)**已在仓库里**,所以 clone 完不用重建、不花钱就能跑。

### .env 需要的变量
```
LLM_API_KEY=<你的 DeepSeek key>
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
```
（仅 A 扩语料抽方法卡、D 调 Tab4 prompt 时才真正调用。日常跑不需要。）

### 付费政策
> **【组长填】** 例:A 扩语料的新方法卡抽取 + D 的 Tab4 prompt 迭代会产生 DeepSeek 调用。统一走组长 / 各自 key + 预算上限 —— 在此写清楚。

---

## 3. 四个 UI 板块在干嘛

1. **🔍 搜索** — 语义 + hybrid 检索,每条结果带 reason tags(dense / bm25 / method_match),显示是哪个信号召回的。*短查询模式下 method_match 不贡献(无锚点)。*
2. **📋 方法卡** — 每篇论文的结构化字段 + "find similar mechanism" 按钮显示 **per-field cosines**(差异化的可视化证据,demo 主场)。
3. **🕸 引文图** — 选论文跑三个推理:ancestors / cross-domain / opposing,出交互子图 + 路径解释。
4. **✍ 相关工作** — 粘想法,RAG 生成带引用标记的 related-work 段落 + fact-check。

---

## 4. 模块分工 + 任务(改什么 / 提交什么)

> 通用规矩:feature 分支 → PR → 合 main 用 `--no-ff`;先读自己模块的 `HANDOFF.md`;**不碰 `spike/`**;提交前过 `ruff` + `black` + `pytest`;commit 加 co-author trailer。

### A — 数据(`data/`)· 起点 `data/HANDOFF.md`
- **P0 去重**:`data/corpus.py` 现在只按标题精确去重,漏前缀变体(W3004578093 vs W3045200674 两条 LightGCN)。修完重建缓存——**注意别 `spike --force`(会覆盖 papers.json)**,而是 `rm data/cache/{embeddings.npy,ids.json,faiss.index,citation_graph.pkl}` 再 `python -m spike` 不带 force。
- **P0 扩 corpus 400→800**:`data/sources/seed_papers.py` 加 15 篇未解析 gold(CCDR/DisenCDR/DDTCDR/PPGN、S3-Rec/CL4SRec、MHCN/SocialLGN、CKE/CKAN、XSimGCL、SURGE/FGNN),放宽抓取量。新论文要抽方法卡(走付费政策)。重建缓存后跑 `--check` 确认解析率上升。
- **P1**:cross-domain 子领域补厚(现仅 32 篇)。
- **提交**:`feat/data-dedup`、`feat/data-scale-800` —— 改后逻辑 + 重建的 papers.json / 方法卡 / 索引。
- **完成标志**:Tab2 不再撞副本;gold 解析率从 70% 上升;ancestors 不再大面积空。

### B — NLP(`nlp/`)· 起点 `nlp/HANDOFF.md`
- **P0 引文意图分类**(把 find_opposing 从 fallback 升级成真功能):
  - `data/sources/s2_contexts.py` 现在是 `NotImplementedError`,实现从 Semantic Scholar 抓每条引用的上下文句子。
  - `nlp/citation_intent/classifier.py` 现在是 stub,接入 SciCite 类意图分类(background / method / comparison)。
  - 新建一个标注脚本(**不能写在冻结的 spike/build_graph.py**):加载 citation_graph.pkl,逐边分类,写回 `g.edges[u,v]["intent"]`,存新缓存。graph_reason 的 `get_edge_intent` 会自动读它。
- **提交**:`feat/nlp-citation-intent` —— s2_contexts 实现 + classifier + 标注脚本 + 意图标注的图缓存 + 测试。
- **完成标志**:图边带 intent;find_opposing 跑出真 "comparison" 引用。

### C — 检索 + 推理(`retrieval/` 含 graph_reason)· 起点 `retrieval/HANDOFF.md`
- **P0 权重调优**(修 eval 的 P3/P5 残留):`retrieval/hybrid.py` 的 DEFAULT_WEIGHTS、`retrieval/method_match.py` 的 FIELD_WEIGHTS。**必须等 D 扩完 gold set**,否则在 n=5 上调参 = 过拟合。
- **P1**:`retrieval/rerank.py` 换 BAAI/bge-reranker-v2-m3 试(ms-marco 是负结果)。
- **P1**:`retrieval/graph_reason.py` 的 sub-area 关键词启发式换成小分类器。
- **集成**:B 的意图落地后验证 find_opposing 升级。
- **提交**:`feat/retrieval-weight-tuning` 等。

### D — 评测 + UI(`eval/` + `ui/`)· 起点 `eval/HANDOFF.md` + `ui/HANDOFF.md`
- **P0 扩 gold set 10→30-50**:`eval/gold_set.json` 加边缘 case、未覆盖子领域、2-3 个对抗 query、≥5 个新 paper-as-query 摘要。每加必跑 `--check`;**加缩写型 gold 标题时同步维护 `DEFAULT_ALIASES`**,否则误报 not-in-corpus。
- **P1 错误分析工具**:nDCG@5 < 0.3 的 query,dump top-10 + gold + 分数,定位是检索 bug 还是 corpus gap。
- **UI**:`ui/related_work_prompt.py`(Tab4 prompt 迭代,标了 TODO(D),调 prompt 花钱→付费政策)+ 整体美化 + find_opposing 综述过滤展示。
- **提交**:`feat/eval-gold-expansion`、`feat/ui-polish`。

### 依赖顺序
```
A 去重 ─────────────→ (立刻,修 Tab2,无依赖)
   ├─ A 扩 corpus ──┐
   ├─ B 引文意图 ───┤  (三条可并行)
   └─ D 扩 gold ────┘
                     ├→ C 权重调优(等 D 的 gold)
                     └→ C 验证 opposing(等 B 的意图)
C 的 bge-rerank / sub-area 分类器:无依赖,随时
```

---

## 5. 绝对不能破的接口契约

并行不撞车的前提:

- `papers.json` schema（A 改 = 全员下游崩）
- `MethodCard` 字段 task/input/output/backbone/loss/key_idea/datasets/metrics（B 改 = method_match 崩）
- `gold_set.json` schema + `metrics.py` 函数签名（D 维护）
- `ui/api.py` 单一后端入口（所有 UI 调用只走它）
- `citation_graph.pkl` 的 `intent` 边属性（B 写入,graph_reason 读;**写显示标签不改 key**,key 永远是英文 background/method/comparison）

---

## 6. 已知坑 / 诚实局限(答辩也要说)

- **spike 缓存失效**:重建语料后别 `spike --force`,见 §4-A。
- **near-dup 去重缺口**:LightGCN 有两条(A 的 P0)。
- **Q1 短查询 nDCG=0**:不是召回坏,是 gold 太严的伪影——gold 只标 5 篇 canonical,corpus 里几十篇同等相关论文合理地把它们挤出 top-10,top-10 全部相关。报告里主动说明。
- **cross-encoder rerank 是诚实负结果**:ms-marco 在学术机制匹配上 −0.106,默认禁用。
- **find_opposing 目前是机制距离 fallback**(没有真 citation intent,B 的活)——答辩别吹成已完成。
- **引文图 OUT-sparsity**:奠基论文 out-edge 少,ancestors 只能"下游→上游";A 扩到 800 后改善。
- **sub-area 推理是关键词启发式**,偶尔误判(C 的 P1 换分类器)。
- **eval 是 n=5 paper query**,小样本——趋势明确、机制可解释,报告里说"种子集,扩充中"。

---

## 7. 答辩话术(NLP 技术 ≥5,实际 8+)

1. **SPECTER2 论文嵌入**(dense 检索)— transformer + 引文对比学习,双 adapter(proximity / adhoc_query),FAISS 近邻。
2. **BM25** — 词项稀疏检索,与 dense 互补。
3. **LLM 结构化信息抽取** — DeepSeek 把摘要抽成方法卡(差异化核心)。
4. **字段级语义匹配** — 方法卡逐字段嵌入 + 加权余弦(Tab2 的 per-field cosines 是它的可视化;eval +0.112)。
5. **RRF 多路融合** — 按排名倒数融合,不依赖分数量纲(hybrid +0.091)。
6. **cross-encoder 重排** — 诚实负结果(加分:敢报负结果)。
7. **引文图多跳推理** — networkx BFS + 方法卡加权路径 + 可读解释。
8. **检索增强生成(RAG)** — Tab4 相关工作生成,带引用标记 + **fact-check expander**(grounding / 反幻觉,有深度的点)。
9. **评测** — nDCG@5 / MRR / Recall@10 + 人工 gold set。

**核心论点**:dense 在话题簇饱和 → 机制层比对破局,method_match 的增益**集中在 session/social 这类 cross-cluster 查询**(dense 失效处),**未调参就 +0.112**(更难被说"在测试集上调过参")。
