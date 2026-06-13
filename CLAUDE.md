# Paper Search & Citation Reasoning System

> 这是 Claude Code 的项目入口(agent 指令)。面向人的项目总览 / 任务分发在 `docs/onboarding.md`,各模块细则在该模块的 `HANDOFF.md`。
> 仓库:`github.com/Pastorale21/paper-search`(private)

## 项目定位
NLP 课程大作业(4 人组),GNN-based recsys 方向的论文语义搜索系统。
核心差异化:**机制级检索**(method-level)+ **引文图多跳推理**,而非词面 / 主题相似。
论点:dense embedding 在同一话题簇里饱和(一堆 GNN 推荐论文余弦挤在 0.95–0.96),用 LLM 把摘要抽成结构化"方法卡"做字段级比对来破局。
核心结果(v2-expanded-d gold set,n=30;817-paper 语料、解析率 98.7%、paper-query same subset n=10、未调参默认权重):在最完整的语料上 **hybrid 0.247 nDCG@5 > dense 0.221(+0.026)**;standalone method_match 0.198、最佳变体 method_match_norm2 0.196 仍略低于 dense。**驱动力是 per-query 机制级差异化**——method-card 在 dense 饱和 / 失效的查询上明显拉开(P2:0.131 → 0.553;P5:0.000 → 0.339;P1 / P4 hybrid 0.509 / 0.485 vs dense 0.214 / 0.170),代价是 KG 类查询(P3 / P9 / P10)dense 更强。诚实补充:n=10、deltas 偏小、dense 仍领先 Recall@10;优势随语料完整度上升(800 时 +0.013、810 补卡前 −0.002、817 card-complete +0.026)。细节见 `docs/eval_findings_d.md`。



## 五层架构(+ 冻结参考实现)
运行依赖顺序:`data → (nlp ∥ retrieval) → eval / graph_reason → ui`。

1. **data/** — 语料摄入与处理
   - `sources/openalex.py` 抓 metadata/corpus、`sources/s2_contexts.py` 抓 Semantic Scholar 引用上下文句子(**目前 NotImplementedError,B 的活**)、`sources/seed_papers.py` 种子论文
   - `parse/grobid_client.py` PDF 解析(GROBID)
   - `corpus.py` 去重 + 质量过滤
2. **nlp/** — 结构化抽取
   - `method_card/`(extractor + prompts):DeepSeek 把摘要抽成方法卡(差异化核心)
   - `scientific_ner/`(extractor + train_ner)
   - `citation_intent/`(classifier + train_scicite):SciCite 类意图分类 background/method/comparison(**classifier 仍是 stub,B 的活**)
3. **retrieval/** — 在线检索与推理(**不存在独立 index/ 层**;FAISS 索引 / 引文图缓存由 `spike/` 构建,落在 `data/cache/`)
   - `dense.py`(SPECTER2)+ `bm25.py` + `method_match.py`(字段级加权余弦)+ `rerank.py`(cross-encoder,**默认禁用,见局限**)+ `hybrid.py`(RRF 融合)
   - `graph_reason.py`:引文图多跳推理 ancestors / cross-domain / opposing
4. **eval/** — `gold_set.py` + `gold_set.json` + `metrics.py`(nDCG@5 / MRR / Recall@10)+ `run.py`(五方法对比)+ `history.py`
5. **ui/** — 4-tab Streamlit,**全部经 `ui/api.py` 单一后端入口**
   - `pages/`:搜索 / 方法卡 / 引文图 / 相关工作
   - `components/`:graph_view(PyVis)、paper_card、reason_tags
   - `related_work_prompt.py`:Tab4 RAG prompt

> **`spike/` 是冻结的 walking skeleton 参考实现,任何人不要改。** 它同时是建索引 / 建图的入口(`build_index.py`、`build_graph.py`、`embed.py`、`fetch.py`)。

## 模块 owner
- `data/`                        → A
- `nlp/`                         → B
- `retrieval/`(含 graph_reason) → C
- `eval/` + `ui/`                → D
- `tests/` 就近归各 owner;`spike/`、`schemas.py` 全队共有

## Commands
均在仓库根目录、用 `uv` 跑。

- `KMP_DUPLICATE_LIB_OK=TRUE uv run streamlit run ui/app.py` — 起 UI(localhost:8501;macOS 需要 KMP fix)
- `uv run python -m spike` — 端到端 smoke + **重建语料 / 索引 / 引文图缓存**(walking skeleton)
- `uv run python -m eval.gold_set --check` — gold set 解析率检查
- `uv run python -m eval.run --method all` — 五方法对比表(无付费)
- `uv run pytest -q` — 全套测试(应全绿)
- `uv run ruff check . && uv run black --check .` — lint(**提交前必跑**)

> 不存在 `data.ingest` / `index.build` 这类模块,别调。摄入与建索引统一走 `spike`。

## 技术栈
- Python 3.10+,环境用 `uv` 管理
- transformers, sentence-transformers(SPECTER2 dense embedding、SciBERT/SciCite 意图)
- faiss-cpu, rank_bm25, networkx
- streamlit, pyvis
- 数据源:**OpenAlex**(metadata/corpus)+ **Semantic Scholar**(引用上下文)+ **GROBID**(PDF 解析)。不是 arXiv。
- LLM API:DeepSeek(默认)/ Claude / OpenAI,通过 `.env` 切换
- 重训练类重依赖(transformers / datasets / accelerate)应放 `pyproject.toml` 的可选依赖组,不进核心依赖

## 数据 schema(改前必须全队同意)
见 `schemas.py`,核心两个:`Paper` 和 `MethodCard`。
`MethodCard` 字段(改 = method_match 崩):`task / input / output / backbone / loss / key_idea / datasets / metrics`。

## 缓存策略(读清楚再操作)
- 基准缓存(papers.json、方法卡、embeddings.npy、faiss.index、ids.json、citation_graph.pkl)**已用 allow-list 方式提交进 git**——新 clone 不用重建、不花钱直接能跑。
- `.gitignore` 只放行这几个 canonical 文件,其余 `data/cache/*`(临时 / 中间产物)仍被忽略。
- **缓存更新是受控操作、走 PR 提交**(默认由 A 在扩语料 / 重建后提交):别把本地随手重建的二进制 churn 提上去——二进制 merge conflict 没法解。
- **重建语料时不要 `spike --force`**:`--force` 会覆盖 `papers.json`。正确做法是 `rm data/cache/{embeddings.npy,ids.json,faiss.index,citation_graph.pkl}` 后跑 `uv run python -m spike`(不带 force)。
- 运行时只把缓存 load 进内存做近邻检索,SPECTER2 不会对文档侧重新编码;实时编码的只有查询本身。

## 付费政策(agent 必须遵守)
- 任何**付费 LLM 调用**(新语料的方法卡抽取、Tab4 prompt 迭代等)**由组长手动授权执行**。
- **agent 在任何付费调用前停下**,把计划交给组长 review,不自行触发。

## 约定
- `main` 受保护,功能分支 `feat/xxx` / `fix/xxx`,PR 至少 1 个 reviewer,合并用 `--no-ff`
- 分支保护:approval 必须来自非最后 push 者(four-eyes);commit 加 co-author trailer
- 所有公开函数有 type hints 和单行 docstring
- 测试就近放:`foo.py` ↔ `test_foo.py`
- Commit message:祈使句、≤72 字符
- LLM API key 统一走 `.env`,**永不写进代码**

## Do NOT
- **不要碰 `spike/`**(冻结参考实现)
- 不要 `spike --force`(覆盖 papers.json,见缓存策略)
- 不要在测试里跑 GROBID(慢、依赖 Docker),用 `tests/fixtures/papers_sample.json` 的预解析 JSON
- 不要提交 PDF、模型权重、以及 allow-list 之外的 `data/cache/` 临时产物;**别 recommit 本地重建的二进制 churn**(注意:`data/` 下的**源码**如 `corpus.py`、`sources/*.py` 是要提交的,别误删跟踪)
- 不要改 `schemas.py` / 下列接口契约 不通知全队
- 不要在 n=5 的 gold set 上调检索权重(过拟合;等 D 扩到 30–50 再调)

## 不能破的接口契约
- `papers.json` schema(A 改 = 全员下游崩)
- `MethodCard` 字段 task/input/output/backbone/loss/key_idea/datasets/metrics(B)
- `gold_set.json` schema + `metrics.py` 函数签名(D)
- `ui/api.py` 单一后端入口(所有 UI 调用只走它)
- `citation_graph.pkl` 的 `intent` 边属性:写显示标签不改 key,key 永远是英文 `background` / `method` / `comparison`(B 写、graph_reason 读)

## 已知坑 / 诚实局限(答辩也要说)
- **find_opposing 目前是机制距离 fallback**——没有真 citation intent(B 的 `s2_contexts` + classifier + 标注脚本落地后才升级),别吹成已完成。
- **cross-encoder rerank 是诚实负结果**:ms-marco 在学术机制匹配上 −0.106,默认禁用。
- **near-dup 去重缺口**:LightGCN 有两条副本(A 的 P0)。
- **引文图 OUT-sparsity**:奠基论文 out-edge 少,ancestors 偏"下游→上游";A 扩到 800 后改善。
- **eval:语料 817、解析率 98.7%、n=30**(by-id 补了 17 篇真论文 + 抽卡;只剩 HGCN / GFair 未识别)。same-subset n=10 card-complete 上 **hybrid 0.247 > dense 0.221(+0.026)**,优势随语料完整度恢复;诚实结论:method-card 为 per-query 机制级互补信号(P1/P2/P4/P5 拉开、KG 类 dense 更强),standalone norm2 仍略低于 dense。n=10、dense 领先 Recall@10。look-alike 用 `GOLD_ANCHORS` exact-id 锚定。见 `docs/eval_findings_d.md`。




## 索引
- 面向人的总览 / 任务分发:`docs/onboarding.md`
- 各模块细则:`data/HANDOFF.md`、`nlp/HANDOFF.md`、`retrieval/HANDOFF.md`、`eval/HANDOFF.md`、`ui/HANDOFF.md`
- 脚手架 slash 命令:`.claude/commands/scaffold-*.md`