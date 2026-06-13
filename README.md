# Paper Search & Citation Reasoning

GNN-recsys 方向的论文**机制级**语义搜索系统(NLP 课程项目)。
普通 dense embedding 在同一话题簇里饱和(一堆 GNN 推荐论文余弦挤在 0.95–0.96),本系统用 LLM
把每篇摘要抽成结构化「方法卡」(task / backbone / loss / key_idea)做**字段级**比对,并用**引文图
多跳推理**给出可追溯的「为什么相关」——而不是词面/主题相似。

> 团队总览 / 任务分发:[docs/onboarding.md](docs/onboarding.md) ·
> 各模块细则:各目录的 `HANDOFF.md` ·
> agent 指令 + 约定:[CLAUDE.md](CLAUDE.md) ·
> 最新评测结论:[docs/eval_findings_d.md](docs/eval_findings_d.md)

---

## 🚀 跑起来(3 步,开箱即用)

前置:[`uv`](https://docs.astral.sh/uv/)(Python 包/环境管理器)。Python 3.10–3.12(uv 会自动装)。

```bash
git clone https://github.com/Pastorale21/paper-search.git
cd paper-search
uv sync                                                    # 装依赖(含 dev: pytest/ruff/black)
KMP_DUPLICATE_LIB_OK=TRUE uv run streamlit run ui/app.py   # 起 UI → http://localhost:8501
```

三件事要知道:

- **开箱即用**:基准缓存(818 篇 `papers.json` + 方法卡 + embeddings/faiss/引文图)**已随仓库提交**,
  所以 clone 完**不用重建、不用 API key、不花钱**就能浏览 / 搜索 / 看引文图。
- **首次运行需要联网**,下载 SPECTER2 嵌入模型(~1–2 GB,从 HuggingFace);之后可离线跑。
- **macOS 必须加** `KMP_DUPLICATE_LIB_OK=TRUE`(faiss 和 torch 的 libomp 冲突,不加会崩)。

跑完确认环境 OK:

```bash
uv run pytest -q     # 全套测试,应全绿
```

---

## 🧭 四个 UI 板块

| Tab | 干嘛 | 要 LLM key? |
| --- | --- | --- |
| 🔍 搜索 | dense / bm25 / hybrid 检索,每条结果带信号标签(谁召回的) | 否 |
| 📋 方法卡 | 每篇的结构化字段 + “find similar mechanism” 显示 per-field cosines | 否 |
| 🕸 引文图 | 选论文跑 ancestors / cross-domain / opposing 推理,出交互子图 + 路径解释 | 否 |
| ✍ 相关工作 | 粘想法 → RAG 生成带引用标记的段落 + fact-check | **是**(没 key 时只展示候选,不生成) |

---

## ⚙️ 常用命令

```bash
uv run pytest -q                                  # 全套测试(应全绿)
uv run ruff check . && uv run black --check .     # lint —— 提交前必跑
uv run python -m eval.gold_set --check            # gold 解析率(当前 99.3%)
uv run python -m eval.run --method all            # 检索方法对比表(无付费)
uv run python -m eval.history                      # 评测随时间变化
```

---

## 🔑 .env(可选,仅付费功能需要)

浏览 / 搜索 / eval **都不需要** key。只有 **Tab4 生成段落** 和 **给新论文抽方法卡** 才会调 LLM:

```bash
cp .env.example .env     # 填你的 DeepSeek key(.env 已 gitignore)
```

变量见 [.env.example](.env.example)。
> ⚠️ **付费政策**:任何付费 LLM 调用由组长授权后手动执行,详见 [CLAUDE.md](CLAUDE.md)。

---

## 🤝 给协作者 / 贡献规范

- 功能分支 `feat/xxx` / `fix/xxx` → PR → 合 `main` 用 `--no-ff`;PR 至少 1 个 reviewer。
- 提交前必过 `uv run ruff check . && uv run black --check .` + `uv run pytest -q`。
- **不要碰 `spike/`**(冻结的参考实现);**不要 `spike --force`**(会覆盖 `papers.json`)。
- 改这些接口契约前**通知全队**:`schemas.py`(`Paper` / `MethodCard`)、`papers.json` schema、
  `gold_set.json` + `metrics.py` 签名、`ui/api.py` 单一后端入口。详见 [CLAUDE.md](CLAUDE.md)。
- 模块 owner:`data/`→A、`nlp/`→B、`retrieval/`→C、`eval/`+`ui/`→D(细则见各 `HANDOFF.md`)。

---

## 🛠 重建 / 扩充语料(一般不需要)

缓存已提交,只有**扩语料**才动它。**别用 `spike --force`**(它会覆盖 `papers.json`):

```bash
# 增量加论文(推荐,不回退、无需重爬整库):
uv run python -m data.corpus --merge-ids W2913560138,W2986515219   # 按 OpenAlex id 精确加
uv run python -m data.corpus --merge-seeds                          # 或按 seed_papers.py 的标题加

# 重建索引(删旧的再不带 --force 跑 spike):
rm data/cache/{embeddings.npy,ids.json,faiss.index,citation_graph.pkl}
uv run python -m spike

# 给新论文抽方法卡(付费,先 dry-run 看成本):
uv run python -m nlp.method_card.extractor --paper-ids W2913560138,W2986515219 --dry-run
uv run python -m eval.gold_set --check                             # 确认解析率
```

> 缓存更新是受控操作、走 PR 提交(二进制 churn,默认由 A / 组长)。

---

## 📊 当前状态

818 篇语料、gold 解析率 **99.3%**、card-complete。最完整语料上 **hybrid 0.247 > dense 0.221
(+0.026 nDCG@5)**,由 **per-query 机制级差异化**驱动(method-card 在 dense 饱和/失效的查询上拉开,
如 P2 0.131→0.553、P5 0.000→0.339)。完整诚实结论见 [docs/eval_findings_d.md](docs/eval_findings_d.md)。

唯一未解析的 gold:`HGCN`(Q9,需真正的 *hypergraph-CF* 论文 —— 不是 hyperbolic 的那篇)。
