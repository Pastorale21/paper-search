# Paper Search & Citation Reasoning System

## 项目定位
NLP 课程大作业(4 人组),做一个 GNN-based recsys 方向的论文语义搜索系统。
核心差异化:**机制级检索**(method-level)+ 引文图多跳推理,而非词面/主题相似。

## 五层架构
1. data/      论文摄入:arXiv API + Semantic Scholar API + GROBID
2. nlp/       结构化抽取:NER、关系抽取、引文意图分类、方法卡 LLM 抽取
3. index/     索引:FAISS 向量库 + NetworkX 引文图 + JSON 方法卡
4. retrieval/ 在线推理:混合检索 + cross-encoder 重排 + 多跳图推理
5. ui/        Streamlit 前端 + PyVis 图可视化

## 模块 owner
- data/      → A
- nlp/       → B
- retrieval/, index/ → C
- ui/, tests/eval/   → D

## Commands
- `uv run python -m spike`     # 100 篇语料端到端 smoke test(必须先通过才能深做)
- `uv run python -m data.ingest`   # 抓取 + 解析论文
- `uv run python -m index.build`   # 建 FAISS + 引文图
- `uv run streamlit run ui/app.py` # 起 UI(localhost:8501)
- `uv run pytest -q`           # 单元测试
- `uv run ruff check . && uv run black --check .`  # lint

## 技术栈
- Python 3.10+,环境用 `uv` 管理
- transformers, sentence-transformers(SPECTER2、SciBERT)
- faiss-cpu, rank_bm25, networkx
- streamlit, pyvis
- LLM API:DeepSeek(默认)/ Claude / OpenAI,通过环境变量切换

## 数据 schema(改前必须全队同意)
见 `schemas.py`,核心两个:`Paper` 和 `MethodCard`。

## 约定
- 所有缓存数据放 `data/cache/`,**永不提交 git**(已在 .gitignore)
- 所有公开函数有 type hints 和单行 docstring
- 测试文件就近放:`foo.py` ↔ `test_foo.py`
- Commit message:祈使句、≤72 字符
- 分支:`main` 受保护,功能分支 `feat/xxx`、`fix/xxx`,PR 至少 1 个 reviewer

## Do NOT
- 不要在测试里跑 GROBID(慢且依赖 Docker),用 `tests/fixtures/` 的预解析 JSON
- 不要提交 PDF、模型权重、任何 `data/` 下的东西
- 不要改 `schemas.py` 不通知队友
- 不要把 LLM API key 写进代码,统一走 `.env`

## 子模块详细规则
更细的规则在各子模块的 CLAUDE.md:
- `data/CLAUDE.md` — GROBID、API 限速、schema
- `nlp/CLAUDE.md`  — prompt 模板、SciCite 微调
- `retrieval/CLAUDE.md` — 混合检索权重、图遍历
- `ui/CLAUDE.md`   — Streamlit + PyVis 集成踩坑