# D 演示脚本

负责人: D (`ui/`, `tests/eval/`)  
用途: 一条短、稳定、可复现的 UI 演示路线，以及实验结果讲法。

## 演示前准备

启动应用:

```powershell
uv run streamlit run ui/app.py
```

打开:

```text
http://localhost:8501
```

可直接使用的搜索深链:

```text
http://localhost:8501?query=graph%20contrastive%20learning%20for%20recommendation&mode=short&method=hybrid&k=10
```

## 演示流程

### 1. 首页

可以这样介绍:

> 这个系统不只是主题相似搜索。它把语义检索、方法卡机制匹配和引文图推理结合起来，
> 所以我们不仅能找论文，还能解释为什么这篇论文被召回。

展示语料统计:

- 论文数量
- 有摘要的论文数量
- 已抽取的方法卡数量
- 引文图节点数和边数

### 2. 搜索页

使用查询:

```text
graph contrastive learning for recommendation
```

设置:

- mode: `short`
- method: `hybrid`
- top-k: `10`

展示:

- 结果列表
- 分数
- hybrid 检索信号标签: `dense`, `bm25`, `method_match`
- 跳转到方法卡 / 引文图的按钮

可以这样讲:

> 每个结果旁边的标签会显示它是由哪些检索信号召回的，所以 hybrid 不是黑盒，
> 我们能看到 dense、BM25 和 method_match 分别起了什么作用。

### 3. 方法卡页

打开一个高置信论文，例如:

```text
LightGCN
```

展示:

- `task`
- `backbone`
- `loss`
- `key_idea`
- 摘要
- “查找机制相似论文”功能

点击:

```text
在全语料上运行机制匹配
```

可以这样讲:

> 每个候选论文会展示逐字段余弦相似度，这就是机制级匹配的可见证据。
> 我们可以看到分数主要来自 backbone、loss、key idea 还是 task 的相似。

### 4. 引文图页

尽量沿用刚才选中的同一篇论文。

按这个顺序点击:

1. 祖先
2. 跨域同机制
3. 对立方法

可以这样讲:

> 引文图页把检索结果变成推理轨迹。它不只返回论文，还返回引文路径和人能读懂的解释。

如果展示“对立方法”，补充说明:

> 对立方法目前使用的是机制距离回退，因为磁盘上还没有完整的逐边 citation intent。
> UI 里已经明确标注了这个限制。

### 5. 相关工作页

点击:

```text
加载 demo 摘要
```

只有在配置了 `LLM_API_KEY` 时再点击生成。

可以这样讲:

> 所有付费 LLM 调用都必须由用户手动点击触发。UI 会先召回真实候选论文和方法卡，
> 再让 LLM 生成带 `[N]` 引用标记的相关工作段落。事实核查区域会展示原始召回证据和
> 原始 LLM 响应，方便检查是否幻觉。

## 实验结果讲法

使用最新记录:

- `docs/eval_findings_d.md`
- eval 结果文件: `data/cache/eval/20260608T153517Z.json`
- gold-title resolution: `118 / 150 = 78.7%`

同一 paper-query 子集上的结果:

| method | nDCG@5 |
| --- | ---: |
| dense | 0.253 |
| method_match | 0.188 |
| hybrid | 0.266 |

稳妥结论:

> 在可直接比较的 paper-query 子集上，hybrid retrieval 超过了 dense。
> 这说明机制级信号在和语义检索融合后是有帮助的。单独的 method-card matching
> 有用，但目前还不足以单独替代 dense，因为它依赖语料覆盖率、方法卡完整度，
> 以及 gold relevance 的定义方式。

不要这样说:

```text
standalone method_match beats dense
```

也就是不要声称“单独 method_match 已经超过 dense”。

## 如果被问 method_match 为什么下降

可以这样回答:

> 误差分析发现主要有三个原因。第一，当前语料还缺少一些经典论文，比如原始 DiffNet
> 和原始 KGCN。第二，有些 gold label 标的是经典 comparator，而 method_match
> 会更倾向召回机制上最近的较新论文。第三，部分经典论文的方法卡比较稀疏，尤其是
> loss 字段缺失，而当前打分没有对缺失字段做重归一化。

可以指向:

- `docs/corpus_gap_request_for_a.md`
- `docs/method_match_followup_for_bc.md`

## 演示前 smoke check

最终展示前运行:

```powershell
uv run pytest tests/test_eval.py tests/test_ui_api.py -q
uv run python -m eval.gold_set --check
uv run python -m eval.run --method all
```

预期:

- gold-title resolution 约等于或高于 `118 / 150 = 78.7%`
- `hybrid` 在 same-subset nDCG@5 上应该仍然高于 dense
- 在 A/B/C 后续完成前，`method_match` 可能仍然低于 dense，这是已知且可解释的结果
