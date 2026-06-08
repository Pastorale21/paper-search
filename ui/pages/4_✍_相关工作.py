"""Related Work Draft tab — wires hybrid retrieval + method cards + LLM into a paragraph.

The LLM call is fully wired but only fires when the user clicks **Generate**.
Prompt template lives in ``ui/related_work_prompt.py`` — that's D's iteration surface.
"""

from __future__ import annotations

import json
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from nlp import config as nlp_config  # noqa: E402
from ui import api  # noqa: E402
from ui.related_work_prompt import (  # noqa: E402
    build_messages,
    extract_citation_markers,
    parse_llm_response,
)

DEMO_DRAFT = (
    "We study graph contrastive learning for collaborative filtering under sparse "
    "user-item interactions. Existing graph recommenders often rely on structural "
    "augmentations or deeper message passing, which can introduce noise and increase "
    "training cost. Our idea is to preserve the lightweight propagation of graph "
    "collaborative filtering while constructing mechanism-aware contrastive views from "
    "user-item neighborhoods and item co-occurrence signals. The goal is to improve "
    "recommendation robustness without depending on expensive graph perturbations."
)

st.set_page_config(page_title="相关工作 · GNN-RecSys", layout="wide")
st.title("✍️ 相关工作草稿")
st.caption(
    "粘贴一段草稿想法或摘要 → 系统通过 hybrid 检索召回 top 候选论文,加载它们的"
    "方法卡,并请 LLM 生成一段带 [N] 引用标记的连贯相关工作段落。"
)

if st.button("加载 demo 摘要"):
    st.session_state["related_work_draft"] = DEMO_DRAFT

user_input = st.text_area(
    "你的草稿想法 / 摘要",
    height=180,
    placeholder="粘贴 1-3 段描述你想法的文字。输入越丰富,召回的候选越好,生成的段落也越有用。",
    key="related_work_draft",
)

cols = st.columns(2)
n_citations = cols[0].slider("使用的引用数量", min_value=5, max_value=15, value=8)
target_words = cols[1].slider("目标段落长度(词数)", min_value=150, max_value=400, value=250)

st.divider()

generate = st.button(
    "生成相关工作段落",
    type="primary",
    disabled=not user_input.strip(),
)

if generate:
    with st.spinner("正在召回候选论文..."):
        retrieved = api.search(user_input.strip(), mode="paper", method="hybrid", k=n_citations)
    if not retrieved:
        st.warning("检索返回 0 个候选。请尝试更长 / 更具体的草稿。")
        st.stop()

    st.subheader(f"已召回 {len(retrieved)} 篇候选论文")
    for i, r in enumerate(retrieved, 1):
        paper = r["paper"] or {}
        st.markdown(
            f"**[{i}]** {paper.get('title') or '?'} · {paper.get('year') or '?'} · "
            f"`{r['paper_id']}` · 分数 `{r['score']:.3f}`"
        )

    messages = build_messages(user_input.strip(), retrieved, target_words=target_words)

    st.subheader("LLM 调用")
    if not nlp_config.LLM_API_KEY:
        st.error("`LLM_API_KEY` 未配置。请按 `nlp/HANDOFF.md` 在 `.env` 中设置,然后重新加载页面。")
        st.stop()

    try:
        client = api.get_llm_client()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    with st.spinner(f"正在调用 {nlp_config.LLM_MODEL} ..."):
        try:
            resp = client.chat.completions.create(
                model=nlp_config.LLM_MODEL,
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or ""
        except Exception as e:
            st.error(f"LLM 调用失败:{e}")
            st.stop()

    parsed = parse_llm_response(raw)
    paragraph = parsed.get("paragraph") or ""
    references = parsed.get("references") or []

    st.subheader("生成的段落")
    if parsed.get("_parse_error"):
        st.warning("LLM 返回了非 JSON 输出(已启用解析回退)。下方显示原始文本。")
    st.markdown(paragraph)

    markers = extract_citation_markers(paragraph)
    if markers:
        st.caption(f"段落中发现的引用标记:{markers}")

    st.subheader("参考文献")
    paper_by_id = api.get_papers_by_id()
    if not references:
        st.info("LLM 未返回参考文献列表——见下方事实核查。")
    else:
        for ref in references:
            n = ref.get("n")
            pid = ref.get("paper_id") or "?"
            reason = ref.get("one_line_reason") or ""
            paper = paper_by_id.get(pid, {})
            st.markdown(
                f"**[{n}]** {paper.get('title') or '?'} · {paper.get('year') or '?'} · "
                f"`{pid}`  \n_{reason}_"
            )

    with st.expander("🔍 事实核查(原始召回论文 + LLM 响应)"):
        st.markdown("**逐引用来源核查**——每个 [N] 标记都应映射到一篇真实召回的论文:")
        for n, item in enumerate(retrieved, 1):
            paper = item["paper"] or {}
            first_sent = (paper.get("abstract") or "").split(". ")[:1]
            first_sent = first_sent[0] + ("." if first_sent else "")
            st.markdown(
                f"- **[{n}]** `{item['paper_id']}` — {paper.get('title') or '?'}  \n"
                f"  摘要开头:_{first_sent or '(无摘要)'}_"
            )
        st.markdown("**原始 LLM 响应(供 D 迭代 prompt):**")
        st.code(raw, language="json")
        st.markdown("**发送给 LLM 的消息(prompt 输入):**")
        st.code(json.dumps(messages, indent=2, ensure_ascii=False), language="json")
