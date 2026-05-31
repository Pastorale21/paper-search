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

st.set_page_config(page_title="Related Work · GNN-RecSys", layout="wide")
st.title("✍️ Related Work Draft")
st.caption(
    "Paste a draft idea or abstract → the system retrieves top candidate papers via "
    "hybrid retrieval, loads their method cards, and asks the LLM for a coherent "
    "related-work paragraph with [N] citation markers."
)

user_input = st.text_area(
    "Your draft idea / abstract",
    height=180,
    placeholder=(
        "Paste 1-3 paragraphs describing your idea. The richer the input, the better the "
        "retrieved candidates and the more useful the generated paragraph."
    ),
)

cols = st.columns(2)
n_citations = cols[0].slider("Number of citations to use", min_value=5, max_value=15, value=8)
target_words = cols[1].slider(
    "Target paragraph length (words)", min_value=150, max_value=400, value=250
)

st.divider()

generate = st.button(
    "Generate related-work paragraph",
    type="primary",
    disabled=not user_input.strip(),
)

if generate:
    with st.spinner("Retrieving candidate papers..."):
        retrieved = api.search(user_input.strip(), mode="paper", method="hybrid", k=n_citations)
    if not retrieved:
        st.warning("Retrieval returned 0 candidates. Try a longer / more specific draft.")
        st.stop()

    st.subheader(f"Retrieved {len(retrieved)} candidate papers")
    for i, r in enumerate(retrieved, 1):
        paper = r["paper"] or {}
        st.markdown(
            f"**[{i}]** {paper.get('title') or '?'} · {paper.get('year') or '?'} · "
            f"`{r['paper_id']}` · score `{r['score']:.3f}`"
        )

    messages = build_messages(user_input.strip(), retrieved, target_words=target_words)

    st.subheader("LLM call")
    if not nlp_config.LLM_API_KEY:
        st.error(
            "`LLM_API_KEY` is not configured. Set it in `.env` per `nlp/HANDOFF.md`, then "
            "reload the page."
        )
        st.stop()

    try:
        client = api.get_llm_client()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    with st.spinner(f"Calling {nlp_config.LLM_MODEL} ..."):
        try:
            resp = client.chat.completions.create(
                model=nlp_config.LLM_MODEL,
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or ""
        except Exception as e:
            st.error(f"LLM call failed: {e}")
            st.stop()

    parsed = parse_llm_response(raw)
    paragraph = parsed.get("paragraph") or ""
    references = parsed.get("references") or []

    st.subheader("Generated paragraph")
    if parsed.get("_parse_error"):
        st.warning("LLM returned non-JSON output (parse fallback engaged). Showing raw text below.")
    st.markdown(paragraph)

    markers = extract_citation_markers(paragraph)
    if markers:
        st.caption(f"Citation markers found in paragraph: {markers}")

    st.subheader("References")
    paper_by_id = api.get_papers_by_id()
    if not references:
        st.info("LLM did not return a references list — see fact-check below.")
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

    with st.expander("🔍 Fact-check (raw retrieved papers + LLM response)"):
        st.markdown(
            "**Per-citation source check** — each [N] marker should map to a real retrieved paper:"
        )
        for n, item in enumerate(retrieved, 1):
            paper = item["paper"] or {}
            first_sent = (paper.get("abstract") or "").split(". ")[:1]
            first_sent = first_sent[0] + ("." if first_sent else "")
            st.markdown(
                f"- **[{n}]** `{item['paper_id']}` — {paper.get('title') or '?'}  \n"
                f"  Abstract opener: _{first_sent or '(no abstract)'}_"
            )
        st.markdown("**Raw LLM response (for D's prompt iteration):**")
        st.code(raw, language="json")
        st.markdown("**Messages sent to the LLM (the prompt input):**")
        st.code(json.dumps(messages, indent=2, ensure_ascii=False), language="json")
