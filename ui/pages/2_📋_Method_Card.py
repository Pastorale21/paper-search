"""Method Card tab — structured view + Find Similar Mechanism (the showcase)."""

from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from ui import api  # noqa: E402

st.set_page_config(page_title="Method Card · GNN-RecSys", layout="wide")
st.title("📋 Method Card")

papers = api.get_papers_by_id()
options = sorted(papers.values(), key=lambda p: -int(p.get("citation_count") or 0))
labels = [
    f"[{p.get('citation_count', 0):>5} cites] {p.get('title') or '?'} ({p.get('year') or '?'})"
    for p in options
]
ids = [p["paper_id"] for p in options]

default_pid = st.session_state.get("selected_paper_id")
default_index = ids.index(default_pid) if default_pid in ids else 0

choice = st.selectbox(
    "Select a paper",
    options=list(range(len(options))),
    index=default_index,
    format_func=lambda i: labels[i],
)
selected_pid = ids[choice]
selected = papers[selected_pid]
st.session_state["selected_paper_id"] = selected_pid

card = api.load_method_card(selected_pid)

# --- Top: structured card + abstract -----------------------------------------------------

st.markdown(f"### {selected.get('title') or '?'} · {selected.get('year') or '?'}")
st.caption(f"`{selected_pid}` · {selected.get('citation_count', 0):,} citations")

col_card, col_abs = st.columns([3, 2])

with col_card:
    if card is None:
        st.warning("No method card extracted for this paper yet.")
        st.code(
            "uv run python -m nlp.method_card.extractor --top 400",
            language="bash",
        )
        st.caption(
            "Per the team's paid-extraction convention, the UI does not fire the LLM "
            "extraction call. Run the CLI above (or pass a smaller `--top N`); see "
            "`nlp/HANDOFF.md` for cost estimates."
        )
    else:
        st.markdown("#### Mechanism-level fields")
        st.markdown(
            f"**🎯 task:** {card.task or '_(empty)_'}  \n"
            f"**📥 input:** {card.input or '_(empty)_'}  \n"
            f"**📤 output:** {card.output or '_(empty)_'}  \n"
            f"**🏗 backbone:** {card.backbone or '_(empty)_'}  \n"
            f"**📐 loss:** {card.loss or '_(empty)_'}  \n"
            f"**💡 key_idea:** *{card.key_idea or '_(empty)_'}*"
        )
        if card.datasets:
            st.markdown("**🗂 datasets:** " + " ".join(f":blue-badge[{d}]" for d in card.datasets))
        if card.metrics:
            st.markdown("**📊 metrics:** " + " ".join(f":violet-badge[{m}]" for m in card.metrics))

with col_abs:
    st.markdown("#### Abstract")
    abstract = selected.get("abstract") or "_(no abstract on disk)_"
    st.write(abstract)

st.divider()

# --- The showcase: Find Similar Mechanism (prominent, not buried) ------------------------

st.header("🔍 Find papers with similar mechanism")
st.markdown(
    "The headline differentiation of this system. **Per-field cosines** are shown for each "
    "candidate — this is the *visible evidence* that mechanism-level matching is what "
    "ranked the paper, not surface similarity. Field weights "
    f"(applied in the aggregate score): {api.field_weights()}."
)

if card is None:
    st.info("Method card needed before similarity ranking can run — extract first (see CLI above).")
else:
    if st.button(
        "Run mechanism match across the corpus",
        type="primary",
        use_container_width=True,
    ):
        st.session_state["_run_match_for"] = selected_pid

if st.session_state.get("_run_match_for") == selected_pid and card is not None:
    with st.spinner("Scoring all corpus papers by weighted field cosine..."):
        matches = api.match_similar_mechanism(selected_pid, k=10)
    if not matches:
        st.info("No matches — anchor's card is empty.")
    else:
        st.markdown(f"### Top-{len(matches)} mechanism-matched papers")
        weights = api.field_weights()
        for i, m in enumerate(matches, 1):
            with st.container(border=True):
                paper = m["paper"] or {}
                pid = m["paper_id"]
                title = paper.get("title") or "?"
                yr = paper.get("year") or "?"
                st.markdown(
                    f"**{i}. {title}** · {yr}  \n`{pid}` · weighted score **`{m['score']:.3f}`**"
                )

                # Per-field cosines as colored chips. This IS the visible evidence.
                pf = m["per_field"]
                chip_row = []
                for f, cos in pf.items():
                    w = weights.get(f, 0.0)
                    if cos is None:
                        chip_row.append(f":gray-badge[{f} (w={w:.2f}): _no data_]")
                    else:
                        color = "green" if cos >= 0.9 else "blue" if cos >= 0.75 else "orange"
                        chip_row.append(f":{color}-badge[{f} (w={w:.2f}): {cos:.3f}]")
                st.markdown(" ".join(chip_row))

                cand_card = m["method_card"]
                if cand_card is not None and cand_card.backbone:
                    st.caption(f"**backbone:** {cand_card.backbone} · **loss:** {cand_card.loss}")
                with st.expander("Abstract"):
                    st.write(paper.get("abstract") or "_(no abstract on disk)_")
