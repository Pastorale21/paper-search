"""PyVis network renderer for citation graph paths — keeps lib/ pollution out of cwd."""

from __future__ import annotations

import pathlib
import tempfile

import streamlit.components.v1 as components
from pyvis.network import Network

# Edge color by intent — warm-palette hues matching the app theme (ui/style.py).
_EDGE_COLOR = {
    "background": "#A8A29A",  # warm gray
    "method": "#2F8077",  # teal (app accent family)
    "comparison": "#C2802F",  # amber
}
# Display-only Chinese labels for the edge hover tooltip. The intent KEYS stay English —
# graph_reason.get_edge_intent + INTENT_WEIGHTS depend on them; never translate the keys.
_INTENT_LABELS = {
    "background": "背景",
    "method": "方法",
    "comparison": "对比",
}
_CENTER_COLOR = "#C75B39"  # warm rust — the focal/anchor node
_NODE_COLOR = "#3E8E83"  # teal — matches the app accent
_CANVAS_BG = "#FCFBF8"  # warm paper, matches backgroundColor
_LABEL_COLOR = "#2A2622"  # warm ink, matches textColor


def render_graph(
    center_id: str,
    paths,
    papers: dict[str, dict],
    height: int = 520,
) -> None:
    """Render the union of nodes/edges across ``paths`` as a PyVis network.

    ``paths`` is an iterable of objects with ``nodes`` and ``edge_intents`` attributes
    (matches ``retrieval.graph_reason.GraphPath`` without importing it here).
    """
    net = Network(
        height=f"{height}px",
        width="100%",
        directed=True,
        notebook=False,
        cdn_resources="in_line",  # critical: keeps lib/ out of cwd (CLAUDE.md gotcha)
        bgcolor=_CANVAS_BG,
        font_color=_LABEL_COLOR,
    )

    seen_nodes: set[str] = set()
    seen_edges: set[tuple[str, str]] = set()

    def _add_node(pid: str) -> None:
        if pid in seen_nodes:
            return
        seen_nodes.add(pid)
        title = papers.get(pid, {}).get("title") or pid
        year = papers.get(pid, {}).get("year") or "?"
        label = title if len(title) <= 40 else title[:39] + "…"
        net.add_node(
            pid,
            label=label,
            title=f"{title} ({year})",
            color=_CENTER_COLOR if pid == center_id else _NODE_COLOR,
        )

    _add_node(center_id)
    for path in paths:
        nodes = list(path.nodes)
        intents = list(path.edge_intents)
        for n in nodes:
            _add_node(n)
        for i in range(len(nodes) - 1):
            edge = (nodes[i], nodes[i + 1])
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            intent = intents[i] if i < len(intents) else "background"
            net.add_edge(
                nodes[i],
                nodes[i + 1],
                color=_EDGE_COLOR.get(intent, _EDGE_COLOR["background"]),
                title=_INTENT_LABELS.get(intent, intent),
            )

    # Write to /tmp (NOT cwd) — never adds clutter to the repo.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w") as f:
        net.write_html(f.name, notebook=False)
        html = pathlib.Path(f.name).read_text(encoding="utf-8")
    components.html(html, height=height + 20)
