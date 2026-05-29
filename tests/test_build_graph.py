"""Unit tests for citation graph construction (induced edges + pickle round-trip)."""

import json
from pathlib import Path

import networkx as nx

from schemas import Paper
from spike.build_graph import build_graph, load_graph, save_graph

FIXTURE = Path(__file__).parent / "fixtures" / "papers_sample.json"


def _papers() -> list[Paper]:
    return [Paper.from_dict(d) for d in json.loads(FIXTURE.read_text())]


def test_induced_edges():
    g = build_graph(_papers())
    assert g.number_of_nodes() == 6
    assert g.number_of_edges() == 6
    assert g.has_edge("P1", "P2")
    assert not g.has_edge("P4", "PX")  # external ref dropped


def test_isolated_node():
    g = build_graph(_papers())
    assert "P6" in set(nx.isolates(g))


def test_pickle_roundtrip(tmp_path):
    g = build_graph(_papers())
    path = tmp_path / "g.pkl"
    save_graph(g, path)
    g2 = load_graph(path)
    assert g2.number_of_nodes() == g.number_of_nodes()
    assert g2.number_of_edges() == g.number_of_edges()
