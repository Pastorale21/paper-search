"""Build a NetworkX citation graph (induced subgraph over the fetched corpus)."""

from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx

from schemas import Paper

from . import config
from .fetch import load_papers


def build_graph(papers: list[Paper]) -> nx.DiGraph:
    """Build a directed citation graph; edges kept only when both endpoints are in the corpus."""
    g = nx.DiGraph()
    ids = {p.paper_id for p in papers}
    for p in papers:
        g.add_node(p.paper_id, title=p.title, year=p.year, citation_count=p.citation_count)
    for p in papers:
        for ref in p.references:
            if ref in ids:
                g.add_edge(p.paper_id, ref)
    return g


def save_graph(g: nx.DiGraph, path: Path = config.GRAPH_PKL) -> None:
    """Persist the graph via plain pickle (networkx 3.x removed write_gpickle)."""
    with open(path, "wb") as f:
        pickle.dump(g, f)


def load_graph(path: Path = config.GRAPH_PKL) -> nx.DiGraph:
    """Load the graph from pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)


def main(force: bool = False) -> nx.DiGraph:
    """Build + cache the citation graph, skipping if cache exists (unless force)."""
    if config.GRAPH_PKL.exists() and not force:
        g = load_graph()
        print(f"[graph] cache hit: |V|={g.number_of_nodes()} |E|={g.number_of_edges()}")
        return g
    papers = load_papers()
    g = build_graph(papers)
    save_graph(g)
    isolated = len(list(nx.isolates(g)))
    print(
        f"[graph] built citation graph: |V|={g.number_of_nodes()} "
        f"|E|={g.number_of_edges()} isolated={isolated}"
    )
    return g
