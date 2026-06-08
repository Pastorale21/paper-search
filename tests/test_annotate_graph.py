"""Unit tests for citation-graph intent annotation (no network, no model)."""

import networkx as nx

from nlp.citation_intent import annotate_graph
from nlp.citation_intent.annotate_graph import annotate_edges, s2_intent_classify
from schemas import Paper

# u cites v (matched by DOI) cites w (matched by MAG); records also reference an out-of-corpus id.
_PAPERS = [
    Paper(paper_id="U", title="citing", source_ids={"doi": "10.1/u"}),
    Paper(paper_id="V", title="cited", source_ids={"doi": "10.1/v"}),
    Paper(
        paper_id="W", title="ancestor", source_ids={"doi": "10.1/w"}, external_ids={"mag": "777"}
    ),
]


def _graph() -> nx.DiGraph:
    g = nx.DiGraph()
    for p in _PAPERS:
        g.add_node(p.paper_id)
    g.add_edge("U", "V")  # matched by DOI
    g.add_edge("V", "W")  # matched by MAG
    return g


def _fetch(paper_id):
    contexts = {
        "U": [
            {"cited_doi": "10.1/V", "intents": ["methodology"], "context": "we use V"},
            {"cited_doi": "10.1/other", "intents": ["background"], "context": "unrelated"},
        ],
        "V": [{"cited_doi": None, "cited_mag": "777", "intents": ["result"], "context": "beats W"}],
    }
    return contexts.get(paper_id, [])


def test_annotate_edges_writes_intents_via_s2_labels():
    g = _graph()
    result = annotate_edges(g, _PAPERS, _fetch, s2_intent_classify)

    assert g.edges["U", "V"]["intent"] == "method"  # methodology -> method
    assert g.edges["V", "W"]["intent"] == "comparison"  # result (matched by MAG) -> comparison
    assert result["edges_annotated"] == 2
    assert result["edge_intents"] == {"U->V": "method", "V->W": "comparison"}
    # Out-of-corpus citation ("10.1/other") never creates an edge.
    assert g.number_of_edges() == 2


def test_unmatched_and_unlabeled_edges_stay_unannotated():
    g = _graph()

    def fetch_no_intents(paper_id):
        if paper_id == "U":
            return [{"cited_doi": "10.1/v", "intents": [], "context": "mentions V"}]
        return []

    result = annotate_edges(g, _PAPERS, fetch_no_intents, s2_intent_classify)
    # S2 gave no intent -> default classifier returns None -> edge left for graph_reason fallback.
    assert "intent" not in g.edges["U", "V"]
    assert result["edges_annotated"] == 0
    assert result["counts"].get("unlabeled") == 1


def test_limit_caps_sources_processed():
    g = _graph()
    result = annotate_edges(g, _PAPERS, _fetch, s2_intent_classify, limit=1)
    # Only "U" (first sorted source) processed; "V"'s edge stays unannotated.
    assert result["sources_processed"] == 1
    assert g.edges["U", "V"]["intent"] == "method"
    assert "intent" not in g.edges["V", "W"]


def test_fetch_error_is_isolated_per_paper():
    g = _graph()

    def fetch_raises(paper_id):
        if paper_id == "U":
            raise RuntimeError("S2 down")
        return _fetch(paper_id)

    result = annotate_edges(g, _PAPERS, fetch_raises, s2_intent_classify)
    assert result["counts"].get("fetch_error") == 1
    # V still processed despite U failing.
    assert g.edges["V", "W"]["intent"] == "comparison"


def test_build_classifier_default_is_free_s2_path():
    classify = annotate_graph._build_classifier(use_scicite=False, allow_llm=False)
    assert classify is s2_intent_classify
    assert classify({"intents": ["methodology"]}) == "method"
    assert classify({"intents": []}) is None


def test_s2_intent_classify_prefers_method_over_background():
    # Multi-intent citation: method must win even when background sorts first.
    assert s2_intent_classify({"intents": ["background", "methodology"]}) == "method"
    assert s2_intent_classify({"intents": ["background", "result"]}) == "comparison"
    assert s2_intent_classify({"intents": ["background"]}) == "background"
    assert s2_intent_classify({"intents": []}) is None


def test_self_citation_edge_is_not_annotated():
    g = nx.DiGraph()
    g.add_node("U")
    g.add_edge("U", "U")  # OpenAlex occasionally lists a paper among its own references
    papers = [Paper(paper_id="U", title="t", source_ids={"doi": "10.1/u"})]

    def fetch(_):
        return [{"cited_doi": "10.1/u", "intents": ["methodology"], "context": "self"}]

    annotate_edges(g, papers, fetch, s2_intent_classify)
    assert "intent" not in g.edges["U", "U"]


def test_skip_labeled_avoids_reclassification():
    g = _graph()
    g.edges["U", "V"]["intent"] = "background"  # pretend a prior run settled this edge
    calls = []

    def recording_classify(record):
        calls.append(record)
        return s2_intent_classify(record)

    result = annotate_edges(g, _PAPERS, _fetch, recording_classify)
    # The pre-labeled U->V edge is served from the graph, not reclassified.
    assert g.edges["U", "V"]["intent"] == "background"
    assert result["counts"].get("cached") == 1
    assert all(r.get("cited_mag") == "777" for r in calls)  # only V->W reached the classifier

    # --reclassify (skip_labeled=False) forces re-labeling.
    g2 = _graph()
    g2.edges["U", "V"]["intent"] = "background"
    annotate_edges(g2, _PAPERS, _fetch, s2_intent_classify, skip_labeled=False)
    assert g2.edges["U", "V"]["intent"] == "method"


def test_build_classifier_scicite_and_llm_branches(monkeypatch):
    built = []

    class FakeClassifier:
        def __init__(self, use_scicite):
            built.append(use_scicite)

        def classify(self, record):
            return "method"

    monkeypatch.setattr(annotate_graph, "CitationIntentClassifier", FakeClassifier)

    scicite = annotate_graph._build_classifier(use_scicite=True, allow_llm=False)
    assert built[-1] is True
    assert scicite({"context": "x"}) == "method"

    llm = annotate_graph._build_classifier(use_scicite=False, allow_llm=True)
    assert built[-1] is False
    assert llm({"context": "x"}) == "method"
