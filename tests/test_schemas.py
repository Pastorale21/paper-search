"""Unit tests for the shared Paper / MethodCard schemas."""

from schemas import MethodCard, Paper


def test_paper_roundtrip():
    p = Paper(
        paper_id="X",
        title="T",
        abstract="A",
        year=2020,
        authors=["a"],
        citation_count=3,
        external_ids={"DOI": "d"},
        references=["R"],
        citations=["C"],
    )
    assert Paper.from_dict(p.to_dict()) == p


def test_paper_from_dict_ignores_unknown():
    p = Paper.from_dict({"paper_id": "X", "title": "T", "junk": 1})
    assert p.paper_id == "X"
    assert p.title == "T"


def test_methodcard_roundtrip():
    m = MethodCard(paper_id="X", name="N", datasets=["d"], metrics=["m"])
    assert MethodCard.from_dict(m.to_dict()) == m
