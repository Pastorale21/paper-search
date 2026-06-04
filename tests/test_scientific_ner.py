"""Unit tests for Scientific NER extraction and cache shape."""

from nlp.scientific_ner.extractor import ScientificNERExtractor
from schemas import Paper


def test_scientific_ner_extracts_expected_entities():
    def fake_pipeline(text):
        assert "LightGCN" in text
        return [
            {"word": "LightGCN", "entity_group": "method", "score": 0.93},
            {"word": "Gowalla", "entity_group": "dataset", "score": 0.91},
            {"word": "NDCG@20", "entity_group": "metric", "score": 0.98},
        ]

    extractor = ScientificNERExtractor(ner_pipeline=fake_pipeline)
    entities = extractor.extract_text("LightGCN is evaluated on Gowalla using NDCG@20.")
    assert [e.label for e in entities] == ["method", "dataset", "metric"]
    assert entities[0].text == "LightGCN"
    assert entities[0].confidence == 0.93


def test_scientific_ner_cache_record_shape():
    def fake_pipeline(text):
        return [{"word": "BPR", "entity_group": "metric", "score": 0.7}]

    extractor = ScientificNERExtractor(ner_pipeline=fake_pipeline)
    record = extractor.extract_paper(
        Paper(paper_id="W1", title="A Paper", abstract="Optimized with BPR.")
    )
    assert record == {
        "paper_id": "W1",
        "entities": [{"text": "BPR", "label": "metric", "confidence": 0.7}],
    }
