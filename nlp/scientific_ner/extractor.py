"""Extract method/dataset/metric entities from paper title + abstract text."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from functools import lru_cache

from schemas import Paper

from .. import config

ENTITY_LABELS = {"method", "dataset", "metric"}


@dataclass(frozen=True)
class ScientificEntity:
    """One extracted scientific entity mention."""

    text: str
    label: str
    confidence: float


def _entity_cache_path(paper_id: str):
    """Cache path for one paper's extracted entities."""
    return config.SCIENTIFIC_ENTITIES_DIR / f"{paper_id.replace('/', '_')}.json"


def _paper_text(paper: Paper) -> str:
    """Join title and abstract into the text shown to the NER model."""
    return f"{paper.title}\n\n{paper.abstract or ''}".strip()


def _normalize_label(label: str) -> str | None:
    """Normalize token-classification labels into method/dataset/metric."""
    normalized = (label or "").lower().replace("b-", "").replace("i-", "")
    normalized = normalized.replace("entity_", "").replace("label_", "")
    aliases = {
        "0": None,
        "1": "method",
        "2": "method",
        "3": "dataset",
        "4": "dataset",
        "5": "metric",
        "6": "metric",
        "method": "method",
        "dataset": "dataset",
        "metric": "metric",
    }
    return aliases.get(normalized, normalized if normalized in ENTITY_LABELS else None)


@lru_cache(maxsize=1)
def _get_ner_pipeline():
    """Load the local Scientific NER pipeline, failing clearly if it is not trained."""
    model_dir = config.SCIENTIFIC_NER_MODEL_DIR
    if not model_dir.exists() or not (model_dir / "config.json").exists():
        raise FileNotFoundError(
            f"Scientific NER model not found at {model_dir}. "
            "Train it with: uv run python -m nlp.scientific_ner.train_ner --epochs 3"
        )
    from transformers import pipeline

    return pipeline(
        "token-classification",
        model=str(model_dir),
        tokenizer=str(model_dir),
        aggregation_strategy="simple",
    )


class ScientificNERExtractor:
    """Extract and cache method/dataset/metric entities."""

    def __init__(self, ner_pipeline=None) -> None:
        self.ner_pipeline = ner_pipeline

    @property
    def pipeline(self):
        """Return the injected or local token-classification pipeline."""
        return self.ner_pipeline or _get_ner_pipeline()

    def extract_text(self, text: str) -> list[ScientificEntity]:
        """Extract scientific entities from free text."""
        if not text.strip():
            return []
        raw_entities = self.pipeline(text)
        seen: set[tuple[str, str]] = set()
        entities: list[ScientificEntity] = []
        for raw in raw_entities:
            label = _normalize_label(str(raw.get("entity_group") or raw.get("entity") or ""))
            word = str(raw.get("word") or "").strip()
            if not label or not word:
                continue
            word = word.replace(" ##", "").replace("##", "")
            key = (word.lower(), label)
            if key in seen:
                continue
            seen.add(key)
            entities.append(
                ScientificEntity(
                    text=word,
                    label=label,
                    confidence=round(float(raw.get("score", 0.0)), 4),
                )
            )
        return entities

    def extract_paper(self, paper: Paper) -> dict:
        """Extract entities for one paper as a cache-serializable dict."""
        return {
            "paper_id": paper.paper_id,
            "entities": [asdict(e) for e in self.extract_text(_paper_text(paper))],
        }

    def extract_batch(self, papers: list[Paper], force: bool = False) -> list[dict]:
        """Extract entities for papers, caching to data/cache/scientific_entities."""
        config.SCIENTIFIC_ENTITIES_DIR.mkdir(parents=True, exist_ok=True)
        results: list[dict] = []
        for paper in papers:
            path = _entity_cache_path(paper.paper_id)
            if path.exists() and not force:
                results.append(json.loads(path.read_text(encoding="utf-8")))
                continue
            record = self.extract_paper(paper)
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            results.append(record)
        return results


def _top_papers(n: int) -> list[Paper]:
    """Return top-n papers by citation count from the cached corpus."""
    papers = config.load_papers()
    return sorted(papers, key=lambda p: p.citation_count, reverse=True)[:n]


def main() -> None:
    """CLI entry point for batch entity extraction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    papers = _top_papers(args.top)
    records = ScientificNERExtractor().extract_batch(papers, force=args.force)
    total = sum(len(r["entities"]) for r in records)
    print(f"Extracted {total} entities from {len(records)} papers.")


if __name__ == "__main__":
    main()
