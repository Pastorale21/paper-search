"""Weakly supervise a SciBERT token-classification model from cached method cards."""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from schemas import MethodCard

from .. import config

MODEL_NAME = "allenai/scibert_scivocab_uncased"
ID_TO_LABEL = {
    0: "O",
    1: "B-method",
    2: "I-method",
    3: "B-dataset",
    4: "I-dataset",
    5: "B-metric",
    6: "I-metric",
}
LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}


def _load_cards() -> dict[str, MethodCard]:
    """Load cached method cards keyed by paper_id."""
    cards: dict[str, MethodCard] = {}
    for path in sorted(config.METHOD_CARDS_DIR.glob("*.json")):
        card = MethodCard.from_dict(json.loads(path.read_text(encoding="utf-8")))
        cards[card.paper_id] = card
    return cards


def _terms(card: MethodCard) -> list[tuple[str, str]]:
    """Build weak supervision terms from method-card fields."""
    out: list[tuple[str, str]] = []
    for value in (card.backbone,):
        if value:
            out.append((value, "method"))
    for value in card.datasets:
        out.append((value, "dataset"))
    for value in card.metrics:
        out.append((value, "metric"))
    return out


def _find_spans(text: str, terms: list[tuple[str, str]]) -> list[dict[str, int | str]]:
    """Find case-insensitive term spans in text."""
    lowered = text.lower()
    spans: list[dict[str, int | str]] = []
    for term, label in terms:
        needle = term.strip().lower()
        if not needle:
            continue
        start = lowered.find(needle)
        if start != -1:
            spans.append({"start": start, "end": start + len(needle), "label": label})
    return spans


def _examples(max_examples: int | None = None) -> list[dict[str, Any]]:
    """Create weak NER examples from cached papers and method cards."""
    cards = _load_cards()
    examples: list[dict[str, Any]] = []
    for paper in config.load_papers():
        card = cards.get(paper.paper_id)
        if card is None:
            continue
        text = f"{paper.title}\n\n{paper.abstract or ''}".strip()
        spans = _find_spans(text, _terms(card))
        if spans:
            examples.append({"text": text, "spans": spans})
        if max_examples and len(examples) >= max_examples:
            break
    return examples


def _tokenize_and_align(example, tokenizer, max_length: int) -> dict:
    """Tokenize one weak NER example and align char spans to token labels."""
    encoded = tokenizer(
        example["text"],
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    labels = [LABEL_TO_ID["O"]] * len(encoded["input_ids"])
    for span in example["spans"]:
        start = int(span["start"])
        end = int(span["end"])
        entity_label = str(span["label"])
        first = True
        for idx, (tok_start, tok_end) in enumerate(encoded["offset_mapping"]):
            if tok_start == tok_end:
                labels[idx] = -100
                continue
            if tok_end <= start or tok_start >= end:
                continue
            prefix = "B" if first else "I"
            labels[idx] = LABEL_TO_ID[f"{prefix}-{entity_label}"]
            first = False
    encoded.pop("offset_mapping")
    encoded["labels"] = labels
    return encoded


def _token_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    """Compute token accuracy over non-masked tokens."""
    mask = labels != -100
    if not mask.any():
        return 0.0
    return float((preds[mask] == labels[mask]).mean())


def main() -> None:
    """CLI entry point for weakly supervised Scientific NER training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-examples", type=int)
    args = parser.parse_args()

    from datasets import Dataset
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )

    examples = _examples(max_examples=args.max_examples)
    if not examples:
        raise RuntimeError(
            "No weak NER examples found. Run method-card extraction before training NER."
        )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = Dataset.from_list(examples)
    tokenized = dataset.map(
        lambda ex: _tokenize_and_align(ex, tokenizer, args.max_length),
        remove_columns=dataset.column_names,
    )
    split = tokenized.train_test_split(test_size=0.2, seed=13) if len(tokenized) > 5 else None
    train_ds = split["train"] if split else tokenized
    eval_ds = split["test"] if split else tokenized

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(ID_TO_LABEL),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"token_accuracy": _token_accuracy(preds, labels)}

    output_dir = config.SCIENTIFIC_NER_MODEL_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=25,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved Scientific NER model to {output_dir}")


if __name__ == "__main__":
    main()
