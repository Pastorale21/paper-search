"""Fine-tune SciBERT on SciCite for citation-intent classification."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from .. import config
from .classifier import map_scicite_label

MODEL_NAME = "allenai/scibert_scivocab_uncased"
DATASET_NAME = "allenai/scicite"
ID_TO_LABEL = {0: "background", 1: "method", 2: "comparison"}
LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}


def _context_text(example: dict[str, Any]) -> str:
    """Return the citation context text from a SciCite example."""
    return str(example.get("string") or example.get("text") or example.get("context") or "")


def _raw_label(example: dict[str, Any], dataset) -> str:
    """Return the raw SciCite label as text, handling ClassLabel datasets."""
    label = example["label"]
    feature = dataset.features.get("label")
    if isinstance(label, int) and hasattr(feature, "int2str"):
        return str(feature.int2str(label))
    return str(label)


def _tokenize_split(split, tokenizer, max_length: int):
    """Tokenize a SciCite split and map labels into the project label space."""

    def convert(example):
        label = map_scicite_label(_raw_label(example, split))
        encoded = tokenizer(_context_text(example), truncation=True, max_length=max_length)
        encoded["labels"] = LABEL_TO_ID[label]
        return encoded

    return split.map(convert, remove_columns=split.column_names)


def _macro_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Compute macro-F1 without pulling in scikit-learn."""
    f1s: list[float] = []
    for label_id in ID_TO_LABEL:
        tp = int(((preds == label_id) & (labels == label_id)).sum())
        fp = int(((preds == label_id) & (labels != label_id)).sum())
        fn = int(((preds != label_id) & (labels == label_id)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(sum(f1s) / len(f1s))


def main() -> None:
    """CLI entry point for training and saving the local SciCite model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--max-train", type=int, help="optional small training subset")
    parser.add_argument("--max-validation", type=int, help="optional small validation subset")
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    raw = load_dataset(DATASET_NAME, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_split = raw["train"]
    validation_split = raw["validation"] if "validation" in raw else raw["test"]
    if args.max_train:
        train_split = train_split.select(range(min(args.max_train, len(train_split))))
    if args.max_validation:
        validation_split = validation_split.select(
            range(min(args.max_validation, len(validation_split)))
        )

    train_ds = _tokenize_split(train_split, tokenizer, args.max_length)
    validation_ds = _tokenize_split(validation_split, tokenizer, args.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(ID_TO_LABEL),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        accuracy = float((preds == labels).mean())
        return {"accuracy": accuracy, "macro_f1": _macro_f1(preds, labels)}

    output_dir = config.SCICITE_MODEL_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=50,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=validation_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved SciCite model to {output_dir}")


if __name__ == "__main__":
    main()
