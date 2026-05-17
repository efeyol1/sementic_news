"""Fine-tune a multilingual sentiment classifier from reviewed JSONL data.

This script is intentionally small and local-first. It trains on reviewed
examples produced by `build_sentiment_gold_dataset.py`, writes a HuggingFace
model directory, and stores metrics next to the model.

Usage:
    python scripts/train_reviewed_sentiment.py \
        --train-jsonl /private/tmp/sentiment_gold_de.jsonl \
        --base-model xlm-roberta-base \
        --output-dir /private/tmp/news_sentiment_multilingual_de
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset
from loguru import logger
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.data.dataset import ID2LABEL, LABEL2ID
from src.training.evaluate import compute_metrics, confusion_matrix_dict


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("label") in LABEL2ID and row.get("text"):
                rows.append(row)
    if not rows:
        raise ValueError(f"No usable rows found in {path}")
    return rows


def _split_rows(
    rows: list[dict[str, Any]],
    eval_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rng = random.Random(seed)
    for row in rows:
        grouped[row["label"]].append(row)

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for label_rows in grouped.values():
        rng.shuffle(label_rows)
        if len(label_rows) <= 1:
            train_rows.extend(label_rows)
            continue
        n_eval = max(1, round(len(label_rows) * eval_ratio))
        n_eval = min(n_eval, len(label_rows) - 1)
        eval_rows.extend(label_rows[:n_eval])
        train_rows.extend(label_rows[n_eval:])

    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)
    if not eval_rows:
        raise ValueError("Need at least two reviewed examples in one label to build an eval split")
    return train_rows, eval_rows


def _split_rows_by_date(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dated = [row for row in rows if row.get("collected_date")]
    dates = sorted({row["collected_date"] for row in dated})
    if len(dates) < 2:
        raise ValueError("Date split requires at least two distinct collected_date values")
    eval_date = dates[-1]
    train_rows = [row for row in rows if row.get("collected_date") != eval_date]
    eval_rows = [row for row in rows if row.get("collected_date") == eval_date]
    if not train_rows or not eval_rows:
        raise ValueError("Date split produced an empty train or eval set")
    return train_rows, eval_rows


def _balance_rows(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rng = random.Random(seed)
    for row in rows:
        grouped[row["label"]].append(row)
    max_count = max(len(label_rows) for label_rows in grouped.values())
    balanced: list[dict[str, Any]] = []
    for label_rows in grouped.values():
        balanced.extend(label_rows)
        needed = max_count - len(label_rows)
        if needed > 0:
            balanced.extend(rng.choice(label_rows) for _ in range(needed))
    rng.shuffle(balanced)
    return balanced


def _to_dataset(rows: list[dict[str, Any]]) -> Dataset:
    return Dataset.from_dict(
        {
            "text": [row["text"] for row in rows],
            "labels": [LABEL2ID[row["label"]] for row in rows],
        }
    )


def _tokenize(ds: Dataset, tokenizer, max_length: int) -> Dataset:
    def _tok(batch):
        tokens = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        tokens["labels"] = batch["labels"]
        return tokens

    tokenized = ds.map(_tok, batched=True, remove_columns=ds.column_names)
    tokenized.set_format("torch")
    return tokenized


def train(args: argparse.Namespace) -> dict[str, Any]:
    rows = _read_jsonl(args.train_jsonl)
    if args.split_strategy == "date":
        train_rows, eval_rows = _split_rows_by_date(rows)
    else:
        train_rows, eval_rows = _split_rows(rows, args.eval_ratio, args.seed)
    if args.balance_train:
        train_rows = _balance_rows(train_rows, args.seed)

    logger.info(
        "Dataset rows — total={}, train={}, eval={}, train_labels={}, eval_labels={}",
        len(rows),
        len(train_rows),
        len(eval_rows),
        dict(Counter(row["label"] for row in train_rows)),
        dict(Counter(row["label"] for row in eval_rows)),
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    train_ds = _tokenize(_to_dataset(train_rows), tokenizer, args.max_length)
    eval_ds = _tokenize(_to_dataset(eval_rows), tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=5,
        max_steps=args.max_steps,
        report_to="none",
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    preds_out = trainer.predict(eval_ds)
    preds = np.argmax(preds_out.predictions, axis=-1)
    cm = confusion_matrix_dict(preds_out.label_ids, preds, ID2LABEL)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    payload = {
        "base_model": args.base_model,
        "train_jsonl": str(args.train_jsonl),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_label_counts": dict(Counter(row["label"] for row in train_rows)),
        "eval_label_counts": dict(Counter(row["label"] for row in eval_rows)),
        "split_strategy": args.split_strategy,
        "metrics": metrics,
        "confusion": cm,
    }
    metrics_path = args.output_dir / "review_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.success("Saved model to {} and metrics to {}", args.output_dir, metrics_path)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a reviewed multilingual sentiment classifier."
    )
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--base-model", default="xlm-roberta-base")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--eval-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--balance-train", action="store_true")
    parser.add_argument(
        "--split-strategy",
        choices=("stratified", "date"),
        default="stratified",
        help="Use label-stratified random split or latest collected_date as eval.",
    )
    return parser.parse_args()


def main() -> int:
    payload = train(_parse_args())
    metrics = payload["metrics"]
    print(
        "eval_accuracy={:.4f} eval_f1_macro={:.4f} eval_recall_neutral={:.4f}".format(
            metrics.get("eval_accuracy", 0.0),
            metrics.get("eval_f1_macro", 0.0),
            metrics.get("eval_recall_neutral", 0.0),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
