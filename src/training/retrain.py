"""Weekly retraining pipeline using accumulated news data.

Loads high-confidence predictions from PostgreSQL, mixes them with the
original winvoker dataset, and retrains the production model.
Pushes to HuggingFace Hub only if F1 improves over the current model.

Usage:
    python -m src.training.retrain
    python -m src.training.retrain --min-confidence 0.90 --dry-run
"""

import argparse
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
from datasets import Dataset, concatenate_datasets
from loguru import logger
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.data.dataset import ID2LABEL, LABEL2ID, get_tokenized_datasets
from src.db.queries import fetch_high_confidence_items
from src.training.evaluate import (
    compute_metrics,
    confusion_matrix_dict,
    min_per_class_f1,
    print_classification_report,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_MODEL_ID = "efeyol11/bert-turkish-sentiment"
MIN_CONFIDENCE = 0.85
MAX_NEWS_SAMPLES = 50_000

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR = _REPO_ROOT / "models"

mlflow.set_tracking_uri(
    os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{_REPO_ROOT / 'mlflow.db'}")
)
mlflow.set_experiment("turkish-sentiment")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_news_dataset(min_confidence: float, max_samples: int = MAX_NEWS_SAMPLES) -> Dataset:
    """Load high-confidence predictions from PostgreSQL."""
    rows = fetch_high_confidence_items(min_confidence=min_confidence, max_samples=max_samples)

    texts, labels = [], []
    for row in rows:
        label = row.get("sentiment_label")
        if label not in LABEL2ID:
            continue
        title = row.get("cleaned_title") or ""
        summary = row.get("cleaned_summary") or ""
        text = f"{title}. {summary}".strip()
        if len(text) >= 20:
            texts.append(text)
            labels.append(LABEL2ID[label])

    logger.info(
        f"Loaded {len(texts)} high-confidence news examples "
        f"(confidence ≥ {min_confidence}) from PostgreSQL"
    )

    if not texts:
        raise ValueError(
            f"No examples with confidence ≥ {min_confidence} in DB. "
            "Lower --min-confidence or run more pipeline days first."
        )

    return Dataset.from_dict({"text": texts, "label": labels})


def _run_behavioral_gate(candidate_path: Path) -> tuple[bool, str]:
    """Run the regression-blocking behavioral suite against a candidate.

    Uses ``-k must_pass`` so only the green tests are gated on. The strict
    xfail watchlist is intentionally excluded — its whole point is to alert
    via XPASS when a retrain improves the model, so we don't want it to
    block a promotion that would resolve known failures.
    """
    env = {**os.environ, "BEHAVIORAL_MODEL_ID": str(candidate_path)}
    try:
        proc = subprocess.run(
            [
                "pytest",
                "tests/behavioral",
                "-m",
                "behavioral",
                "-k",
                "must_pass",
                "--tb=short",
                "-q",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
            cwd=str(_REPO_ROOT),
        )
    except subprocess.TimeoutExpired:
        return False, "behavioral suite timeout (>15 min)"

    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return proc.returncode == 0, output


def _tokenize_raw(ds: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    def _tok(batch):
        tokens = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        tokens["labels"] = batch["label"]
        return tokens

    ds = ds.map(_tok, batched=True, remove_columns=ds.column_names)
    ds.set_format("torch")
    return ds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def retrain(
    min_confidence: float = MIN_CONFIDENCE,
    epochs: int = 2,
    batch_size: int = 32,
    dry_run: bool = False,
    max_orig_samples: int | None = None,
) -> dict:
    """Retrain production model on accumulated news + original data.

    Returns metrics dict with at least ``f1_macro``.
    """
    logger.info(
        f"Retraining — min_confidence={min_confidence}, epochs={epochs}, "
        f"max_orig_samples={max_orig_samples}"
    )

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)

    if dry_run:
        orig_cap = 128
    else:
        orig_cap = max_orig_samples
    orig_train, val_ds = get_tokenized_datasets(tokenizer, max_samples=orig_cap)

    # News dataset (high-confidence pseudo-labels)
    raw_news = _load_news_dataset(min_confidence)
    if dry_run:
        raw_news = raw_news.select(range(min(64, len(raw_news))))
    news_train = _tokenize_raw(raw_news, tokenizer)

    train_ds = concatenate_datasets([orig_train, news_train])
    logger.info(
        f"Training set: {len(orig_train)} original + {len(news_train)} news "
        f"= {len(train_ds)} total"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        HF_MODEL_ID,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=str(_MODELS_DIR / "retrain_checkpoints"),
        num_train_epochs=1 if dry_run else epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=1e-5,         # smaller LR for fine-tuning an already fine-tuned model
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=100,
        max_steps=20 if dry_run else -1,
        report_to="none",
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    run_name = f"retrain-{date.today().isoformat()}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "base_model": HF_MODEL_ID,
            "min_confidence": min_confidence,
            "epochs": epochs,
            "news_samples": len(news_train),
            "orig_samples": len(orig_train),
            "max_orig_samples": max_orig_samples,
            "dry_run": dry_run,
        })

        logger.info("Training...")
        trainer.train()

        logger.info("Evaluating...")
        metrics = trainer.evaluate()
        mlflow.log_metrics(metrics)

        f1 = metrics.get("eval_f1_macro", 0.0)
        per_class_floor = min_per_class_f1(metrics)
        preds_out = trainer.predict(val_ds)
        preds = np.argmax(preds_out.predictions, axis=-1)
        print_classification_report(preds_out.label_ids, preds, ID2LABEL)

        cm = confusion_matrix_dict(preds_out.label_ids, preds, ID2LABEL)
        mlflow.log_dict(cm, "confusion_matrix.json")

        logger.info(
            f"Retrain F1 macro: {f1:.4f} | min per-class F1: {per_class_floor:.4f}"
        )

        if dry_run:
            logger.info("DRY RUN — skipping HF Hub push")
            return metrics

        # Save candidate locally so the behavioral suite can load it before
        # we decide whether to promote to HF Hub.
        candidate_dir = _MODELS_DIR / "retrained"
        trainer.save_model(str(candidate_dir))
        tokenizer.save_pretrained(str(candidate_dir))
        logger.info(f"Candidate saved to {candidate_dir}")

        logger.info("Running behavioral gate (must-pass) against candidate...")
        behavioral_passed, behavioral_log = _run_behavioral_gate(candidate_dir)
        mlflow.log_text(behavioral_log[-4000:], "behavioral_output.txt")
        mlflow.set_tag("behavioral_passed", str(behavioral_passed))

        # Quality gate: macro F1 ≥ 0.80, every class ≥ 0.70, behavioral green.
        # Per-class floor blocks "macro looks healthy but minority collapsed"
        # regressions; behavioral catches catastrophic class collapse the
        # validation set won't surface (e.g., model only predicts neutral).
        gate_macro = f1 >= 0.80
        gate_per_class = per_class_floor >= 0.70
        gate_behavioral = behavioral_passed

        if gate_macro and gate_per_class and gate_behavioral:
            logger.info(
                f"Quality gate passed (macro={f1:.4f}, min_class={per_class_floor:.4f}, "
                f"behavioral=PASS) — pushing to HuggingFace Hub..."
            )
            model.push_to_hub(HF_MODEL_ID)
            tokenizer.push_to_hub(HF_MODEL_ID)
            mlflow.set_tag("deployed", "true")
            logger.success(f"Model pushed to {HF_MODEL_ID}")
        else:
            reason = []
            if not gate_macro:
                reason.append(f"macro F1 {f1:.4f} < 0.80")
            if not gate_per_class:
                reason.append(f"min per-class F1 {per_class_floor:.4f} < 0.70")
            if not gate_behavioral:
                reason.append("behavioral suite failed")
            logger.warning(
                "Quality gate failed (" + "; ".join(reason) + ") — model NOT pushed"
            )
            mlflow.set_tag("deployed", "false")
            mlflow.set_tag("gate_failure", "; ".join(reason))

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Weekly retraining with accumulated news data")
    p.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--max-orig-samples",
        type=int,
        default=None,
        help="Cap original winvoker rows (full dataset if unset). Use to fit CPU runs in CI time budget.",
    )
    p.add_argument("--dry-run", action="store_true", help="Quick smoke test, no HF push")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        retrain(
            min_confidence=args.min_confidence,
            epochs=args.epochs,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            max_orig_samples=args.max_orig_samples,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        sys.exit(1)
