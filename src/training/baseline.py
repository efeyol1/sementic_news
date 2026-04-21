"""Baseline evaluation for savasy/bert-base-turkish-sentiment-cased.

Evaluates the pre-trained model (zero fine-tuning) on the test split of
winvoker/turkish-sentiment-analysis-dataset and logs metrics to MLflow.

Usage:
    python -m src.training.baseline
    python -m src.training.baseline --max-samples 500
"""

import argparse
from pathlib import Path

import mlflow
import numpy as np
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from src.data.dataset import LABEL2ID, MODEL_NAME, get_tokenized_datasets
from src.training.evaluate import compute_metrics, print_classification_report
from src.data.dataset import ID2LABEL

ROOT = Path(__file__).resolve().parents[2]
mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
mlflow.set_experiment("turkish-sentiment")


def run_baseline(max_samples: int | None = None) -> dict:
    logger.info(f"Loading pre-trained model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    _, val_ds = get_tokenized_datasets(tokenizer, max_samples=max_samples)

    eval_args = TrainingArguments(
        output_dir=str(ROOT / "models" / "baseline_eval"),
        per_device_eval_batch_size=32,
        report_to="none",
        use_cpu=True,
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    logger.info(f"Evaluating baseline on {len(val_ds)} samples...")
    metrics = trainer.evaluate()

    preds_out = trainer.predict(val_ds)
    preds = np.argmax(preds_out.predictions, axis=-1)
    labels = preds_out.label_ids
    print_classification_report(labels, preds, ID2LABEL)

    with mlflow.start_run(run_name="baseline-pretrained"):
        mlflow.log_params({
            "model_name": MODEL_NAME,
            "fine_tuned": False,
            "eval_samples": len(val_ds),
        })
        clean = {k.replace("eval_", ""): v for k, v in metrics.items() if not k.startswith("eval_runtime")}
        mlflow.log_metrics(clean)
        mlflow.set_tag("type", "baseline")

    logger.success(
        f"Baseline — accuracy={clean.get('accuracy', 0):.4f} "
        f"f1_macro={clean.get('f1_macro', 0):.4f}"
    )
    return clean


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate baseline model without fine-tuning")
    p.add_argument("--max-samples", type=int, default=None, help="Limit eval samples (smoke test)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_baseline(max_samples=args.max_samples)
