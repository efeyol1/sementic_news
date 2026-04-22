import argparse
from pathlib import Path

import mlflow
import mlflow.pytorch
from loguru import logger
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.data.dataset import ID2LABEL, LABEL2ID, MODEL_NAME, get_tokenized_datasets
from src.training.evaluate import compute_metrics

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
MLFLOW_DIR = ROOT / "mlruns"
MODELS_DIR.mkdir(exist_ok=True)

mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
mlflow.set_experiment("turkish-sentiment")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-steps", type=int, default=-1, help="Set low for smoke test")
    p.add_argument("--dry-run", action="store_true", help="5-step smoke test")
    return p.parse_args()


def main():
    args = parse_args()

    max_samples = None
    if args.dry_run:
        logger.info("DRY RUN MODE — 5 steps, 128 samples only")
        args.max_steps = 5
        args.epochs = 1
        max_samples = 128

    logger.info(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    train_ds, val_ds = get_tokenized_datasets(tokenizer, max_samples=max_samples)

    training_args = TrainingArguments(
        output_dir=str(MODELS_DIR / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
        max_steps=args.max_steps,
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

    with mlflow.start_run(run_name="bert-turkish-sentiment"):
        mlflow.log_params({
            "model_name": MODEL_NAME,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "max_length": 128,
        })

        logger.info("Training started...")
        trainer.train()

        logger.info("Evaluating...")
        metrics = trainer.evaluate()
        mlflow.log_metrics(metrics)

        logger.info(f"Metrics: {metrics}")

        save_path = MODELS_DIR / "best"
        trainer.save_model(str(save_path))
        tokenizer.save_pretrained(str(save_path))
        mlflow.pytorch.log_model(trainer.model, artifact_path="best")
        logger.success(f"Model saved to {save_path}")

        f1 = metrics.get("eval_f1_macro", 0)
        logger.info(f"Final F1 macro: {f1:.4f}")

        if not args.dry_run and f1 < 0.75:
            raise ValueError(
                f"F1 macro {f1:.4f} below threshold 0.75 — model not saved"
            )


if __name__ == "__main__":
    main()