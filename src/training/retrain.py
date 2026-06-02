"""Weekly retraining pipeline using accumulated news data.

Loads high-confidence predictions from PostgreSQL, mixes them with the
original winvoker dataset, and retrains the production model.
Pushes to HuggingFace Hub only if F1 improves over the current model.

Usage:
    python -m src.training.retrain
    python -m src.training.retrain --min-confidence 0.90 --dry-run
"""

import argparse
import csv
import json
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
# Gold-anchored retrain: the new model must beat the *current* model's macro-F1
# on the held-out human-labeled eval set by at least this margin before we
# consider it an improvement (small enough to catch real gains, large enough to
# ignore ~50-item noise).
HUMAN_GATE_MARGIN = 0.02

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


def _row_text(title: str, summary: str) -> str:
    """Same title+summary shape the model sees in production / the seed CSVs."""
    return f"{(title or '').strip()}. {(summary or '').strip()}".strip()


def _load_gold_dataset(paths: list[Path], exclude_ids: set[str]) -> Dataset:
    """Load Claude-labeled (``reviewed_label``) gold CSVs as a training set.

    These human-grade-but-machine-produced labels replace the self-labeled news
    pseudo-labels that reinforce the model's over-polarization. ``exclude_ids``
    holds out the human-eval rows so the gate set never leaks into training.
    """
    texts: list[str] = []
    labels: list[int] = []
    seen: set[str] = set()
    for path in paths:
        with open(path, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                rid = str(row.get("id") or "")
                if not rid or rid in exclude_ids or rid in seen:
                    continue
                label = (row.get("reviewed_label") or "").strip().lower()
                if label not in LABEL2ID:
                    continue
                text = _row_text(row.get("title", ""), row.get("summary", ""))
                if len(text) < 20:
                    continue
                seen.add(rid)
                texts.append(text)
                labels.append(LABEL2ID[label])
    if not texts:
        raise ValueError(f"No usable gold rows in {[str(p) for p in paths]}")
    logger.info(f"Loaded {len(texts)} gold (Claude-labeled) training examples")
    return Dataset.from_dict({"text": texts, "label": labels})


def _balance_gold(ds: Dataset) -> Dataset:
    """Up-sample each label to the size of the largest gold class.

    The raw gold mirrors the corpus prior (neutral-heavy, positive-sparse —
    here ~46/38/16). Replicating it flat with ``gold_weight`` propagates that
    skew and starves the positive class, which is what dragged positive F1 down
    on the human-eval set. Equalizing per-class counts gives every class the
    same voice before ``gold_weight`` scales the whole block.
    """
    by_class: dict[int, list[int]] = {}
    for idx, label in enumerate(ds["label"]):
        by_class.setdefault(int(label), []).append(idx)
    target = max(len(idxs) for idxs in by_class.values())
    picks: list[int] = []
    for _, idxs in sorted(by_class.items()):
        reps = -(-target // len(idxs))  # ceil: cover target with cyclic repeats
        picks.extend((idxs * reps)[:target])
    before = {ID2LABEL[label]: len(idxs) for label, idxs in sorted(by_class.items())}
    logger.info(f"Balanced gold: {before} → {target} each ({len(picks)} rows total)")
    return ds.select(picks)


def _load_human_eval(json_path: Path, seed_csv: Path) -> tuple[Dataset, float, set[str]]:
    """Held-out human-labeled eval set + the *current* model's baseline macro-F1.

    The human labels are the gate's ground truth; the baseline is the live
    model's predictions (``current_label`` in the seed) scored against them, so
    the gate measures real improvement over what is deployed today.
    """
    from sklearn.metrics import f1_score

    doc = json.loads(Path(json_path).read_text(encoding="utf-8"))
    human = {
        str(it["id"]): it["human_label"].strip().lower()
        for it in doc["items"]
        if (it.get("human_label") or "").strip().lower() in LABEL2ID
    }
    seed = {str(r["id"]): r for r in csv.DictReader(open(seed_csv, encoding="utf-8"))}

    texts: list[str] = []
    gold: list[int] = []
    current: list[int] = []
    for rid, hlabel in human.items():
        row = seed.get(rid)
        if not row:
            continue
        text = _row_text(row.get("title", ""), row.get("summary", ""))
        if len(text) < 20:
            continue
        texts.append(text)
        gold.append(LABEL2ID[hlabel])
        current.append(LABEL2ID.get((row.get("current_label") or "").strip().lower(), 1))

    baseline_f1 = float(f1_score(gold, current, average="macro", labels=[0, 1, 2], zero_division=0))
    logger.info(
        f"Human-eval: {len(texts)} items | current-model baseline macro-F1={baseline_f1:.4f}"
    )
    return Dataset.from_dict({"text": texts, "label": gold}), baseline_f1, set(human)


def _macro_f1_on(trainer, tokenizer, raw_ds: Dataset) -> tuple[float, list[float]]:
    """Predict on a raw text/label dataset and return (macro-F1, per-class F1)."""
    from sklearn.metrics import f1_score

    tok = _tokenize_raw(raw_ds, tokenizer)
    out = trainer.predict(tok)
    preds = np.argmax(out.predictions, axis=-1)
    gold = out.label_ids
    macro = float(f1_score(gold, preds, average="macro", labels=[0, 1, 2], zero_division=0))
    per_class = [
        float(f1_score(gold, preds, average=None, labels=[i], zero_division=0)[0])
        for i in (0, 1, 2)
    ]
    return macro, per_class


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
    gold_paths: list[Path] | None = None,
    human_eval_path: Path | None = None,
    human_eval_seed_csv: Path | None = None,
    gold_weight: int = 3,
    gold_balance: bool = False,
    use_self_news: bool = True,
    push: bool = True,
) -> dict:
    """Retrain production model on accumulated news + original data.

    Gold-anchored mode (``gold_paths`` set): mixes Claude-labeled news gold into
    training (up-weighted ``gold_weight``×) and gates on a held-out *human*-eval
    set instead of the self-friendly winvoker split. Pass ``use_self_news=False``
    to drop the model-pseudo-labeled news that reinforces over-polarization.

    Returns metrics dict with at least ``f1_macro``.
    """
    logger.info(
        f"Retraining — min_confidence={min_confidence}, epochs={epochs}, "
        f"max_orig_samples={max_orig_samples}, gold_paths={bool(gold_paths)}, "
        f"gold_weight={gold_weight}, use_self_news={use_self_news}, push={push}"
    )

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)

    orig_cap = 128 if dry_run else max_orig_samples
    orig_train, val_ds = get_tokenized_datasets(tokenizer, max_samples=orig_cap)

    # Load the held-out human-eval gate first so its ids can be excluded from
    # the gold training set (no leakage).
    human_raw = None
    baseline_human_f1 = 0.0
    human_eval_ids: set[str] = set()
    if human_eval_path and human_eval_seed_csv:
        human_raw, baseline_human_f1, human_eval_ids = _load_human_eval(
            Path(human_eval_path), Path(human_eval_seed_csv)
        )

    parts = [orig_train]
    part_desc = [f"{len(orig_train)} original"]

    if gold_paths:
        gold_raw = _load_gold_dataset([Path(p) for p in gold_paths], exclude_ids=human_eval_ids)
        if gold_balance:
            gold_raw = _balance_gold(gold_raw)
        if dry_run:
            gold_raw = gold_raw.select(range(min(64, len(gold_raw))))
        gold_tok = _tokenize_raw(gold_raw, tokenizer)
        weight = max(1, int(gold_weight))
        parts.extend([gold_tok] * weight)
        part_desc.append(f"{len(gold_tok)}×{weight} gold{' (balanced)' if gold_balance else ''}")

    if use_self_news:
        raw_news = _load_news_dataset(min_confidence)
        if dry_run:
            raw_news = raw_news.select(range(min(64, len(raw_news))))
        news_train = _tokenize_raw(raw_news, tokenizer)
        parts.append(news_train)
        part_desc.append(f"{len(news_train)} self-labeled news")

    train_ds = concatenate_datasets(parts)
    logger.info(f"Training set: {' + '.join(part_desc)} = {len(train_ds)} total")

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
            "orig_samples": len(orig_train),
            "train_total": len(train_ds),
            "gold_mode": bool(gold_paths),
            "gold_weight": gold_weight if gold_paths else 0,
            "gold_balance": gold_balance if gold_paths else False,
            "use_self_news": use_self_news,
            "max_orig_samples": max_orig_samples,
            "dry_run": dry_run,
        })

        logger.info("Training...")
        trainer.train()

        logger.info("Evaluating (winvoker val)...")
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
            f"Retrain F1 macro (winvoker val): {f1:.4f} | min per-class F1: {per_class_floor:.4f}"
        )

        # Held-out human-eval — the meaningful gate for over-polarization. The
        # winvoker split is self-friendly (~1.0) and only catches catastrophic
        # collapse; the human set measures real improvement over the live model.
        human_f1: float | None = None
        gate_human = True
        if human_raw is not None:
            human_f1, human_per_class = _macro_f1_on(trainer, tokenizer, human_raw)
            gate_human = human_f1 >= baseline_human_f1 + HUMAN_GATE_MARGIN
            mlflow.log_metrics({
                "human_eval_macro_f1": human_f1,
                "human_eval_baseline_f1": baseline_human_f1,
                "human_eval_f1_negative": human_per_class[0],
                "human_eval_f1_neutral": human_per_class[1],
                "human_eval_f1_positive": human_per_class[2],
            })
            logger.info(
                f"Human-eval macro-F1: {human_f1:.4f} (baseline {baseline_human_f1:.4f}, "
                f"neg/neu/pos={human_per_class[0]:.2f}/{human_per_class[1]:.2f}/{human_per_class[2]:.2f}) "
                f"— gate {'PASS' if gate_human else 'FAIL'}"
            )

        if dry_run:
            logger.info("DRY RUN — skipping behavioral gate and HF Hub push")
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

        # Quality gate: winvoker macro ≥ 0.80 & per-class ≥ 0.70 (regression net),
        # behavioral green, and — in gold mode — human-eval beats the live model.
        gate_macro = f1 >= 0.80
        gate_per_class = per_class_floor >= 0.70
        gate_behavioral = behavioral_passed
        all_pass = gate_macro and gate_per_class and gate_behavioral and gate_human

        reason = []
        if not gate_macro:
            reason.append(f"winvoker macro F1 {f1:.4f} < 0.80")
        if not gate_per_class:
            reason.append(f"min per-class F1 {per_class_floor:.4f} < 0.70")
        if not gate_behavioral:
            reason.append("behavioral suite failed")
        if not gate_human and human_f1 is not None:
            reason.append(
                f"human-eval F1 {human_f1:.4f} < baseline {baseline_human_f1:.4f}+{HUMAN_GATE_MARGIN}"
            )

        if not push:
            verdict = "WOULD PASS" if all_pass else "WOULD FAIL"
            logger.info(
                f"EVAL-ONLY (push disabled) — gate {verdict}"
                + ("" if all_pass else f": {'; '.join(reason)}")
            )
            mlflow.set_tag("deployed", "false")
            mlflow.set_tag("eval_only", "true")
            mlflow.set_tag("would_pass", str(all_pass))
        elif all_pass:
            logger.info(
                f"Quality gate passed (winvoker macro={f1:.4f}, min_class={per_class_floor:.4f}, "
                f"behavioral=PASS, human_f1={human_f1}) — pushing to HuggingFace Hub..."
            )
            model.push_to_hub(HF_MODEL_ID)
            tokenizer.push_to_hub(HF_MODEL_ID)
            mlflow.set_tag("deployed", "true")
            logger.success(f"Model pushed to {HF_MODEL_ID}")
        else:
            logger.warning("Quality gate failed (" + "; ".join(reason) + ") — model NOT pushed")
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
    p.add_argument(
        "--gold-paths", nargs="*", default=None,
        help="Claude-labeled gold CSV(s) to mix into training (enables gold-anchored mode).",
    )
    p.add_argument("--human-eval", default=None, help="Human-labeled eval JSON (held-out gate).")
    p.add_argument(
        "--human-eval-seed", default=None,
        help="Seed CSV providing title/summary/current_label for the human-eval ids.",
    )
    p.add_argument("--gold-weight", type=int, default=3, help="Up-sample factor for gold rows.")
    p.add_argument(
        "--gold-balance", action="store_true",
        help="Up-sample each gold class to the largest class before weighting "
             "(fixes positive-class starvation from the neutral-heavy gold prior).",
    )
    p.add_argument(
        "--no-self-news", action="store_true",
        help="Drop model-pseudo-labeled news (recommended in gold mode — it reinforces over-polarization).",
    )
    p.add_argument("--no-push", action="store_true", help="Eval-only: run gates but never push to HF Hub.")
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
            gold_paths=args.gold_paths,
            human_eval_path=args.human_eval,
            human_eval_seed_csv=args.human_eval_seed,
            gold_weight=args.gold_weight,
            gold_balance=args.gold_balance,
            use_self_news=not args.no_self_news,
            push=not args.no_push,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        sys.exit(1)
