"""Sentiment analysis for preprocessed Turkish news items.

Reads data/processed/YYYY-MM-DD.json, runs inference via the
savasy/bert-base-turkish-sentiment model, and writes enriched results to
data/analyzed/YYYY-MM-DD.json.

Usage:
    python -m src.analysis.sentiment               # analyze today's file
    python -m src.analysis.sentiment --date 2026-04-17
"""

import argparse
import json
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import torch
from loguru import logger
from transformers import pipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_MODEL_ID = "savasy/bert-base-turkish-sentiment-cased"

# Normalize the model's raw label strings to canonical form
_LABEL_MAP: dict[str, str] = {
    "negative": "negative",
    "negatif": "negative",
    "notr": "neutral",
    "nötr": "neutral",
    "neutral": "neutral",
    "positive": "positive",
    "pozitif": "positive",
}

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
_DATA_ANALYZED_DIR = _REPO_ROOT / "data" / "analyzed"

mlflow.set_tracking_uri(f"sqlite:///{_REPO_ROOT / 'mlflow.db'}")
mlflow.set_experiment("news-sentiment")

# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def _load_pipeline():
    """Load the HuggingFace sentiment pipeline (CPU or CUDA automatically)."""
    device = 0 if torch.cuda.is_available() else -1
    logger.info(f"Loading model {HF_MODEL_ID!r} on {'CUDA' if device == 0 else 'CPU'}")
    pipe = pipeline(
        "text-classification",
        model=HF_MODEL_ID,
        top_k=None,
        device=device,
        truncation=True,
        max_length=512,
    )
    logger.success("Model loaded.")
    return pipe


def _build_text(item: dict[str, Any]) -> str:
    """Combine cleaned title and summary into a single inference input."""
    title = item.get("cleaned_title", "") or ""
    summary = item.get("cleaned_summary", "") or ""
    return f"{title}. {summary}".strip()


def _predict(text: str, pipe) -> dict[str, Any]:
    """Run inference and return normalized scores dict plus top label."""
    raw: list[dict] = pipe(text)[0]  # top_k=None → list of {label, score}
    scores: dict[str, float] = {}
    for entry in raw:
        canonical = _LABEL_MAP.get(entry["label"].lower(), entry["label"].lower())
        scores[canonical] = round(entry["score"], 6)

    top_label = max(scores, key=lambda k: scores[k])
    return {
        "sentiment_label": top_label,
        "sentiment_score": scores[top_label],
        "sentiment_scores": scores,
    }


# ---------------------------------------------------------------------------
# Per-item processing
# ---------------------------------------------------------------------------


def _to_analyzed(item: dict[str, Any], pipe) -> dict[str, Any]:
    """Add sentiment fields to a single preprocessed item."""
    result = dict(item)
    if not item.get("is_turkish", True):
        result["sentiment_label"] = None
        result["sentiment_score"] = None
        result["sentiment_scores"] = None
        result["analyzed_at"] = None
        return result

    text = _build_text(item)
    prediction = _predict(text, pipe)
    result.update(prediction)
    result["analyzed_at"] = datetime.now(timezone.utc).isoformat()
    return result


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load_processed(date_str: str, processed_dir: Path) -> list[dict[str, Any]]:
    """Load preprocessed items for *date_str*."""
    path = processed_dir / f"{date_str}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed file not found: {path}\n"
            f"Run `python -m src.data.preprocessor --date {date_str}` first."
        )
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    logger.info(f"Loaded {len(data)} processed items from {path.name}")
    return data


def _save_analyzed(
    items: list[dict[str, Any]],
    date_str: str,
    analyzed_dir: Path,
) -> Path:
    """Write analyzed items to *analyzed_dir*/YYYY-MM-DD.json."""
    analyzed_dir.mkdir(parents=True, exist_ok=True)
    path = analyzed_dir / f"{date_str}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(items, fh, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(items)} analyzed items → {path}")
    return path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze(
    date_str: str | None = None,
    processed_dir: Path | None = None,
    analyzed_dir: Path | None = None,
) -> Path:
    """Run the full sentiment analysis pipeline for a single day's file.

    Args:
        date_str: ISO date string (``"YYYY-MM-DD"``).  Defaults to today.
        processed_dir: Source directory.  Defaults to ``data/processed/``.
        analyzed_dir: Destination directory.  Defaults to ``data/analyzed/``.

    Returns:
        Path to the written analyzed JSON file.
    """
    date_str = date_str or date.today().isoformat()
    processed_dir = processed_dir or _DATA_PROCESSED_DIR
    analyzed_dir = analyzed_dir or _DATA_ANALYZED_DIR

    items = _load_processed(date_str, processed_dir)
    pipe = _load_pipeline()

    t0 = time.perf_counter()
    analyzed: list[dict[str, Any]] = []
    for item in items:
        analyzed.append(_to_analyzed(item, pipe))
    duration = time.perf_counter() - t0

    turkish_items = [a for a in analyzed if a.get("sentiment_label") is not None]
    label_counts: dict[str, int] = {"negative": 0, "neutral": 0, "positive": 0}
    scores_sum = 0.0
    for a in turkish_items:
        label_counts[a["sentiment_label"]] = label_counts.get(a["sentiment_label"], 0) + 1
        scores_sum += a["sentiment_score"]

    n_tr = len(turkish_items)
    avg_score = scores_sum / n_tr if n_tr else 0.0

    logger.info(
        f"Sentiment done — {n_tr} Turkish items: "
        f"neg={label_counts['negative']}, neu={label_counts['neutral']}, "
        f"pos={label_counts['positive']} in {duration:.1f}s"
    )

    out_path = _save_analyzed(analyzed, date_str, analyzed_dir)

    with mlflow.start_run(run_name=f"sentiment-{date_str}"):
        mlflow.log_params({
            "model_id": HF_MODEL_ID,
            "date": date_str,
            "total_items": len(items),
            "turkish_items": n_tr,
        })
        mlflow.log_metrics({
            "negative_pct": label_counts["negative"] / n_tr if n_tr else 0.0,
            "neutral_pct": label_counts["neutral"] / n_tr if n_tr else 0.0,
            "positive_pct": label_counts["positive"] / n_tr if n_tr else 0.0,
            "avg_confidence": avg_score,
            "duration_sec": duration,
        })

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sentiment analysis on preprocessed Turkish news data."
    )
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        metavar="YYYY-MM-DD",
        help="Date of the processed file to analyze (default: today)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    try:
        analyze(date_str=args.date)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)
