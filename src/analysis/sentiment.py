"""Multilingual sentiment analysis for news items.

Uses zero-shot classification (MoritzLaurer/mDeBERTa-v3-base-zeroshot-v1)
so the same model works for any language without fine-tuning.
Labels are defined per language — extend SENTIMENT_LABELS to add new countries.

Reads data/processed/YYYY-MM-DD.json, writes enriched results to
data/analyzed/YYYY-MM-DD.json.

Usage:
    python -m src.analysis.sentiment               # analyze today's file
    python -m src.analysis.sentiment --date 2026-04-17
    python -m src.analysis.sentiment --lang de     # German news
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

HF_MODEL_ID = "joeddav/xlm-roberta-large-xnli"

# Extend this dict to support new countries/languages.
# Keys must be ISO 639-1 language codes.
SENTIMENT_LABELS: dict[str, dict[str, str]] = {
    "tr": {
        "positive": "olumlu haber",
        "negative": "olumsuz haber",
        "neutral":  "tarafsız haber",
    },
    "de": {
        "positive": "positive Nachricht",
        "negative": "negative Nachricht",
        "neutral":  "neutrale Nachricht",
    },
    "fr": {
        "positive": "nouvelle positive",
        "negative": "nouvelle négative",
        "neutral":  "nouvelle neutre",
    },
    "es": {
        "positive": "noticia positiva",
        "negative": "noticia negativa",
        "neutral":  "noticia neutra",
    },
    "en": {
        "positive": "positive news",
        "negative": "negative news",
        "neutral":  "neutral news",
    },
}

DEFAULT_LANG = "tr"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
_DATA_ANALYZED_DIR = _REPO_ROOT / "data" / "analyzed"

mlflow.set_tracking_uri(f"sqlite:///{_REPO_ROOT / 'mlflow.db'}")
mlflow.set_experiment("news-sentiment")

# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def _load_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    logger.info(f"Loading model {HF_MODEL_ID!r} on {'CUDA' if device == 0 else 'CPU/MPS'}")
    pipe = pipeline(
        "zero-shot-classification",
        model=HF_MODEL_ID,
        device=device,
    )
    logger.success("Model loaded.")
    return pipe


def _get_labels(lang: str) -> dict[str, str]:
    """Return {canonical: label_text} for the given language, fallback to English."""
    return SENTIMENT_LABELS.get(lang, SENTIMENT_LABELS["en"])


def _build_text(item: dict[str, Any]) -> str:
    title = item.get("cleaned_title", "") or ""
    summary = item.get("cleaned_summary", "") or ""
    return f"{title}. {summary}".strip()


def _predict(text: str, pipe, lang: str) -> dict[str, Any]:
    """Run zero-shot inference and return normalized scores dict plus top label."""
    label_map = _get_labels(lang)
    candidate_labels = list(label_map.values())

    result = pipe(text, candidate_labels, multi_label=False)

    # Map label text back to canonical name
    reverse_map = {v: k for k, v in label_map.items()}
    scores: dict[str, float] = {}
    for lbl, score in zip(result["labels"], result["scores"]):
        canonical = reverse_map[lbl]
        scores[canonical] = round(score, 6)

    top_label = result["labels"][0]  # zero-shot returns sorted by score
    top_canonical = reverse_map[top_label]

    return {
        "sentiment_label": top_canonical,
        "sentiment_score": round(result["scores"][0], 6),
        "sentiment_scores": scores,
    }


# ---------------------------------------------------------------------------
# Per-item processing
# ---------------------------------------------------------------------------


def _to_analyzed(item: dict[str, Any], pipe, lang: str) -> dict[str, Any]:
    result = dict(item)
    if not item.get("is_turkish", True):
        result["sentiment_label"] = None
        result["sentiment_score"] = None
        result["sentiment_scores"] = None
        result["analyzed_at"] = None
        return result

    text = _build_text(item)
    prediction = _predict(text, pipe, lang)
    result.update(prediction)
    result["analyzed_at"] = datetime.now(timezone.utc).isoformat()
    return result


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load_processed(date_str: str, processed_dir: Path) -> list[dict[str, Any]]:
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
    lang: str = DEFAULT_LANG,
    processed_dir: Path | None = None,
    analyzed_dir: Path | None = None,
) -> Path:
    """Run the full sentiment analysis pipeline for a single day's file.

    Args:
        date_str:      ISO date string (``"YYYY-MM-DD"``).  Defaults to today.
        lang:          ISO 639-1 language code (``"tr"``, ``"de"``, etc.).
        processed_dir: Source directory.  Defaults to ``data/processed/``.
        analyzed_dir:  Destination directory.  Defaults to ``data/analyzed/``.
    """
    date_str = date_str or date.today().isoformat()
    processed_dir = processed_dir or _DATA_PROCESSED_DIR
    analyzed_dir = analyzed_dir or _DATA_ANALYZED_DIR

    if lang not in SENTIMENT_LABELS:
        logger.warning(f"Lang '{lang}' not in SENTIMENT_LABELS, falling back to English")
        lang = "en"

    items = _load_processed(date_str, processed_dir)
    pipe = _load_pipeline()

    t0 = time.perf_counter()
    analyzed: list[dict[str, Any]] = []
    for item in items:
        analyzed.append(_to_analyzed(item, pipe, lang))
    duration = time.perf_counter() - t0

    turkish_items = [a for a in analyzed if a.get("sentiment_label") is not None]
    label_counts: dict[str, int] = {"negative": 0, "neutral": 0, "positive": 0}
    scores_sum = 0.0
    for a in turkish_items:
        lbl = a["sentiment_label"]
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
        scores_sum += a["sentiment_score"]

    n_tr = len(turkish_items)
    avg_score = scores_sum / n_tr if n_tr else 0.0

    logger.info(
        f"Sentiment done — {n_tr} items [{lang}]: "
        f"neg={label_counts['negative']}, neu={label_counts['neutral']}, "
        f"pos={label_counts['positive']} in {duration:.1f}s"
    )

    out_path = _save_analyzed(analyzed, date_str, analyzed_dir)

    with mlflow.start_run(run_name=f"sentiment-{date_str}"):
        mlflow.log_params({
            "model_id": HF_MODEL_ID,
            "lang": lang,
            "date": date_str,
            "total_items": len(items),
            "analyzed_items": n_tr,
        })
        mlflow.log_metrics({
            "negative_pct": label_counts["negative"] / n_tr if n_tr else 0.0,
            "neutral_pct":  label_counts["neutral"] / n_tr if n_tr else 0.0,
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
        description="Run multilingual zero-shot sentiment analysis on news data."
    )
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        metavar="YYYY-MM-DD",
        help="Date of the processed file to analyze (default: today)",
    )
    parser.add_argument(
        "--lang",
        default=DEFAULT_LANG,
        metavar="LANG",
        help=f"Language code: {list(SENTIMENT_LABELS.keys())} (default: {DEFAULT_LANG})",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    try:
        analyze(date_str=args.date, lang=args.lang)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)
