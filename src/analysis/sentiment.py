"""Multilingual sentiment analysis for news items.

Two modes (controlled by SENTIMENT_MODEL_ID env var):
  - Zero-shot (default): joeddav/xlm-roberta-large-xnli — no fine-tuning needed,
    works for any language via SENTIMENT_LABELS candidate labels.
  - Fine-tuned: set SENTIMENT_MODEL_ID=efeyol11/bert-turkish-sentiment (or local
    path) to use the domain-adapted BERT model for Turkish news.

Reads items from PostgreSQL, writes enriched results back to the same rows.

Usage:
    python -m src.analysis.sentiment               # analyze today's items
    python -m src.analysis.sentiment --date 2026-04-17
    SENTIMENT_MODEL_ID=efeyol11/bert-turkish-sentiment python -m src.analysis.sentiment
"""

import argparse
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import torch
from loguru import logger
from transformers import pipeline

from src.db.queries import bulk_update_sentiment, fetch_processed_by_date

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ZERO_SHOT_MODEL = "joeddav/xlm-roberta-large-xnli"
_FINETUNED_LABEL_MAP = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive",
                        "negative": "negative", "neutral": "neutral", "positive": "positive"}

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
mlflow.set_tracking_uri(f"sqlite:///{_REPO_ROOT / 'mlflow.db'}")
mlflow.set_experiment("news-sentiment")

# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def _is_finetuned_mode() -> bool:
    return bool(os.getenv("SENTIMENT_MODEL_ID"))


def _active_model_id() -> str:
    return os.getenv("SENTIMENT_MODEL_ID") or _ZERO_SHOT_MODEL


def _load_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    model_id = _active_model_id()
    if _is_finetuned_mode():
        logger.info(f"Loading fine-tuned model {model_id!r} on {'CUDA' if device == 0 else 'CPU'}")
        pipe = pipeline("text-classification", model=model_id, device=device, top_k=None)
    else:
        logger.info(f"Loading zero-shot model {model_id!r} on {'CUDA' if device == 0 else 'CPU'}")
        pipe = pipeline("zero-shot-classification", model=model_id, device=device)
    logger.success("Model loaded.")
    return pipe


def _get_labels(lang: str) -> dict[str, str]:
    return SENTIMENT_LABELS.get(lang, SENTIMENT_LABELS["en"])


def _build_text(item: dict[str, Any]) -> str:
    title = item.get("cleaned_title", "") or ""
    summary = item.get("cleaned_summary", "") or ""
    return f"{title}. {summary}".strip()


def _predict(text: str, pipe, lang: str) -> dict[str, Any]:
    if _is_finetuned_mode():
        # Fine-tuned text-classification pipeline returns list of {label, score} dicts
        raw = pipe(text, truncation=True, max_length=128)[0]
        scores: dict[str, float] = {
            _FINETUNED_LABEL_MAP.get(r["label"], r["label"]): round(r["score"], 6)
            for r in raw
        }
        top = max(scores, key=lambda k: scores[k])
        return {
            "sentiment_label": top,
            "sentiment_score": round(scores[top], 6),
            "sentiment_scores": scores,
        }
    else:
        label_map = _get_labels(lang)
        candidate_labels = list(label_map.values())
        result = pipe(text, candidate_labels, multi_label=False)
        reverse_map = {v: k for k, v in label_map.items()}
        scores = {
            reverse_map[lbl]: round(score, 6)
            for lbl, score in zip(result["labels"], result["scores"])
        }
        top_canonical = reverse_map[result["labels"][0]]
        return {
            "sentiment_label": top_canonical,
            "sentiment_score": round(result["scores"][0], 6),
            "sentiment_scores": scores,
        }


# ---------------------------------------------------------------------------
# Per-item processing
# ---------------------------------------------------------------------------


def _to_analyzed(item: dict[str, Any], pipe, lang: str) -> dict[str, Any]:
    result = {"id": item["id"]}
    if not item.get("is_turkish", True):
        result.update({
            "sentiment_label": None,
            "sentiment_score": None,
            "sentiment_scores": None,
            "analyzed_at": None,
        })
        return result
    text = _build_text(item)
    prediction = _predict(text, pipe, lang)
    result.update(prediction)
    result["analyzed_at"] = datetime.now(timezone.utc).isoformat()
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze(date_str: str | None = None, lang: str = DEFAULT_LANG) -> int:
    """Run sentiment analysis for a single day's items.

    Returns:
        Number of Turkish items analyzed.
    """
    date_str = date_str or date.today().isoformat()

    if lang not in SENTIMENT_LABELS:
        logger.warning(f"Lang '{lang}' not in SENTIMENT_LABELS, falling back to English")
        lang = "en"

    items = fetch_processed_by_date(date_str)
    if not items:
        logger.warning(f"No preprocessed items found for {date_str}")
        return 0

    pipe = _load_pipeline()

    t0 = time.perf_counter()
    updates: list[dict[str, Any]] = [_to_analyzed(item, pipe, lang) for item in items]
    duration = time.perf_counter() - t0

    bulk_update_sentiment(updates)

    turkish_updates = [u for u in updates if u.get("sentiment_label") is not None]
    label_counts: dict[str, int] = {"negative": 0, "neutral": 0, "positive": 0}
    scores_sum = 0.0
    for u in turkish_updates:
        lbl = u["sentiment_label"]
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
        scores_sum += u["sentiment_score"]

    n_tr = len(turkish_updates)
    avg_score = scores_sum / n_tr if n_tr else 0.0

    logger.info(
        f"Sentiment done — {n_tr} items [{lang}]: "
        f"neg={label_counts['negative']}, neu={label_counts['neutral']}, "
        f"pos={label_counts['positive']} in {duration:.1f}s"
    )

    with mlflow.start_run(run_name=f"sentiment-{date_str}"):
        mlflow.log_params({
            "model_id": _active_model_id(),
            "finetuned": _is_finetuned_mode(),
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

    return n_tr


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multilingual zero-shot sentiment analysis on news data."
    )
    parser.add_argument("--date", default=date.today().isoformat(), metavar="YYYY-MM-DD")
    parser.add_argument(
        "--lang",
        default=DEFAULT_LANG,
        metavar="LANG",
        help=f"Language code: {list(SENTIMENT_LABELS.keys())} (default: {DEFAULT_LANG})",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    analyze(date_str=args.date, lang=args.lang)
    sys.exit(0)
