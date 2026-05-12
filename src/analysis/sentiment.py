"""Multilingual sentiment analysis for news items.

Three backends, picked via env vars:
  - **Zero-shot** (default): joeddav/xlm-roberta-large-xnli — no fine-tuning,
    multilingual via SENTIMENT_LABELS candidate labels. Slowest.
  - **PyTorch fine-tuned**: ``SENTIMENT_MODEL_ID=efeyol11/bert-turkish-sentiment``
    uses the domain-adapted BERT model directly via transformers.
  - **ONNX int8** (production): ``SENTIMENT_BACKEND=onnx_int8`` loads the
    quantized ONNX build from ``$SENTIMENT_ONNX_DIR`` (default
    ``models/onnx_int8/``). ~3.8× faster than PyTorch on CPU; cuts the
    daily-pipeline sentiment step from ~16 min to ~4 min and shrinks the
    Neon SSL idle window. The ONNX build is created by
    ``python -m src.serving.onnx_export`` — daily_pipeline.yml runs that
    before the pipeline so the artifact is fresh on every run.

Reads items from PostgreSQL, writes enriched results back to the same rows.

Usage:
    python -m src.analysis.sentiment               # analyze today's items
    python -m src.analysis.sentiment --date 2026-04-17
    python -m src.analysis.sentiment --date 2026-05-06 --only-stale-after-body --limit 100
    python -m src.analysis.sentiment --date 2026-05-06 --only-with-body --limit 100
    SENTIMENT_MODEL_ID=efeyol11/bert-turkish-sentiment python -m src.analysis.sentiment
    SENTIMENT_BACKEND=onnx_int8 python -m src.analysis.sentiment
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

from src.analysis.text_inputs import build_sentiment_text
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


_DEFAULT_ONNX_DIR = _REPO_ROOT / "models" / "onnx_int8"
_DEFAULT_FINETUNED_MODEL_ID = "efeyol11/bert-turkish-sentiment"


def _is_onnx_mode() -> bool:
    return os.getenv("SENTIMENT_BACKEND", "").lower() == "onnx_int8"


def _is_finetuned_mode() -> bool:
    """True for any text-classification path (PyTorch fine-tuned OR ONNX).
    Both share the same label map and prediction post-processing — only the
    underlying executor differs."""
    return bool(os.getenv("SENTIMENT_MODEL_ID")) or _is_onnx_mode()


def _active_model_id() -> str:
    """Identifier logged to MLflow. ONNX mode prefixes the source repo so a
    backend switch is visible in the experiment history."""
    if _is_onnx_mode():
        source = os.getenv("SENTIMENT_MODEL_ID") or _DEFAULT_FINETUNED_MODEL_ID
        return f"onnx_int8:{source}"
    return os.getenv("SENTIMENT_MODEL_ID") or _ZERO_SHOT_MODEL


def _onnx_dir() -> Path:
    return Path(os.getenv("SENTIMENT_ONNX_DIR") or _DEFAULT_ONNX_DIR)


def _load_onnx_pipeline():
    """Build a HF text-classification pipeline backed by ONNX Runtime.

    Uses optimum's ORT model wrapper so the resulting pipe has the same
    interface as the PyTorch fine-tuned path — ``_predict`` doesn't need
    to know which backend produced the scores.
    """
    onnx_dir = _onnx_dir()
    if not onnx_dir.exists():
        raise FileNotFoundError(
            f"ONNX dir not found: {onnx_dir}. Run "
            "'python -m src.serving.onnx_export' first, or unset "
            "SENTIMENT_BACKEND to fall back to PyTorch."
        )

    # Lazy import: optimum lives in the [serving] extra. Importing here
    # keeps zero-shot / PyTorch users on a lighter dependency footprint.
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    logger.info(f"Loading ONNX int8 model from {onnx_dir}")
    model = ORTModelForSequenceClassification.from_pretrained(str(onnx_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir))
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None)
    logger.success("ONNX int8 model loaded.")
    return pipe


def _load_pipeline():
    if _is_onnx_mode():
        return _load_onnx_pipeline()
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
    return build_sentiment_text(item)


def _predict(text: str, pipe, lang: str) -> dict[str, Any]:
    if _is_finetuned_mode():
        # Fine-tuned text-classification pipeline returns list of {label, score} dicts
        # 256 captures title + summary + 800-char body (~230-300 Turkish tokens
        # for typical news items) while keeping CPU attention cost ~2× the
        # title-only baseline rather than the ~5× hit of max_length=512.
        raw = pipe(text, truncation=True, max_length=256)[0]
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


def analyze(
    date_str: str | None = None,
    lang: str = DEFAULT_LANG,
    country_code: str = "TR",
    only_missing: bool = False,
    only_stale_after_body: bool = False,
    only_with_body: bool = False,
    limit: int | None = None,
) -> int:
    """Run sentiment analysis for a single day's items.

    Returns:
        Number of Turkish items analyzed.
    """
    scoped_modes = [only_missing, only_stale_after_body, only_with_body]
    if sum(bool(mode) for mode in scoped_modes) > 1:
        raise ValueError("only_missing, only_stale_after_body, and only_with_body are mutually exclusive")
    if limit is not None and limit < 1:
        raise ValueError("limit must be a positive integer")
    if (only_missing or only_stale_after_body or only_with_body) and not _is_finetuned_mode():
        raise ValueError(
            "scoped sentiment writes require SENTIMENT_MODEL_ID or SENTIMENT_BACKEND=onnx_int8 "
            "to avoid writing with the default zero-shot model"
        )

    date_str = date_str or date.today().isoformat()

    if lang not in SENTIMENT_LABELS:
        logger.warning(f"Lang '{lang}' not in SENTIMENT_LABELS, falling back to English")
        lang = "en"

    items = fetch_processed_by_date(
        date_str,
        only_missing=only_missing,
        only_stale_after_body=only_stale_after_body,
        only_with_body=only_with_body,
        limit=limit,
        country_code=country_code,
    )
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
            "country_code": country_code,
            "date": date_str,
            "total_items": len(items),
            "analyzed_items": n_tr,
            "only_missing": only_missing,
            "only_stale_after_body": only_stale_after_body,
            "only_with_body": only_with_body,
            "limit": limit,
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
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Analyze only rows that do not have a sentiment label yet.",
    )
    parser.add_argument(
        "--only-stale-after-body",
        action="store_true",
        help="Re-score rows whose sentiment was produced before article body fetch.",
    )
    parser.add_argument(
        "--only-with-body",
        action="store_true",
        help=(
            "Re-score every body-bearing row, OVERWRITING existing sentiment. "
            "Destructive — prefer --only-missing or --only-stale-after-body for incremental work."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum rows to analyze.")
    args = parser.parse_args(argv)
    scoped_modes = [args.only_missing, args.only_stale_after_body, args.only_with_body]
    if sum(bool(mode) for mode in scoped_modes) > 1:
        parser.error("--only-missing, --only-stale-after-body, and --only-with-body are mutually exclusive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    if (args.only_missing or args.only_stale_after_body or args.only_with_body) and not _is_finetuned_mode():
        parser.error(
            "scoped sentiment writes require SENTIMENT_MODEL_ID or SENTIMENT_BACKEND=onnx_int8"
        )
    return args


if __name__ == "__main__":
    args = _parse_args()
    analyze(
        date_str=args.date,
        lang=args.lang,
        only_missing=args.only_missing,
        only_stale_after_body=args.only_stale_after_body,
        only_with_body=args.only_with_body,
        limit=args.limit,
    )
    sys.exit(0)
