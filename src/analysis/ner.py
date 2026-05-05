"""Named Entity Recognition for Turkish news items.

Reads items from PostgreSQL, extracts PER/ORG/LOC entities via
savasy/bert-base-turkish-ner-cased, and writes enriched results back to the
same rows.

Usage:
    python -m src.analysis.ner               # process today's items
    python -m src.analysis.ner --date 2026-04-17
"""

import argparse
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import mlflow
import torch
from loguru import logger
from transformers import pipeline

from src.analysis.text_inputs import build_ner_text
from src.db.queries import bulk_update_ner, fetch_for_ner

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_MODEL_ID = "savasy/bert-base-turkish-ner-cased"

_KEEP_LABELS = {"PER", "ORG", "LOC"}

_REPO_ROOT = Path(__file__).resolve().parents[2]

mlflow.set_tracking_uri(f"sqlite:///{_REPO_ROOT / 'mlflow.db'}")
mlflow.set_experiment("news-ner")

# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def _load_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    logger.info(f"Loading model {HF_MODEL_ID!r} on {'CUDA' if device == 0 else 'CPU'}")
    pipe = pipeline(
        "ner",
        model=HF_MODEL_ID,
        aggregation_strategy="simple",
        device=device,
    )
    logger.success("Model loaded.")
    return pipe


def _build_text(item: dict[str, Any]) -> str:
    return build_ner_text(item)


def _extract_entities(text: str, pipe) -> dict[str, list[str]]:
    raw: list[dict] = pipe(text)
    grouped: dict[str, set[str]] = {label: set() for label in _KEEP_LABELS}
    for ent in raw:
        label = ent.get("entity_group", "")
        if label in _KEEP_LABELS:
            word = ent["word"].strip()
            if word:
                grouped[label].add(word)
    return {label: sorted(words) for label, words in grouped.items()}


# ---------------------------------------------------------------------------
# Per-item processing
# ---------------------------------------------------------------------------


def _to_ner(item: dict[str, Any], pipe) -> dict[str, Any]:
    result: dict[str, Any] = {"id": item["id"]}
    if not item.get("is_turkish", True):
        result["entities"] = {label: [] for label in _KEEP_LABELS}
        result["entity_count"] = 0
        return result
    text = _build_text(item)
    entities = _extract_entities(text, pipe)
    result["entities"] = entities
    result["entity_count"] = sum(len(v) for v in entities.values())
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_entities(date_str: str | None = None) -> int:
    """Run NER pipeline for a single day's items.

    Returns:
        Total number of entities extracted across all Turkish items.
    """
    date_str = date_str or date.today().isoformat()

    items = fetch_for_ner(date_str)
    if not items:
        logger.warning(f"No items found for NER on {date_str}")
        return 0

    pipe = _load_pipeline()

    t0 = time.perf_counter()
    updates: list[dict[str, Any]] = []
    for i, item in enumerate(items, 1):
        updates.append(_to_ner(item, pipe))
        if i % 50 == 0:
            logger.debug(f"NER progress: {i}/{len(items)}")
    duration = time.perf_counter() - t0

    bulk_update_ner(updates)

    total_entities = sum(u.get("entity_count", 0) for u in updates)
    per_count = sum(len(u["entities"].get("PER", [])) for u in updates)
    org_count = sum(len(u["entities"].get("ORG", [])) for u in updates)
    loc_count = sum(len(u["entities"].get("LOC", [])) for u in updates)

    logger.info(
        f"NER done — {len(items)} items, {total_entities} entities "
        f"(PER={per_count}, ORG={org_count}, LOC={loc_count}) in {duration:.1f}s"
    )

    n_tr = len([i for i in items if i.get("is_turkish")])
    with mlflow.start_run(run_name=f"ner-{date_str}"):
        mlflow.log_params({
            "model_id": HF_MODEL_ID,
            "date": date_str,
            "total_items": len(items),
            "turkish_items": n_tr,
        })
        mlflow.log_metrics({
            "total_entities": total_entities,
            "per_count": per_count,
            "org_count": org_count,
            "loc_count": loc_count,
            "avg_entities_per_item": total_entities / n_tr if n_tr else 0,
            "duration_sec": duration,
        })

    return total_entities


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract named entities from Turkish news data."
    )
    parser.add_argument("--date", default=date.today().isoformat(), metavar="YYYY-MM-DD")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    extract_entities(date_str=args.date)
    sys.exit(0)
