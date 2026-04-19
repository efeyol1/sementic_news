"""Named Entity Recognition for analyzed Turkish news items.

Reads data/analyzed/YYYY-MM-DD.json, extracts PER/ORG/LOC entities via
savasy/bert-base-turkish-ner-cased, and writes enriched results back to the
same file.

Usage:
    python -m src.analysis.ner               # process today's file
    python -m src.analysis.ner --date 2026-04-17
"""

import argparse
import json
import re
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import mlflow
import torch
from loguru import logger
from transformers import pipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_MODEL_ID = "savasy/bert-base-turkish-ner-cased"

# Entity types we care about (others are silently dropped)
_KEEP_LABELS = {"PER", "ORG", "LOC"}

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_ANALYZED_DIR = _REPO_ROOT / "data" / "analyzed"

_HTML_TAG_RE = re.compile(r"<[^>]+>")

mlflow.set_tracking_uri(f"sqlite:///{_REPO_ROOT / 'mlflow.db'}")
mlflow.set_experiment("news-ner")

# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def _load_pipeline():
    """Load the NER pipeline (CPU or CUDA automatically)."""
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


def _strip_html(text: str) -> str:
    """Remove HTML tags — keeps original casing for NER accuracy."""
    return _HTML_TAG_RE.sub(" ", text).strip()


def _build_text(item: dict[str, Any]) -> str:
    """Build NER input from original (cased) title + summary."""
    title = _strip_html(item.get("title", "") or "")
    summary = _strip_html(item.get("summary", "") or "")
    return f"{title}. {summary}".strip()


def _extract_entities(text: str, pipe) -> dict[str, list[str]]:
    """Run NER and return deduplicated entities grouped by label type."""
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
    """Add NER entity fields to a single analyzed item."""
    result = dict(item)
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
# I/O helpers
# ---------------------------------------------------------------------------


def _load_analyzed(date_str: str, analyzed_dir: Path) -> list[dict[str, Any]]:
    """Load analyzed items for *date_str*."""
    path = analyzed_dir / f"{date_str}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Analyzed file not found: {path}\n"
            f"Run `python -m src.analysis.sentiment --date {date_str}` first."
        )
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    logger.info(f"Loaded {len(data)} analyzed items from {path.name}")
    return data


def _save_analyzed(
    items: list[dict[str, Any]],
    date_str: str,
    analyzed_dir: Path,
) -> Path:
    """Write NER-enriched items back to *analyzed_dir*/YYYY-MM-DD.json."""
    analyzed_dir.mkdir(parents=True, exist_ok=True)
    path = analyzed_dir / f"{date_str}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(items, fh, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(items)} NER-enriched items → {path}")
    return path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_entities(
    date_str: str | None = None,
    analyzed_dir: Path | None = None,
) -> Path:
    """Run NER pipeline for a single day's analyzed file.

    Reads data/analyzed/YYYY-MM-DD.json, adds ``entities`` and
    ``entity_count`` fields to each item, and overwrites the file.

    Args:
        date_str: ISO date string (``"YYYY-MM-DD"``).  Defaults to today.
        analyzed_dir: Source/destination directory.  Defaults to
            ``data/analyzed/``.

    Returns:
        Path to the updated analyzed JSON file.
    """
    date_str = date_str or date.today().isoformat()
    analyzed_dir = analyzed_dir or _DATA_ANALYZED_DIR

    items = _load_analyzed(date_str, analyzed_dir)
    pipe = _load_pipeline()

    t0 = time.perf_counter()
    enriched: list[dict[str, Any]] = []
    for i, item in enumerate(items, 1):
        enriched.append(_to_ner(item, pipe))
        if i % 50 == 0:
            logger.debug(f"NER progress: {i}/{len(items)}")
    duration = time.perf_counter() - t0

    turkish_items = [e for e in enriched if e.get("is_turkish", True)]
    total_entities = sum(e.get("entity_count", 0) for e in turkish_items)
    per_count = sum(len(e["entities"].get("PER", [])) for e in turkish_items)
    org_count = sum(len(e["entities"].get("ORG", [])) for e in turkish_items)
    loc_count = sum(len(e["entities"].get("LOC", [])) for e in turkish_items)

    logger.info(
        f"NER done — {len(turkish_items)} items, {total_entities} entities "
        f"(PER={per_count}, ORG={org_count}, LOC={loc_count}) in {duration:.1f}s"
    )

    out_path = _save_analyzed(enriched, date_str, analyzed_dir)

    with mlflow.start_run(run_name=f"ner-{date_str}"):
        mlflow.log_params({
            "model_id": HF_MODEL_ID,
            "date": date_str,
            "total_items": len(items),
            "turkish_items": len(turkish_items),
        })
        mlflow.log_metrics({
            "total_entities": total_entities,
            "per_count": per_count,
            "org_count": org_count,
            "loc_count": loc_count,
            "avg_entities_per_item": total_entities / len(turkish_items) if turkish_items else 0,
            "duration_sec": duration,
        })

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract named entities from analyzed Turkish news data."
    )
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        metavar="YYYY-MM-DD",
        help="Date of the analyzed file to process (default: today)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    try:
        extract_entities(date_str=args.date)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)
