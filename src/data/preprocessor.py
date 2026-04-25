"""Preprocessor for raw Turkish news RSS data.

Reads raw items from PostgreSQL, applies cleaning and enrichment, and writes
the result back to the same rows.

Processing pipeline (per item):
    1. HTML tag removal from title and summary
    2. Whitespace normalization
    3. Short-item filtering: delete if title or summary < MIN_CHARS characters
    4. ISO-8601 date normalization
    5. Turkish language detection (is_turkish flag)
    6. char_count = len(cleaned_title) + len(cleaned_summary)

Usage:
    python -m src.data.preprocessor               # process today's items
    python -m src.data.preprocessor --date 2026-04-17
"""

import argparse
import re
import sys
from datetime import date, datetime, timezone
from typing import Any

from bs4 import BeautifulSoup
from langdetect import DetectorFactory, LangDetectException, detect
from loguru import logger

from src.db.queries import bulk_update_preprocessed, fetch_raw_by_date

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_CHARS: int = 10

DetectorFactory.seed = 42

# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------


def _strip_html(text: str) -> str:
    if not text or "<" not in text:
        return text
    return BeautifulSoup(text, "html.parser").get_text(separator=" ")


_WHITESPACE_RE = re.compile(r"[ \t\r\n]+")


def _normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _clean(text: str) -> str:
    return _normalize_whitespace(_strip_html(text))


# ---------------------------------------------------------------------------
# Date normalization
# ---------------------------------------------------------------------------

_DATE_FORMATS: list[str] = [
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%SZ",
    "%a, %d %b %Y %H:%M:%S %z",
    "%a, %d %b %Y %H:%M:%S %Z",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
]


def _normalize_date(raw) -> str:
    if raw is None:
        return datetime.now(timezone.utc).isoformat()
    # psycopg2 may return datetime objects directly from the DB
    if isinstance(raw, datetime):
        if raw.tzinfo is None:
            raw = raw.replace(tzinfo=timezone.utc)
        return raw.isoformat()
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(str(raw).strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except ValueError:
            continue
    logger.debug(f"Could not parse date {raw!r}, using current UTC time")
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


def _is_turkish(text: str) -> bool:
    sample = text[:300].strip()
    if not sample:
        return True
    try:
        return detect(sample) == "tr"
    except LangDetectException:
        return True


# ---------------------------------------------------------------------------
# Per-item processing
# ---------------------------------------------------------------------------


def process_item(item: dict[str, Any]) -> dict[str, Any] | None:
    """Apply the full preprocessing pipeline to a single news item.

    Returns None if the item should be discarded (too short).
    """
    raw_title: str = item.get("title", "") or ""
    raw_summary: str = item.get("summary", "") or ""

    cleaned_title = _clean(raw_title)
    cleaned_summary = _clean(raw_summary)

    if len(cleaned_title) < MIN_CHARS or len(cleaned_summary) < MIN_CHARS:
        return None

    normalized_date = _normalize_date(item.get("published_date"))
    combined = f"{cleaned_title} {cleaned_summary}"
    turkish = _is_turkish(combined)

    return {
        "id": item["id"],
        "cleaned_title": cleaned_title.lower(),
        "cleaned_summary": cleaned_summary.lower(),
        "is_turkish": turkish,
        "char_count": len(cleaned_title) + len(cleaned_summary),
        "published_date": normalized_date,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def preprocess(date_str: str | None = None) -> int:
    """Run the full preprocessing pipeline for a single day's raw items.

    Returns:
        Number of items kept after filtering.
    """
    date_str = date_str or date.today().isoformat()

    raw_items = fetch_raw_by_date(date_str)
    if not raw_items:
        logger.warning(f"No raw items found for {date_str} — run rss_collector first")
        return 0

    updates: list[dict[str, Any]] = []
    deletes: list[int] = []
    non_turkish = 0

    for item in raw_items:
        result = process_item(item)
        if result is None:
            deletes.append(item["id"])
        else:
            if not result["is_turkish"]:
                non_turkish += 1
            updates.append(result)

    bulk_update_preprocessed(updates, deletes)

    logger.info(
        f"Preprocessing done — {len(updates)} kept, {len(deletes)} deleted (too short), "
        f"{non_turkish} flagged as non-Turkish"
    )
    return len(updates)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess raw Turkish news RSS data.")
    parser.add_argument("--date", default=date.today().isoformat(), metavar="YYYY-MM-DD")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    preprocess(date_str=args.date)
    sys.exit(0)
