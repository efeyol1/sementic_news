"""Preprocessor for raw Turkish news RSS data.

Reads data/raw/YYYY-MM-DD.json, applies cleaning and enrichment steps, and
writes the result to data/processed/YYYY-MM-DD.json.

Processing pipeline (per item):
    1. HTML tag removal from title and summary
    2. Whitespace normalization (collapse runs, strip leading/trailing)
    3. Lowercase copies stored in cleaned_title / cleaned_summary
       (originals preserved as-is)
    4. Short-item filtering: skip if title or summary < MIN_CHARS characters
    5. ISO-8601 date normalization (handles missing / malformed timestamps)
    6. Turkish language detection via langdetect (is_turkish flag)
    7. char_count = len(cleaned_title) + len(cleaned_summary)

Usage:
    python -m src.data.preprocessor               # process today's file
    python -m src.data.preprocessor --date 2026-04-17
"""

import argparse
import json
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from langdetect import DetectorFactory, LangDetectException, detect
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_CHARS: int = 10  # items shorter than this (title OR summary) are skipped

# Make langdetect deterministic across runs
DetectorFactory.seed = 42

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_RAW_DIR = _REPO_ROOT / "data" / "raw"
_DATA_PROCESSED_DIR = _REPO_ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------


def _strip_html(text: str) -> str:
    """Remove all HTML/XML tags from *text* using BeautifulSoup.

    Args:
        text: Raw string that may contain HTML markup.

    Returns:
        Plain text with tags removed and HTML entities decoded.
    """
    if not text or "<" not in text:
        return text
    return BeautifulSoup(text, "html.parser").get_text(separator=" ")


_WHITESPACE_RE = re.compile(r"[ \t\r\n]+")


def _normalize_whitespace(text: str) -> str:
    """Collapse consecutive whitespace characters into a single space."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def _clean(text: str) -> str:
    """Full cleaning pass: strip HTML then normalize whitespace."""
    return _normalize_whitespace(_strip_html(text))


# ---------------------------------------------------------------------------
# Date normalization
# ---------------------------------------------------------------------------

_DATE_FORMATS: list[str] = [
    "%Y-%m-%dT%H:%M:%S%z",   # ISO 8601 with offset  (already our format)
    "%Y-%m-%dT%H:%M:%SZ",    # ISO 8601 UTC Z-suffix
    "%a, %d %b %Y %H:%M:%S %z",  # RFC 2822 (common in RSS)
    "%a, %d %b %Y %H:%M:%S %Z",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
]


def _normalize_date(raw: str | None) -> str:
    """Return an ISO-8601 UTC timestamp string from *raw*.

    Tries a list of known formats in order.  Falls back to the current UTC
    time and emits a debug log if the value cannot be parsed.

    Args:
        raw: Date string from the raw JSON record (may be None or empty).

    Returns:
        ISO-8601 string, e.g. ``"2026-04-17T07:00:00+00:00"``.
    """
    if raw:
        for fmt in _DATE_FORMATS:
            try:
                dt = datetime.strptime(raw.strip(), fmt)
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
    """Return True if *text* is detected as Turkish.

    Uses the first 300 characters for speed.  Defaults to True on any
    detection error (Turkish corpus is dominant — false negatives are worse
    than false positives here).

    Args:
        text: Cleaned plain-text content.

    Returns:
        Boolean language flag.
    """
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

    Args:
        item: Raw news record with keys: title, summary, source_name,
            published_date, link, category.

    Returns:
        Enriched record, or None if the item is filtered out (too short).
    """
    raw_title: str = item.get("title", "") or ""
    raw_summary: str = item.get("summary", "") or ""

    cleaned_title = _clean(raw_title)
    cleaned_summary = _clean(raw_summary)

    # --- Filter: skip items that are too short after cleaning ---
    if len(cleaned_title) < MIN_CHARS or len(cleaned_summary) < MIN_CHARS:
        return None

    # --- Date normalization ---
    normalized_date = _normalize_date(item.get("published_date"))

    # --- Language detection (run on combined text for better accuracy) ---
    combined = f"{cleaned_title} {cleaned_summary}"
    turkish = _is_turkish(combined)

    return {
        # Original fields (unchanged)
        "title": raw_title,
        "summary": raw_summary,
        "source_name": item.get("source_name", ""),
        "published_date": normalized_date,
        "link": item.get("link", ""),
        "category": item.get("category"),
        # New enrichment fields
        "cleaned_title": cleaned_title.lower(),
        "cleaned_summary": cleaned_summary.lower(),
        "is_turkish": turkish,
        "char_count": len(cleaned_title) + len(cleaned_summary),
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_raw(date_str: str, raw_dir: Path) -> list[dict[str, Any]]:
    """Load raw items for *date_str* from *raw_dir*.

    Args:
        date_str: ISO date string, e.g. ``"2026-04-17"``.
        raw_dir: Directory containing YYYY-MM-DD.json raw files.

    Returns:
        List of raw item dicts.

    Raises:
        FileNotFoundError: If the raw file does not exist.
    """
    path = raw_dir / f"{date_str}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Raw file not found: {path}\n"
            f"Run `python -m src.data.rss_collector` first."
        )
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    logger.info(f"Loaded {len(data)} raw items from {path.name}")
    return data


def save_processed(
    items: list[dict[str, Any]],
    date_str: str,
    processed_dir: Path,
) -> Path:
    """Write processed items to *processed_dir*/YYYY-MM-DD.json.

    Always overwrites — preprocessor output is deterministic for a given
    raw file, so re-running is safe.

    Args:
        items: Processed item dicts.
        date_str: ISO date string used as the filename stem.
        processed_dir: Output directory.

    Returns:
        Absolute path of the written file.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = processed_dir / f"{date_str}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(items, fh, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(items)} processed items → {path}")
    return path


# ---------------------------------------------------------------------------
# Public API / CLI entrypoint
# ---------------------------------------------------------------------------


def preprocess(
    date_str: str | None = None,
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
) -> Path:
    """Run the full preprocessing pipeline for a single day's raw file.

    Args:
        date_str: ISO date string (``"YYYY-MM-DD"``).  Defaults to today.
        raw_dir: Source directory.  Defaults to ``data/raw/``.
        processed_dir: Destination directory.  Defaults to
            ``data/processed/``.

    Returns:
        Path to the written processed JSON file.
    """
    date_str = date_str or date.today().isoformat()
    raw_dir = raw_dir or _DATA_RAW_DIR
    processed_dir = processed_dir or _DATA_PROCESSED_DIR

    raw_items = load_raw(date_str, raw_dir)

    processed: list[dict[str, Any]] = []
    filtered = 0
    non_turkish = 0

    for item in raw_items:
        result = process_item(item)
        if result is None:
            filtered += 1
            continue
        if not result["is_turkish"]:
            non_turkish += 1
        processed.append(result)

    logger.info(
        f"Pipeline complete — "
        f"{len(processed)} kept, {filtered} filtered (too short), "
        f"{non_turkish} flagged as non-Turkish"
    )

    return save_processed(processed, date_str, processed_dir)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess raw Turkish news RSS data."
    )
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        metavar="YYYY-MM-DD",
        help="Date of the raw file to process (default: today)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    try:
        preprocess(date_str=args.date)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)
