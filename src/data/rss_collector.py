"""RSS feed collector for Turkish news sources.

Fetches articles from 10 major Turkish news outlets and persists them to
data/raw/YYYY-MM-DD.json.  Duplicate detection is link-based, so re-running
on the same day only appends genuinely new articles.

Usage:
    python -m src.data.rss_collector
"""

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import feedparser
from loguru import logger

# ---------------------------------------------------------------------------
# RSS feed registry
# ---------------------------------------------------------------------------

RSS_FEEDS: dict[str, str] = {
    "Habertürk": "https://www.haberturk.com/rss",
    "Hürriyet": "https://www.hurriyet.com.tr/rss/anasayfa",
    "NTV": "https://www.ntv.com.tr/gundem.rss",
    "CNN Türk": "https://www.cnnturk.com/feed/rss/news",
    "Sözcü": "https://www.sozcu.com.tr/rss/son-dakika.xml",
    "Milliyet": "https://www.milliyet.com.tr/rss/rssnew/gundemrss.xml",
    "Sabah": "https://www.sabah.com.tr/rss/anasayfa.xml",
    "TRT Haber": "https://www.trthaber.com/sondakika.rss",
    "Cumhuriyet": "https://www.cumhuriyet.com.tr/rss",
    "Yeni Şafak": "https://www.yenisafak.com/rss",
}

# Resolved relative to this file: <repo_root>/data/raw/
_DATA_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# ---------------------------------------------------------------------------
# Entry parsing helpers
# ---------------------------------------------------------------------------


def _parse_date(entry: Any) -> str:
    """Return an ISO-8601 UTC timestamp from a feedparser entry.

    Tries published_parsed → updated_parsed → created_parsed in order.
    Falls back to the current UTC time if none are present.
    """
    for attr in ("published_parsed", "updated_parsed", "created_parsed"):
        t = getattr(entry, attr, None)
        if t is not None:
            try:
                return datetime(*t[:6], tzinfo=timezone.utc).isoformat()
            except Exception:
                pass
    return datetime.now(timezone.utc).isoformat()


def _extract_category(entry: Any) -> str | None:
    """Return the first category tag term from an entry, or None."""
    tags = getattr(entry, "tags", None)
    if tags:
        return tags[0].get("term") or tags[0].get("label") or None
    return None


def _entry_to_item(entry: Any, source_name: str) -> dict[str, Any]:
    """Normalize a feedparser entry into a plain dict.

    Args:
        entry: A single feedparser entry object.
        source_name: Human-readable outlet name written into the record.

    Returns:
        Dict with keys: title, summary, source_name, published_date,
        link, category.
    """
    return {
        "title": getattr(entry, "title", "").strip(),
        "summary": getattr(entry, "summary", "").strip(),
        "source_name": source_name,
        "published_date": _parse_date(entry),
        "link": getattr(entry, "link", "").strip(),
        "category": _extract_category(entry),
    }


# ---------------------------------------------------------------------------
# Per-source fetch
# ---------------------------------------------------------------------------


def fetch_feed(source_name: str, url: str) -> list[dict[str, Any]]:
    """Fetch and parse a single RSS feed.

    Errors are caught and logged so a single bad source never halts the
    whole collection run.

    Args:
        source_name: Human-readable outlet name (used in log output).
        url: RSS/Atom feed URL.

    Returns:
        List of normalized news item dicts; empty list on any failure.
    """
    logger.info(f"Fetching  [{source_name}]  {url}")
    try:
        feed = feedparser.parse(url)

        # bozo=1 means the feed was malformed XML.  feedparser often still
        # returns partial results, so we continue but emit a warning.
        if feed.bozo and feed.bozo_exception:
            logger.warning(
                f"[{source_name}] Malformed feed "
                f"({type(feed.bozo_exception).__name__}: {feed.bozo_exception})"
                " — continuing with partial results"
            )

        if not feed.entries:
            logger.warning(f"[{source_name}] Feed parsed but contained 0 entries")
            return []

        items = [_entry_to_item(e, source_name) for e in feed.entries]
        logger.success(f"[{source_name}] {len(items)} items fetched")
        return items

    except Exception as exc:
        logger.error(f"[{source_name}] Unrecoverable error: {exc}")
        return []


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _load_existing(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    """Read today's existing JSON file (if any).

    Args:
        path: Absolute path to the target JSON file.

    Returns:
        Tuple of (existing_items, seen_links).  Both are empty when the
        file does not exist or is unreadable.
    """
    if not path.exists():
        return [], set()
    try:
        with path.open(encoding="utf-8") as fh:
            data: list[dict[str, Any]] = json.load(fh)
        seen_links = {item["link"] for item in data if item.get("link")}
        logger.debug(f"Loaded {len(data)} existing items from {path.name}")
        return data, seen_links
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"Could not read {path}: {exc} — starting fresh")
        return [], set()


def save_items(items: list[dict[str, Any]], output_dir: Path) -> Path:
    """Append new items to today's JSON file, skipping link-based duplicates.

    If the file for today already exists the function loads it, deduplicates
    incoming items by their *link* field, then writes the merged list back.

    Args:
        items: Normalized news item dicts to persist.
        output_dir: Directory where YYYY-MM-DD.json files are written.

    Returns:
        Absolute path of the written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    today_str = date.today().isoformat()
    path = output_dir / f"{today_str}.json"

    existing, seen_links = _load_existing(path)

    new_items: list[dict[str, Any]] = []
    for item in items:
        link = item.get("link", "")
        if link and link in seen_links:
            continue
        new_items.append(item)
        if link:
            seen_links.add(link)

    all_items = existing + new_items

    with path.open("w", encoding="utf-8") as fh:
        json.dump(all_items, fh, ensure_ascii=False, indent=2)

    logger.info(
        f"Wrote {path.name}: "
        f"{len(new_items)} new + {len(existing)} existing = {len(all_items)} total"
    )
    return path


# ---------------------------------------------------------------------------
# Public API / CLI entrypoint
# ---------------------------------------------------------------------------


def collect_all(
    feeds: dict[str, str] | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Collect news from all configured RSS feeds and persist to disk.

    Each source is fetched independently; a failure in one never prevents
    others from running.  Summary statistics and any failed sources are
    logged at the end.

    Args:
        feeds: ``{source_name: rss_url}`` mapping.  Defaults to
            :data:`RSS_FEEDS`.
        output_dir: Directory for output files.  Defaults to
            ``<repo_root>/data/raw/``.

    Returns:
        Absolute path to the JSON file that was written.
    """
    feeds = feeds or RSS_FEEDS
    output_dir = output_dir or _DATA_RAW_DIR

    all_items: list[dict[str, Any]] = []
    failed_sources: list[str] = []

    for source_name, url in feeds.items():
        items = fetch_feed(source_name, url)
        if items:
            all_items.extend(items)
        else:
            failed_sources.append(source_name)

    if failed_sources:
        logger.warning(
            f"{len(failed_sources)} source(s) returned no items: "
            + ", ".join(failed_sources)
        )

    output_path = save_items(all_items, output_dir)

    logger.success(
        f"Collection complete — "
        f"{len(all_items)} items from "
        f"{len(feeds) - len(failed_sources)}/{len(feeds)} sources"
    )
    return output_path


if __name__ == "__main__":
    collect_all()
    sys.exit(0)
