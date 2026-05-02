"""RSS feed collector for news sources.

V2 Phase 2: feed registry now lives in ``configs/countries/<slug>.yaml``
under the ``sources`` section. ``collect_all`` defaults to loading
``turkey.yaml`` so existing callers (pipeline, CLI) keep working without
arguments. Multi-URL sources are still supported (NTV's 5 category
feeds, Milliyet's 3) — feeds are aggregated and deduped by link before
insert. Duplicate detection at DB level via ``ON CONFLICT (link)``
remains, so re-running on the same day only inserts genuinely new
articles.

Usage:
    python -m src.data.rss_collector             # turkey by default
    python -m src.data.rss_collector --country turkey
"""

import argparse
import sys
from datetime import date, datetime, timezone
from typing import Any

import feedparser
from loguru import logger

from src.config import load_country_config
from src.db.queries import insert_raw_items

# ---------------------------------------------------------------------------
# Country config → feed registry
# ---------------------------------------------------------------------------


def _feeds_from_config(country_config: dict[str, Any]) -> dict[str, list[str]]:
    """Translate a country config's ``sources`` block into name → urls.

    Only ``enabled`` sources are returned. URL lists are preserved so
    multi-feed outlets (NTV, Milliyet) keep their fan-out behavior.
    """
    feeds: dict[str, list[str]] = {}
    for src in country_config.get("sources", []):
        if not src.get("enabled", True):
            continue
        if src.get("type", "rss") != "rss":
            # Future: HTTP API, sitemap, etc. Skip non-RSS for now.
            logger.warning(
                f"Skipping {src.get('name')!r}: unsupported type {src.get('type')!r}"
            )
            continue
        name = src["name"]
        urls = list(src.get("urls") or [])
        if name in feeds:
            feeds[name].extend(urls)
        else:
            feeds[name] = urls
    return feeds


# ---------------------------------------------------------------------------
# Entry parsing helpers
# ---------------------------------------------------------------------------


def _parse_date(entry: Any) -> str:
    for attr in ("published_parsed", "updated_parsed", "created_parsed"):
        t = getattr(entry, attr, None)
        if t is not None:
            try:
                return datetime(*t[:6], tzinfo=timezone.utc).isoformat()
            except Exception:
                pass
    return datetime.now(timezone.utc).isoformat()


def _extract_category(entry: Any) -> str | None:
    tags = getattr(entry, "tags", None)
    if tags:
        return tags[0].get("term") or tags[0].get("label") or None
    return None


def _entry_to_item(entry: Any, source_name: str) -> dict[str, Any]:
    return {
        "title": getattr(entry, "title", "").strip(),
        "summary": getattr(entry, "summary", "").strip(),
        "source_name": source_name,
        "published_date": _parse_date(entry),
        "link": getattr(entry, "link", "").strip(),
        "category": _extract_category(entry),
    }


# ---------------------------------------------------------------------------
# Per-feed / per-source fetch
# ---------------------------------------------------------------------------


def fetch_feed(source_name: str, url: str) -> list[dict[str, Any]]:
    """Fetch a single RSS URL. Returns parsed items or [] on failure."""
    logger.info(f"Fetching  [{source_name}]  {url}")
    try:
        feed = feedparser.parse(url)
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


def fetch_source(source_name: str, urls: list[str]) -> list[dict[str, Any]]:
    """Fetch all configured URLs for one source and dedupe by link.

    Multi-feed sources (NTV, Milliyet) get a substantial coverage boost
    when their per-category endpoints are aggregated; single-URL sources
    pass through unchanged.
    """
    if len(urls) == 1:
        return fetch_feed(source_name, urls[0])

    aggregated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for url in urls:
        for item in fetch_feed(source_name, url):
            link = item.get("link") or ""
            if link and link in seen:
                continue
            if link:
                seen.add(link)
            aggregated.append(item)
    logger.success(
        f"[{source_name}] {len(aggregated)} unique items "
        f"after dedupe across {len(urls)} feeds"
    )
    return aggregated


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def collect_all(
    feeds: dict[str, list[str]] | None = None,
    date_str: str | None = None,
    country: str = "turkey",
) -> int:
    """Collect news from configured RSS feeds and persist to PostgreSQL.

    Args:
        feeds: explicit name → urls map. If omitted, loaded from the
               country config (Phase 2 default behavior).
        date_str: ISO date; defaults to today.
        country: country slug or code (default ``"turkey"``). Phase 3
                 will plumb the real selection through the orchestrator.

    Returns:
        Number of new rows inserted.
    """
    date_str = date_str or date.today().isoformat()

    if feeds is None:
        config = load_country_config(country)
        feeds = _feeds_from_config(config)
        logger.info(
            f"Loaded {len(feeds)} sources from country config "
            f"[{config['country_code']}/{config['country_slug']}]"
        )

    all_items: list[dict[str, Any]] = []
    failed_sources: list[str] = []

    for source_name, urls in feeds.items():
        items = fetch_source(source_name, urls)
        if items:
            all_items.extend(items)
        else:
            failed_sources.append(source_name)

    if failed_sources:
        logger.warning(
            f"{len(failed_sources)} source(s) returned no items: "
            + ", ".join(failed_sources)
        )

    inserted = insert_raw_items(all_items, date_str)

    logger.success(
        f"Collection complete — {inserted} new items inserted "
        f"from {len(feeds) - len(failed_sources)}/{len(feeds)} sources"
    )
    return inserted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect news from a country's RSS feeds")
    p.add_argument("--country", default="turkey", help="Country slug or code (default: turkey)")
    p.add_argument("--date", default=date.today().isoformat(), metavar="YYYY-MM-DD")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    collect_all(country=args.country, date_str=args.date)
    sys.exit(0)
