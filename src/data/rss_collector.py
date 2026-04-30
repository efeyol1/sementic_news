"""RSS feed collector for Turkish news sources.

Fetches articles from 10 major Turkish news outlets and persists them to
the PostgreSQL news_items table. Duplicate detection is link-based, so
re-running on the same day only inserts genuinely new articles.

Usage:
    python -m src.data.rss_collector
"""

import sys
from datetime import date, datetime, timezone
from typing import Any

import feedparser
from loguru import logger

from src.db.queries import insert_raw_items

# ---------------------------------------------------------------------------
# RSS feed registry
# ---------------------------------------------------------------------------

# RSS endpoint'leri 2026-04-30'da audit edildi. Item count'lar hâlihazırda
# büyük varyans gösteriyor (10–100); çoğu kaynağın tek tip "ana feed"i yok,
# bu sayılar publisher'ın o günkü yayın yoğunluğuyla değişiyor. Audit
# notları:
#   - Milliyet ``gundemrss.xml`` 301 ile ``sondakikarss.xml``e redirect
#     ediyordu; doğrudan hedefi kullanıyoruz.
#   - Sabah ``anasayfa.xml`` (10 item) yerine ``sondakika.xml`` (50 item)
#     kullanıyoruz — 5× daha geniş kapsam.
#   - CNN Türk ve Yeni Şafak feed'leri zaman zaman gelecek tarihli (negatif
#     yaş) entry barındırıyor (publisher embargo). Bu davranış collector
#     seviyesinde clip'lenmiyor; downstream'de ``published_date``'i filtre
#     olarak kullanırken dikkat.
RSS_FEEDS: dict[str, str] = {
    "Habertürk": "https://www.haberturk.com/rss",
    "Hürriyet": "https://www.hurriyet.com.tr/rss/anasayfa",
    "NTV": "https://www.ntv.com.tr/gundem.rss",
    "CNN Türk": "https://www.cnnturk.com/feed/rss/news",
    "Sözcü": "https://www.sozcu.com.tr/rss/son-dakika.xml",
    "Milliyet": "https://www.milliyet.com.tr/rss/rssnew/sondakikarss.xml",
    "Sabah": "https://www.sabah.com.tr/rss/sondakika.xml",
    "TRT Haber": "https://www.trthaber.com/sondakika.rss",
    "Cumhuriyet": "https://www.cumhuriyet.com.tr/rss",
    "Yeni Şafak": "https://www.yenisafak.com/rss",
}

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
# Per-source fetch
# ---------------------------------------------------------------------------


def fetch_feed(source_name: str, url: str) -> list[dict[str, Any]]:
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def collect_all(
    feeds: dict[str, str] | None = None,
    date_str: str | None = None,
) -> int:
    """Collect news from all configured RSS feeds and persist to PostgreSQL.

    Returns:
        Number of new rows inserted.
    """
    feeds = feeds or RSS_FEEDS
    date_str = date_str or date.today().isoformat()

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

    inserted = insert_raw_items(all_items, date_str)

    logger.success(
        f"Collection complete — {inserted} new items inserted "
        f"from {len(feeds) - len(failed_sources)}/{len(feeds)} sources"
    )
    return inserted


if __name__ == "__main__":
    collect_all()
    sys.exit(0)
