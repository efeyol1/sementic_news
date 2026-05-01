"""RSS feed collector for Turkish news sources.

Fetches articles from 10 major Turkish news outlets and persists them to
the PostgreSQL news_items table. Some outlets shard their content across
category-specific feeds (NTV, Milliyet) — for those we list multiple URLs
per source and dedupe in-memory by link before insert. Duplicate detection
is link-based, so re-running on the same day only inserts genuinely new
articles.

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

# RSS endpoint'leri 2026-04-30 → 2026-05-01'de iki kez audit edildi.
# Item count varyansı (10–100) çoğunlukla publisher tarafından feed-layer
# kısıtlamasından geliyor. Audit notları:
#
#   - **NTV** ana ``gundem.rss``'de 20 item ile cap'lı. Her kategori ayrı
#     20-item feed sunuyor; 5 kategori birleştirip dedupe ediyoruz → ~80-100.
#   - **Milliyet** ``sondakikarss.xml`` 20 item, ama ``dunyarss.xml`` 50
#     item veriyor; üçünü birleştirip dedupe ediyoruz.
#   - **Yeni Şafak** ana feed'de 15 ile sert sınırlı; kategori feed'leri
#     301 + boş XML, fonksiyonel değil. Duruyor.
#   - **CNN Türk** tüm kategori path'leri aynı 35 item'i alias gibi
#     dönüyor — birleştirmek faydasız. 35'te kalır.
#   - **Sabah** ``anasayfa.xml`` (10) → ``sondakika.xml`` (50) geçişi
#     2026-04-30'da yapıldı.
#   - **CNN Türk + Yeni Şafak** zaman zaman gelecek tarihli (negatif yaş)
#     entry'ler barındırıyor (publisher embargo / scheduled posts).
#     Collector seviyesinde clip'lenmiyor; downstream'de ``published_date``
#     filtre olarak kullanırken dikkat.
RSS_FEEDS: dict[str, list[str]] = {
    "Habertürk": ["https://www.haberturk.com/rss"],
    "Hürriyet":  ["https://www.hurriyet.com.tr/rss/anasayfa"],
    "NTV": [
        "https://www.ntv.com.tr/gundem.rss",
        "https://www.ntv.com.tr/son-dakika.rss",
        "https://www.ntv.com.tr/turkiye.rss",
        "https://www.ntv.com.tr/dunya.rss",
        "https://www.ntv.com.tr/ekonomi.rss",
    ],
    "CNN Türk":  ["https://www.cnnturk.com/feed/rss/news"],
    "Sözcü":     ["https://www.sozcu.com.tr/rss/son-dakika.xml"],
    "Milliyet": [
        "https://www.milliyet.com.tr/rss/rssnew/sondakikarss.xml",
        "https://www.milliyet.com.tr/rss/rssnew/dunyarss.xml",
        "https://www.milliyet.com.tr/rss/rssnew/ekonomirss.xml",
    ],
    "Sabah":      ["https://www.sabah.com.tr/rss/sondakika.xml"],
    "TRT Haber":  ["https://www.trthaber.com/sondakika.rss"],
    "Cumhuriyet": ["https://www.cumhuriyet.com.tr/rss"],
    "Yeni Şafak": ["https://www.yenisafak.com/rss"],
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
) -> int:
    """Collect news from all configured RSS feeds and persist to PostgreSQL.

    Returns:
        Number of new rows inserted.
    """
    feeds = feeds or RSS_FEEDS
    date_str = date_str or date.today().isoformat()

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


if __name__ == "__main__":
    collect_all()
    sys.exit(0)
