"""RSS / sitemap collector for news sources.

V2 Phase 2 + followup: source registry lives in
``configs/countries/<slug>.yaml`` under ``sources``. Each entry declares
a ``type``:

  - ``rss``                 — default; feedparser-based.
  - ``googlenews_sitemap``  — Google's news-extended sitemap (title +
                              publication_date inline, no per-article
                              fetch needed).
  - ``html_sitemap``        — plain sitemap whose ``<url>`` entries
                              carry only ``<loc>``; titles come from
                              fetching each article's HTML and reading
                              ``<title>`` / og:description. Used when an
                              outlet's RSS is capped and they don't
                              expose a Google News sitemap (e.g. Yeni
                              Şafak). Concurrent ThreadPool fetch.

``collect_all`` defaults to ``turkey.yaml`` so existing callers
(pipeline, CLI) keep working without arguments. Multi-URL sources are
still supported (NTV's category feeds, Milliyet's, etc.) — items are
aggregated and deduped by link before insert. DB-level dedup via
``ON CONFLICT (link)`` provides a second safety net so two source
entries pointing at overlapping content (e.g. Habertürk RSS + Habertürk
googlenews_sitemap) merge cleanly.

Usage:
    python -m src.data.rss_collector             # turkey by default
    python -m src.data.rss_collector --country turkey
"""

import argparse
import html
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from typing import Any
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

import feedparser
from loguru import logger

from src.config import load_country_config
from src.db.queries import insert_raw_items

_HTTP_USER_AGENT = (
    "Mozilla/5.0 (compatible; SemanticNewsBot/1.0; +https://github.com/efeyol11/sementic_news)"
)
_SITEMAP_TIMEOUT_SEC = 15
_HTML_SCRAPE_TIMEOUT_SEC = 10
_HTML_SCRAPE_WORKERS = 15  # parallel article fetches per html_sitemap source

# Title / meta description regex for HTML scraping. Loose by design —
# news pages vary in attribute order and quoting style. ``DOTALL`` lets
# title text wrap across lines. Anchored on attributes rather than tag
# nesting to avoid false matches inside ``<script>`` blocks.
_TITLE_TAG_RE = re.compile(r"<title[^>]*>(.+?)</title>", re.IGNORECASE | re.DOTALL)
_OG_DESC_RE = re.compile(
    r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_META_DESC_RE = re.compile(
    r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_OG_TITLE_RE = re.compile(
    r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)

# Sitemap XML namespaces we care about. ``sm`` is the standard sitemap
# protocol; ``news`` is Google News' extension which carries the
# ``<news:title>`` + ``<news:publication_date>`` we need to skip per-
# article HTML scraping.
_SITEMAP_NS = {
    "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
    "news": "http://www.google.com/schemas/sitemap-news/0.9",
}

# ---------------------------------------------------------------------------
# Country config → source descriptor list
# ---------------------------------------------------------------------------


def _sources_from_config(country_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate the ``sources`` block into runnable descriptors.

    Each descriptor has ``name``, ``type``, ``urls``. Multiple entries
    with the same name (e.g. Habertürk listed twice — once as RSS, once
    as googlenews_sitemap) are kept distinct so each fetcher runs;
    duplicate articles deduplicate at insert time via ON CONFLICT(link).
    """
    descriptors: list[dict[str, Any]] = []
    for src in country_config.get("sources", []):
        if not src.get("enabled", True):
            continue
        descriptors.append(
            {
                "name": src["name"],
                "type": src.get("type", "rss"),
                "urls": list(src.get("urls") or []),
            }
        )
    return descriptors


# Backward-compat shim: callers still calling ``_feeds_from_config`` expect
# ``dict[name, urls]`` shape covering only RSS sources.
def _feeds_from_config(country_config: dict[str, Any]) -> dict[str, list[str]]:
    feeds: dict[str, list[str]] = {}
    for src in _sources_from_config(country_config):
        if src["type"] != "rss":
            continue
        if src["name"] in feeds:
            feeds[src["name"]].extend(src["urls"])
        else:
            feeds[src["name"]] = list(src["urls"])
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
# RSS fetcher
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


# ---------------------------------------------------------------------------
# Google News sitemap fetcher
# ---------------------------------------------------------------------------


def _http_get_bytes(url: str, timeout: int = _SITEMAP_TIMEOUT_SEC) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _HTTP_USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _sitemap_entry_to_item(url_el: ET.Element, source_name: str) -> dict[str, Any] | None:
    """Convert one ``<url>`` element from a Google News sitemap into our
    canonical item dict. Returns None when the entry lacks a usable
    title (plain non-news sitemaps fall through here).
    """
    loc = url_el.find("sm:loc", _SITEMAP_NS)
    title_el = url_el.find(".//news:title", _SITEMAP_NS)
    if loc is None or title_el is None:
        return None
    title = (title_el.text or "").strip()
    link = (loc.text or "").strip()
    if not title or not link:
        return None

    pub_date_el = url_el.find(".//news:publication_date", _SITEMAP_NS)
    if pub_date_el is not None and pub_date_el.text:
        published = pub_date_el.text.strip()
    else:
        published = datetime.now(timezone.utc).isoformat()

    return {
        "title": title,
        # Sitemap doesn't carry a summary; pipeline's preprocessor falls
        # back to title for langdetect / cleaning, which is fine.
        "summary": "",
        "source_name": source_name,
        "published_date": published,
        "link": link,
        "category": None,
    }


def fetch_sitemap(source_name: str, url: str) -> list[dict[str, Any]]:
    """Fetch a Google News sitemap and parse entries into items.

    Skips entries that don't carry ``<news:title>`` (i.e. plain
    sitemaps without the news extension). For those use
    ``fetch_html_sitemap`` instead — it pays the per-article HTML cost.
    """
    logger.info(f"Fetching sitemap [{source_name}]  {url}")
    try:
        body = _http_get_bytes(url)
        root = ET.fromstring(body)
    except Exception as exc:
        logger.error(f"[{source_name}] Sitemap fetch/parse failed: {exc}")
        return []

    items: list[dict[str, Any]] = []
    for url_el in root.findall(".//sm:url", _SITEMAP_NS):
        item = _sitemap_entry_to_item(url_el, source_name)
        if item is not None:
            items.append(item)

    if not items:
        logger.warning(
            f"[{source_name}] Sitemap parsed but contained 0 news-extended entries — "
            "likely a plain sitemap without news:title fields"
        )
    else:
        logger.success(f"[{source_name}] {len(items)} items from Google News sitemap")
    return items


# ---------------------------------------------------------------------------
# HTML sitemap fetcher (plain sitemap → per-article HTML scrape)
# ---------------------------------------------------------------------------


def _strip_title_suffix(title: str) -> str:
    """Trim trailing ``" | Section Name"`` style suffixes from HTML titles.

    Many Turkish news sites format ``<title>`` as
    ``"Article Title | Politika Haberleri"`` etc. Keeping just the
    article portion makes downstream NER/embedding cleaner.
    """
    if "|" in title:
        title = title.split("|")[0]
    return title.strip()


def _category_from_url(url: str) -> str | None:
    """Extract the first non-empty path segment as a category guess.

    For ``https://example.com/gundem/article-slug-123`` returns
    ``"gundem"``. Returns ``None`` for root URLs.
    """
    parts = urlparse(url).path.strip("/").split("/")
    return parts[0] or None if parts else None


def _http_get_text(url: str, timeout: int = _HTML_SCRAPE_TIMEOUT_SEC) -> str | None:
    """Best-effort article HTML fetch. Returns text body, or ``None`` on
    any error (404, timeout, malformed) so the caller can drop the entry
    silently."""
    try:
        body = _http_get_bytes(url, timeout=timeout)
    except Exception:
        return None
    return body.decode("utf-8", errors="replace")


def _extract_article_meta(page_html: str) -> tuple[str, str]:
    """Pull ``(title, summary)`` from a generic news HTML page.

    Title preference: ``og:title`` then ``<title>``. Both have any
    ``" | Section Name"`` suffix stripped (Yeni Şafak / Hürriyet /
    others all do this) and HTML entities unescaped (``&quot;``,
    ``&#x27;`` etc.) so downstream NER/embedding sees clean text.

    Summary preference: ``og:description`` then ``<meta name=description>``.
    Both fall back to empty string when missing.
    """
    og_title = _OG_TITLE_RE.search(page_html)
    if og_title:
        title = _strip_title_suffix(og_title.group(1))
    else:
        title_match = _TITLE_TAG_RE.search(page_html)
        title = _strip_title_suffix(title_match.group(1)) if title_match else ""

    desc_match = _OG_DESC_RE.search(page_html) or _META_DESC_RE.search(page_html)
    summary = desc_match.group(1).strip() if desc_match else ""

    return html.unescape(title), html.unescape(summary)


def _scrape_article(entry: dict[str, str], source_name: str) -> dict[str, Any] | None:
    """Fetch one article URL and assemble an item dict, or ``None`` on
    fetch failure / missing title."""
    html = _http_get_text(entry["url"])
    if not html:
        return None
    title, summary = _extract_article_meta(html)
    if not title:
        return None
    return {
        "title": title,
        "summary": summary,
        "source_name": source_name,
        "published_date": entry.get("lastmod") or datetime.now(timezone.utc).isoformat(),
        "link": entry["url"],
        "category": _category_from_url(entry["url"]),
    }


def fetch_html_sitemap(source_name: str, sitemap_url: str) -> list[dict[str, Any]]:
    """Fetch a plain sitemap, then concurrently scrape each article's HTML.

    Used when the outlet's RSS is capped and the sitemap lacks
    ``<news:title>`` (i.e. not Google News format). Issues one request
    per article via ``ThreadPoolExecutor`` (default 15 workers,
    10 s per-request timeout) — total wall time ≈ articles / workers ×
    avg latency, e.g. 200 articles / 15 / 0.5 s ≈ 7 s.

    Articles whose HTML is unreachable or whose title can't be parsed
    are silently dropped (returned count < discovered count).
    """
    logger.info(f"Discovering URLs [{source_name}]  {sitemap_url}")
    try:
        body = _http_get_bytes(sitemap_url)
        root = ET.fromstring(body)
    except Exception as exc:
        logger.error(f"[{source_name}] sitemap fetch/parse failed: {exc}")
        return []

    entries: list[dict[str, str]] = []
    for url_el in root.findall(".//sm:url", _SITEMAP_NS):
        loc = url_el.find("sm:loc", _SITEMAP_NS)
        if loc is None or not loc.text:
            continue
        lastmod_el = url_el.find("sm:lastmod", _SITEMAP_NS)
        entries.append(
            {
                "url": loc.text.strip(),
                "lastmod": (lastmod_el.text or "").strip()
                if lastmod_el is not None
                else "",
            }
        )

    if not entries:
        logger.warning(f"[{source_name}] sitemap had 0 URL entries")
        return []

    logger.info(
        f"[{source_name}] scraping {len(entries)} articles "
        f"({_HTML_SCRAPE_WORKERS} workers, {_HTML_SCRAPE_TIMEOUT_SEC}s timeout)..."
    )

    items: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=_HTML_SCRAPE_WORKERS) as pool:
        for result in pool.map(lambda e: _scrape_article(e, source_name), entries):
            if result is not None:
                items.append(result)

    dropped = len(entries) - len(items)
    if dropped:
        logger.warning(
            f"[{source_name}] scraped {len(items)}/{len(entries)} articles "
            f"({dropped} dropped: 4xx, timeout, or missing title)"
        )
    else:
        logger.success(f"[{source_name}] {len(items)} articles scraped from sitemap")
    return items


# ---------------------------------------------------------------------------
# Per-source dispatcher
# ---------------------------------------------------------------------------


def fetch_source(
    source_name: str,
    urls: list[str],
    source_type: str = "rss",
) -> list[dict[str, Any]]:
    """Fetch all URLs for one source, dedupe by link, return items.

    Dispatches on ``source_type``: ``rss`` (default, feedparser),
    ``googlenews_sitemap`` (XML namespace-aware parse, no scrape) or
    ``html_sitemap`` (plain sitemap + concurrent article HTML scrape).
    Multi-URL sources of any type get in-memory link dedupe.

    Dispatch via direct symbol reference (resolved at call time) so
    tests can ``monkeypatch.setattr(rss_collector, "fetch_sitemap", ...)``
    and have it picked up.
    """
    if not urls:
        return []

    if source_type == "googlenews_sitemap":
        fetcher = fetch_sitemap
    elif source_type == "html_sitemap":
        fetcher = fetch_html_sitemap
    else:
        fetcher = fetch_feed

    if len(urls) == 1:
        return fetcher(source_name, urls[0])

    aggregated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for url in urls:
        for item in fetcher(source_name, url):
            link = item.get("link") or ""
            if link and link in seen:
                continue
            if link:
                seen.add(link)
            aggregated.append(item)
    logger.success(
        f"[{source_name}] {len(aggregated)} unique items "
        f"after dedupe across {len(urls)} {source_type} feeds"
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
    """Collect news from configured sources and persist to PostgreSQL.

    Args:
        feeds: explicit ``name -> urls`` map (RSS only, backward-compat
               for callers that drove the collector directly). When
               omitted, sources come from the country config — that
               path supports mixed RSS / sitemap entries.
        date_str: ISO date; defaults to today.
        country: country slug or code (default ``"turkey"``). Phase 3
                 will plumb the real selection through the orchestrator.

    Returns:
        Number of new rows inserted.
    """
    date_str = date_str or date.today().isoformat()

    if feeds is None:
        config = load_country_config(country)
        sources = _sources_from_config(config)
        logger.info(
            f"Loaded {len(sources)} source descriptors from country config "
            f"[{config['country_code']}/{config['country_slug']}]"
        )
    else:
        # Backward-compat: dict path implies all-RSS, single descriptor per name.
        sources = [
            {"name": name, "type": "rss", "urls": list(urls)}
            for name, urls in feeds.items()
        ]

    all_items: list[dict[str, Any]] = []
    failed: list[str] = []
    seen_names: set[str] = set()

    for src in sources:
        items = fetch_source(src["name"], src["urls"], src["type"])
        if items:
            all_items.extend(items)
            seen_names.add(src["name"])
        else:
            failed.append(f"{src['name']}({src['type']})")

    if failed:
        logger.warning(
            f"{len(failed)} source feed(s) returned no items: " + ", ".join(failed)
        )

    inserted = insert_raw_items(all_items, date_str)

    logger.success(
        f"Collection complete — {inserted} new items inserted "
        f"from {len(seen_names)} distinct sources"
    )
    return inserted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect news from a country's RSS / sitemap sources")
    p.add_argument("--country", default="turkey", help="Country slug or code (default: turkey)")
    p.add_argument("--date", default=date.today().isoformat(), metavar="YYYY-MM-DD")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    collect_all(country=args.country, date_str=args.date)
    sys.exit(0)
