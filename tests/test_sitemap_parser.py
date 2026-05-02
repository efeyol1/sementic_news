"""Tests for the Google News sitemap parser in src.data.rss_collector.

Exercises ``_sitemap_entry_to_item`` directly with synthetic XML so the
suite stays offline. ``fetch_sitemap`` itself is a thin urllib + ET
wrapper that's easier to verify with one integration smoke than to mock.
"""

from __future__ import annotations

from xml.etree import ElementTree as ET

from src.data import rss_collector

# Sample fragment mirroring Sözcü's Google News sitemap structure.
_GOOGLENEWS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<urlset
    xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
    xmlns:news="http://www.google.com/schemas/sitemap-news/0.9">
  <url>
    <loc>https://example.com/article-1</loc>
    <news:news>
      <news:publication>
        <news:name>Example Source</news:name>
        <news:language>tr</news:language>
      </news:publication>
      <news:publication_date>2026-05-02T14:40:00+03:00</news:publication_date>
      <news:title>Example article title one</news:title>
    </news:news>
  </url>
  <url>
    <loc>https://example.com/article-2</loc>
    <news:news>
      <news:publication>
        <news:name>Example Source</news:name>
        <news:language>tr</news:language>
      </news:publication>
      <news:publication_date>2026-05-02T15:48:31+03:00</news:publication_date>
      <news:title>Example article title two</news:title>
    </news:news>
  </url>
  <url>
    <!-- Plain sitemap entry without news:title — must be skipped. -->
    <loc>https://example.com/article-3</loc>
    <lastmod>2026-05-02T16:00:00+03:00</lastmod>
  </url>
</urlset>
"""


def _parse_url_elements():
    root = ET.fromstring(_GOOGLENEWS_XML)
    return root.findall(".//sm:url", rss_collector._SITEMAP_NS)


def test_sitemap_entry_extracts_title_link_and_date():
    url_el = _parse_url_elements()[0]
    item = rss_collector._sitemap_entry_to_item(url_el, "Example Source")
    assert item is not None
    assert item["title"] == "Example article title one"
    assert item["link"] == "https://example.com/article-1"
    assert item["source_name"] == "Example Source"
    assert item["published_date"] == "2026-05-02T14:40:00+03:00"
    # Sitemap doesn't carry a summary; we leave it empty so downstream
    # preprocessing falls back to title for langdetect / cleaning.
    assert item["summary"] == ""
    assert item["category"] is None


def test_sitemap_entry_skipped_when_news_title_missing():
    """Plain sitemaps (no news: namespace) should be silently skipped —
    they need per-article HTML scraping which this handler doesn't do."""
    plain_url_el = _parse_url_elements()[2]
    assert rss_collector._sitemap_entry_to_item(plain_url_el, "Example Source") is None


def test_sitemap_entry_handles_blank_title_or_link():
    """Defensive: even a partially-formed news entry should be rejected
    rather than producing an item with empty fields downstream."""
    bad_xml = """<?xml version="1.0"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
            xmlns:news="http://www.google.com/schemas/sitemap-news/0.9">
      <url>
        <loc></loc>
        <news:news>
          <news:title>Has title but blank loc</news:title>
        </news:news>
      </url>
    </urlset>"""
    url_el = ET.fromstring(bad_xml).findall(".//sm:url", rss_collector._SITEMAP_NS)[0]
    assert rss_collector._sitemap_entry_to_item(url_el, "Source") is None


def test_fetch_source_dispatches_to_sitemap_for_googlenews_type(monkeypatch):
    """``fetch_source`` should route googlenews_sitemap entries to
    ``fetch_sitemap`` rather than feedparser-based ``fetch_feed``."""
    rss_called = []
    sitemap_called = []
    monkeypatch.setattr(
        rss_collector, "fetch_feed",
        lambda name, url: rss_called.append((name, url)) or [],
    )
    monkeypatch.setattr(
        rss_collector, "fetch_sitemap",
        lambda name, url: sitemap_called.append((name, url)) or [{"link": "x", "title": "t",
                                                                  "summary": "", "source_name": name,
                                                                  "published_date": "", "category": None}],
    )
    items = rss_collector.fetch_source("X", ["https://x/sitemap.xml"], source_type="googlenews_sitemap")
    assert items and items[0]["title"] == "t"
    assert sitemap_called and not rss_called


def test_sources_from_config_preserves_type_per_entry():
    """Multi-type configs (Habertürk has rss AND googlenews_sitemap entries
    in turkey.yaml) should yield two distinct descriptors, not get merged
    into one RSS-only entry."""
    config = {
        "sources": [
            {"name": "X", "type": "rss", "urls": ["https://x/rss"]},
            {"name": "X", "type": "googlenews_sitemap", "urls": ["https://x/news.xml"]},
            {"name": "Y", "type": "rss", "urls": ["https://y/rss"], "enabled": False},
        ]
    }
    descriptors = rss_collector._sources_from_config(config)
    # Y is disabled → dropped.
    assert len(descriptors) == 2
    types = {(d["name"], d["type"]) for d in descriptors}
    assert types == {("X", "rss"), ("X", "googlenews_sitemap")}


# ---------------------------------------------------------------------------
# HTML scraping helpers (used by html_sitemap fetcher)
# ---------------------------------------------------------------------------


def test_strip_title_suffix_removes_section_after_pipe():
    assert rss_collector._strip_title_suffix("Article Title | Politika Haberleri") == "Article Title"
    assert rss_collector._strip_title_suffix("No Suffix Here") == "No Suffix Here"
    assert rss_collector._strip_title_suffix("  whitespace  | section  ") == "whitespace"


def test_category_from_url_picks_first_path_segment():
    assert rss_collector._category_from_url("https://example.com/gundem/article-slug-123") == "gundem"
    assert rss_collector._category_from_url("https://example.com/spor/futbol/match") == "spor"
    # Root URL: no category to derive.
    assert rss_collector._category_from_url("https://example.com/") is None


def test_extract_article_meta_prefers_og_title_and_strips_suffix():
    page = """<html><head>
        <meta property="og:title" content="Real Title | Section Name">
        <title>Fallback Title | Section Name</title>
        <meta property="og:description" content="The article summary.">
    </head></html>"""
    title, summary = rss_collector._extract_article_meta(page)
    assert title == "Real Title"
    assert summary == "The article summary."


def test_extract_article_meta_falls_back_to_title_tag():
    """When og:title is missing, the <title> tag is used (also suffix-stripped)."""
    page = """<html><head>
        <title>Only Title Tag | Some Section</title>
        <meta name="description" content="Plain description.">
    </head></html>"""
    title, summary = rss_collector._extract_article_meta(page)
    assert title == "Only Title Tag"
    assert summary == "Plain description."


def test_extract_article_meta_unescapes_html_entities():
    """``&quot;`` / ``&#x27;`` come through in raw HTML and would pollute
    NER / embeddings if left intact."""
    page = """<html><head>
        <meta property="og:title" content="It&#x27;s a &quot;Test&quot;">
        <meta property="og:description" content="Quote: &quot;hello&quot;.">
    </head></html>"""
    title, summary = rss_collector._extract_article_meta(page)
    assert title == "It's a \"Test\""
    assert summary == 'Quote: "hello".'


def test_extract_article_meta_returns_empty_when_no_meta():
    title, summary = rss_collector._extract_article_meta("<html><body>No metadata.</body></html>")
    assert title == ""
    assert summary == ""
