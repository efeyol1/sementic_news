from __future__ import annotations

import json

from src.data.canonical_categories import (
    infer_category_from_url,
    infer_discovery_role,
    normalize_article_url,
)
from src.data.rss_validator import build_discovery_record, build_feed_health, write_discovery_reports


def test_url_inference_marks_general_and_category_feeds():
    assert infer_discovery_role("https://www.haberturk.com/rss") == "general_discovery"
    assert infer_category_from_url("https://www.haberturk.com/rss/ekonomi.xml") == "economy_finance"
    assert infer_discovery_role("https://www.haberturk.com/rss/ekonomi.xml") == "category"
    assert infer_category_from_url("https://www.milliyet.com.tr/rss/rssnew/ekonomirss.xml") == "economy_finance"
    assert infer_category_from_url("https://www.cnnturk.com/feed/rss/turkiye/news") == "politics_governance"


def test_normalize_article_url_removes_tracking_params():
    url = "HTTPS://Example.com/path/article/?utm_source=x&b=2&a=1#section"
    assert normalize_article_url(url) == "https://example.com/path/article?a=1&b=2"


def test_write_discovery_reports(tmp_path):
    item = {
        "link": "https://example.com/news/1",
        "title": "Title",
        "summary": "Summary",
        "published_date": "2026-05-05T08:00:00+00:00",
        "category": "Ekonomi",
    }
    discovered = [
        build_discovery_record(
            item=item,
            source_name="Example",
            source_type="rss",
            discovery_url="https://example.com/rss/ekonomi.xml",
            discovery_role="category",
            canonical_category="economy_finance",
            canonical_url="https://example.com/news/1",
        )
    ]
    health = [
        build_feed_health(
            source_name="Example",
            source_type="rss",
            discovery_url="https://example.com/rss/ekonomi.xml",
            discovery_role="category",
            canonical_category="economy_finance",
            min_expected_items=2,
            http_status=200,
            parse_ok=True,
            item_count=1,
            unique_url_count=1,
            latest_date="2026-05-05T08:00:00+00:00",
        )
    ]

    discovered_path, health_path = write_discovery_reports(
        date_str="2026-05-05",
        country_code="TR",
        country_slug="turkey",
        feed_health=health,
        discovered_records=discovered,
        inserted_count=1,
        report_dir=tmp_path,
    )

    discovered_payload = json.loads(discovered_path.read_text(encoding="utf-8"))
    health_payload = json.loads(health_path.read_text(encoding="utf-8"))
    assert discovered_payload["unique_urls"] == 1
    assert health_payload["sources"]["Example"]["low_categories"] == ["economy_finance"]
    assert health_payload["sources"]["Example"]["final_parsed_article_count"] == 1
