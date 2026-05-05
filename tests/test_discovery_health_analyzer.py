from __future__ import annotations

from analyze_discovery_health import format_summary, summarize_report


def test_summarize_report_flags_low_warning_and_error_feeds():
    report = {
        "_path": "data/discovery/source_health_2026-05-05.json",
        "date": "2026-05-05",
        "totals": {
            "feed_checks": 2,
            "discovered_url_records": 12,
            "unique_discovered_urls": 10,
            "inserted_news_items": 8,
            "failed_fetch_count": 1,
            "failed_parse_count": 1,
        },
        "sources": {
            "Example": {
                "discovery_checked_count": 2,
                "raw_item_count": 12,
                "unique_url_count": 10,
                "low_categories": ["economy_finance"],
                "failed_fetch_count": 1,
                "failed_parse_count": 1,
                "categories": {
                    "economy_finance": 8,
                    "other": 4,
                },
            }
        },
        "feeds": [
            {
                "source_name": "Example",
                "source_type": "rss",
                "canonical_category": "economy_finance",
                "unique_url_count": 1,
                "min_expected_items": 2,
                "below_min_expected": True,
                "parse_ok": True,
                "error": None,
                "warning": "encoding warning",
                "discovery_url": "https://example.com/rss",
            },
            {
                "source_name": "Example",
                "source_type": "rss",
                "canonical_category": "other",
                "unique_url_count": 0,
                "min_expected_items": 2,
                "below_min_expected": True,
                "parse_ok": False,
                "error": "timeout",
                "warning": None,
                "discovery_url": "https://example.com/broken",
            },
        ],
    }

    summary = summarize_report(report)
    assert summary["sources"][0]["source"] == "Example"
    assert summary["sources"][0]["dedupe_loss"] == 2
    assert summary["sources"][0]["other_pct"] == 33.3
    assert len(summary["warning_feeds"]) == 1
    assert len(summary["error_feeds"]) == 1
    assert len(summary["low_feeds"]) == 2

    text = format_summary(summary)
    assert "Discovery health: 2026-05-05" in text
    assert "Below minimum feeds:" in text
    assert "Warning feeds:" in text
    assert "Error feeds:" in text
