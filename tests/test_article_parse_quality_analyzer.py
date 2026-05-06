from __future__ import annotations

from analyze_article_parse_quality import format_summary, summarize_rows


def test_summarize_rows_reports_source_quality_and_attention():
    rows = [
        {
            "source_name": "Good",
            "category": "economy_finance",
            "parse_status": "parsed",
            "article_chars": 1000,
        },
        {
            "source_name": "Good",
            "category": "economy_finance",
            "parse_status": "parsed",
            "article_chars": 2000,
        },
        {
            "source_name": "Bad",
            "category": "other",
            "parse_status": "parsed",
            "article_chars": 300,
        },
        {
            "source_name": "Bad",
            "category": "other",
            "parse_status": "empty",
            "article_chars": 0,
        },
        {
            "source_name": "Bad",
            "category": "other",
            "parse_status": None,
            "article_chars": 0,
        },
    ]

    summary = summarize_rows(rows, min_fetched=2, min_parsed_pct=60.0)

    assert summary["totals"]["total"] == 5
    assert summary["totals"]["fetched"] == 4
    assert summary["totals"]["coverage_pct"] == 80.0
    assert summary["totals"]["parsed_pct_fetched"] == 75.0

    bad = next(row for row in summary["sources"] if row["source"] == "Bad")
    assert bad["fetched"] == 2
    assert bad["statuses"]["unfetched"] == 1
    assert bad["parsed_pct_fetched"] == 50.0
    assert bad["needs_attention"] is True

    good = next(row for row in summary["sources"] if row["source"] == "Good")
    assert good["avg_chars"] == 1500.0
    assert good["needs_attention"] is False

    text = format_summary(summary, "2026-05-05")
    assert "Article parse quality: 2026-05-05" in text
    assert "Needs attention:" in text
    assert "Bad" in text
