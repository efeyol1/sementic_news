from __future__ import annotations

import json

import src.data.article_fetcher as article_fetcher
from src.data.article_fetcher import load_discovery_metadata


def test_load_discovery_metadata_prefers_category_over_general(tmp_path):
    report = {
        "records": [
            {
                "canonical_url": "https://example.com/news/1?utm_source=x",
                "canonical_category": "other",
                "discovery_role": "general_discovery",
                "discovery_url": "https://example.com/rss",
            },
            {
                "canonical_url": "https://example.com/news/1",
                "canonical_category": "economy_finance",
                "discovery_role": "category",
                "discovery_url": "https://example.com/rss/economy",
            },
        ]
    }
    path = tmp_path / "discovered_urls_2026-05-05.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    metadata = load_discovery_metadata("2026-05-05", path)

    assert metadata["https://example.com/news/1"]["canonical_category"] == "economy_finance"
    assert metadata["https://example.com/news/1"]["discovery_role"] == "category"


def test_fetch_articles_forwards_per_source_limit(monkeypatch):
    captured = {}

    def fake_fetch_for_article_fetching(date_str, limit, retry_failed, per_source_limit):
        captured.update(
            {
                "date_str": date_str,
                "limit": limit,
                "retry_failed": retry_failed,
                "per_source_limit": per_source_limit,
            }
        )
        return []

    monkeypatch.setattr(article_fetcher, "load_discovery_metadata", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(article_fetcher, "fetch_for_article_fetching", fake_fetch_for_article_fetching)

    result = article_fetcher.fetch_articles(
        date_str="2026-05-05",
        limit=50,
        retry_failed=True,
        per_source_limit=5,
        ensure_schema=False,
    )

    assert result == {}
    assert captured == {
        "date_str": "2026-05-05",
        "limit": 50,
        "retry_failed": True,
        "per_source_limit": 5,
    }
