from __future__ import annotations

import json

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
