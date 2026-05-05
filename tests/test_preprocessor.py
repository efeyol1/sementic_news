from __future__ import annotations

from src.data.preprocessor import process_item


def test_process_item_keeps_title_only_sitemap_records():
    item = {
        "id": 1,
        "title": "Merkez Bankası faiz kararını açıkladı",
        "summary": "",
        "published_date": "2026-05-05T08:00:00+00:00",
    }

    result = process_item(item)

    assert result is not None
    assert result["cleaned_title"] == "merkez bankası faiz kararını açıkladı"
    assert result["cleaned_summary"] == ""
    assert result["char_count"] == len("Merkez Bankası faiz kararını açıkladı")


def test_process_item_drops_items_with_no_usable_text():
    item = {
        "id": 1,
        "title": "Kısa",
        "summary": "",
        "published_date": "2026-05-05T08:00:00+00:00",
    }

    assert process_item(item) is None


def test_process_item_cleans_article_text():
    item = {
        "id": 1,
        "title": "Kısa",
        "summary": "",
        "article_text": "<p>Bu haberin gövde metni yeterince uzun ve anlamlı.</p>",
        "published_date": "2026-05-05T08:00:00+00:00",
    }

    result = process_item(item)

    assert result is not None
    assert result["cleaned_article_text"] == "bu haberin gövde metni yeterince uzun ve anlamlı."
    assert result["char_count"] > len(result["cleaned_title"])
