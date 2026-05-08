from __future__ import annotations

import csv

from export_sentiment_review_sample import build_sample, write_csv


def test_build_sample_is_source_label_stratified_and_review_ready(tmp_path):
    rows = [
        {
            "id": 1,
            "source_name": "A",
            "category": "other",
            "sentiment_label": "positive",
            "sentiment_score": 0.91,
            "cleaned_title": "baslik 1",
            "cleaned_summary": "ozet 1",
            "cleaned_article_text": "x" * 20,
            "link": "https://example.com/1",
        },
        {
            "id": 2,
            "source_name": "A",
            "category": "other",
            "sentiment_label": "positive",
            "sentiment_score": 0.92,
            "cleaned_title": "baslik 2",
            "cleaned_summary": "ozet 2",
            "cleaned_article_text": "y" * 20,
            "link": "https://example.com/2",
        },
        {
            "id": 3,
            "source_name": "A",
            "category": "other",
            "sentiment_label": "negative",
            "sentiment_score": 0.88,
            "cleaned_title": "baslik 3",
            "cleaned_summary": "ozet 3",
            "cleaned_article_text": "z" * 20,
            "link": "https://example.com/3",
        },
        {
            "id": 4,
            "source_name": "B",
            "category": "economy_finance",
            "sentiment_label": "neutral",
            "sentiment_score": 0.77,
            "cleaned_title": "baslik 4",
            "cleaned_summary": "ozet 4",
            "cleaned_article_text": "body text",
            "link": "https://example.com/4",
        },
        {
            "id": 5,
            "source_name": "B",
            "category": "economy_finance",
            "sentiment_label": None,
            "sentiment_score": None,
            "cleaned_title": "not scored",
            "cleaned_summary": "",
            "cleaned_article_text": "",
            "link": "",
        },
    ]

    sample = build_sample(rows, per_source_label=1, seed=1, body_chars=5)

    assert len(sample) == 3
    assert {row["current_label"] for row in sample} == {"negative", "neutral", "positive"}
    assert all(row["reviewed_label"] == "" for row in sample)
    assert all(len(row["body_snippet"]) <= 5 for row in sample)

    output = tmp_path / "review.csv"
    write_csv(sample, output)

    with open(output, encoding="utf-8", newline="") as fh:
        written = list(csv.DictReader(fh))

    assert len(written) == 3
    assert "reviewed_label" in written[0]
    assert "review_notes" in written[0]
