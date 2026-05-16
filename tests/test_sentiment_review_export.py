from __future__ import annotations

import csv
import sys

from export_sentiment_review_sample import build_sample, main, write_csv


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
    assert all(row["review_status"] == "pending" for row in sample)
    assert all(row["label_source"] == "" for row in sample)
    assert all(len(row["body_snippet"]) <= 5 for row in sample)
    assert {row["confidence_bucket"] for row in sample} == {"high"}
    assert {row["has_body"] for row in sample} == {"true"}

    output = tmp_path / "review.csv"
    write_csv(sample, output)

    with open(output, encoding="utf-8", newline="") as fh:
        written = list(csv.DictReader(fh))

    assert len(written) == 3
    assert "reviewed_label" in written[0]
    assert "review_notes" in written[0]
    assert "confidence_bucket" in written[0]
    assert "review_status" in written[0]


def test_main_resolves_country_and_scopes_default_output(monkeypatch, tmp_path, capsys):
    import export_sentiment_review_sample as script

    captured = {}
    monkeypatch.setattr(script, "_DEFAULT_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(
        script,
        "load_country_config",
        lambda country: {
            "country_code": "DE",
            "country_slug": "germany",
            "language": "de",
        },
    )

    def fake_fetch(date_str, country_code="TR"):
        captured["date"] = date_str
        captured["country_code"] = country_code
        return []

    monkeypatch.setattr(script, "fetch_sentiment_quality_rows", fake_fetch)
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--country", "germany", "--date", "2026-05-08"],
    )

    assert main() == 0
    assert captured == {"date": "2026-05-08", "country_code": "DE"}
    assert (tmp_path / "sentiment_review_germany_2026-05-08.csv").exists()
    assert "for DE" in capsys.readouterr().out


def test_main_supports_date_range(monkeypatch, tmp_path):
    import export_sentiment_review_sample as script

    requested: list[str] = []
    monkeypatch.setattr(script, "_DEFAULT_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(
        script,
        "load_country_config",
        lambda country: {
            "country_code": "DE",
            "country_slug": "germany",
            "language": "de",
        },
    )

    def fake_fetch(date_str, country_code="TR"):
        requested.append(date_str)
        return [
            {
                "id": len(requested),
                "source_name": "A",
                "category": "other",
                "sentiment_label": "neutral",
                "sentiment_score": 0.5,
                "cleaned_title": "titel",
                "cleaned_summary": "zusammenfassung",
                "cleaned_article_text": "",
                "link": "https://example.com",
            }
        ]

    monkeypatch.setattr(script, "fetch_sentiment_quality_rows", fake_fetch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--country",
            "germany",
            "--date-from",
            "2026-05-08",
            "--date-to",
            "2026-05-09",
        ],
    )

    assert main() == 0
    assert requested == ["2026-05-08", "2026-05-09"]
    assert (tmp_path / "sentiment_review_germany_2026-05-08_2026-05-09.csv").exists()
