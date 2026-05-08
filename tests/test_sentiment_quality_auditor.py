from __future__ import annotations

from datetime import datetime, timezone

from audit_sentiment_quality import format_summary, summarize_rows


def test_summarize_rows_flags_stale_low_neutral_and_suspicious_labels():
    before_body = datetime(2026, 5, 6, 8, 0, tzinfo=timezone.utc)
    after_body = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    rows = [
        {
            "id": 1,
            "source_name": "A",
            "category": "security_justice",
            "cleaned_title": "depremde 146 kisi hayatini kaybetti",
            "cleaned_summary": "",
            "cleaned_article_text": "deprem nedeniyle yikim buyuk",
            "sentiment_label": "positive",
            "sentiment_score": 0.98,
            "analyzed_at": before_body,
            "article_fetched_at": after_body,
        },
        {
            "id": 2,
            "source_name": "A",
            "category": "economy_finance",
            "cleaned_title": "borsa gunu yukselisle kapatti",
            "cleaned_summary": "",
            "cleaned_article_text": "piyasalarda sakin seyir izlendi",
            "sentiment_label": "positive",
            "sentiment_score": 0.9,
            "analyzed_at": after_body,
            "article_fetched_at": before_body,
        },
        {
            "id": 3,
            "source_name": "B",
            "category": "other",
            "cleaned_title": "bakanlik aciklama yapti",
            "cleaned_summary": "",
            "cleaned_article_text": None,
            "sentiment_label": None,
            "sentiment_score": None,
            "analyzed_at": None,
            "article_fetched_at": None,
        },
    ]

    summary = summarize_rows(rows, min_neutral_pct=10.0, high_confidence=0.85)

    assert summary["totals"]["total"] == 3
    assert summary["totals"]["scored"] == 2
    assert summary["totals"]["with_body"] == 2
    assert summary["totals"]["stale_after_body"] == 1
    assert summary["totals"]["scored_after_body"] == 1
    assert summary["totals"]["label_counts"] == {
        "negative": 0,
        "neutral": 0,
        "positive": 2,
    }
    assert "stale_sentiment_after_body_fetch" in summary["warnings"]
    assert "low_neutral_share" in summary["warnings"]
    assert "high_confidence_label_cue_mismatch" in summary["warnings"]
    assert summary["suspicious"]["counts"]["positive_with_negative_cue"] == 1

    text = format_summary(summary, "2026-05-06")
    assert "Sentiment quality audit: 2026-05-06" in text
    assert "stale after body fetch: 1" in text
    assert "positive_with_negative_cue" in text


def test_cue_match_handles_uppercase_turkish_diacritics():
    rows = [
        {
            "id": 10,
            "source_name": "X",
            "cleaned_title": "BAKAN İSTİFA ETTİ",
            "cleaned_summary": "",
            "cleaned_article_text": "",
            "sentiment_label": "positive",
            "sentiment_score": 0.97,
            "analyzed_at": None,
            "article_fetched_at": None,
        },
    ]
    summary = summarize_rows(rows)
    assert summary["suspicious"]["counts"].get("positive_with_negative_cue") == 1


def test_kaza_cue_does_not_collide_with_kazandi():
    rows = [
        {
            "id": 11,
            "source_name": "Y",
            "cleaned_title": "Galatasaray maçı kazandı",
            "cleaned_summary": "",
            "cleaned_article_text": "",
            "sentiment_label": "positive",
            "sentiment_score": 0.97,
            "analyzed_at": None,
            "article_fetched_at": None,
        },
    ]
    summary = summarize_rows(rows)
    assert summary["suspicious"]["counts"] == {}
