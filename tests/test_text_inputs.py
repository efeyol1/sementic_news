from __future__ import annotations

from src.analysis.text_inputs import (
    EMBEDDING_BODY_CHARS,
    NER_BODY_CHARS,
    SENTIMENT_BODY_CHARS,
    build_clustering_text,
    build_embedding_text,
    build_ner_text,
    build_sentiment_text,
)


def test_build_sentiment_text_prefers_body_snippet_with_fallback():
    item = {
        "cleaned_title": "başlık",
        "cleaned_summary": "özet",
        "cleaned_article_text": "a" * (SENTIMENT_BODY_CHARS + 20),
    }

    text = build_sentiment_text(item)

    assert text.startswith("başlık. özet. ")
    assert len(text.split(". ")[-1]) == SENTIMENT_BODY_CHARS


def test_build_analysis_text_falls_back_to_title_summary_without_body():
    item = {"cleaned_title": "başlık", "cleaned_summary": "özet"}

    assert build_sentiment_text(item) == "başlık. özet"
    assert build_clustering_text(item) == "başlık. özet"


def test_different_analysis_steps_use_different_body_windows():
    item = {
        "cleaned_title": "başlık",
        "cleaned_summary": "",
        "cleaned_article_text": "b" * (NER_BODY_CHARS + 20),
    }

    embedding_body = build_embedding_text(item).split(". ")[-1]
    ner_body = build_ner_text(item).split(". ")[-1]

    assert len(embedding_body) == EMBEDDING_BODY_CHARS
    assert len(ner_body) == NER_BODY_CHARS
