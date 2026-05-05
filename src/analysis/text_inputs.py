"""Shared text construction for analysis modules.

All analysis steps should make the same fallback decision: prefer article body
when it exists, but keep title + summary working for RSS-only rows.
"""

from __future__ import annotations

from typing import Any

SENTIMENT_BODY_CHARS = 800
CLUSTERING_BODY_CHARS = 1800
EMBEDDING_BODY_CHARS = 1800
NER_BODY_CHARS = 3000


def _clean_part(value: Any) -> str:
    return str(value or "").strip()


def _snippet(value: Any, max_chars: int) -> str:
    text = _clean_part(value)
    if max_chars <= 0:
        return ""
    return text[:max_chars].strip()


def build_analysis_text(item: dict[str, Any], body_chars: int) -> str:
    """Return title + summary + optional article-body snippet."""
    title = _clean_part(item.get("cleaned_title") or item.get("title"))
    summary = _clean_part(item.get("cleaned_summary") or item.get("summary"))
    body = _snippet(item.get("cleaned_article_text") or item.get("article_text"), body_chars)
    return ". ".join(part for part in (title, summary, body) if part)


def build_sentiment_text(item: dict[str, Any]) -> str:
    return build_analysis_text(item, body_chars=SENTIMENT_BODY_CHARS)


def build_clustering_text(item: dict[str, Any]) -> str:
    return build_analysis_text(item, body_chars=CLUSTERING_BODY_CHARS)


def build_embedding_text(item: dict[str, Any]) -> str:
    return build_analysis_text(item, body_chars=EMBEDDING_BODY_CHARS)


def build_ner_text(item: dict[str, Any]) -> str:
    return build_analysis_text(item, body_chars=NER_BODY_CHARS)
