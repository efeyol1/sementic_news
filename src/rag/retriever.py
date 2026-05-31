"""Entity-scoped evidence retrieval for the RAG layer.

Structured-first: pull in-country, in-window articles that mention the
entity (``fetch_entity_snippet_rows``), extract a short snippet centered
on a top-collocate lemma, and rank by how many top collocates the snippet
covers (then recency). This makes the citations *real* — every snippet we
feed the generator actually contains the words the explanation will cite.

No torch import at module load (mirrors ``vector_store._get_embed_model``);
the API process stays lightweight. A dense-rerank leg can be added later
behind an ImportError guard, but structured retrieval is the always-on path.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any

from loguru import logger

from src.db.queries import fetch_entity_snippet_rows
from src.rag.context import Snippet

_SNIPPET_CHARS = 320
_TOP_LEMMA_LOOKUP = 15   # how many of the profile's top collocates to match on
_WS = re.compile(r"\s+")


def _text_of(row: dict[str, Any]) -> str:
    """Best available body text, fallback chain."""
    for key in ("cleaned_article_text", "cleaned_summary", "summary", "title"):
        val = row.get(key)
        if val and str(val).strip():
            return _WS.sub(" ", str(val)).strip()
    return ""


def _iso_date(row: dict[str, Any]) -> str:
    d = row.get("collected_date") or row.get("published_date")
    if d is None:
        return ""
    return d.isoformat()[:10] if hasattr(d, "isoformat") else str(d)[:10]


def _extract_snippet(text: str, focus: str | None) -> str:
    """~320-char window around the first occurrence of ``focus`` (case-
    insensitive), trimmed to word boundaries. Falls back to the head of
    the text when ``focus`` is absent."""
    if not text:
        return ""
    if len(text) <= _SNIPPET_CHARS:
        return text
    idx = text.lower().find(focus.lower()) if focus else -1
    if idx < 0:
        return text[:_SNIPPET_CHARS].rsplit(" ", 1)[0] + "…"
    half = _SNIPPET_CHARS // 2
    start = max(0, idx - half)
    end = min(len(text), idx + half)
    chunk = text[start:end]
    # Trim partial words at both ends.
    if start > 0:
        chunk = "…" + chunk.split(" ", 1)[-1]
    if end < len(text):
        chunk = chunk.rsplit(" ", 1)[0] + "…"
    return chunk.strip()


def retrieve_snippets(
    profile: dict[str, Any],
    *,
    max_snippets: int = 8,
    candidate_limit: int = 40,
) -> list[Snippet]:
    """Return up to ``max_snippets`` ranked, collocate-bearing snippets.

    Reads ``window_days`` / ``end_date`` / qid / canonical / top_collocates
    off the profile row. Snippets are ranked by collocate coverage, then by
    candidate recency (the query already returns recency-ordered rows).
    """
    window_days = int(profile["window_days"])
    end = profile["end_date"]
    end_date = end if hasattr(end, "isoformat") else date.fromisoformat(str(end))
    start_date = end_date - timedelta(days=window_days - 1)

    top_lemmas = [
        str(c.get("lemma", "")).lower().strip()
        for c in (profile.get("top_collocates") or [])[:_TOP_LEMMA_LOOKUP]
        if c.get("lemma")
    ]
    entity_text = str(profile.get("canonical") or "")

    try:
        rows = fetch_entity_snippet_rows(
            country_code=profile["country_code"],
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            qid=profile.get("wikidata_qid"),
            canonical=profile.get("canonical"),
            limit=candidate_limit,
        )
    except Exception as exc:  # fail-soft: empty retrieval, never crash the request
        logger.warning(f"snippet retrieval failed: {exc}")
        return []

    scored: list[tuple[int, int, Snippet]] = []  # (coverage, -recency_rank, snippet)
    for rank, row in enumerate(rows):
        text = _text_of(row)
        if not text:
            continue
        lower = text.lower()
        matched = [lem for lem in top_lemmas if lem and lem in lower]
        focus = matched[0] if matched else (entity_text or None)
        snippet_text = _extract_snippet(text, focus)
        if not snippet_text:
            continue
        scored.append(
            (
                len(matched),
                -rank,  # earlier (more recent) candidates win ties
                Snippet(
                    id="",  # assigned after ranking
                    text=snippet_text,
                    source_name=row.get("source_name") or "?",
                    date=_iso_date(row),
                    link=row.get("link"),
                    article_id=int(row["article_id"]),
                    matched_collocates=matched,
                ),
            )
        )

    # Coverage desc, then recency. Snippets with ≥1 matched collocate first.
    scored.sort(key=lambda t: (-t[0], t[1]))
    chosen = [s for _, _, s in scored[:max_snippets]]
    # Assign stable citation ids.
    return [
        Snippet(
            id=f"S{i + 1}",
            text=s.text,
            source_name=s.source_name,
            date=s.date,
            link=s.link,
            article_id=s.article_id,
            matched_collocates=s.matched_collocates,
        )
        for i, s in enumerate(chosen)
    ]
