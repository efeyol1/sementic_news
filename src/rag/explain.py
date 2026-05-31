"""Entry point for the RAG explainability layer (Sprint 11).

Orchestrates the cache → retrieve → context → generate → cache pipeline for
one entity profile. ``explain_entity`` is the single function the API (and
``src.rag`` re-export) calls.

Cache semantics mirror the migration docstring: a row is fetched by its
identity key (country/canonical/type/window/end_date/lang) and counts as a
HIT only when its stored ``profile_hash`` still equals the hash of the
current profile. A mismatch (the Sprint 7 aggregator rewrote the profile) is
STALE → regenerate + upsert. A cache *write* failure never fails the request:
the freshly generated payload is returned regardless.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.db.queries import (
    fetch_explanation_cache,
    upsert_explanation_cache,
)
from src.rag.config import max_snippets
from src.rag.context import build_context, profile_hash
from src.rag.generator import get_generator
from src.rag.retriever import retrieve_snippets


def explain_entity(
    profile: dict[str, Any],
    country_config: dict[str, Any],
    lang: str,
) -> tuple[dict[str, Any], bool]:
    """Return ``(payload, cached)`` for one entity profile.

    ``payload`` is the JSONB-serializable ``Explanation`` dict; ``cached`` is
    ``True`` only on a fresh cache hit (``profile_hash`` matches).
    """
    h = profile_hash(profile)
    cached_row = fetch_explanation_cache(
        profile["country_code"],
        profile["canonical"],
        profile["entity_type"],
        profile["window_days"],
        str(profile["end_date"]),
        lang,
    )
    if cached_row and cached_row.get("profile_hash") == h:
        return cached_row["explanation"], True

    # Miss or stale → regenerate.
    snippets = retrieve_snippets(profile, max_snippets=max_snippets(country_config))
    ctx = build_context(profile, snippets, lang)
    gen = get_generator(country_config)
    expl = gen.generate(ctx)
    payload = expl.to_payload()

    # Don't cache a *transient* Anthropic→extractive fallback (API error,
    # rate limit, truncated JSON): freezing it would serve the degraded
    # result until the profile changes. Genuine extractive (no key/SDK) is
    # cached normally.
    if getattr(gen, "backend", "") == "anthropic" and payload["backend"] != "anthropic":
        logger.warning("Anthropic→extractive fallback — not caching; next call retries Claude.")
        return payload, False

    try:
        upsert_explanation_cache(
            {
                "country_code": profile["country_code"],
                "wikidata_qid": profile.get("wikidata_qid"),
                "canonical": profile["canonical"],
                "entity_type": profile["entity_type"],
                "window_days": profile["window_days"],
                "end_date": str(profile["end_date"]),
                "profile_hash": h,
                "backend": payload["backend"],
                "model": payload.get("model"),
                "lang": lang,
                "explanation": payload,
            }
        )
    except Exception as exc:  # cache write must never fail the request
        logger.warning(f"explanation cache upsert failed: {exc}")

    return payload, False
