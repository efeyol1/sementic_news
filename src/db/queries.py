"""All database query functions for the pipeline and API."""

from __future__ import annotations

import json
from typing import Any

import psycopg2.extras
from loguru import logger

from src.db.client import get_conn, retry_on_connection_loss  # noqa: E402

# ---------------------------------------------------------------------------
# RSS Collector
# ---------------------------------------------------------------------------

@retry_on_connection_loss()
def insert_raw_items(
    items: list[dict[str, Any]],
    date_str: str,
    country_code: str = "TR",
    language: str = "tr",
) -> int:
    """Bulk-insert raw RSS items; skip duplicates by link. Returns rows inserted.

    ``country_code`` / ``language`` default to the V1 Turkey pilot values so
    legacy callers stay correct; Phase 3 will plumb the real selection through
    from the country config.
    """
    if not items:
        return 0
    rows = [
        (
            date_str,
            item.get("title") or None,
            item.get("summary") or None,
            item.get("source_name") or "",
            item.get("published_date") or None,
            item.get("link") or None,  # empty string → NULL (avoids false conflicts)
            item.get("category") or None,
            country_code,
            language,
        )
        for item in items
    ]
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM news_items WHERE collected_date = %s AND country_code = %s",
                (date_str, country_code),
            )
            before = cur.fetchone()[0]
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO news_items
                    (collected_date, title, summary, source_name, published_date, link, category,
                     country_code, language)
                VALUES %s
                ON CONFLICT (link) DO NOTHING
                """,
                rows,
            )
            cur.execute(
                "SELECT count(*) FROM news_items WHERE collected_date = %s AND country_code = %s",
                (date_str, country_code),
            )
            after = cur.fetchone()[0]
    inserted = after - before
    logger.info(
        f"Inserted {inserted}/{len(rows)} items for {date_str} "
        f"[{country_code}/{language}] (total in DB: {after})"
    )
    return inserted


# ---------------------------------------------------------------------------
# Article body fetcher
# ---------------------------------------------------------------------------

def fetch_for_article_fetching(
    date_str: str,
    limit: int = 50,
    retry_failed: bool = False,
    per_source_limit: int | None = None,
    country_code: str = "TR",
) -> list[dict[str, Any]]:
    """Return rows whose article body should be fetched."""
    status_filter = (
        "AND (parse_status IS NULL OR parse_status IN ('fetch_error', 'parse_error', 'empty'))"
        if retry_failed
        else "AND parse_status IS NULL"
    )
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if per_source_limit is not None:
                cur.execute(
                    f"""
                    SELECT id, title, summary, source_name, link, category,
                           canonical_category, discovery_role, parse_status
                    FROM (
                        SELECT
                            id, title, summary, source_name, link, category,
                            canonical_category, discovery_role, parse_status,
                            ROW_NUMBER() OVER (PARTITION BY source_name ORDER BY id) AS source_rank
                        FROM news_items
                        WHERE collected_date = %s
                          AND country_code = %s
                          AND link IS NOT NULL
                          {status_filter}
                    ) ranked
                    WHERE source_rank <= %s
                    ORDER BY source_name, id
                    LIMIT %s
                    """,
                    (date_str, country_code, per_source_limit, limit),
                )
            else:
                cur.execute(
                    f"""
                    SELECT id, title, summary, source_name, link, category,
                           canonical_category, discovery_role, parse_status
                    FROM news_items
                    WHERE collected_date = %s
                      AND country_code = %s
                      AND link IS NOT NULL
                      {status_filter}
                    ORDER BY id
                    LIMIT %s
                    """,
                    (date_str, country_code, limit),
                )
            return [dict(row) for row in cur.fetchall()]


@retry_on_connection_loss()
def bulk_update_article_parse(updates: list[dict[str, Any]]) -> None:
    """Persist article body parse results onto ``news_items`` rows."""
    if not updates:
        return
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                UPDATE news_items SET
                    article_text         = %s,
                    cleaned_article_text = %s,
                    canonical_category   = COALESCE(%s, canonical_category),
                    discovery_role       = COALESCE(%s, discovery_role),
                    parse_status         = %s,
                    parse_error          = %s,
                    article_fetched_at   = NOW()
                WHERE id = %s
                """,
                [
                    (
                        u.get("article_text"),
                        u.get("cleaned_article_text"),
                        u.get("canonical_category"),
                        u.get("discovery_role"),
                        u.get("parse_status"),
                        u.get("parse_error"),
                        u["id"],
                    )
                    for u in updates
                ],
            )
            logger.info(f"Updated {cur.rowcount} article parse fields")


def fetch_article_parse_quality_rows(
    date_str: str,
    country_code: str = "TR",
) -> list[dict[str, Any]]:
    """Return row-level article parse quality signals for one collection date."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    source_name,
                    COALESCE(canonical_category, 'unknown') AS category,
                    COALESCE(parse_status, 'unfetched') AS parse_status,
                    LENGTH(COALESCE(cleaned_article_text, '')) AS article_chars
                FROM news_items
                WHERE collected_date = %s
                  AND country_code = %s
                ORDER BY source_name, category, parse_status
                """,
                (date_str, country_code),
            )
            return [dict(row) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

def fetch_raw_by_date(date_str: str, country_code: str = "TR") -> list[dict[str, Any]]:
    """Return all items collected on *date_str* with their raw fields."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT *
                FROM news_items
                WHERE collected_date = %s
                  AND country_code = %s
                ORDER BY id
                """,
                (date_str, country_code),
            )
            return [dict(row) for row in cur.fetchall()]


@retry_on_connection_loss()
def bulk_update_preprocessed(updates: list[dict[str, Any]], deletes: list[int]) -> None:
    """Apply preprocessor results: update valid items, delete too-short ones."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            if deletes:
                cur.execute("DELETE FROM news_items WHERE id = ANY(%s)", (deletes,))
                logger.info(f"Deleted {cur.rowcount} too-short items")
            if updates:
                cur.executemany(
                    """
                    UPDATE news_items SET
                        cleaned_title        = %s,
                        cleaned_summary      = %s,
                        cleaned_article_text = %s,
                        is_turkish           = %s,
                        char_count           = %s,
                        published_date       = %s
                    WHERE id = %s
                    """,
                    [
                        (
                            u["cleaned_title"],
                            u["cleaned_summary"],
                            u.get("cleaned_article_text"),
                            u["is_turkish"],
                            u["char_count"],
                            u["published_date"],
                            u["id"],
                        )
                        for u in updates
                    ],
                )
                logger.info(f"Updated {cur.rowcount} preprocessed items")


# ---------------------------------------------------------------------------
# Sentiment
# ---------------------------------------------------------------------------

def fetch_processed_by_date(
    date_str: str,
    only_missing: bool = False,
    only_stale_after_body: bool = False,
    only_with_body: bool = False,
    limit: int | None = None,
    country_code: str = "TR",
) -> list[dict[str, Any]]:
    """Return items with cleaned fields needed for sentiment analysis."""
    # ``country_code`` is the multi-country filter; the legacy ``is_turkish``
    # langdetect flag is intentionally NOT used here — DE/FR rows would be
    # is_turkish=false and the scoping queries would return zero rows. Per-row
    # language gating still happens downstream in ``_to_analyzed`` for the
    # zero-shot path.
    filters = ["collected_date = %s", "country_code = %s"]
    params: list[Any] = [date_str, country_code]
    if only_missing:
        filters.append("sentiment_label IS NULL")
    if only_stale_after_body:
        filters.extend(
            [
                "sentiment_label IS NOT NULL",
                "article_fetched_at IS NOT NULL",
                "analyzed_at IS NOT NULL",
                "analyzed_at < article_fetched_at",
            ]
        )
    if only_with_body:
        filters.append("LENGTH(COALESCE(cleaned_article_text, '')) > 0")
    limit_clause = ""
    if limit is not None:
        limit_clause = " LIMIT %s"
        params.append(limit)
    order_clause = "analyzed_at NULLS FIRST, id" if only_with_body else "id"

    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT id, cleaned_title, cleaned_summary, cleaned_article_text, is_turkish
                FROM news_items
                WHERE {" AND ".join(filters)}
                ORDER BY {order_clause}
                {limit_clause}
                """,
                params,
            )
            return [dict(row) for row in cur.fetchall()]


def fetch_sentiment_quality_rows(
    date_str: str,
    country_code: str = "TR",
) -> list[dict[str, Any]]:
    """Return row-level sentiment QA signals for one collection date."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    id,
                    source_name,
                    link,
                    COALESCE(canonical_category, 'unknown') AS category,
                    cleaned_title,
                    cleaned_summary,
                    cleaned_article_text,
                    sentiment_label,
                    sentiment_score,
                    sentiment_scores,
                    raw_sentiment_score,
                    calibrated_sentiment_score,
                    calibrated_sentiment_scores,
                    calibration_method,
                    analyzed_at,
                    article_fetched_at,
                    is_turkish
                FROM news_items
                WHERE collected_date = %s
                  AND country_code = %s
                ORDER BY source_name, id
                """,
                (date_str, country_code),
            )
            return [dict(row) for row in cur.fetchall()]


@retry_on_connection_loss()
def bulk_update_sentiment(updates: list[dict[str, Any]]) -> None:
    if not updates:
        return
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                UPDATE news_items SET
                    sentiment_label              = %s,
                    sentiment_score              = %s,
                    sentiment_scores             = %s,
                    raw_sentiment_score          = %s,
                    calibrated_sentiment_score   = %s,
                    calibrated_sentiment_scores  = %s,
                    calibration_method           = %s,
                    analyzed_at                  = %s
                WHERE id = %s
                """,
                [
                    (
                        u.get("sentiment_label"),
                        u.get("sentiment_score"),
                        psycopg2.extras.Json(u.get("sentiment_scores")),
                        u.get("raw_sentiment_score"),
                        u.get("calibrated_sentiment_score"),
                        psycopg2.extras.Json(u.get("calibrated_sentiment_scores")),
                        u.get("calibration_method"),
                        u.get("analyzed_at"),
                        u["id"],
                    )
                    for u in updates
                ],
            )
            logger.info(f"Updated {cur.rowcount} sentiment fields")


@retry_on_connection_loss()
def bulk_update_sentiment_calibration(updates: list[dict[str, Any]]) -> None:
    """Persist calibrated confidence fields without changing sentiment labels."""
    if not updates:
        return
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                UPDATE news_items SET
                    raw_sentiment_score          = %s,
                    calibrated_sentiment_score   = %s,
                    calibrated_sentiment_scores  = %s,
                    calibration_method           = %s
                WHERE id = %s
                """,
                [
                    (
                        u.get("raw_sentiment_score"),
                        u.get("calibrated_sentiment_score"),
                        psycopg2.extras.Json(u.get("calibrated_sentiment_scores")),
                        u.get("calibration_method"),
                        u["id"],
                    )
                    for u in updates
                ],
            )
            logger.info(f"Updated {cur.rowcount} sentiment calibration fields")


# ---------------------------------------------------------------------------
# NER
# ---------------------------------------------------------------------------

def fetch_for_ner(date_str: str, country_code: str = "TR") -> list[dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, title, summary, article_text, cleaned_article_text, is_turkish
                FROM news_items
                WHERE collected_date = %s
                  AND country_code = %s
                ORDER BY id
                """,
                (date_str, country_code),
            )
            return [dict(row) for row in cur.fetchall()]


@retry_on_connection_loss()
def bulk_update_ner(updates: list[dict[str, Any]]) -> None:
    if not updates:
        return
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                UPDATE news_items SET
                    entities     = %s,
                    entity_count = %s
                WHERE id = %s
                """,
                [
                    (
                        psycopg2.extras.Json(u.get("entities", {})),
                        u.get("entity_count", 0),
                        u["id"],
                    )
                    for u in updates
                ],
            )
            logger.info(f"Updated {cur.rowcount} NER fields")


# ---------------------------------------------------------------------------
# Entity mentions (Sprint 1: multilingual NER track, lives alongside the
# legacy news_items.entities JSONB column. Sprint 2 fills wikidata_qid;
# Sprint 5 reads position_in_article for collocation windows.)
# ---------------------------------------------------------------------------


def fetch_for_entity_extraction(
    date_str: str,
    country_code: str,
) -> list[dict[str, Any]]:
    """Return rows the entity_extraction step should process for one day."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, title, summary, article_text, cleaned_article_text,
                       cleaned_title, cleaned_summary
                FROM news_items
                WHERE collected_date = %s
                  AND country_code = %s
                ORDER BY id
                """,
                (date_str, country_code),
            )
            return [dict(row) for row in cur.fetchall()]


@retry_on_connection_loss()
def bulk_insert_entity_mentions(
    mentions: list[dict[str, Any]],
    article_ids: list[int],
) -> int:
    """Replace and bulk-insert mentions for *article_ids*.

    Idempotent re-run: we DELETE every existing mention for the article
    IDs touched in this batch, then insert the fresh set. Same shape as
    ``bulk_update_ner`` (callers pass the article IDs they're rewriting),
    so re-processing a day never duplicates rows.
    """
    if not article_ids:
        return 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM entity_mentions WHERE article_id = ANY(%s)",
                (list(article_ids),),
            )
            if not mentions:
                logger.info(
                    f"Cleared entity_mentions for {len(article_ids)} articles "
                    "(no new mentions to insert)"
                )
                return 0
            rows = [
                (
                    m["article_id"],
                    m["country_code"],
                    m["collected_date"],
                    m["entity_text"],
                    m["entity_type"],
                    m.get("wikidata_qid"),
                    m.get("canonical"),
                    m.get("resolution_confidence"),
                    m.get("resolver_method"),
                    m.get("position_in_article"),
                )
                for m in mentions
            ]
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO entity_mentions
                    (article_id, country_code, collected_date, entity_text,
                     entity_type, wikidata_qid, canonical, resolution_confidence,
                     resolver_method, position_in_article)
                VALUES %s
                """,
                rows,
            )
            # ``execute_values`` runs N INSERT batches under the hood (one
            # per ``page_size`` chunk) and ``cur.rowcount`` only reflects
            # the last batch — so it under-counts when ``len(rows) >
            # page_size``. There's no ON CONFLICT clause here, so every
            # row in ``rows`` is inserted; ``len(rows)`` is the truthful
            # count for both the return value and the log line.
            inserted = len(rows)
            logger.info(
                f"Inserted {inserted} entity mentions across "
                f"{len(article_ids)} articles"
            )
            return inserted


@retry_on_connection_loss()
def fetch_top_entity_texts_for_cache(
    country_code: str,
    since_date: str,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Return the most frequent still-unresolved (entity_text, entity_type) pairs.

    Used by the preemptive Wikidata cache warmer: picks high-volume mentions
    since ``since_date`` that have no QID and were not resolved beyond the
    deterministic ``normalized`` fallback, so they can be linked ahead of time.
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT entity_text, entity_type, COUNT(*) AS mention_count
                FROM entity_mentions
                WHERE country_code = %s
                  AND collected_date >= %s
                  AND wikidata_qid IS NULL
                  AND (resolver_method IS NULL OR resolver_method = 'normalized')
                GROUP BY entity_text, entity_type
                ORDER BY mention_count DESC
                LIMIT %s
                """,
                (country_code, since_date, limit),
            )
            return [dict(row) for row in cur.fetchall()]


def fetch_unresolved_entity_mentions(
    date_str: str,
    country_code: str,
    limit: int = 1000,
    include_normalized: bool = False,
) -> list[dict[str, Any]]:
    """Return mentions that still need the resolver/backfill step."""
    normalized_filter = "OR resolver_method = 'normalized'" if include_normalized else ""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT id, entity_text, entity_type, canonical, wikidata_qid
                FROM entity_mentions
                WHERE collected_date = %s
                  AND country_code = %s
                  AND (
                    canonical IS NULL
                    OR resolver_method IS NULL
                    {normalized_filter}
                  )
                ORDER BY id
                LIMIT %s
                """,
                (date_str, country_code, limit),
            )
            return [dict(row) for row in cur.fetchall()]


@retry_on_connection_loss()
def bulk_update_entity_resolution(updates: list[dict[str, Any]]) -> None:
    if not updates:
        return
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                UPDATE entity_mentions SET
                    canonical             = %s,
                    wikidata_qid          = %s,
                    resolution_confidence = %s,
                    resolver_method       = %s
                WHERE id = %s
                """,
                [
                    (
                        u.get("canonical"),
                        u.get("wikidata_qid"),
                        u.get("resolution_confidence"),
                        u.get("resolver_method"),
                        u["id"],
                    )
                    for u in updates
                ],
            )
            logger.info(f"Updated {cur.rowcount} entity resolution fields")


def fetch_entity_resolution_cache(
    normalized_text: str,
    entity_type: str,
    country_code: str,
) -> dict[str, Any] | None:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT normalized_text, entity_type, country_code, canonical,
                       wikidata_qid, confidence, resolver_method
                FROM entity_resolution_cache
                WHERE normalized_text = %s
                  AND entity_type = %s
                  AND country_code = %s
                """,
                (normalized_text, entity_type, country_code),
            )
            row = cur.fetchone()
            return dict(row) if row else None


@retry_on_connection_loss()
def upsert_entity_resolution_cache(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO entity_resolution_cache (
                    normalized_text, entity_type, country_code, canonical,
                    wikidata_qid, confidence, resolver_method
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (normalized_text, entity_type, country_code)
                DO UPDATE SET
                    canonical       = EXCLUDED.canonical,
                    wikidata_qid    = EXCLUDED.wikidata_qid,
                    confidence      = EXCLUDED.confidence,
                    resolver_method = EXCLUDED.resolver_method,
                    updated_at      = NOW()
                """,
                [
                    (
                        r["normalized_text"],
                        r["entity_type"],
                        r["country_code"],
                        r.get("canonical"),
                        r.get("wikidata_qid"),
                        r.get("confidence"),
                        r.get("resolver_method"),
                    )
                    for r in rows
                ],
            )


# ---------------------------------------------------------------------------
# Entity collocations (Sprint 5: lemma cooccurrence counts; Sprint 6 PMI)
# ---------------------------------------------------------------------------


def fetch_for_collocation_extraction(
    date_str: str,
    country_code: str,
) -> list[dict[str, Any]]:
    """Return one row per (mention, article) pair the collocation step
    should process for one day.

    Joins ``entity_mentions`` to ``news_items`` so the extractor can
    rebuild the exact NER input string with
    :func:`src.analysis.text_inputs.build_ner_text`. Only mentions with
    a non-NULL ``position_in_article`` (the HuggingFace NER ``start``
    offset into that same string) are returned; older rows without a
    position can't be windowed.

    Mentions without a resolved canonical fall back to ``entity_text``
    so the row is still addressable. ``wikidata_qid`` is kept nullable.
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    em.id                  AS mention_id,
                    em.article_id          AS article_id,
                    em.entity_text         AS entity_text,
                    em.entity_type         AS entity_type,
                    em.wikidata_qid        AS wikidata_qid,
                    COALESCE(em.canonical, em.entity_text) AS canonical,
                    em.position_in_article AS position_in_article,
                    em.country_code        AS country_code,
                    em.collected_date      AS collected_date,
                    ni.title               AS title,
                    ni.summary             AS summary,
                    ni.article_text        AS article_text,
                    ni.cleaned_title       AS cleaned_title,
                    ni.cleaned_summary     AS cleaned_summary,
                    ni.cleaned_article_text AS cleaned_article_text
                FROM entity_mentions em
                JOIN news_items ni ON ni.id = em.article_id
                WHERE em.collected_date = %s
                  AND em.country_code   = %s
                  AND em.position_in_article IS NOT NULL
                ORDER BY em.article_id, em.position_in_article
                """,
                (date_str, country_code),
            )
            return [dict(row) for row in cur.fetchall()]


@retry_on_connection_loss()
def bulk_upsert_entity_collocations(
    rows: list[dict[str, Any]],
    country_code: str,
    date_str: str,
) -> int:
    """Replace one (country_code, collected_date) slice with *rows*.

    Idempotent: DELETE the day's existing collocations for the country,
    then INSERT the freshly aggregated rows. Mirrors
    :func:`bulk_insert_entity_mentions` but keys on (country, date)
    instead of article_id because the aggregate is computed across all
    articles for the day.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM entity_collocations
                WHERE country_code = %s AND collected_date = %s
                """,
                (country_code, date_str),
            )
            if not rows:
                logger.info(
                    f"Cleared entity_collocations for {country_code} "
                    f"{date_str} (no new rows to insert)"
                )
                return 0
            tuples = [
                (
                    country_code,
                    date_str,
                    r.get("wikidata_qid"),
                    r["canonical"],
                    r["entity_type"],
                    r["lemma"],
                    r["pos"],
                    r["cooccurrence_count"],
                )
                for r in rows
            ]
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO entity_collocations
                    (country_code, collected_date, wikidata_qid, canonical,
                     entity_type, lemma, pos, cooccurrence_count)
                VALUES %s
                """,
                tuples,
            )
            inserted = len(tuples)
            logger.info(
                f"Inserted {inserted} entity collocations for "
                f"{country_code} {date_str}"
            )
            return inserted


# ---------------------------------------------------------------------------
# Entity collocation stats (Sprint 6: PMI / log-likelihood backfill)
# ---------------------------------------------------------------------------


def fetch_collocations_for_stats(
    date_str: str,
    country_code: str,
) -> list[dict[str, Any]]:
    """Return one (country, date) slice of ``entity_collocations`` for PMI.

    Selects the columns Sprint 6's PMI step needs to compute marginals
    and write back stats. ``id`` is included so the UPDATE keys on the
    table's primary key (the natural unique constraint would also work
    but the BIGSERIAL is cheaper to address).
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, canonical, entity_type, lemma, pos, cooccurrence_count
                FROM entity_collocations
                WHERE country_code = %s AND collected_date = %s
                ORDER BY id
                """,
                (country_code, date_str),
            )
            return [dict(row) for row in cur.fetchall()]


def fetch_entity_mention_counts(
    date_str: str,
    country_code: str,
) -> dict[tuple[str, str], int]:
    """Distinct mention count per (canonical, entity_type) for one day.

    Powers the ``entity_total`` column — distinct prominence signal,
    independent of the lemma-weighted ``cooccurrence_with_entity_total``
    marginal. Sprint 7's rolling aggregator and Sprint 8's
    ``/api/entity/{qid}/profile`` consume this value.

    Mentions without a resolved ``canonical`` fall back to ``entity_text``,
    matching :func:`fetch_for_collocation_extraction`.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COALESCE(canonical, entity_text) AS canonical,
                       entity_type,
                       COUNT(*) AS mention_count
                FROM entity_mentions
                WHERE collected_date = %s AND country_code = %s
                GROUP BY COALESCE(canonical, entity_text), entity_type
                """,
                (date_str, country_code),
            )
            return {(c, t): int(n) for c, t, n in cur.fetchall()}


@retry_on_connection_loss()
def bulk_update_collocation_stats(
    rows: list[dict[str, Any]],
    country_code: str,
    date_str: str,
) -> int:
    """UPDATE the six Sprint-6 columns for the given collocation rows.

    Uses ``UPDATE … FROM (VALUES %s)`` so a single round trip rewrites
    every row in the day's slice. ``id`` is the PK and uniquely
    addresses each row — ``country_code`` / ``date_str`` are accepted
    only for logging context.

    Rows that came in below the ``min_cooccurrence_count`` threshold are
    *not* passed here; their stats columns stay NULL.
    """
    if not rows:
        logger.info(
            f"No collocation stats to update for {country_code} {date_str}"
        )
        return 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            tuples = [
                (
                    r["id"],
                    r["cooccurrence_with_entity_total"],
                    r["entity_total"],
                    r["token_total"],
                    r["window_total"],
                    r["pmi"],
                    r["log_likelihood"],
                )
                for r in rows
            ]
            psycopg2.extras.execute_values(
                cur,
                """
                UPDATE entity_collocations AS ec SET
                    cooccurrence_with_entity_total = v.cwet,
                    entity_total                   = v.et,
                    token_total                    = v.tt,
                    window_total                   = v.wt,
                    pmi                            = v.pmi,
                    log_likelihood                 = v.llr
                FROM (VALUES %s) AS v(id, cwet, et, tt, wt, pmi, llr)
                WHERE ec.id = v.id
                """,
                tuples,
            )
            updated = len(tuples)
            logger.info(
                f"Updated PMI/LLR for {updated} entity_collocations rows "
                f"({country_code} {date_str})"
            )
            return updated


# ---------------------------------------------------------------------------
# Entity country profile (Sprint 7: rolling 7d/30d aggregator)
# ---------------------------------------------------------------------------


def fetch_entity_collocations_date_range(
    country_code: str,
    start_date: str,
    end_date: str,
) -> list[dict[str, Any]]:
    """Return ``entity_collocations`` rows in ``[start_date, end_date]``.

    Sprint 7's rolling aggregator reads multiple consecutive days and
    re-derives window-level marginals (R_w, T_w, N_w) from the raw
    ``cooccurrence_count`` values. The Sprint-6 per-row stats
    (``pmi``, ``log_likelihood``, ``cooccurrence_with_entity_total``)
    are intentionally NOT selected — naively summing them across days
    would double-count entries that span multiple days. The window
    stats are recomputed from scratch.

    ``entity_total`` IS returned because it's the Option-B distinct
    mention count, useful for ``total_mentions`` aggregation; summing
    daily distinct counts gives an approximation (an entity counted
    twice if mentioned on two days) which is the documented Sprint 7
    semantics.
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    collected_date,
                    canonical,
                    entity_type,
                    wikidata_qid,
                    lemma,
                    pos,
                    cooccurrence_count,
                    entity_total
                FROM entity_collocations
                WHERE country_code = %s
                  AND collected_date BETWEEN %s AND %s
                ORDER BY collected_date, canonical, entity_type
                """,
                (country_code, start_date, end_date),
            )
            return [dict(row) for row in cur.fetchall()]


@retry_on_connection_loss()
def bulk_upsert_entity_country_profile(
    rows: list[dict[str, Any]],
    country_code: str,
    end_date: str,
) -> int:
    """Replace one (country_code, end_date) slice with *rows*.

    Idempotent: DELETE the day's profile rows for the country, then
    INSERT the freshly rolled-up entries. Each row carries both 7d and
    30d windows for the same entity — the aggregator decides what to
    emit per entity per window.

    ``top_collocates`` is a Python list of dicts; psycopg2 adapts it to
    a JSONB literal via ``psycopg2.extras.Json``.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM entity_country_profile
                WHERE country_code = %s AND end_date = %s
                """,
                (country_code, end_date),
            )
            if not rows:
                logger.info(
                    f"Cleared entity_country_profile for {country_code} "
                    f"{end_date} (no new rows to insert)"
                )
                return 0
            tuples = [
                (
                    country_code,
                    r["canonical"],
                    r["entity_type"],
                    r.get("wikidata_qid"),
                    r["window_days"],
                    end_date,
                    r["coverage_days"],
                    r["total_cooccurrences"],
                    r["total_mentions"],
                    r["cooccurrence_with_entity_total"],
                    r["window_total"],
                    r["avg_pmi"],
                    r["avg_log_likelihood"],
                    psycopg2.extras.Json(r["top_collocates"]),
                )
                for r in rows
            ]
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO entity_country_profile
                    (country_code, canonical, entity_type, wikidata_qid,
                     window_days, end_date, coverage_days,
                     total_cooccurrences, total_mentions,
                     cooccurrence_with_entity_total, window_total,
                     avg_pmi, avg_log_likelihood, top_collocates)
                VALUES %s
                """,
                tuples,
            )
            inserted = len(tuples)
            logger.info(
                f"Inserted {inserted} entity_country_profile rows for "
                f"{country_code} {end_date}"
            )
            return inserted


# ---------------------------------------------------------------------------
# Entity profile reads (Sprint 8 — API serving layer)
# ---------------------------------------------------------------------------

# Columns served by the profile endpoints; kept in one place so the three
# profile readers below stay in sync.
_PROFILE_COLS = """
    country_code, canonical, entity_type, wikidata_qid,
    window_days, end_date, coverage_days,
    total_cooccurrences, total_mentions, window_total,
    avg_pmi, avg_log_likelihood, top_collocates
"""


def fetch_latest_profile_end_date(country_code: str | None = None) -> str | None:
    """Most recent ``entity_country_profile.end_date`` (optionally per country).

    Sprint 8's profile endpoints default ``end_date`` to the freshest rolling
    window the Sprint 7 aggregator produced. Returns an ISO date string or
    ``None`` when no profile rows exist yet.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            if country_code:
                cur.execute(
                    "SELECT MAX(end_date) FROM entity_country_profile "
                    "WHERE country_code = %s",
                    (country_code,),
                )
            else:
                cur.execute("SELECT MAX(end_date) FROM entity_country_profile")
            row = cur.fetchone()
    if not row or row[0] is None:
        return None
    return row[0].isoformat()


def fetch_entity_profile(
    country_code: str,
    window_days: int,
    end_date: str | None = None,
    qid: str | None = None,
    canonical: str | None = None,
) -> dict[str, Any] | None:
    """One ``entity_country_profile`` row for an entity in one country/window.

    Looked up by ``qid`` (Wikidata QID, uses ``idx_ecp_qid_window``) when
    given, else by ``canonical`` text (``idx_ecp_country_end_canonical``).
    ``end_date`` defaults to the country's most recent window. ``top_collocates``
    comes back as a Python ``list[dict]`` (JSONB) and ``end_date`` as a
    ``datetime.date`` — the API serializer stringifies it.
    """
    if not qid and not canonical:
        raise ValueError("fetch_entity_profile needs either qid or canonical")
    if end_date is None:
        end_date = fetch_latest_profile_end_date(country_code)
        if end_date is None:
            return None
    clause, param = (
        ("wikidata_qid = %s", qid) if qid else ("canonical = %s", canonical)
    )
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT {_PROFILE_COLS}
                FROM entity_country_profile
                WHERE country_code = %s AND window_days = %s AND end_date = %s
                  AND {clause}
                ORDER BY total_mentions DESC
                LIMIT 1
                """,
                (country_code, window_days, end_date, param),
            )
            row = cur.fetchone()
    return dict(row) if row else None


def fetch_entity_profiles_all_countries(
    window_days: int,
    end_date: str | None = None,
    qid: str | None = None,
    canonical: str | None = None,
    countries: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Same entity's profile across every country (Sprint 8 ``/compare``).

    QID-primary: cross-country identity only holds for Wikidata-linked
    entities, so a ``canonical`` lookup matches the literal string per country
    and is best-effort. When ``end_date`` is omitted, ``DISTINCT ON`` picks
    each country's *own* latest window (countries can lag a day); pass an
    explicit ``end_date`` to pin a single date. ``countries`` (ISO codes)
    narrows the result.
    """
    if not qid and not canonical:
        raise ValueError(
            "fetch_entity_profiles_all_countries needs either qid or canonical"
        )
    clause, param = (
        ("wikidata_qid = %s", qid) if qid else ("canonical = %s", canonical)
    )
    params: list[Any] = [window_days, param]
    sql = f"""
        SELECT DISTINCT ON (country_code) {_PROFILE_COLS}
        FROM entity_country_profile
        WHERE window_days = %s AND {clause}
    """
    if end_date is not None:
        sql += " AND end_date = %s"
        params.append(end_date)
    if countries:
        sql += " AND country_code = ANY(%s)"
        params.append(countries)
    sql += " ORDER BY country_code, end_date DESC, total_mentions DESC"
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]


def fetch_entity_directory(
    country_code: str,
    window_days: int,
    end_date: str | None = None,
    q: str | None = None,
    entity_type: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Top entities for a country's latest window (Sprint 8 ``/api/entities``).

    Ranked by ``total_mentions`` then ``avg_log_likelihood``. ``q`` filters
    canonical names (case-insensitive substring); ``entity_type`` restricts to
    PER/ORG/LOC. Powers the dashboard's entity discovery/search.
    """
    if end_date is None:
        end_date = fetch_latest_profile_end_date(country_code)
        if end_date is None:
            return []
    params: list[Any] = [country_code, window_days, end_date]
    sql = """
        SELECT canonical, wikidata_qid, entity_type, coverage_days,
               total_mentions, total_cooccurrences, avg_pmi, avg_log_likelihood
        FROM entity_country_profile
        WHERE country_code = %s AND window_days = %s AND end_date = %s
    """
    if q:
        sql += " AND canonical ILIKE %s"
        params.append(f"%{q}%")
    if entity_type:
        sql += " AND entity_type = %s"
        params.append(entity_type)
    sql += (
        " ORDER BY total_mentions DESC, avg_log_likelihood DESC NULLS LAST "
        "LIMIT %s"
    )
    params.append(limit)
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]


def fetch_entity_timeline(
    country_code: str,
    start_date: str,
    end_date: str,
    qid: str | None = None,
    canonical: str | None = None,
) -> list[dict[str, Any]]:
    """Per-day mention volume + salience for an entity (Sprint 8 ``/timeline``).

    Reads daily ``entity_collocations`` (not the rolling profile — it carries
    deeper history) over ``[start_date, end_date]``. ``entity_total`` is
    constant across a day's lemma rows, so ``MAX`` recovers the distinct
    mention count; cooccurrences sum and PMI/LLR average over the day's
    collocates. QID-primary, canonical fallback.
    """
    if not qid and not canonical:
        raise ValueError("fetch_entity_timeline needs either qid or canonical")
    clause, param = (
        ("wikidata_qid = %s", qid) if qid else ("canonical = %s", canonical)
    )
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT collected_date,
                       MAX(entity_total)        AS mention_count,
                       SUM(cooccurrence_count)  AS total_cooccurrences,
                       AVG(pmi)                 AS avg_pmi,
                       AVG(log_likelihood)      AS avg_log_likelihood
                FROM entity_collocations
                WHERE country_code = %s
                  AND collected_date BETWEEN %s AND %s
                  AND {clause}
                GROUP BY collected_date
                ORDER BY collected_date
                """,
                (country_code, start_date, end_date, param),
            )
            return [dict(r) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def fetch_for_clustering(date_str: str, country_code: str = "TR") -> list[dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, cleaned_title, cleaned_summary, cleaned_article_text, is_turkish,
                       source_name, category, canonical_category, entities, sentiment_label
                FROM news_items
                WHERE collected_date = %s
                  AND country_code = %s
                ORDER BY id
                """,
                (date_str, country_code),
            )
            return [dict(row) for row in cur.fetchall()]


@retry_on_connection_loss()
def bulk_update_clustering(updates: list[dict[str, Any]]) -> None:
    if not updates:
        return
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                UPDATE news_items SET
                    cluster_id       = %s,
                    cluster_keywords = %s,
                    cluster_title    = %s
                WHERE id = %s
                """,
                [
                    (
                        u.get("cluster_id"),
                        psycopg2.extras.Json(u.get("cluster_keywords", [])),
                        u.get("cluster_title", ""),
                        u["id"],
                    )
                    for u in updates
                ],
            )
            logger.info(f"Updated {cur.rowcount} clustering fields")


def upsert_cluster_summaries(
    summaries: list[dict[str, Any]],
    date_str: str,
    country_code: str = "TR",
) -> None:
    if not summaries:
        return
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO cluster_summaries
                    (country_code, date, cluster_id, title, size, keywords, sources, sentiment_distribution)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (country_code, date, cluster_id) DO UPDATE SET
                    title                  = EXCLUDED.title,
                    size                   = EXCLUDED.size,
                    keywords               = EXCLUDED.keywords,
                    sources                = EXCLUDED.sources,
                    sentiment_distribution = EXCLUDED.sentiment_distribution
                """,
                [
                    (
                        country_code,
                        date_str,
                        s["cluster_id"],
                        s.get("title", ""),
                        s.get("size", 0),
                        psycopg2.extras.Json(s.get("keywords", [])),
                        psycopg2.extras.Json(s.get("sources", {})),
                        psycopg2.extras.Json(s.get("sentiment_distribution", {})),
                    )
                    for s in summaries
                ],
            )
            logger.info(f"Upserted {len(summaries)} cluster summaries for {date_str}")


# ---------------------------------------------------------------------------
# Vector store (pgvector)
# ---------------------------------------------------------------------------

def fetch_for_indexing(date_str: str, country_code: str = "TR") -> list[dict[str, Any]]:
    """Return Turkish items with sentiment to be embedded."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, title, cleaned_title, cleaned_summary, cleaned_article_text, source_name,
                       link, sentiment_label, sentiment_score, cluster_id
                FROM news_items
                WHERE collected_date = %s
                  AND country_code = %s
                  AND sentiment_label IS NOT NULL
                ORDER BY id
                """,
                (date_str, country_code),
            )
            rows = [dict(row) for row in cur.fetchall()]
    for row in rows:
        row["date"] = date_str
    return rows


@retry_on_connection_loss()
def bulk_update_embeddings(updates: list[dict[str, Any]]) -> None:
    """Store sentence-transformer embeddings in the news_items.embedding column.

    Each update dict must have: id (int), embedding (list[float] of length 384).
    """
    if not updates:
        return
    # pgvector expects the vector as a string literal: '[0.1,0.2,...]'
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE news_items SET embedding = %s::vector WHERE id = %s",
                [
                    ("[" + ",".join(str(x) for x in u["embedding"]) + "]", u["id"])
                    for u in updates
                ],
            )
            logger.info(f"Updated {cur.rowcount} embeddings")


def find_similar_pgvector(
    query_embedding: list[float],
    n: int = 5,
    country_code: str = "TR",
) -> list[dict[str, Any]]:
    """Return top-n news items by cosine similarity to *query_embedding*."""
    vec_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    title,
                    source_name,
                    collected_date::text AS date,
                    sentiment_label,
                    link,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM news_items
                WHERE embedding IS NOT NULL
                  AND country_code = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (vec_str, country_code, vec_str, n),
            )
            return [dict(row) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def fetch_all_for_api(date_str: str, country_code: str = "TR") -> list[dict[str, Any]]:
    """Fetch all items for a date with every field the API needs."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    id, title, source_name, published_date, link,
                    is_turkish, sentiment_label, sentiment_score,
                    calibrated_sentiment_score,
                    entities, cluster_id, cluster_title, cluster_keywords
                FROM news_items
                WHERE collected_date = %s
                  AND country_code = %s
                ORDER BY id
                """,
                (date_str, country_code),
            )
            return [dict(row) for row in cur.fetchall()]


def fetch_top_entities(
    date_str: str,
    country_code: str = "TR",
    limit: int = 10,
) -> dict[str, list[str]]:
    """Return top canonical entity names from the entity_mentions table."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                WITH grouped AS (
                    SELECT
                        entity_type,
                        COALESCE(NULLIF(canonical, ''), entity_text) AS name,
                        COUNT(*) AS mention_count,
                        ROW_NUMBER() OVER (
                            PARTITION BY entity_type
                            ORDER BY COUNT(*) DESC, COALESCE(NULLIF(canonical, ''), entity_text)
                        ) AS rank
                    FROM entity_mentions
                    WHERE collected_date = %s
                      AND country_code = %s
                      AND entity_type IN ('PER', 'ORG', 'LOC')
                    GROUP BY entity_type, COALESCE(NULLIF(canonical, ''), entity_text)
                )
                SELECT entity_type, name, mention_count
                FROM grouped
                WHERE rank <= %s
                ORDER BY entity_type, mention_count DESC, name
                """,
                (date_str, country_code, limit),
            )
            result: dict[str, list[str]] = {"PER": [], "ORG": [], "LOC": []}
            for row in cur.fetchall():
                result.setdefault(row["entity_type"], []).append(row["name"])
            return result


def fetch_entity_resolution_audit_rows(
    date_str: str,
    country_code: str,
) -> list[dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    entity_text,
                    entity_type,
                    COALESCE(NULLIF(canonical, ''), entity_text) AS canonical,
                    wikidata_qid,
                    resolver_method,
                    resolution_confidence,
                    COUNT(*) AS mention_count
                FROM entity_mentions
                WHERE collected_date = %s
                  AND country_code = %s
                GROUP BY entity_text, entity_type, COALESCE(NULLIF(canonical, ''), entity_text),
                         wikidata_qid, resolver_method, resolution_confidence
                ORDER BY mention_count DESC, canonical, entity_text
                """,
                (date_str, country_code),
            )
            return [dict(row) for row in cur.fetchall()]


def fetch_cluster_summaries(
    date_str: str,
    country_code: str = "TR",
) -> list[dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT cluster_id, title, size, keywords, sources, sentiment_distribution
                FROM cluster_summaries
                WHERE date = %s
                  AND country_code = %s
                ORDER BY size DESC
                """,
                (date_str, country_code),
            )
            return [dict(row) for row in cur.fetchall()]


def fetch_available_dates(country_code: str = "TR") -> list[str]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT collected_date
                FROM news_items
                WHERE country_code = %s
                ORDER BY collected_date
                """,
                (country_code,),
            )
            return [str(row[0]) for row in cur.fetchall()]


def fetch_high_confidence_items(
    min_confidence: float = 0.85,
    max_samples: int = 50_000,
    country_code: str = "TR",
) -> list[dict[str, Any]]:
    """Return high-confidence Turkish news items for retraining.

    Returns id, cleaned_title, cleaned_summary, sentiment_label, sentiment_score.
    Ordered newest-first so the recency cap (LIMIT) keeps the most recent examples.
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT cleaned_title, cleaned_summary, sentiment_label, sentiment_score
                FROM news_items
                WHERE country_code = %s
                  AND sentiment_label IS NOT NULL
                  AND sentiment_score >= %s
                ORDER BY collected_date DESC
                LIMIT %s
                """,
                (country_code, min_confidence, max_samples),
            )
            return [dict(row) for row in cur.fetchall()]


def fetch_sentiment_trend(
    days: int = 30,
    country_code: str = "TR",
) -> list[dict[str, Any]]:
    """Return daily sentiment counts for the last *days* days, ordered ascending."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    collected_date::text AS date,
                    COUNT(*) FILTER (WHERE sentiment_label = 'positive') AS positive,
                    COUNT(*) FILTER (WHERE sentiment_label = 'negative') AS negative,
                    COUNT(*) FILTER (WHERE sentiment_label = 'neutral')  AS neutral,
                    COUNT(*) AS total
                FROM news_items
                WHERE country_code = %s
                  AND sentiment_label IS NOT NULL
                  AND collected_date >= CURRENT_DATE - %s::int
                GROUP BY collected_date
                ORDER BY collected_date
                """,
                (country_code, days),
            )
            return [dict(row) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Drift reports
# ---------------------------------------------------------------------------


def upsert_drift_report(report: dict[str, Any], country_code: str = "TR") -> None:
    """Persist a drift report so the API can expose it as a Prometheus gauge.

    Idempotent on ``(country_code, date)``. Re-running the daily drift step
    overwrites the prior row for that country/day.
    """
    today = report.get("today") or {}
    baseline = report.get("baseline") or {}
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO drift_reports (
                    country_code, date, status, psi, severity, baseline_days,
                    today_total, today_ratios, baseline_ratios, per_class_delta
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (country_code, date) DO UPDATE SET
                    status          = EXCLUDED.status,
                    psi             = EXCLUDED.psi,
                    severity        = EXCLUDED.severity,
                    baseline_days   = EXCLUDED.baseline_days,
                    today_total     = EXCLUDED.today_total,
                    today_ratios    = EXCLUDED.today_ratios,
                    baseline_ratios = EXCLUDED.baseline_ratios,
                    per_class_delta = EXCLUDED.per_class_delta,
                    computed_at     = NOW()
                """,
                (
                    country_code,
                    report["date"],
                    report.get("status"),
                    report.get("psi"),
                    report.get("severity"),
                    report.get("baseline_days"),
                    today.get("total"),
                    json.dumps(today.get("ratios")) if today.get("ratios") else None,
                    json.dumps(baseline.get("ratios")) if baseline.get("ratios") else None,
                    json.dumps(report.get("per_class_delta")) if report.get("per_class_delta") else None,
                ),
            )


def fetch_latest_drift_report(country_code: str = "TR") -> dict[str, Any] | None:
    """Most recent drift row, or None if the table is empty."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT date::text AS date, country_code, status, psi, severity, baseline_days,
                       today_total, today_ratios, baseline_ratios, per_class_delta,
                       computed_at
                FROM drift_reports
                WHERE country_code = %s
                ORDER BY date DESC
                LIMIT 1
                """,
                (country_code,),
            )
            row = cur.fetchone()
            return dict(row) if row else None


def fetch_drift_history(
    days: int = 30,
    country_code: str = "TR",
) -> list[dict[str, Any]]:
    """Recent drift rows for trend charts; oldest first so charts plot left-to-right."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT date::text AS date, country_code, status, psi, severity, baseline_days,
                       today_total, today_ratios, baseline_ratios, per_class_delta
                FROM drift_reports
                WHERE date >= CURRENT_DATE - %s::int
                  AND country_code = %s
                ORDER BY date
                """,
                (days, country_code),
            )
            return [dict(row) for row in cur.fetchall()]
