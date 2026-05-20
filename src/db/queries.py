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
                  AND is_turkish = true
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
                WHERE is_turkish = true
                  AND country_code = %s
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
                WHERE is_turkish = true
                  AND country_code = %s
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
