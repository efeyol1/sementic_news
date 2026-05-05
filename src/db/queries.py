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

def insert_raw_items(items: list[dict[str, Any]], date_str: str) -> int:
    """Bulk-insert raw RSS items; skip duplicates by link. Returns rows inserted."""
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
        )
        for item in items
    ]
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM news_items WHERE collected_date = %s", (date_str,))
            before = cur.fetchone()[0]
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO news_items
                    (collected_date, title, summary, source_name, published_date, link, category)
                VALUES %s
                ON CONFLICT (link) DO NOTHING
                """,
                rows,
            )
            cur.execute("SELECT count(*) FROM news_items WHERE collected_date = %s", (date_str,))
            after = cur.fetchone()[0]
    inserted = after - before
    logger.info(f"Inserted {inserted}/{len(rows)} items for {date_str} (total in DB: {after})")
    return inserted


# ---------------------------------------------------------------------------
# Article body fetcher
# ---------------------------------------------------------------------------

def fetch_for_article_fetching(
    date_str: str,
    limit: int = 50,
    retry_failed: bool = False,
) -> list[dict[str, Any]]:
    """Return rows whose article body should be fetched."""
    status_filter = (
        "AND (parse_status IS NULL OR parse_status IN ('fetch_error', 'parse_error', 'empty'))"
        if retry_failed
        else "AND parse_status IS NULL"
    )
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT id, title, summary, source_name, link, category,
                       canonical_category, discovery_role, parse_status
                FROM news_items
                WHERE collected_date = %s
                  AND link IS NOT NULL
                  {status_filter}
                ORDER BY id
                LIMIT %s
                """,
                (date_str, limit),
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


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

def fetch_raw_by_date(date_str: str) -> list[dict[str, Any]]:
    """Return all items collected on *date_str* with their raw fields."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM news_items WHERE collected_date = %s ORDER BY id",
                (date_str,),
            )
            return [dict(row) for row in cur.fetchall()]


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

def fetch_processed_by_date(date_str: str) -> list[dict[str, Any]]:
    """Return items with cleaned fields needed for sentiment analysis."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, cleaned_title, cleaned_summary, cleaned_article_text, is_turkish"
                " FROM news_items WHERE collected_date = %s ORDER BY id",
                (date_str,),
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
                    sentiment_label  = %s,
                    sentiment_score  = %s,
                    sentiment_scores = %s,
                    analyzed_at      = %s
                WHERE id = %s
                """,
                [
                    (
                        u.get("sentiment_label"),
                        u.get("sentiment_score"),
                        psycopg2.extras.Json(u.get("sentiment_scores")),
                        u.get("analyzed_at"),
                        u["id"],
                    )
                    for u in updates
                ],
            )
            logger.info(f"Updated {cur.rowcount} sentiment fields")


# ---------------------------------------------------------------------------
# NER
# ---------------------------------------------------------------------------

def fetch_for_ner(date_str: str) -> list[dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, title, summary, article_text, cleaned_article_text, is_turkish
                FROM news_items
                WHERE collected_date = %s
                ORDER BY id
                """,
                (date_str,),
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
# Clustering
# ---------------------------------------------------------------------------

def fetch_for_clustering(date_str: str) -> list[dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, cleaned_title, cleaned_summary, cleaned_article_text, is_turkish,
                       source_name, entities, sentiment_label
                FROM news_items
                WHERE collected_date = %s
                ORDER BY id
                """,
                (date_str,),
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


def upsert_cluster_summaries(summaries: list[dict[str, Any]], date_str: str) -> None:
    if not summaries:
        return
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO cluster_summaries
                    (date, cluster_id, title, size, keywords, sources, sentiment_distribution)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (date, cluster_id) DO UPDATE SET
                    title                  = EXCLUDED.title,
                    size                   = EXCLUDED.size,
                    keywords               = EXCLUDED.keywords,
                    sources                = EXCLUDED.sources,
                    sentiment_distribution = EXCLUDED.sentiment_distribution
                """,
                [
                    (
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

def fetch_for_indexing(date_str: str) -> list[dict[str, Any]]:
    """Return Turkish items with sentiment to be embedded."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, title, cleaned_title, cleaned_summary, cleaned_article_text, source_name,
                       link, sentiment_label, sentiment_score, cluster_id
                FROM news_items
                WHERE collected_date = %s
                  AND is_turkish = true
                  AND sentiment_label IS NOT NULL
                ORDER BY id
                """,
                (date_str,),
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
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (vec_str, vec_str, n),
            )
            return [dict(row) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def fetch_all_for_api(date_str: str) -> list[dict[str, Any]]:
    """Fetch all items for a date with every field the API needs."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    id, title, source_name, published_date, link,
                    is_turkish, sentiment_label, sentiment_score,
                    entities, cluster_id, cluster_title, cluster_keywords
                FROM news_items
                WHERE collected_date = %s
                ORDER BY id
                """,
                (date_str,),
            )
            return [dict(row) for row in cur.fetchall()]


def fetch_cluster_summaries(date_str: str) -> list[dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT cluster_id, title, size, keywords, sources, sentiment_distribution
                FROM cluster_summaries
                WHERE date = %s
                ORDER BY size DESC
                """,
                (date_str,),
            )
            return [dict(row) for row in cur.fetchall()]


def fetch_available_dates() -> list[str]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT collected_date FROM news_items ORDER BY collected_date"
            )
            return [str(row[0]) for row in cur.fetchall()]


def fetch_high_confidence_items(min_confidence: float = 0.85, max_samples: int = 50_000) -> list[dict[str, Any]]:
    """Return high-confidence Turkish news items for retraining.

    Returns id, cleaned_title, cleaned_summary, sentiment_label, sentiment_score.
    Ordered oldest-first so recency cap (LIMIT) keeps the most recent examples.
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT cleaned_title, cleaned_summary, sentiment_label, sentiment_score
                FROM news_items
                WHERE is_turkish = true
                  AND sentiment_label IS NOT NULL
                  AND sentiment_score >= %s
                ORDER BY collected_date DESC
                LIMIT %s
                """,
                (min_confidence, max_samples),
            )
            return [dict(row) for row in cur.fetchall()]


def fetch_sentiment_trend(days: int = 30) -> list[dict[str, Any]]:
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
                  AND sentiment_label IS NOT NULL
                  AND collected_date >= CURRENT_DATE - %s::int
                GROUP BY collected_date
                ORDER BY collected_date
                """,
                (days,),
            )
            return [dict(row) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Drift reports
# ---------------------------------------------------------------------------


def upsert_drift_report(report: dict[str, Any]) -> None:
    """Persist a drift report so the API can expose it as a Prometheus gauge.

    Idempotent on ``(date)``. Re-running the daily drift step overwrites the
    prior row for that day, which matches our daily-pipeline semantics.
    """
    today = report.get("today") or {}
    baseline = report.get("baseline") or {}
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO drift_reports (
                    date, status, psi, severity, baseline_days,
                    today_total, today_ratios, baseline_ratios, per_class_delta
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (date) DO UPDATE SET
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


def fetch_latest_drift_report() -> dict[str, Any] | None:
    """Most recent drift row, or None if the table is empty."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT date::text AS date, status, psi, severity, baseline_days,
                       today_total, today_ratios, baseline_ratios, per_class_delta,
                       computed_at
                FROM drift_reports
                ORDER BY date DESC
                LIMIT 1
                """,
            )
            row = cur.fetchone()
            return dict(row) if row else None


def fetch_drift_history(days: int = 30) -> list[dict[str, Any]]:
    """Recent drift rows for trend charts; oldest first so charts plot left-to-right."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT date::text AS date, status, psi, severity, baseline_days,
                       today_total, today_ratios, baseline_ratios, per_class_delta
                FROM drift_reports
                WHERE date >= CURRENT_DATE - %s::int
                ORDER BY date
                """,
                (days,),
            )
            return [dict(row) for row in cur.fetchall()]
