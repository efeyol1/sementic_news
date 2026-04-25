"""All database query functions for the pipeline and API."""

from __future__ import annotations

import json
from typing import Any

import psycopg2.extras
from loguru import logger

from src.db.client import get_conn


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
            inserted = cur.rowcount
    logger.info(f"Inserted {inserted}/{len(rows)} items for {date_str}")
    return inserted


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
                        cleaned_title   = %s,
                        cleaned_summary = %s,
                        is_turkish      = %s,
                        char_count      = %s,
                        published_date  = %s
                    WHERE id = %s
                    """,
                    [
                        (
                            u["cleaned_title"],
                            u["cleaned_summary"],
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
                "SELECT id, cleaned_title, cleaned_summary, is_turkish FROM news_items WHERE collected_date = %s ORDER BY id",
                (date_str,),
            )
            return [dict(row) for row in cur.fetchall()]


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
                "SELECT id, title, summary, is_turkish FROM news_items WHERE collected_date = %s ORDER BY id",
                (date_str,),
            )
            return [dict(row) for row in cur.fetchall()]


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
                SELECT id, cleaned_title, cleaned_summary, is_turkish,
                       source_name, entities, sentiment_label
                FROM news_items
                WHERE collected_date = %s
                ORDER BY id
                """,
                (date_str,),
            )
            return [dict(row) for row in cur.fetchall()]


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
# Vector store
# ---------------------------------------------------------------------------

def fetch_for_indexing(date_str: str) -> list[dict[str, Any]]:
    """Return Turkish items with sentiment for ChromaDB indexing."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, title, cleaned_title, cleaned_summary, source_name,
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
