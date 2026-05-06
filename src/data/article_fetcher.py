"""Fetch article body text for discovered news items.

This is intentionally a standalone CLI step while the parser is being
validated. It backfills optional article-body columns without changing the
main daily pipeline behavior.

Usage:
    python -m src.data.article_fetcher --date 2026-05-05 --limit 20
    python -m src.data.article_fetcher --date 2026-05-05 --limit 100 --workers 8
    python -m src.data.article_fetcher --date 2026-05-05 --per-source-limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

from loguru import logger

from src.data.article_parser import parse_article_html
from src.data.canonical_categories import normalize_article_url
from src.db.queries import bulk_update_article_parse, fetch_for_article_fetching
from src.db.schema import init_db

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DISCOVERY_DIR = _REPO_ROOT / "data" / "discovery"
_HTTP_USER_AGENT = (
    "Mozilla/5.0 (compatible; SemanticNewsBot/1.0; +https://github.com/efeyol11/sementic_news)"
)
_DEFAULT_TIMEOUT_SEC = 12


def _default_discovery_file(date_str: str) -> Path:
    return _DISCOVERY_DIR / f"discovered_urls_{date_str}.json"


def _prefer_metadata(current: dict[str, Any] | None, candidate: dict[str, Any]) -> dict[str, Any]:
    if current is None:
        return candidate
    current_role = current.get("discovery_role")
    candidate_role = candidate.get("discovery_role")
    current_category = current.get("canonical_category")
    candidate_category = candidate.get("canonical_category")

    if current_role == "general_discovery" and candidate_role == "category":
        return candidate
    if current_category == "other" and candidate_category and candidate_category != "other":
        return candidate
    return current


def load_discovery_metadata(date_str: str, path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Load URL-level discovery metadata keyed by canonical article URL."""
    report_path = Path(path) if path else _default_discovery_file(date_str)
    if not report_path.exists():
        logger.warning(f"Discovery URL report not found: {report_path}")
        return {}

    with open(report_path, encoding="utf-8") as fh:
        payload = json.load(fh)

    metadata: dict[str, dict[str, Any]] = {}
    for record in payload.get("records", []):
        url = normalize_article_url(record.get("canonical_url") or record.get("url") or "")
        if not url:
            continue
        candidate = {
            "canonical_category": record.get("canonical_category"),
            "discovery_role": record.get("discovery_role"),
            "discovery_url": record.get("discovery_url"),
        }
        metadata[url] = _prefer_metadata(metadata.get(url), candidate)
    logger.info(f"Loaded discovery metadata for {len(metadata)} URLs from {report_path}")
    return metadata


def _http_get_text(url: str, timeout: int = _DEFAULT_TIMEOUT_SEC) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _HTTP_USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    return body.decode("utf-8", errors="replace")


def _to_update(
    row: dict[str, Any],
    metadata_by_url: dict[str, dict[str, Any]],
    timeout: int,
) -> dict[str, Any]:
    row_id = row["id"]
    url = normalize_article_url(row.get("link") or "")
    metadata = metadata_by_url.get(url, {})

    base = {
        "id": row_id,
        "canonical_category": metadata.get("canonical_category"),
        "discovery_role": metadata.get("discovery_role"),
    }
    if not url:
        return {
            **base,
            "article_text": None,
            "cleaned_article_text": None,
            "parse_status": "skipped",
            "parse_error": "missing link",
        }

    try:
        html = _http_get_text(url, timeout=timeout)
    except Exception as exc:
        return {
            **base,
            "article_text": None,
            "cleaned_article_text": None,
            "parse_status": "fetch_error",
            "parse_error": str(exc)[:500],
        }

    try:
        parsed = parse_article_html(html)
    except Exception as exc:
        return {
            **base,
            "article_text": None,
            "cleaned_article_text": None,
            "parse_status": "parse_error",
            "parse_error": str(exc)[:500],
        }

    return {
        **base,
        "article_text": parsed.article_text or None,
        "cleaned_article_text": parsed.cleaned_article_text or None,
        "parse_status": parsed.parse_status,
        "parse_error": parsed.parse_error,
    }


def fetch_articles(
    date_str: str | None = None,
    limit: int = 50,
    workers: int = 8,
    timeout: int = _DEFAULT_TIMEOUT_SEC,
    retry_failed: bool = False,
    per_source_limit: int | None = None,
    discovery_file: str | Path | None = None,
    ensure_schema: bool = True,
) -> dict[str, int]:
    """Fetch and persist article bodies for one date. Returns status counts."""
    date_str = date_str or date.today().isoformat()
    if ensure_schema:
        init_db()

    metadata_by_url = load_discovery_metadata(date_str, discovery_file)
    rows = fetch_for_article_fetching(
        date_str,
        limit=limit,
        retry_failed=retry_failed,
        per_source_limit=per_source_limit,
    )
    if not rows:
        logger.warning(f"No article rows to fetch for {date_str}")
        return {}

    logger.info(f"Fetching article bodies for {len(rows)} rows ({workers} workers)")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        updates = list(
            pool.map(
                lambda row: _to_update(row, metadata_by_url, timeout),
                rows,
            )
        )

    bulk_update_article_parse(updates)
    counts = Counter(update["parse_status"] for update in updates)
    logger.info("Article fetch results: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return dict(counts)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch full article text for collected news items.")
    parser.add_argument("--date", default=date.today().isoformat(), metavar="YYYY-MM-DD")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=_DEFAULT_TIMEOUT_SEC)
    parser.add_argument(
        "--per-source-limit",
        type=int,
        default=None,
        help="Maximum rows to fetch per source before applying --limit.",
    )
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--discovery-file", default=None)
    parser.add_argument(
        "--skip-init-db",
        action="store_true",
        help="Do not run Alembic before fetching. Use only when schema is already current.",
    )
    args = parser.parse_args(argv)
    if args.limit < 1 or args.workers < 1 or args.timeout < 1:
        parser.error("--limit, --workers, and --timeout must be positive integers")
    if args.per_source_limit is not None and args.per_source_limit < 1:
        parser.error("--per-source-limit must be a positive integer")
    return args


if __name__ == "__main__":
    args = _parse_args()
    fetch_articles(
        date_str=args.date,
        limit=args.limit,
        workers=args.workers,
        timeout=args.timeout,
        retry_failed=args.retry_failed,
        per_source_limit=args.per_source_limit,
        discovery_file=args.discovery_file,
        ensure_schema=not args.skip_init_db,
    )
    sys.exit(0)
