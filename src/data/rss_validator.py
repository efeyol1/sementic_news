"""Discovery health reporting for RSS / sitemap ingestion."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data.canonical_categories import CANONICAL_CATEGORIES

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_REPORT_DIR = _REPO_ROOT / "data" / "discovery"


def latest_item_date(items: list[dict[str, Any]]) -> str | None:
    dates = [str(item.get("published_date") or "") for item in items if item.get("published_date")]
    return max(dates) if dates else None


def build_feed_health(
    *,
    source_name: str,
    source_type: str,
    discovery_url: str,
    discovery_role: str,
    canonical_category: str,
    min_expected_items: int,
    http_status: int | None,
    parse_ok: bool,
    item_count: int,
    unique_url_count: int,
    latest_date: str | None,
    error: str | None = None,
    warning: str | None = None,
) -> dict[str, Any]:
    """Build one serializable feed-health record."""
    return {
        "source_name": source_name,
        "source_type": source_type,
        "discovery_url": discovery_url,
        "discovery_role": discovery_role,
        "canonical_category": canonical_category,
        "min_expected_items": min_expected_items,
        "http_status": http_status,
        "parse_ok": parse_ok,
        "item_count": item_count,
        "unique_url_count": unique_url_count,
        "latest_item_date": latest_date,
        "below_min_expected": unique_url_count < min_expected_items,
        "error": error,
        "warning": warning,
    }


def build_discovery_record(
    *,
    item: dict[str, Any],
    source_name: str,
    source_type: str,
    discovery_url: str,
    discovery_role: str,
    canonical_category: str,
    canonical_url: str,
) -> dict[str, Any]:
    """Build one URL-pool record for later article parsing."""
    return {
        "source_name": source_name,
        "source_type": source_type,
        "discovery_url": discovery_url,
        "discovery_role": discovery_role,
        "canonical_category": canonical_category,
        "canonical_url": canonical_url,
        "url": item.get("link") or "",
        "title": item.get("title") or "",
        "summary": item.get("summary") or "",
        "published_date": item.get("published_date"),
        "rss_category": item.get("category"),
    }


def build_source_health_report(
    *,
    date_str: str,
    country_code: str,
    country_slug: str,
    feed_health: list[dict[str, Any]],
    discovered_records: list[dict[str, Any]],
    inserted_count: int,
) -> dict[str, Any]:
    """Aggregate feed and URL records into a source-level health report."""
    by_source: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "rss_checked_count": 0,
            "discovery_checked_count": 0,
            "raw_item_count": 0,
            "unique_url_count": 0,
            "final_parsed_article_count": 0,
            "failed_fetch_count": 0,
            "failed_parse_count": 0,
            "low_categories": [],
            "categories": {category: 0 for category in CANONICAL_CATEGORIES},
        }
    )

    urls_by_source: dict[str, set[str]] = defaultdict(set)
    categories_by_source: dict[str, Counter] = defaultdict(Counter)
    for record in discovered_records:
        source = record["source_name"]
        canonical_url = record.get("canonical_url") or record.get("url") or ""
        if canonical_url:
            urls_by_source[source].add(canonical_url)
        categories_by_source[source][record["canonical_category"]] += 1

    for health in feed_health:
        source = health["source_name"]
        row = by_source[source]
        row["discovery_checked_count"] += 1
        if health["source_type"] == "rss":
            row["rss_checked_count"] += 1
        row["raw_item_count"] += health["item_count"]
        if health.get("error") or (health.get("http_status") and health["http_status"] >= 400):
            row["failed_fetch_count"] += 1
        if not health["parse_ok"]:
            row["failed_parse_count"] += 1
        if health["below_min_expected"]:
            category = health["canonical_category"]
            if category not in row["low_categories"]:
                row["low_categories"].append(category)

    for source, urls in urls_by_source.items():
        row = by_source[source]
        row["unique_url_count"] = len(urls)
        # Until the article-body parser lands, discovered URL count is the
        # closest proxy for final parsed article count.
        row["final_parsed_article_count"] = len(urls)
        row["categories"].update(dict(categories_by_source[source]))

    return {
        "date": date_str,
        "country_code": country_code,
        "country_slug": country_slug,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "totals": {
            "feed_checks": len(feed_health),
            "discovered_url_records": len(discovered_records),
            "unique_discovered_urls": len(
                {r.get("canonical_url") or r.get("url") for r in discovered_records}
            ),
            "inserted_news_items": inserted_count,
            "failed_fetch_count": sum(1 for h in feed_health if h.get("error")),
            "failed_parse_count": sum(1 for h in feed_health if not h["parse_ok"]),
        },
        "sources": dict(sorted(by_source.items())),
        "feeds": feed_health,
    }


def write_discovery_reports(
    *,
    date_str: str,
    country_code: str,
    country_slug: str,
    feed_health: list[dict[str, Any]],
    discovered_records: list[dict[str, Any]],
    inserted_count: int,
    report_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write discovered URL pool and source health JSON reports."""
    out_dir = Path(report_dir) if report_dir else _DEFAULT_REPORT_DIR
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    discovered_path = out_dir / f"discovered_urls_{date_str}.json"
    health_path = out_dir / f"source_health_{date_str}.json"

    discovered_payload = {
        "date": date_str,
        "country_code": country_code,
        "country_slug": country_slug,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_records": len(discovered_records),
        "unique_urls": len({r.get("canonical_url") or r.get("url") for r in discovered_records}),
        "records": discovered_records,
    }
    health_payload = build_source_health_report(
        date_str=date_str,
        country_code=country_code,
        country_slug=country_slug,
        feed_health=feed_health,
        discovered_records=discovered_records,
        inserted_count=inserted_count,
    )

    with open(discovered_path, "w", encoding="utf-8") as fh:
        json.dump(discovered_payload, fh, ensure_ascii=False, indent=2)
    with open(health_path, "w", encoding="utf-8") as fh:
        json.dump(health_payload, fh, ensure_ascii=False, indent=2)

    return discovered_path, health_path
