"""Summarize Semantic News discovery health reports.

Usage:
    python scripts/analyze_discovery_health.py
    python scripts/analyze_discovery_health.py data/discovery/source_health_2026-05-05.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DISCOVERY_DIR = _REPO_ROOT / "data" / "discovery"


def _latest_health_report() -> Path:
    reports = sorted(_DISCOVERY_DIR.glob("source_health_*.json"))
    if not reports:
        raise FileNotFoundError(
            f"No source_health_*.json reports found under {_DISCOVERY_DIR}. "
            "Run `python -m src.data.rss_collector --country turkey` first."
        )
    return reports[-1]


def load_report(path: str | Path | None = None) -> dict[str, Any]:
    report_path = Path(path) if path else _latest_health_report()
    with open(report_path, encoding="utf-8") as fh:
        report = json.load(fh)
    report["_path"] = str(report_path)
    return report


def _nonzero_categories(categories: dict[str, int]) -> dict[str, int]:
    return {category: count for category, count in categories.items() if count}


def _top_categories(categories: dict[str, int], limit: int = 4) -> str:
    nonzero = sorted(_nonzero_categories(categories).items(), key=lambda kv: kv[1], reverse=True)
    if not nonzero:
        return "-"
    return ", ".join(f"{category}:{count}" for category, count in nonzero[:limit])


def summarize_report(report: dict[str, Any]) -> dict[str, Any]:
    sources = report.get("sources", {})
    feeds = report.get("feeds", [])

    warning_feeds = [feed for feed in feeds if feed.get("warning")]
    error_feeds = [feed for feed in feeds if feed.get("error") or not feed.get("parse_ok", True)]
    low_feeds = [feed for feed in feeds if feed.get("below_min_expected")]

    rows = []
    for source_name, data in sources.items():
        raw_count = data.get("raw_item_count", 0) or 0
        unique_count = data.get("unique_url_count", 0) or 0
        other_count = (data.get("categories") or {}).get("other", 0) or 0
        other_pct = round(other_count / raw_count * 100, 1) if raw_count else 0.0
        rows.append(
            {
                "source": source_name,
                "feeds": data.get("discovery_checked_count", 0),
                "raw": raw_count,
                "unique": unique_count,
                "dedupe_loss": max(raw_count - unique_count, 0),
                "other_pct": other_pct,
                "low_categories": data.get("low_categories") or [],
                "top_categories": _top_categories(data.get("categories") or {}),
                "failed_fetch": data.get("failed_fetch_count", 0),
                "failed_parse": data.get("failed_parse_count", 0),
            }
        )

    rows.sort(key=lambda row: row["unique"], reverse=True)
    return {
        "path": report.get("_path"),
        "date": report.get("date"),
        "totals": report.get("totals") or {},
        "sources": rows,
        "warning_feeds": warning_feeds,
        "error_feeds": error_feeds,
        "low_feeds": low_feeds,
    }


def format_summary(summary: dict[str, Any]) -> str:
    totals = summary["totals"]
    lines = [
        f"Discovery health: {summary['date']} ({summary['path']})",
        "",
        "Totals:",
        f"  feed checks: {totals.get('feed_checks', 0)}",
        f"  discovered records: {totals.get('discovered_url_records', 0)}",
        f"  unique discovered URLs: {totals.get('unique_discovered_urls', 0)}",
        f"  inserted news_items: {totals.get('inserted_news_items', 0)}",
        f"  failed fetches: {totals.get('failed_fetch_count', 0)}",
        f"  failed parses: {totals.get('failed_parse_count', 0)}",
        "",
        "Sources:",
    ]

    for row in summary["sources"]:
        low = ",".join(row["low_categories"]) if row["low_categories"] else "-"
        lines.append(
            "  "
            f"{row['source']}: unique={row['unique']} raw={row['raw']} "
            f"feeds={row['feeds']} dedupe_loss={row['dedupe_loss']} "
            f"other={row['other_pct']}% low={low}"
        )
        lines.append(f"    top categories: {row['top_categories']}")

    if summary["low_feeds"]:
        lines.extend(["", "Below minimum feeds:"])
        for feed in summary["low_feeds"]:
            lines.append(
                "  "
                f"{feed['source_name']} | {feed['canonical_category']} | "
                f"{feed['unique_url_count']}/{feed['min_expected_items']} | "
                f"{feed['discovery_url']}"
            )

    if summary["warning_feeds"]:
        lines.extend(["", "Warning feeds:"])
        for feed in summary["warning_feeds"]:
            lines.append(
                "  "
                f"{feed['source_name']} | {feed['source_type']} | "
                f"{feed.get('warning')} | {feed['discovery_url']}"
            )

    if summary["error_feeds"]:
        lines.extend(["", "Error feeds:"])
        for feed in summary["error_feeds"]:
            lines.append(
                "  "
                f"{feed['source_name']} | {feed['source_type']} | "
                f"{feed.get('error')} | {feed['discovery_url']}"
            )

    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize discovery source health JSON.")
    parser.add_argument(
        "report",
        nargs="?",
        help="Path to source_health_YYYY-MM-DD.json. Defaults to latest report.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable summary JSON instead of text.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = summarize_report(load_report(args.report))
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
