"""Summarize article body parse quality by source and category.

Usage:
    python scripts/analyze_article_parse_quality.py --date 2026-05-05
    python scripts/analyze_article_parse_quality.py --date 2026-05-05 --json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import date
from statistics import mean
from typing import Any

from src.db.queries import fetch_article_parse_quality_rows

_STATUSES = ("parsed", "empty", "fetch_error", "parse_error", "skipped", "unfetched")


def _pct(part: int, whole: int) -> float:
    return round(part / whole * 100, 1) if whole else 0.0


def _body_stats(chars: list[int]) -> dict[str, int | float]:
    if not chars:
        return {"avg_chars": 0.0, "max_chars": 0}
    return {
        "avg_chars": round(mean(chars), 1),
        "max_chars": max(chars),
    }


def _summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(row.get("parse_status") or "unfetched") for row in rows)
    total = len(rows)
    fetched = total - statuses.get("unfetched", 0)
    parsed = statuses.get("parsed", 0)
    parsed_chars = [
        int(row.get("article_chars") or 0)
        for row in rows
        if (row.get("parse_status") or "unfetched") == "parsed"
    ]

    result: dict[str, Any] = {
        "total": total,
        "fetched": fetched,
        "coverage_pct": _pct(fetched, total),
        "parsed_pct_fetched": _pct(parsed, fetched),
        "statuses": {status: statuses.get(status, 0) for status in _STATUSES},
    }
    result.update(_body_stats(parsed_chars))
    return result


def summarize_rows(
    rows: list[dict[str, Any]],
    min_fetched: int = 10,
    min_parsed_pct: float = 85.0,
) -> dict[str, Any]:
    """Build source/category quality summary from DB rows."""
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row.get("source_name") or "unknown")].append(row)
        by_category[str(row.get("category") or "unknown")].append(row)

    source_rows = []
    for source, source_items in by_source.items():
        summary = _summarize_group(source_items)
        needs_attention = (
            summary["fetched"] >= min_fetched
            and summary["parsed_pct_fetched"] < min_parsed_pct
        )
        source_rows.append(
            {
                "source": source,
                **summary,
                "needs_attention": needs_attention,
            }
        )
    source_rows.sort(key=lambda row: (row["needs_attention"], row["fetched"], row["total"]), reverse=True)

    category_rows = []
    for category, category_items in by_category.items():
        category_rows.append({"category": category, **_summarize_group(category_items)})
    category_rows.sort(key=lambda row: row["fetched"], reverse=True)

    totals = _summarize_group(rows)
    return {
        "totals": totals,
        "sources": source_rows,
        "categories": category_rows,
        "thresholds": {
            "min_fetched": min_fetched,
            "min_parsed_pct": min_parsed_pct,
        },
    }


def format_summary(summary: dict[str, Any], date_str: str) -> str:
    totals = summary["totals"]
    lines = [
        f"Article parse quality: {date_str}",
        "",
        "Totals:",
        f"  total rows: {totals['total']}",
        f"  fetched rows: {totals['fetched']} ({totals['coverage_pct']}% coverage)",
        f"  parsed of fetched: {totals['statuses']['parsed']} ({totals['parsed_pct_fetched']}%)",
        f"  empty/fetch_error/parse_error: "
        f"{totals['statuses']['empty']}/"
        f"{totals['statuses']['fetch_error']}/"
        f"{totals['statuses']['parse_error']}",
        f"  avg parsed chars: {totals['avg_chars']}",
        "",
        "Sources:",
    ]

    for row in summary["sources"]:
        marker = " !" if row["needs_attention"] else ""
        lines.append(
            "  "
            f"{row['source']}{marker}: fetched={row['fetched']}/{row['total']} "
            f"coverage={row['coverage_pct']}% parsed={row['parsed_pct_fetched']}% "
            f"statuses={row['statuses']} avg_chars={row['avg_chars']}"
        )

    attention = [row for row in summary["sources"] if row["needs_attention"]]
    if attention:
        lines.extend(["", "Needs attention:"])
        for row in attention:
            lines.append(
                "  "
                f"{row['source']}: parsed_pct_fetched={row['parsed_pct_fetched']}% "
                f"below threshold {summary['thresholds']['min_parsed_pct']}%"
            )

    lines.extend(["", "Categories:"])
    for row in summary["categories"]:
        lines.append(
            "  "
            f"{row['category']}: fetched={row['fetched']}/{row['total']} "
            f"parsed={row['parsed_pct_fetched']}% avg_chars={row['avg_chars']}"
        )

    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize article body parse quality from PostgreSQL.")
    parser.add_argument("--date", default=date.today().isoformat(), metavar="YYYY-MM-DD")
    parser.add_argument("--min-fetched", type=int, default=10)
    parser.add_argument("--min-parsed-pct", type=float, default=85.0)
    parser.add_argument("--json", action="store_true", help="Print machine-readable summary JSON.")
    args = parser.parse_args()
    if args.min_fetched < 1:
        parser.error("--min-fetched must be positive")
    if not 0 <= args.min_parsed_pct <= 100:
        parser.error("--min-parsed-pct must be between 0 and 100")
    return args


def main() -> int:
    args = _parse_args()
    rows = fetch_article_parse_quality_rows(args.date)
    summary = summarize_rows(
        rows,
        min_fetched=args.min_fetched,
        min_parsed_pct=args.min_parsed_pct,
    )
    if args.json:
        print(json.dumps({"date": args.date, **summary}, ensure_ascii=False, indent=2))
    else:
        print(format_summary(summary, args.date))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
