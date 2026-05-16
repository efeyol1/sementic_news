"""Audit entity canonicalization and resolution quality.

Usage:
    python scripts/audit_entity_resolution.py --country germany --date 2026-05-14
    python scripts/audit_entity_resolution.py --country germany --date 2026-05-14 --json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import date
from typing import Any

from src.config import load_country_config
from src.db.queries import fetch_entity_resolution_audit_rows


def summarize_rows(rows: list[dict[str, Any]], max_examples: int = 20) -> dict[str, Any]:
    total_mentions = sum(int(row.get("mention_count") or 0) for row in rows)
    by_canonical: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_canonical[(str(row.get("entity_type")), str(row.get("canonical")))].append(row)

    alias_merges = []
    suspicious_merges = []
    for (entity_type, canonical), group in by_canonical.items():
        aliases = sorted({str(row.get("entity_text")) for row in group})
        mentions = sum(int(row.get("mention_count") or 0) for row in group)
        if len(aliases) > 1:
            entry = {
                "entity_type": entity_type,
                "canonical": canonical,
                "aliases": aliases,
                "alias_count": len(aliases),
                "mention_count": mentions,
            }
            alias_merges.append(entry)
            methods = {row.get("resolver_method") for row in group}
            qids = {row.get("wikidata_qid") for row in group if row.get("wikidata_qid")}
            if len(qids) > 1 or (len(aliases) >= 5 and "local_alias" not in methods):
                suspicious_merges.append(entry)

    unresolved = [
        row
        for row in rows
        if not row.get("wikidata_qid") and row.get("resolver_method") != "local_alias"
    ]
    unresolved.sort(key=lambda row: int(row.get("mention_count") or 0), reverse=True)

    top_by_type: dict[str, list[dict[str, Any]]] = {}
    type_counts = Counter(str(row.get("entity_type")) for row in rows)
    for entity_type in sorted(type_counts):
        grouped = [
            {
                "canonical": canonical,
                "mention_count": sum(int(row.get("mention_count") or 0) for row in group),
                "alias_count": len({str(row.get("entity_text")) for row in group}),
            }
            for (etype, canonical), group in by_canonical.items()
            if etype == entity_type
        ]
        grouped.sort(key=lambda row: row["mention_count"], reverse=True)
        top_by_type[entity_type] = grouped[:max_examples]

    alias_merges.sort(key=lambda row: (row["alias_count"], row["mention_count"]), reverse=True)
    suspicious_merges.sort(key=lambda row: (row["alias_count"], row["mention_count"]), reverse=True)

    return {
        "totals": {
            "mention_rows": len(rows),
            "mentions": total_mentions,
            "canonical_count": len(by_canonical),
            "alias_merge_count": len(alias_merges),
            "unresolved_count": len(unresolved),
            "suspicious_merge_count": len(suspicious_merges),
        },
        "alias_merges": alias_merges[:max_examples],
        "unresolved_top": unresolved[:max_examples],
        "suspicious_merges": suspicious_merges[:max_examples],
        "top_by_type": top_by_type,
    }


def format_summary(summary: dict[str, Any], date_str: str, country_code: str) -> str:
    totals = summary["totals"]
    lines = [
        f"Entity resolution audit: {date_str} [{country_code}]",
        "",
        "Totals:",
        f"  mention groups: {totals['mention_rows']}",
        f"  mentions: {totals['mentions']}",
        f"  canonical entities: {totals['canonical_count']}",
        f"  alias merges: {totals['alias_merge_count']}",
        f"  unresolved groups: {totals['unresolved_count']}",
        f"  suspicious merges: {totals['suspicious_merge_count']}",
    ]

    lines.extend(["", "Alias merges:"])
    for row in summary["alias_merges"]:
        lines.append(
            "  "
            f"{row['entity_type']} | {row['canonical']} | mentions={row['mention_count']} "
            f"aliases={row['aliases']}"
        )

    lines.extend(["", "Unresolved top:"])
    for row in summary["unresolved_top"]:
        lines.append(
            "  "
            f"{row['entity_type']} | {row['entity_text']} | canonical={row['canonical']} "
            f"mentions={row['mention_count']} method={row.get('resolver_method')}"
        )

    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit entity canonicalization/resolution.")
    parser.add_argument("--date", default=date.today().isoformat(), metavar="YYYY-MM-DD")
    parser.add_argument("--country", default="turkey", help="Country slug or ISO code")
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.max_examples < 1:
        parser.error("--max-examples must be positive")
    return args


def main() -> int:
    args = _parse_args()
    country_config = load_country_config(args.country)
    country_code = country_config["country_code"]
    rows = fetch_entity_resolution_audit_rows(args.date, country_code=country_code)
    summary = summarize_rows(rows, max_examples=args.max_examples)
    if args.json:
        print(json.dumps(
            {"date": args.date, "country_code": country_code, **summary},
            ensure_ascii=False,
            indent=2,
            default=str,
        ))
    else:
        print(format_summary(summary, args.date, country_code))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
