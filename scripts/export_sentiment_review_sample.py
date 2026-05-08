"""Export a balanced CSV for manual sentiment review.

Review workflow:
  1. Run this script for a pipeline date.
  2. Fill ``reviewed_label`` with negative / neutral / positive.
  3. Use the reviewed file as the gold set for model QA and retraining.

Usage:
    python scripts/export_sentiment_review_sample.py --date 2026-05-06
"""

from __future__ import annotations

import argparse
import csv
import random
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from src.db.queries import fetch_sentiment_quality_rows

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "data" / "qa"
_LABELS = ("negative", "neutral", "positive")
_CSV_FIELDS = (
    "id",
    "source_name",
    "category",
    "current_label",
    "current_score",
    "reviewed_label",
    "review_notes",
    "title",
    "summary",
    "body_snippet",
    "link",
)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def _snippet(value: Any, max_chars: int) -> str:
    text = str(value or "").strip()
    return text[:max_chars].strip()


def build_sample(
    rows: list[dict[str, Any]],
    per_source_label: int = 2,
    seed: int = 7,
    body_chars: int = 900,
) -> list[dict[str, Any]]:
    """Return a deterministic source x label stratified review sample."""
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        label = row.get("sentiment_label")
        if label not in _LABELS:
            continue
        source = str(row.get("source_name") or "unknown")
        groups.setdefault((source, str(label)), []).append(row)

    sample: list[dict[str, Any]] = []
    for key in sorted(groups):
        group_rows = groups[key][:]
        rng.shuffle(group_rows)
        for row in group_rows[:per_source_label]:
            score = _as_float(row.get("sentiment_score"))
            sample.append(
                {
                    "id": row.get("id"),
                    "source_name": row.get("source_name") or "",
                    "category": row.get("category") or "unknown",
                    "current_label": row.get("sentiment_label") or "",
                    "current_score": round(score, 6) if score is not None else "",
                    "reviewed_label": "",
                    "review_notes": "",
                    "title": row.get("cleaned_title") or "",
                    "summary": row.get("cleaned_summary") or "",
                    "body_snippet": _snippet(row.get("cleaned_article_text"), body_chars),
                    "link": row.get("link") or "",
                }
            )

    sample.sort(key=lambda row: (row["source_name"], row["current_label"], row["id"]))
    return sample


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export sentiment review sample CSV.")
    parser.add_argument("--date", default=date.today().isoformat(), metavar="YYYY-MM-DD")
    parser.add_argument("--per-source-label", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--body-chars", type=int, default=900)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.per_source_label < 1:
        parser.error("--per-source-label must be positive")
    if args.body_chars < 0:
        parser.error("--body-chars must be non-negative")
    return args


def main() -> int:
    args = _parse_args()
    rows = fetch_sentiment_quality_rows(args.date)
    sample = build_sample(
        rows,
        per_source_label=args.per_source_label,
        seed=args.seed,
        body_chars=args.body_chars,
    )
    output_path = args.output or _DEFAULT_OUTPUT_DIR / f"sentiment_review_{args.date}.csv"
    write_csv(sample, output_path)
    print(f"Wrote {len(sample)} review rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
