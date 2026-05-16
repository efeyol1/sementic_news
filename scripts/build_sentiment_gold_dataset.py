"""Build a JSONL gold dataset from reviewed sentiment QA CSV files.

The output is intended for local training/evaluation. Prefer writing it to
`/private/tmp` because the rows may contain news text snippets.

Usage:
    python scripts/build_sentiment_gold_dataset.py \
        data/qa/sentiment_review_germany_2026-05-14.csv \
        --country DE \
        --output /private/tmp/sentiment_gold_de.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

LABELS = {"negative", "neutral", "positive"}


def _build_text(row: dict[str, str]) -> str:
    parts = [
        row.get("title", ""),
        row.get("summary", ""),
        row.get("body_snippet", ""),
    ]
    return "\n".join(part.strip() for part in parts if part and part.strip())


def build_dataset(
    inputs: list[Path],
    output: Path,
    country: str,
    min_chars: int = 20,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    seen_ids: set[tuple[str, str]] = set()
    with output.open("w", encoding="utf-8") as out:
        for path in inputs:
            with path.open(encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    label = (row.get("reviewed_label") or "").strip().lower()
                    if label not in LABELS:
                        continue
                    text = _build_text(row)
                    if len(text) < min_chars:
                        continue
                    key = (country, row.get("id") or text)
                    if key in seen_ids:
                        continue
                    seen_ids.add(key)
                    out.write(
                        json.dumps(
                            {
                                "text": text,
                                "label": label,
                                "country": country,
                                "source_name": row.get("source_name"),
                                "row_id": row.get("id"),
                                "collected_date": row.get("collected_date") or "",
                                "label_source": row.get("label_source") or "",
                                "review_status": row.get("review_status") or "",
                                "source_file": str(path),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    written += 1

    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build JSONL sentiment gold data from reviewed CSV exports."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--country", required=True, help="Country code, e.g. DE")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/sentiment_gold.jsonl"),
    )
    parser.add_argument("--min-chars", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    written = build_dataset(
        inputs=args.inputs,
        output=args.output,
        country=args.country,
        min_chars=args.min_chars,
    )
    print(f"Wrote {written} reviewed examples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
