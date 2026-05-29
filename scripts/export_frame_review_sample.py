"""Export a stratified CSV for manual frame-bridge review (Sprint 10 QA).

Review workflow:
  1. Run this script for a country (and optionally window/date).
  2. For each row, read the top collocates and fill ``reviewed_frame`` with
     the frame a human judges dominant (economic / security / identity /
     governance / humanitarian / conflict), or ``none`` if no frame fits.
  3. Feed the file to evaluate_frame_review.py for top-1 accuracy + macro F1.

Stratified by predicted dominant frame so every frame gets representation
even when one frame dominates the corpus.

Usage:
    python scripts/export_frame_review_sample.py --country germany
    python scripts/export_frame_review_sample.py --country germany --per-frame 8 --window 30
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

# Reuse the auditor's dominant-frame rule so predicted == what the audit sees.
from audit_frame_bridge import dominant_frame  # noqa: E402  (scripts/ on path)

from src.analysis.frame_bridge import FRAME_NAMES
from src.config import load_country_config
from src.db.queries import (
    fetch_country_profiles_for_frame_audit,
    fetch_latest_profile_end_date,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "data" / "qa"
_CSV_FIELDS = (
    "country_code",
    "window_days",
    "end_date",
    "canonical",
    "wikidata_qid",
    "entity_type",
    "top_collocates",
    "predicted_frame",
    "reviewed_frame",
    "review_notes",
)


def _collocates_str(top_collocates: list[dict[str, Any]] | None, k: int = 12) -> str:
    """Render top-k collocate lemmas (LLR-ranked already) for the reviewer."""
    out = []
    for c in (top_collocates or [])[:k]:
        out.append(f"{c.get('lemma')}({round(float(c.get('llr', 0)), 1)})")
    return ", ".join(out)


def build_sample(
    rows: list[dict[str, Any]],
    per_frame: int,
    *,
    seed: int = 13,
) -> list[dict[str, Any]]:
    """Stratify rows by predicted dominant frame, sample up to per_frame each."""
    by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        dom = dominant_frame(row.get("frame_intensities"))
        if dom is None:
            continue
        by_frame[dom].append(row)

    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    for frame in FRAME_NAMES:
        bucket = by_frame.get(frame, [])
        rng.shuffle(bucket)
        for row in bucket[:per_frame]:
            end_date = row.get("end_date")
            sample.append(
                {
                    "country_code": row["country_code"],
                    "window_days": row["window_days"],
                    "end_date": (
                        end_date.isoformat()
                        if hasattr(end_date, "isoformat")
                        else str(end_date)
                    ),
                    "canonical": row["canonical"],
                    "wikidata_qid": row.get("wikidata_qid") or "",
                    "entity_type": row["entity_type"],
                    "top_collocates": _collocates_str(row.get("top_collocates")),
                    "predicted_frame": frame,
                    "reviewed_frame": "",
                    "review_notes": "",
                }
            )
    return sample


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export frame review sample CSV.")
    parser.add_argument("--country", required=True, help="country slug or code")
    parser.add_argument("--window", type=int, default=30, choices=(7, 30))
    parser.add_argument("--date", default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--per-frame", type=int, default=8)
    parser.add_argument("--output", default=None, help="output CSV path")
    args = parser.parse_args(argv)

    config = load_country_config(args.country)
    country_code = config["country_code"]
    end_date = args.date or fetch_latest_profile_end_date(country_code)
    rows = fetch_country_profiles_for_frame_audit(
        country_code, args.window, end_date=end_date
    )
    sample = build_sample(rows, args.per_frame)

    out_path = (
        Path(args.output)
        if args.output
        else _DEFAULT_OUTPUT_DIR
        / f"frame_review_{country_code}_{args.window}d_{end_date}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(sample)

    print(f"Wrote {len(sample)} rows → {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
