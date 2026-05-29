"""Evaluate a reviewed frame CSV against the frame bridge's predictions.

Reads the CSV produced by export_frame_review_sample.py after a human has
filled ``reviewed_frame``. Reports top-1 frame ranking accuracy + macro F1
+ confusion, and applies the Sprint 10 acceptance gate (≥70% accuracy).

A row counts only when ``predicted_frame`` and ``reviewed_frame`` are both
real frames; ``reviewed_frame == "none"`` (reviewer saw no fitting frame)
and blanks are skipped and reported separately.

Usage:
    python scripts/evaluate_frame_review.py data/qa/frame_review_DE_30d_2026-05-29.csv
    python scripts/evaluate_frame_review.py <csv> --gate 0.70 --json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from src.analysis.frame_bridge import FRAME_NAMES

ACCEPTANCE_DEFAULT = 0.70


def load_reviewed_rows(path: Path) -> tuple[list[dict[str, str]], int]:
    """Return (usable rows, skipped count). Usable = both labels are frames."""
    with open(path, encoding="utf-8", newline="") as fh:
        all_rows = list(csv.DictReader(fh))
    usable = [
        r
        for r in all_rows
        if r.get("reviewed_frame") in FRAME_NAMES
        and r.get("predicted_frame") in FRAME_NAMES
    ]
    return usable, len(all_rows) - len(usable)


def evaluate_rows(rows: list[dict[str, str]], skipped: int = 0) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for r in rows if r["predicted_frame"] == r["reviewed_frame"])
    confusion = {g: {p: 0 for p in FRAME_NAMES} for g in FRAME_NAMES}
    for r in rows:
        confusion[r["reviewed_frame"]][r["predicted_frame"]] += 1

    per_frame = {}
    for frame in FRAME_NAMES:
        tp = confusion[frame][frame]
        fp = sum(confusion[g][frame] for g in FRAME_NAMES if g != frame)
        fn = sum(confusion[frame][p] for p in FRAME_NAMES if p != frame)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_frame[frame] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(confusion[frame].values()),
        }

    macro_f1 = sum(per_frame[f]["f1"] for f in FRAME_NAMES) / len(FRAME_NAMES)
    accuracy = correct / total if total else 0.0
    return {
        "evaluated": total,
        "skipped": skipped,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_frame": per_frame,
        "confusion": confusion,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate reviewed frame CSV.")
    parser.add_argument("csv_path", help="reviewed frame_review_*.csv")
    parser.add_argument("--gate", type=float, default=ACCEPTANCE_DEFAULT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    rows, skipped = load_reviewed_rows(Path(args.csv_path))
    if not rows:
        print("No usable rows (need predicted_frame + reviewed_frame). Skipped "
              f"{skipped}.")
        return 1
    report = evaluate_rows(rows, skipped)
    passed = report["accuracy"] >= args.gate
    report["gate"] = args.gate
    report["passed"] = passed

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"\nFrame review evaluation ({report['evaluated']} rows, "
              f"{report['skipped']} skipped)")
        print("=" * 48)
        print(f"accuracy : {report['accuracy']:.1%}  (gate ≥ {args.gate:.0%})")
        print(f"macro F1 : {report['macro_f1']:.4f}")
        for frame in FRAME_NAMES:
            pf = report["per_frame"][frame]
            print(f"    {frame:<14} P={pf['precision']:.2f} R={pf['recall']:.2f} "
                  f"F1={pf['f1']:.2f} (n={pf['support']})")
        print("=" * 48)
        print("GATE: " + ("PASS ✅" if passed else "FAIL ❌ → Sprint 12 classifier"))
    return 0 if passed else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
