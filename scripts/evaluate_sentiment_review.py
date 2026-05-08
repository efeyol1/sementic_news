"""Evaluate reviewed sentiment CSV against current model labels.

Usage:
    python scripts/evaluate_sentiment_review.py data/qa/sentiment_review_2026-05-06.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

_LABELS = ("negative", "neutral", "positive")


def _pct(part: int, whole: int) -> float:
    return round(part / whole * 100, 1) if whole else 0.0


def load_reviewed_rows(path: Path) -> tuple[list[dict[str, str]], int]:
    """Return (usable rows, count of skipped rows missing a usable label pair)."""
    with open(path, encoding="utf-8", newline="") as fh:
        all_rows = list(csv.DictReader(fh))
    usable = [
        row for row in all_rows
        if row.get("reviewed_label") in _LABELS and row.get("current_label") in _LABELS
    ]
    return usable, len(all_rows) - len(usable)


def evaluate_rows(rows: list[dict[str, str]], skipped: int = 0) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if row["current_label"] == row["reviewed_label"])
    confusion: dict[str, dict[str, int]] = {
        gold: {pred: 0 for pred in _LABELS}
        for gold in _LABELS
    }
    for row in rows:
        confusion[row["reviewed_label"]][row["current_label"]] += 1

    per_label = {}
    for label in _LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[gold][label] for gold in _LABELS if gold != label)
        fn = sum(confusion[label][pred] for pred in _LABELS if pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(confusion[label].values()),
        }

    macro_f1 = sum(per_label[label]["f1"] for label in _LABELS) / len(_LABELS)
    return {
        "total_reviewed": total,
        "skipped_rows": skipped,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "accuracy_pct": _pct(correct, total),
        "macro_f1": round(macro_f1, 4),
        "gold_counts": dict(Counter(row["reviewed_label"] for row in rows)),
        "pred_counts": dict(Counter(row["current_label"] for row in rows)),
        "confusion": confusion,
        "per_label": per_label,
    }


def format_summary(summary: dict[str, Any], path: Path) -> str:
    lines = [
        f"Sentiment review evaluation: {path}",
        "",
        f"Reviewed rows: {summary['total_reviewed']}",
        f"Skipped rows (no reviewed_label): {summary['skipped_rows']}",
        f"Accuracy: {summary['accuracy_pct']}%",
        f"Macro F1: {summary['macro_f1']}",
        f"Gold counts: {summary['gold_counts']}",
        f"Pred counts: {summary['pred_counts']}",
        "",
        "Per label:",
    ]
    for label, metrics in summary["per_label"].items():
        lines.append(
            "  "
            f"{label}: precision={metrics['precision']} recall={metrics['recall']} "
            f"f1={metrics['f1']} support={metrics['support']}"
        )
    lines.extend(["", "Confusion matrix (gold -> predicted):"])
    for gold, preds in summary["confusion"].items():
        lines.append(f"  {gold}: {preds}")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate reviewed sentiment CSV labels.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--json", action="store_true", help="Print machine-readable summary JSON.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    rows, skipped = load_reviewed_rows(args.path)
    summary = evaluate_rows(rows, skipped=skipped)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(format_summary(summary, args.path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
