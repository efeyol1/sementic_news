from __future__ import annotations

import csv

from evaluate_sentiment_review import evaluate_rows, format_summary, load_reviewed_rows


def test_evaluate_rows_reports_accuracy_macro_f1_and_confusion(tmp_path):
    rows = [
        {"current_label": "positive", "reviewed_label": "positive"},
        {"current_label": "positive", "reviewed_label": "negative"},
        {"current_label": "neutral", "reviewed_label": "neutral"},
        {"current_label": "negative", "reviewed_label": ""},
    ]
    path = tmp_path / "review.csv"
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["current_label", "reviewed_label"])
        writer.writeheader()
        writer.writerows(rows)

    reviewed, skipped = load_reviewed_rows(path)
    summary = evaluate_rows(reviewed, skipped=skipped)

    assert summary["total_reviewed"] == 3
    assert summary["skipped_rows"] == 1
    assert summary["accuracy_pct"] == 66.7
    assert summary["confusion"]["negative"]["positive"] == 1
    assert summary["confusion"]["positive"]["positive"] == 1
    assert summary["confusion"]["neutral"]["neutral"] == 1
    assert summary["per_label"]["positive"]["precision"] == 0.5

    text = format_summary(summary, path)
    assert "Sentiment review evaluation:" in text
    assert "Skipped rows (no reviewed_label): 1" in text
    assert "Macro F1:" in text
