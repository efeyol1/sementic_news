from __future__ import annotations

import csv
import json

from build_sentiment_gold_dataset import build_dataset


def test_build_dataset_preserves_review_metadata(tmp_path):
    source = tmp_path / "review.csv"
    with source.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "id",
                "collected_date",
                "source_name",
                "reviewed_label",
                "label_source",
                "review_status",
                "title",
                "summary",
                "body_snippet",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": "1",
                "collected_date": "2026-05-14",
                "source_name": "tagesschau",
                "reviewed_label": "neutral",
                "label_source": "assistant_silver",
                "review_status": "silver",
                "title": "regierung trifft sich",
                "summary": "die verhandlungen gehen weiter",
                "body_snippet": "",
            }
        )

    output = tmp_path / "gold.jsonl"

    assert build_dataset([source], output, country="DE") == 1
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["label"] == "neutral"
    assert row["country"] == "DE"
    assert row["collected_date"] == "2026-05-14"
    assert row["label_source"] == "assistant_silver"
    assert row["review_status"] == "silver"
