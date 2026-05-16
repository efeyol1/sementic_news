"""Confidence calibration for persisted sentiment scores.

The first rollout is heuristic by design: it preserves the predicted label and
only makes the reported confidence more conservative when the top two classes
are close. Raw model scores remain stored for debugging and future learned
calibration.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from decimal import Decimal
from typing import Any

from loguru import logger

from src.config import load_country_config
from src.db.queries import bulk_update_sentiment_calibration, fetch_sentiment_quality_rows

CALIBRATION_METHOD = "margin_heuristic_v1"
_LABELS = ("negative", "neutral", "positive")


def calibrate_sentiment(
    sentiment_scores: dict[str, Any] | None,
    sentiment_label: str | None = None,
) -> dict[str, Any]:
    """Return calibrated confidence fields without changing the label."""
    scores = _coerce_scores(sentiment_scores)
    if not scores:
        return {
            "calibrated_sentiment_score": None,
            "calibrated_sentiment_scores": None,
            "calibration_method": None,
        }

    label = sentiment_label if sentiment_label in scores else max(scores, key=scores.get)
    top_score = _clip(scores[label])
    second_score = max((_clip(v) for k, v in scores.items() if k != label), default=0.0)
    margin = max(0.0, top_score - second_score)

    # A confident model should be confident because the winner is both high and
    # clearly separated from the runner-up. Margins below ~0.10 get pulled down
    # hard; margins above ~0.75 are left almost unchanged.
    margin_factor = 0.65 + 0.35 * min(1.0, margin / 0.75)
    calibrated = _clip(top_score * margin_factor)

    calibrated_scores = dict(scores)
    calibrated_scores[label] = calibrated

    return {
        "calibrated_sentiment_score": round(calibrated, 6),
        "calibrated_sentiment_scores": {
            key: round(_clip(value), 6) for key, value in calibrated_scores.items()
        },
        "calibration_method": CALIBRATION_METHOD,
    }


def calibrate_date(
    date_str: str | None = None,
    country_config: dict[str, Any] | None = None,
) -> int:
    """Backfill calibrated confidence fields for one country/date."""
    if country_config is None:
        raise ValueError("country_config is required for sentiment calibration")
    date_str = date_str or date.today().isoformat()
    country_code = country_config["country_code"]
    rows = fetch_sentiment_quality_rows(date_str, country_code=country_code)
    updates = []
    for row in rows:
        if not row.get("sentiment_label"):
            continue
        calibrated = calibrate_sentiment(
            row.get("sentiment_scores"),
            sentiment_label=row.get("sentiment_label"),
        )
        updates.append(
            {
                "id": row["id"],
                "raw_sentiment_score": row.get("sentiment_score"),
                **calibrated,
            }
        )
    bulk_update_sentiment_calibration(updates)
    logger.info(
        f"sentiment_calibration done — {len(updates)} rows for {date_str} [{country_code}]"
    )
    return len(updates)


def _coerce_scores(scores: dict[str, Any] | None) -> dict[str, float]:
    if not isinstance(scores, dict):
        return {}
    coerced: dict[str, float] = {}
    for label in _LABELS:
        if scores.get(label) is None:
            continue
        value = scores[label]
        if isinstance(value, Decimal):
            value = float(value)
        coerced[label] = _clip(float(value))
    return coerced


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill calibrated sentiment confidence scores.")
    parser.add_argument("--date", default=date.today().isoformat(), metavar="YYYY-MM-DD")
    parser.add_argument("--country", default="turkey", help="Country slug or ISO code (default: turkey)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_country_config(args.country)
    calibrate_date(date_str=args.date, country_config=cfg)
    sys.exit(0)
