"""Daily drift detection for the news-sentiment pipeline.

Compares today's sentiment-label distribution against a rolling 30-day
baseline using Population Stability Index (PSI). Writes a JSON artifact
to ``data/drift_reports/<country_code>_<date>.json`` and, when running under GitHub
Actions, appends a one-screen summary to ``$GITHUB_STEP_SUMMARY``.

Designed to fail-soft: if the baseline is too thin (cold start, after a
prolonged outage) or the data layer is unreachable, the script logs and
exits 0 — drift checks should never block the daily pipeline.

PSI thresholds (industry standard):
    PSI < 0.10  → stable
    PSI < 0.25  → moderate shift, watch
    PSI ≥ 0.25  → significant shift, investigate

Usage:
    python -m src.monitoring.drift                # today vs prior 30 days
    python -m src.monitoring.drift --date 2026-04-26
    python -m src.monitoring.drift --window 14    # shorter baseline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import mlflow
import numpy as np
from loguru import logger

from src.db.queries import fetch_sentiment_trend, upsert_drift_report

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPORTS_DIR = _REPO_ROOT / "data" / "drift_reports"

_LABEL_KEYS = ("negative", "neutral", "positive")
_PSI_EPSILON = 1e-6
_BASELINE_MIN_DAYS = 7  # below this we don't trust the baseline
_BASELINE_MIN_ROWS_PER_DAY = 50

mlflow.set_tracking_uri(
    os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{_REPO_ROOT / 'mlflow.db'}")
)
mlflow.set_experiment("news-sentiment")


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------


def population_stability_index(actual: np.ndarray, baseline: np.ndarray) -> float:
    """Standard PSI between two probability vectors over the same bins.

    Both inputs must sum to ~1 and have identical shape. Zero bins are
    smoothed by ``_PSI_EPSILON`` to keep the log term finite.
    """
    a = np.clip(actual.astype(float), _PSI_EPSILON, None)
    b = np.clip(baseline.astype(float), _PSI_EPSILON, None)
    return float(np.sum((a - b) * np.log(a / b)))


def severity_label(psi: float) -> str:
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "moderate"
    return "significant"


# ---------------------------------------------------------------------------
# Data shaping
# ---------------------------------------------------------------------------


def _ratios(row: dict) -> np.ndarray:
    """Convert one trend row to a (neg, neu, pos) probability vector."""
    total = max(int(row["total"]), 1)
    return np.array(
        [int(row[k]) / total for k in _LABEL_KEYS],
        dtype=float,
    )


def _split_today_vs_baseline(
    rows: list[dict], target: str, window: int
) -> tuple[dict | None, list[dict]]:
    """Pull today's row out of the trend; the rest forms the baseline window."""
    today_row = next((r for r in rows if r["date"] == target), None)
    cutoff = (datetime.fromisoformat(target) - timedelta(days=window)).date().isoformat()
    baseline = [
        r
        for r in rows
        if r["date"] != target
        and r["date"] >= cutoff
        and int(r["total"]) >= _BASELINE_MIN_ROWS_PER_DAY
    ]
    return today_row, baseline


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_drift_check(
    target_date: str,
    window: int = 30,
    country_code: str = "TR",
) -> dict:
    """Compute drift for *target_date* against the prior *window* days.

    Returns a JSON-serializable report dict. Always writes the report to
    ``data/drift_reports/<country_code>_<target_date>.json``; never raises on data
    issues — instead reports ``status='insufficient_data'``.
    """
    logger.info(f"Drift check — target={target_date}, country={country_code}, window={window}d")

    rows = fetch_sentiment_trend(days=window + 1, country_code=country_code)
    today_row, baseline_rows = _split_today_vs_baseline(rows, target_date, window)

    report: dict = {
        "date": target_date,
        "country_code": country_code,
        "window_days": window,
        "baseline_days": len(baseline_rows),
        "computed_at": datetime.now(UTC).isoformat(),
    }

    if today_row is None or int(today_row.get("total", 0)) < 1:
        report["status"] = "insufficient_data"
        report["reason"] = f"no rows for {target_date}"
        _persist(report)
        _persist_to_db(report)
        _emit_step_summary(report)
        return report

    if len(baseline_rows) < _BASELINE_MIN_DAYS:
        report["status"] = "insufficient_data"
        report["reason"] = (
            f"baseline has {len(baseline_rows)} qualifying day(s); "
            f"need ≥ {_BASELINE_MIN_DAYS}"
        )
        _persist(report)
        _persist_to_db(report)
        _emit_step_summary(report)
        return report

    today_ratios = _ratios(today_row)
    baseline_avg = np.mean([_ratios(r) for r in baseline_rows], axis=0)

    psi = population_stability_index(today_ratios, baseline_avg)
    severity = severity_label(psi)

    per_class_delta = {
        label: round(float(today_ratios[i] - baseline_avg[i]), 4)
        for i, label in enumerate(_LABEL_KEYS)
    }

    report.update(
        {
            "status": "ok",
            "psi": round(psi, 4),
            "severity": severity,
            "today": {
                "total": int(today_row["total"]),
                "ratios": {k: round(float(today_ratios[i]), 4) for i, k in enumerate(_LABEL_KEYS)},
            },
            "baseline": {
                "ratios": {k: round(float(baseline_avg[i]), 4) for i, k in enumerate(_LABEL_KEYS)},
                "days_used": len(baseline_rows),
            },
            "per_class_delta": per_class_delta,
        }
    )

    _persist(report)
    _persist_to_db(report)
    _log_to_mlflow(target_date, psi, severity, per_class_delta)
    _emit_step_summary(report)

    if severity == "significant":
        logger.warning(f"PSI={psi:.4f} ({severity}) — sentiment distribution shifted")
    else:
        logger.info(f"PSI={psi:.4f} ({severity})")

    return report


# ---------------------------------------------------------------------------
# Side effects (persist / surface)
# ---------------------------------------------------------------------------


def _persist(report: dict) -> Path:
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = _REPORTS_DIR / f"{report.get('country_code', 'TR')}_{report['date']}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info(f"Wrote drift report → {out.relative_to(_REPO_ROOT)}")
    return out


def _persist_to_db(report: dict) -> None:
    """Mirror the JSON report into Postgres so the API can expose it.

    Fail-soft: a DB hiccup mustn't kill the daily pipeline's drift step.
    """
    try:
        upsert_drift_report(report, country_code=report.get("country_code", "TR"))
        logger.info(f"Drift report persisted to DB ({report.get('status')})")
    except Exception as exc:
        logger.warning(f"DB persist skipped: {exc}")


def _log_to_mlflow(target_date: str, psi: float, severity: str, deltas: dict) -> None:
    try:
        with mlflow.start_run(run_name=f"drift-{target_date}"):
            mlflow.log_metric("psi", psi)
            mlflow.set_tag("severity", severity)
            mlflow.set_tag("type", "drift")
            for label, delta in deltas.items():
                mlflow.log_metric(f"delta_{label}", delta)
    except Exception as exc:  # MLflow store can be flaky on CI; don't kill the run
        logger.warning(f"MLflow drift log skipped: {exc}")


def _emit_step_summary(report: dict) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    if report.get("status") == "insufficient_data":
        md = [
            f"### Drift report — {report.get('country_code', 'TR')} {report['date']}",
            "",
            "- **Status**: `insufficient_data`",
            f"- **Reason**: {report.get('reason', 'unknown')}",
            f"- **Baseline available**: {report.get('baseline_days', 0)} day(s)",
            "",
            "_PSI is suppressed until the baseline window has enough qualifying days "
            f"(need ≥ {_BASELINE_MIN_DAYS}). The daily run keeps building history "
            "automatically; no action required._",
            "",
        ]
    else:
        today = report["today"]["ratios"]
        base = report["baseline"]["ratios"]
        md = [
            f"### Drift report — {report.get('country_code', 'TR')} {report['date']}",
            "",
            f"- **PSI**: `{report['psi']}` → **{report['severity'].upper()}**",
            f"- **Today**: {report['today']['total']} items "
            f"(neg={today['negative']}, neu={today['neutral']}, pos={today['positive']})",
            f"- **Baseline**: {report['baseline']['days_used']} day rolling avg "
            f"(neg={base['negative']}, neu={base['neutral']}, pos={base['positive']})",
            "",
            "| Class | Δ vs baseline |",
            "|---|---|",
        ]
        for label, delta in report["per_class_delta"].items():
            md.append(f"| {label} | {delta:+.4f} |")
        md.append("")

    with open(summary_path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Daily sentiment-distribution drift check")
    p.add_argument("--date", default=date.today().isoformat(), metavar="YYYY-MM-DD")
    p.add_argument("--window", type=int, default=30, help="Baseline window in days")
    return p.parse_args(argv)


def main() -> int:
    args = _parse_args()
    try:
        run_drift_check(args.date, window=args.window)
    except Exception as exc:
        logger.warning(f"Drift check failed soft: {exc}")
        return 0  # never block the pipeline on drift instrumentation
    return 0


if __name__ == "__main__":
    sys.exit(main())
