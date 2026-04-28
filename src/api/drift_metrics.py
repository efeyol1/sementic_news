"""Prometheus exporter for the latest drift report.

Registers a custom collector with the global ``prometheus_client`` registry
so the existing ``/metrics`` endpoint (exposed by
``prometheus_fastapi_instrumentator``) also serves drift gauges. Grafana
panels then read these like any other Prometheus series — no extra
datasource needed.

A scrape-time DB query is wasteful (Prometheus scrapes every 15s by default)
so we cache the latest row in-process for 5 minutes. The API process is
small and single-tenant; a simple module-level cache is enough.
"""

from __future__ import annotations

import time

from loguru import logger
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import REGISTRY

from src.db.queries import fetch_latest_drift_report

_CACHE_TTL_SEC = 300  # 5 minutes — drift is daily, no benefit from fresher reads
_SEVERITY_NUMERIC = {"stable": 0, "moderate": 1, "significant": 2}
# -1 means "no useful PSI yet" (insufficient_data) so a Grafana stat panel
# can render a distinct color from the severity bands.
_SEVERITY_INSUFFICIENT = -1


class _Cache:
    __slots__ = ("data", "ts")

    def __init__(self) -> None:
        self.data: dict | None = None
        self.ts: float = 0.0

    def get(self) -> dict | None:
        now = time.time()
        if self.data is None or now - self.ts > _CACHE_TTL_SEC:
            try:
                self.data = fetch_latest_drift_report()
                self.ts = now
            except Exception as exc:
                # Keep the previous cached value; never let a transient DB
                # error knock /metrics offline.
                logger.warning(f"Drift cache refresh skipped: {exc}")
        return self.data


_cache = _Cache()


class DriftCollector:
    """Custom Prometheus collector — yields one gauge family per metric."""

    def collect(self):  # type: ignore[override]
        report = _cache.get()
        if not report:
            return

        psi = report.get("psi")
        if psi is not None:
            yield GaugeMetricFamily(
                "news_sentiment_drift_psi",
                "PSI of today's sentiment-label distribution vs 30-day rolling baseline",
                value=float(psi),
            )

        severity = report.get("severity")
        sev_value = _SEVERITY_NUMERIC.get(severity, _SEVERITY_INSUFFICIENT)
        yield GaugeMetricFamily(
            "news_sentiment_drift_severity",
            "0=stable, 1=moderate, 2=significant, -1=insufficient_data",
            value=float(sev_value),
        )

        baseline_days = report.get("baseline_days")
        if baseline_days is not None:
            yield GaugeMetricFamily(
                "news_sentiment_drift_baseline_days",
                "Qualifying baseline days available for the latest drift check",
                value=float(baseline_days),
            )

        deltas = report.get("per_class_delta") or {}
        for cls in ("negative", "neutral", "positive"):
            if cls in deltas:
                yield GaugeMetricFamily(
                    f"news_sentiment_drift_delta_{cls}",
                    f"Today minus baseline ratio for class={cls}",
                    value=float(deltas[cls]),
                )


def register() -> None:
    """Idempotent: safe to call from FastAPI startup or module import."""
    if any(isinstance(c, DriftCollector) for c in list(REGISTRY._collector_to_names)):
        return
    REGISTRY.register(DriftCollector())
    logger.info("Registered DriftCollector for /metrics")


# Register on import so the collector is live before the first scrape.
register()
