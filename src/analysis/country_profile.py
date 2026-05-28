"""Rolling 7d/30d entity profile rollup (Sprint 7 of the entity-narrative track).

For each (entity, country, window_days, end_date) emits one
``entity_country_profile`` row by aggregating Sprint 6's daily
``entity_collocations`` over a window of ``window_days`` ending on
``end_date``.

Aggregate-first math (avoids the naive double-counting trap of
summing daily ``cooccurrence_with_entity_total`` values across days):

  c11_w(e, l) = Σ daily cooccurrence_count over window
  R_w(e)     = Σ c11_w over all lemmas of e
  T_w(l)     = Σ c11_w over all entities of l
  N_w        = Σ c11_w over the whole (country, window) slice
  pmi_w      = compute_pmi(c11_w, R_w(e), T_w(l), N_w)   ← same Sprint 6 fn
  llr_w      = compute_llr(c11_w, R_w(e), T_w(l), N_w)   ← same Sprint 6 fn

Partial windows: if a (country, entity) has data on only k of N days,
emit a row with ``coverage_days = k``. Skip only when k == 0 (entity
absent from the entire window). The TR slice — Sprint 6 just activated
collocations — will sit at ``coverage_days ∈ {3, 4}`` in early days.

Idempotency: per (country_code, end_date) DELETE+INSERT (see
``bulk_upsert_entity_country_profile``).

Usage:
    python -m src.analysis.country_profile --country germany --date 2026-05-29
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from datetime import date, timedelta
from typing import Any

from loguru import logger

from src.analysis.collocation_stats import compute_llr, compute_pmi
from src.analysis.collocations import is_enabled
from src.config import load_country_config
from src.db.queries import (
    bulk_upsert_entity_country_profile,
    fetch_entity_collocations_date_range,
)

# ---------------------------------------------------------------------------
# Module constants — hardcoded for MVP. Per-country override is Sprint 8 work
# if dashboards need it; until then we keep the contract simple.
# ---------------------------------------------------------------------------

WINDOWS: tuple[int, ...] = (7, 30)
TOP_K: int = 20
MIN_TOP_K_LLR: float = 0.0  # collocates with LLR below this are dropped


# ---------------------------------------------------------------------------
# Window aggregation
# ---------------------------------------------------------------------------


def _aggregate_window(
    window_rows: list[dict[str, Any]],
    window_days: int,
) -> list[dict[str, Any]]:
    """Aggregate one window of daily rows into per-entity profile dicts.

    ``window_rows`` is the subset of ``fetch_entity_collocations_date_range``
    output that falls within this window's date range. Returns one dict
    per (canonical, entity_type) that produced ≥1 collocate. Returns an
    empty list when the window has no rows at all.
    """
    if not window_rows:
        return []

    # Per-entity, per-lemma c11_w sums (the only marginal we trust from
    # the daily rows — everything else is recomputed from these).
    per_entity_lemma_c11: dict[
        tuple[str, str], dict[tuple[str, str], int]
    ] = defaultdict(lambda: defaultdict(int))
    # Per-lemma marginal T_w(l).
    per_lemma_c11: dict[tuple[str, str], int] = defaultdict(int)
    # Distinct dates per entity → coverage_days.
    per_entity_dates: dict[tuple[str, str], set] = defaultdict(set)
    # First non-NULL wikidata_qid wins (mention may resolve in a later
    # day after the first appearance with NULL qid).
    per_entity_qid: dict[tuple[str, str], str | None] = {}
    # Sum entity_total per (entity, date) once even though it duplicates
    # across lemma rows for the same day.
    seen_entity_date: set[tuple[str, str, Any]] = set()
    per_entity_mentions_sum: dict[tuple[str, str], int] = defaultdict(int)
    n_w = 0

    for row in window_rows:
        e = (row["canonical"], row["entity_type"])
        lemma_key = (row["lemma"], row["pos"])
        c = int(row["cooccurrence_count"])

        per_entity_lemma_c11[e][lemma_key] += c
        per_lemma_c11[lemma_key] += c
        n_w += c
        per_entity_dates[e].add(row["collected_date"])

        # qid: keep first non-NULL we see for this entity.
        if e not in per_entity_qid or per_entity_qid[e] is None:
            per_entity_qid[e] = row.get("wikidata_qid")

        # entity_total is per (entity, date) — same value across all
        # lemma rows for that (entity, date), so we record it once.
        ed_key = (e[0], e[1], row["collected_date"])
        if ed_key not in seen_entity_date:
            seen_entity_date.add(ed_key)
            mc = row.get("entity_total")
            per_entity_mentions_sum[e] += int(mc) if mc is not None else 0

    if n_w <= 0:
        return []

    # --- Build per-entity profile rows ----------------------------------
    output: list[dict[str, Any]] = []
    for e, lemma_counts in per_entity_lemma_c11.items():
        canonical, entity_type = e
        r_w = sum(lemma_counts.values())

        # Per-collocate window PMI/LLR.
        collocates: list[dict[str, Any]] = []
        for lemma_key, c11_w in lemma_counts.items():
            lemma, pos = lemma_key
            t_w = per_lemma_c11[lemma_key]
            pmi = compute_pmi(c11_w, r_w, t_w, n_w)
            llr = compute_llr(c11_w, r_w, t_w, n_w)
            if llr < MIN_TOP_K_LLR:
                continue
            collocates.append(
                {
                    "lemma": lemma,
                    "pos": pos,
                    "c11_window": c11_w,
                    "pmi": pmi,
                    "llr": llr,
                }
            )

        if not collocates:
            # Every collocate filtered by MIN_TOP_K_LLR; skip the row.
            continue

        avg_pmi = sum(c["pmi"] for c in collocates) / len(collocates)
        avg_llr = sum(c["llr"] for c in collocates) / len(collocates)

        # Deterministic sort: LLR desc, PMI desc, lemma asc.
        collocates.sort(key=lambda x: (-x["llr"], -x["pmi"], x["lemma"]))
        top = collocates[:TOP_K]

        output.append(
            {
                "canonical": canonical,
                "entity_type": entity_type,
                "wikidata_qid": per_entity_qid.get(e),
                "window_days": window_days,
                "coverage_days": len(per_entity_dates[e]),
                "total_cooccurrences": r_w,
                "total_mentions": per_entity_mentions_sum.get(e, 0),
                "cooccurrence_with_entity_total": r_w,
                "window_total": n_w,
                "avg_pmi": avg_pmi,
                "avg_log_likelihood": avg_llr,
                "top_collocates": top,
            }
        )
    return output


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_country_profile_batch(
    date_str: str | None = None,
    country_config: dict[str, Any] | None = None,
) -> int:
    """Roll up daily collocations into 7d + 30d profiles for one country.

    Returns the number of profile rows upserted (across all windows).
    Soft no-op when collocations are disabled for the country (Sprint 7
    runs under the same parent gate as PMI — no separate flag).
    """
    if country_config is None:
        raise ValueError("country_config is required for country_profile")
    if not is_enabled(country_config):
        logger.info(
            f"entity_narrative.collocations disabled for "
            f"{country_config.get('country_slug', '?')} — skipping country_profile"
        )
        return 0

    date_str = date_str or date.today().isoformat()
    country_code: str = country_config["country_code"]
    end_date = date.fromisoformat(date_str)
    max_window = max(WINDOWS)
    start_date = end_date - timedelta(days=max_window - 1)

    t0 = time.perf_counter()

    # Single fetch over the largest window; filter in Python per window.
    all_rows = fetch_entity_collocations_date_range(
        country_code=country_code,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )

    if not all_rows:
        logger.warning(
            f"No entity_collocations rows in [{start_date}, {end_date}] "
            f"for {country_code} — country_profile is a no-op"
        )
        # Still clear the day's slice so a previously bad run can't
        # leave stale rows around.
        bulk_upsert_entity_country_profile([], country_code, end_date.isoformat())
        return 0

    profile_rows: list[dict[str, Any]] = []
    for window_days in WINDOWS:
        window_start = end_date - timedelta(days=window_days - 1)
        window_rows = [
            r for r in all_rows if r["collected_date"] >= window_start
        ]
        profile_rows.extend(_aggregate_window(window_rows, window_days))

    upserted = bulk_upsert_entity_country_profile(
        profile_rows, country_code, end_date.isoformat()
    )
    elapsed = time.perf_counter() - t0
    logger.success(
        f"Country profile [{country_code} {end_date}]: "
        f"{len(all_rows)} daily rows over [{start_date}, {end_date}], "
        f"{upserted} profile rows upserted "
        f"(windows={list(WINDOWS)}, top_k={TOP_K}) in {elapsed:.1f}s"
    )
    return upserted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roll up daily collocations into 7d/30d entity profiles.",
    )
    parser.add_argument("--country", required=True, help="country slug or code")
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        metavar="YYYY-MM-DD",
        help="End date of the rolling window (default: today)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    country_config = load_country_config(args.country)
    compute_country_profile_batch(date_str=args.date, country_config=country_config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
