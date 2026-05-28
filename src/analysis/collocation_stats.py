"""PMI + log-likelihood backfill (Sprint 6 of the entity-narrative track).

Sprint 5 wrote raw ``cooccurrence_count`` into ``entity_collocations``
and left the six statistical columns NULL. This module fills them via
in-place UPDATE:

  - ``cooccurrence_with_entity_total``  R(e) = sum c11 over (canonical, entity_type)
  - ``entity_total``                    N_mentions(e) — distinct mention count from entity_mentions
  - ``token_total``                     T(l) = sum c11 over (lemma, pos)
  - ``window_total``                    N    = sum c11 over the (country, date) slice
  - ``pmi``                             log( c11 * N / (R(e) * T(l)) )
  - ``log_likelihood``                  Dunning 1993 2x2 G^2

Gate: same ``entity_narrative.collocations.enabled`` flag as Sprint 5.
PMI is on by default whenever extraction is on — no separate config
block. Rows below ``min_cooccurrence_count`` are left NULL.

Usage:
    python -m src.analysis.collocation_stats --country germany --date 2026-05-27
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from datetime import date
from typing import Any

from loguru import logger

from src.analysis.collocations import _setting, is_enabled
from src.config import load_country_config
from src.db.queries import (
    bulk_update_collocation_stats,
    fetch_collocations_for_stats,
    fetch_entity_mention_counts,
)

# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------


def compute_pmi(c11: int, r_e: int, t_l: int, n: int) -> float:
    """Pointwise Mutual Information.

    PMI = log( c11 * N / (R(e) * T(l)) ), natural log.

    Caller is responsible for ensuring c11 >= 1 (and therefore R(e), T(l)
    >= 1 by definition of the daily slice). With those guarantees no
    denominator can be zero.
    """
    return math.log(c11 * n / (r_e * t_l))


def _xlogx(x: float) -> float:
    """``x * log(x)`` with the ``0 * log(0) = 0`` convention used by LLR.

    Negative values are clamped to 0 — they only occur with corrupt
    inputs (e.g. marginals < c11), and silently swallowing them keeps
    the LLR sum non-negative as Dunning intends.
    """
    if x <= 0:
        return 0.0
    return x * math.log(x)


def compute_llr(c11: int, r_e: int, t_l: int, n: int) -> float:
    """Dunning 1993 log-likelihood ratio (G^2) for a 2x2 contingency table.

    Cells:
        c11 = co-occurrence count
        c12 = R(e) - c11      (entity present, this lemma absent)
        c21 = T(l) - c11      (lemma present, this entity absent)
        c22 = N - R(e) - T(l) + c11  (neither)

    G^2 = 2 * sum( o_ij * log(o_ij / e_ij) ) with the ``0*log0=0``
    convention. The closed-form simplification used here is:

        G^2 = 2 * [ sum(o_ij * log o_ij) - sum(row_marginals * log row_marginals)
                  - sum(col_marginals * log col_marginals) + N * log N ]

    Always >= 0. Higher = stronger association (positive or negative).
    """
    c12 = r_e - c11
    c21 = t_l - c11
    c22 = n - r_e - t_l + c11

    cells = _xlogx(c11) + _xlogx(c12) + _xlogx(c21) + _xlogx(c22)
    row_marginals = _xlogx(r_e) + _xlogx(n - r_e)
    col_marginals = _xlogx(t_l) + _xlogx(n - t_l)
    grand = _xlogx(n)

    g2 = 2.0 * (cells - row_marginals - col_marginals + grand)
    # Guard against tiny negative drift from float math.
    return max(0.0, g2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_collocation_stats_batch(
    date_str: str | None = None,
    country_config: dict[str, Any] | None = None,
) -> int:
    """Backfill PMI / LLR / totals for one day's collocations.

    Returns the number of rows updated. Soft no-op when collocations are
    disabled for the country — the pipeline calls this unconditionally
    under the parent ``collocations.enabled`` gate, so a disabled return
    here is a defensive guard, not the common path.
    """
    if country_config is None:
        raise ValueError("country_config is required for collocation stats")
    if not is_enabled(country_config):
        logger.info(
            f"entity_narrative.collocations disabled for "
            f"{country_config.get('country_slug', '?')} — skipping PMI"
        )
        return 0

    date_str = date_str or date.today().isoformat()
    country_code: str = country_config["country_code"]
    min_cooccurrence = int(_setting(country_config, "min_cooccurrence_count"))

    t0 = time.perf_counter()

    rows = fetch_collocations_for_stats(date_str, country_code=country_code)
    if not rows:
        logger.warning(
            f"No entity_collocations rows for {country_code} {date_str} — "
            "PMI step is a no-op (did Sprint 5 extraction run?)"
        )
        return 0

    mention_counts = fetch_entity_mention_counts(
        date_str, country_code=country_code
    )

    # --- Marginals over the (country, date) slice -------------------------
    r_e: dict[tuple[str, str], int] = defaultdict(int)  # entity row marginal
    t_l: dict[tuple[str, str], int] = defaultdict(int)  # lemma col marginal
    n_total = 0
    for row in rows:
        c = int(row["cooccurrence_count"])
        r_e[(row["canonical"], row["entity_type"])] += c
        t_l[(row["lemma"], row["pos"])] += c
        n_total += c

    if n_total <= 0:
        logger.warning(
            f"window_total computed as 0 for {country_code} {date_str} — "
            "all rows have cooccurrence_count <= 0; aborting"
        )
        return 0

    # --- Per-row derived values + threshold ------------------------------
    updates: list[dict[str, Any]] = []
    filtered = 0
    for row in rows:
        c11 = int(row["cooccurrence_count"])
        if c11 < min_cooccurrence:
            # Leave totals/PMI/LLR NULL — Sprint 5 should have already
            # filtered these, but a defensive check protects against
            # manual inserts or future config drift.
            filtered += 1
            continue
        e_key = (row["canonical"], row["entity_type"])
        l_key = (row["lemma"], row["pos"])
        r_e_val = r_e[e_key]
        t_l_val = t_l[l_key]
        updates.append(
            {
                "id": row["id"],
                "cooccurrence_with_entity_total": r_e_val,
                "entity_total": mention_counts.get(e_key, 0),
                "token_total": t_l_val,
                "window_total": n_total,
                "pmi": compute_pmi(c11, r_e_val, t_l_val, n_total),
                "log_likelihood": compute_llr(c11, r_e_val, t_l_val, n_total),
            }
        )

    updated = bulk_update_collocation_stats(updates, country_code, date_str)
    elapsed = time.perf_counter() - t0
    logger.success(
        f"PMI/LLR [{country_code} {date_str}]: "
        f"{len(rows)} rows fetched, {filtered} below "
        f"min_cooccurrence={min_cooccurrence}, {updated} updated "
        f"(N={n_total}, |entities|={len(r_e)}, |lemmas|={len(t_l)}) "
        f"in {elapsed:.1f}s"
    )
    return updated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill PMI / log-likelihood for one country/day."
    )
    parser.add_argument("--country", required=True, help="country slug or code")
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        metavar="YYYY-MM-DD",
        help="Date to process (default: today)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    country_config = load_country_config(args.country)
    compute_collocation_stats_batch(date_str=args.date, country_config=country_config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
