"""Audit frame-bridge coverage and distribution health (Sprint 10 QA).

Reads the rolling ``entity_country_profile`` rows for one (country, window,
end_date) and reports:

  * frame coverage — % of profiles that got a non-NULL frame_intensities
  * unassigned-collocate rate — % of top collocates whose lemma matched no
    frame seed (high → lexicon is too sparse for this language/corpus)
  * dominant-frame distribution + Shannon entropy (low entropy → one frame
    dominates everything, a red flag for the seed lists)

This is the automated half of the Sprint 10 quality gate; the manual
accuracy probe lives in export_frame_review_sample.py + evaluate_frame_review.py.

Usage:
    python scripts/audit_frame_bridge.py --country germany
    python scripts/audit_frame_bridge.py --country germany --window 7 --date 2026-05-29
    python scripts/audit_frame_bridge.py --country germany --json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from typing import Any

from src.analysis.frame_bridge import FRAME_NAMES, load_frame_lexicon
from src.config import load_country_config
from src.db.queries import fetch_country_profiles_for_frame_audit


def dominant_frame(frame_intensities: dict[str, float] | None) -> str | None:
    """Highest-intensity frame, tie-broken by canonical FRAME_NAMES order.

    Returns ``None`` when there is no signal (NULL or all-zero).
    """
    if not frame_intensities:
        return None
    order = [n for n in FRAME_NAMES if n in frame_intensities]
    order += [n for n in frame_intensities if n not in FRAME_NAMES]
    best: str | None = None
    best_val = 0.0
    for name in order:
        val = float(frame_intensities[name])
        if val > best_val:
            best_val = val
            best = name
    return best


def _entropy_bits(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    return h


def compute_frame_audit(
    rows: list[dict[str, Any]],
    lexicon: dict[str, set[str]],
) -> dict[str, Any]:
    """Pure health summary over profile rows (no DB) — see module docstring."""
    n = len(rows)
    all_seeds: set[str] = set()
    for seeds in lexicon.values():
        all_seeds |= seeds

    with_frames = 0
    total_collocates = 0
    unassigned = 0
    dominant: Counter = Counter()
    for row in rows:
        fi = row.get("frame_intensities")
        if fi:
            with_frames += 1
        dom = dominant_frame(fi)
        if dom:
            dominant[dom] += 1
        for collocate in row.get("top_collocates") or []:
            total_collocates += 1
            lemma = str(collocate.get("lemma", "")).lower().strip()
            if lemma not in all_seeds:
                unassigned += 1

    return {
        "profiles": n,
        "profiles_with_frames": with_frames,
        "frame_coverage_pct": round(with_frames / n * 100, 1) if n else 0.0,
        "top_collocates_total": total_collocates,
        "unassigned_collocates": unassigned,
        "unassigned_collocate_pct": (
            round(unassigned / total_collocates * 100, 1) if total_collocates else 0.0
        ),
        "dominant_frame_distribution": dict(dominant),
        "dominant_frame_entropy_bits": round(_entropy_bits(dominant), 3),
        "max_entropy_bits": round(math.log2(len(FRAME_NAMES)), 3),
    }


def _print_human(report: dict[str, Any], country_code: str, window: int) -> None:
    print(f"\nFrame-bridge audit — {country_code} {window}d")
    print("=" * 48)
    print(f"profiles                : {report['profiles']}")
    print(
        f"frame coverage          : {report['profiles_with_frames']}"
        f"/{report['profiles']} ({report['frame_coverage_pct']}%)"
    )
    print(
        f"unassigned collocates   : {report['unassigned_collocates']}"
        f"/{report['top_collocates_total']} ({report['unassigned_collocate_pct']}%)"
    )
    print(
        f"dominant-frame entropy  : {report['dominant_frame_entropy_bits']} "
        f"/ {report['max_entropy_bits']} bits"
    )
    print("dominant-frame counts   :")
    dist = report["dominant_frame_distribution"]
    for name in FRAME_NAMES:
        if name in dist:
            print(f"    {name:<14}{dist[name]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit frame-bridge health.")
    parser.add_argument("--country", required=True, help="country slug or code")
    parser.add_argument("--window", type=int, default=30, choices=(7, 30))
    parser.add_argument("--date", default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    args = parser.parse_args(argv)

    config = load_country_config(args.country)
    country_code = config["country_code"]
    language = config.get("language") or "en"
    lexicon = load_frame_lexicon(language)

    rows = fetch_country_profiles_for_frame_audit(
        country_code, args.window, end_date=args.date
    )
    report = compute_frame_audit(rows, lexicon)
    report["country_code"] = country_code
    report["window_days"] = args.window
    report["language"] = language

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_human(report, country_code, args.window)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
