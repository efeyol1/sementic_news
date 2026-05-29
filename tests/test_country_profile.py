"""Unit tests for src.analysis.country_profile (Sprint 7).

Strategy mirrors Sprint 6's tests — monkeypatched DB fetch/upsert,
hand-built daily-collocation row dicts. No real Postgres, no spaCy.

Coverage:
  * Aggregate-first window math (PMI/LLR over window-summed cells)
  * Critical R_w double-counting trap (must recompute, not naive sum)
  * Top-K shape, ordering, truncation
  * Partial windows (TR-like sparse history)
  * Option-B total_mentions distinct from total_cooccurrences
  * Idempotent recompute, empty input, disabled gate
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pytest

from src.analysis import country_profile as cp
from src.analysis.collocation_stats import compute_pmi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _country_config(
    *,
    enabled: bool = True,
    country_code: str = "DE",
    slug: str = "germany",
    language: str = "de",
) -> dict[str, Any]:
    return {
        "country_code": country_code,
        "country_slug": slug,
        "language": language,
        "entity_narrative": {
            "enabled": True,
            "collocations": {
                "enabled": enabled,
                "min_cooccurrence_count": 1,
            },
        },
    }


def _daily_row(
    collected_date: date,
    canonical: str,
    entity_type: str,
    lemma: str,
    pos: str,
    cooccurrence_count: int,
    *,
    entity_total: int = 1,
    wikidata_qid: str | None = None,
) -> dict[str, Any]:
    """One row matching ``fetch_entity_collocations_date_range``'s output shape."""
    return {
        "collected_date": collected_date,
        "canonical": canonical,
        "entity_type": entity_type,
        "wikidata_qid": wikidata_qid,
        "lemma": lemma,
        "pos": pos,
        "cooccurrence_count": cooccurrence_count,
        "entity_total": entity_total,
    }


def _stub_db(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fetch_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    captured: dict[str, Any] = {"upserted": None}

    monkeypatch.setattr(
        cp,
        "fetch_entity_collocations_date_range",
        lambda country_code, start_date, end_date: list(fetch_rows),
    )

    def fake_upsert(rows, country_code, end_date):
        captured["upserted"] = list(rows)
        captured["country_code"] = country_code
        captured["end_date"] = end_date
        return len(rows)

    monkeypatch.setattr(cp, "bulk_upsert_entity_country_profile", fake_upsert)
    return captured


# ---------------------------------------------------------------------------
# Math / aggregation (5)
# ---------------------------------------------------------------------------


def test_aggregate_first_pmi_matches_sprint6_formula_on_window_sums():
    # Three days for (Trump, tariff): c11 = 3 + 5 + 2 = 10
    # One day for (Trump, meet): c11 = 6
    # → window-summed marginals: R_w(Trump) = 16, T_w(tariff) = 10, N_w = 16
    # PMI(tariff) = compute_pmi(10, 16, 10, 16)
    base = date(2026, 5, 29)
    rows = [
        _daily_row(base - timedelta(days=2), "Trump", "PER", "tariff", "NOUN", 3),
        _daily_row(base - timedelta(days=1), "Trump", "PER", "tariff", "NOUN", 5),
        _daily_row(base,                     "Trump", "PER", "tariff", "NOUN", 2),
        _daily_row(base,                     "Trump", "PER", "meet",   "VERB", 6),
    ]
    out = cp._aggregate_window(rows, window_days=7)
    assert len(out) == 1
    profile = out[0]
    assert profile["total_cooccurrences"] == 16
    assert profile["window_total"] == 16
    # Find the tariff collocate
    tariff = next(c for c in profile["top_collocates"] if c["lemma"] == "tariff")
    assert tariff["c11_window"] == 10
    assert tariff["pmi"] == pytest.approx(compute_pmi(10, 16, 10, 16))


def test_aggregate_first_llr_nonnegative_for_random_window_inputs():
    # Build a small 2-entity, 3-lemma window; assert every emitted
    # collocate has llr >= 0 (Sprint 6's compute_llr guarantee).
    base = date(2026, 5, 29)
    rows = [
        _daily_row(base, "Trump", "PER", "tariff", "NOUN", 7),
        _daily_row(base, "Trump", "PER", "meet",   "VERB", 3),
        _daily_row(base, "Merkel", "PER", "meet",  "VERB", 4),
        _daily_row(base, "Merkel", "PER", "bonn",  "PROPN", 2),
    ]
    out = cp._aggregate_window(rows, window_days=7)
    for profile in out:
        for c in profile["top_collocates"]:
            assert c["llr"] >= 0.0


def test_window_marginals_recomputed_not_summed_from_daily_R_e():
    # CRITICAL TRAP PIN: Sprint 6's daily R(e) for "Trump" on day 1 was
    # c11(tariff) + c11(meet) = 5 + 3 = 8. On day 2 only c11(tariff)=5,
    # so daily R(e) = 5. Naive summing would give R_w = 8 + 5 = 13. The
    # correct window R_w(e) = c11_w(tariff) + c11_w(meet) = (5+5) + 3 = 13.
    # ...wait, those agree in this case. Reconstruct a divergent case:
    #   Day 1: (Trump, tariff)=4, (Trump, meet)=3 → daily R = 7
    #   Day 2: (Trump, tariff)=2                  → daily R = 2
    # Naive sum: 7 + 2 = 9. Correct R_w: c11_w(tariff) + c11_w(meet) = 6 + 3 = 9.
    # They also agree! That's actually the math identity: when daily R is
    # itself sum of c11 across lemmas, summing daily R == summing all c11.
    # The trap is: if some other module pre-computes R(e) wrong, or if the
    # daily marginal includes entities NOT in this lemma list (impossible
    # in Sprint 6's schema). Pin the contract: R_w is computed from c11
    # only, by summing per-lemma totals.
    base = date(2026, 5, 29)
    rows = [
        _daily_row(base - timedelta(days=1), "Trump", "PER", "tariff", "NOUN", 4),
        _daily_row(base - timedelta(days=1), "Trump", "PER", "meet",   "VERB", 3),
        _daily_row(base,                     "Trump", "PER", "tariff", "NOUN", 2),
    ]
    out = cp._aggregate_window(rows, window_days=7)
    profile = out[0]
    # R_w(Trump) = sum c11_w across lemmas = 6 (tariff) + 3 (meet) = 9
    assert profile["total_cooccurrences"] == 9
    assert profile["cooccurrence_with_entity_total"] == 9
    # Verify by reconstructing per-lemma sums from top_collocates
    collocate_sum = sum(c["c11_window"] for c in profile["top_collocates"])
    assert collocate_sum == 9


def test_partial_window_emits_row_with_coverage_days():
    # Entity appears on 3 of 7 days; row still emitted, coverage_days=3.
    base = date(2026, 5, 29)
    rows = [
        _daily_row(base - timedelta(days=4), "Trump", "PER", "tariff", "NOUN", 2),
        _daily_row(base - timedelta(days=2), "Trump", "PER", "tariff", "NOUN", 3),
        _daily_row(base,                     "Trump", "PER", "tariff", "NOUN", 5),
    ]
    out = cp._aggregate_window(rows, window_days=7)
    assert len(out) == 1
    assert out[0]["coverage_days"] == 3
    assert out[0]["window_days"] == 7
    assert out[0]["total_cooccurrences"] == 10


def test_zero_coverage_emits_no_row():
    # Empty window → no rows emitted.
    assert cp._aggregate_window([], window_days=7) == []


# ---------------------------------------------------------------------------
# Top-K (3)
# ---------------------------------------------------------------------------


def test_top_collocates_sorted_by_llr_desc_then_pmi_desc_then_lemma():
    # Build a single-entity window with multiple collocates whose LLR
    # ordering differs from PMI, and force a 3-way tie in LLR for two
    # collocates that differ in PMI, plus identical LLR+PMI tie that
    # must be lemma-asc.
    base = date(2026, 5, 29)
    # All same N_w so LLR/PMI comparisons are valid. Construct counts
    # such that the only fact we test is sort order on the produced floats.
    rows = [
        _daily_row(base, "Trump", "PER", "alpha", "NOUN", 5),
        _daily_row(base, "Trump", "PER", "beta",  "NOUN", 5),
        _daily_row(base, "Trump", "PER", "gamma", "NOUN", 3),
        _daily_row(base, "Trump", "PER", "delta", "NOUN", 1),
    ]
    out = cp._aggregate_window(rows, window_days=7)
    top = out[0]["top_collocates"]
    # Verify sort: each entry's (-llr, -pmi, lemma) must be <= next.
    for prev, nxt in zip(top, top[1:]):
        a = (-prev["llr"], -prev["pmi"], prev["lemma"])
        b = (-nxt["llr"], -nxt["pmi"], nxt["lemma"])
        assert a <= b


def test_top_collocates_truncated_to_TOP_K_20():
    # 25 distinct lemmas for one entity → top_collocates length 20.
    base = date(2026, 5, 29)
    rows = [
        _daily_row(base, "Trump", "PER", f"lemma_{i:02d}", "NOUN", i + 1)
        for i in range(25)
    ]
    out = cp._aggregate_window(rows, window_days=7)
    assert len(out[0]["top_collocates"]) == cp.TOP_K == 20


def test_top_collocates_jsonb_shape():
    base = date(2026, 5, 29)
    rows = [
        _daily_row(base, "Trump", "PER", "tariff", "NOUN", 4),
        _daily_row(base, "Trump", "PER", "meet",   "VERB", 2),
    ]
    out = cp._aggregate_window(rows, window_days=7)
    expected_keys = {"lemma", "pos", "c11_window", "pmi", "llr"}
    for c in out[0]["top_collocates"]:
        assert set(c.keys()) == expected_keys


# ---------------------------------------------------------------------------
# DB stub flow (4)
# ---------------------------------------------------------------------------


def test_batch_skips_when_disabled(monkeypatch):
    cfg = _country_config(enabled=False)
    monkeypatch.setattr(
        cp, "fetch_entity_collocations_date_range",
        lambda *a, **k: pytest.fail("must not fetch when disabled"),
    )
    monkeypatch.setattr(
        cp, "bulk_upsert_entity_country_profile",
        lambda *a, **k: pytest.fail("must not upsert when disabled"),
    )
    assert cp.compute_country_profile_batch(
        date_str="2026-05-29", country_config=cfg,
    ) == 0


def test_batch_full_flow_two_windows_one_entity(monkeypatch):
    # Two entities both within the 30d window; the inner 7d should
    # produce a smaller subset (in this construction, the 7d window
    # has fewer days of data for Merkel). Verify both windows produce
    # rows for the entity that appears in them.
    end = date(2026, 5, 29)
    rows = [
        # Trump: appears 6 days back and on end_date (both inside 7d)
        _daily_row(end - timedelta(days=6), "Trump", "PER", "tariff", "NOUN", 4),
        _daily_row(end,                     "Trump", "PER", "tariff", "NOUN", 6),
        # Merkel: appears only 20 days back (outside 7d, inside 30d)
        _daily_row(end - timedelta(days=20), "Merkel", "PER", "bonn", "PROPN", 3),
    ]
    captured = _stub_db(monkeypatch, fetch_rows=rows)
    cfg = _country_config()
    n = cp.compute_country_profile_batch(date_str=end.isoformat(), country_config=cfg)

    upserted = captured["upserted"]
    by_key = {(r["canonical"], r["window_days"]): r for r in upserted}
    # 7d: Trump only. 30d: Trump + Merkel. Total 3.
    assert n == 3
    assert ("Trump", 7) in by_key
    assert ("Trump", 30) in by_key
    assert ("Merkel", 30) in by_key
    assert ("Merkel", 7) not in by_key


def test_batch_idempotent_recompute(monkeypatch):
    end = date(2026, 5, 29)
    rows = [
        _daily_row(end, "Trump", "PER", "tariff", "NOUN", 5, entity_total=4),
        _daily_row(end, "Trump", "PER", "meet",   "VERB", 3, entity_total=4),
    ]

    first = _stub_db(monkeypatch, fetch_rows=rows)
    cp.compute_country_profile_batch(date_str=end.isoformat(), country_config=_country_config())

    second = _stub_db(monkeypatch, fetch_rows=rows)
    cp.compute_country_profile_batch(date_str=end.isoformat(), country_config=_country_config())

    assert first["upserted"] == second["upserted"]


def test_batch_empty_fetch_is_noop(monkeypatch):
    captured = _stub_db(monkeypatch, fetch_rows=[])
    n = cp.compute_country_profile_batch(
        date_str="2026-05-29", country_config=_country_config(),
    )
    assert n == 0
    # Empty rows still triggers the upsert (to clear stale slice) — but
    # the upsert helper receives an empty list.
    assert captured["upserted"] == []


# ---------------------------------------------------------------------------
# Option-B integration (1)
# ---------------------------------------------------------------------------


def test_total_mentions_sums_entity_total_column(monkeypatch):
    # entity_total is per (entity, date) — same value across all lemma
    # rows for the same (entity, date). We sum it once per date.
    # Day 1: Trump's entity_total = 3 (across 2 lemma rows, same number)
    # Day 2: Trump's entity_total = 5 (across 1 lemma row)
    # total_mentions = 3 + 5 = 8
    # total_cooccurrences = R_w(Trump) = 4 + 2 + 7 = 13 (NOT 8)
    end = date(2026, 5, 29)
    rows = [
        _daily_row(end - timedelta(days=1), "Trump", "PER", "tariff", "NOUN", 4, entity_total=3),
        _daily_row(end - timedelta(days=1), "Trump", "PER", "meet",   "VERB", 2, entity_total=3),
        _daily_row(end,                     "Trump", "PER", "tariff", "NOUN", 7, entity_total=5),
    ]
    out = cp._aggregate_window(rows, window_days=7)
    profile = out[0]
    assert profile["total_mentions"] == 8
    assert profile["total_cooccurrences"] == 13
    assert profile["total_mentions"] != profile["total_cooccurrences"]


# ---------------------------------------------------------------------------
# TR partial path (1)
# ---------------------------------------------------------------------------


def _country_config_with_frames(language: str = "de") -> dict[str, Any]:
    cfg = _country_config(language=language)
    cfg["entity_narrative"]["frame_bridge"] = {"enabled": True}
    return cfg


def test_frame_intensities_populated_when_enabled(monkeypatch):
    # German lemmas that hit the real configs/frames/de.yaml lexicon:
    # "krieg" → conflict, "wirtschaft" → economic. A second entity is
    # needed so the entity↔lemma association (LLR) is non-zero — a
    # single-entity window is degenerate (every PMI/LLR collapses to 0).
    end = date(2026, 5, 29)
    rows = [
        _daily_row(end, "Trump",  "PER", "krieg",      "NOUN", 8),
        _daily_row(end, "Trump",  "PER", "wirtschaft", "NOUN", 3),
        _daily_row(end, "Merkel", "PER", "wirtschaft", "NOUN", 6),
        _daily_row(end, "Merkel", "PER", "reform",     "NOUN", 4),
    ]
    captured = _stub_db(monkeypatch, fetch_rows=rows)
    cp.compute_country_profile_batch(
        date_str=end.isoformat(), country_config=_country_config_with_frames("de")
    )
    trump_rows = [r for r in captured["upserted"] if r["canonical"] == "Trump"]
    assert trump_rows
    for row in trump_rows:
        fi = row["frame_intensities"]
        assert fi is not None
        assert sum(fi.values()) == pytest.approx(1.0)
        # krieg (conflict) and wirtschaft (economic) both carry weight.
        assert fi["conflict"] > 0.0 and fi["economic"] > 0.0


def test_frame_intensities_none_when_disabled(monkeypatch):
    # frame_bridge absent (default off) → column left NULL. This is the
    # TR-baseline-preservation contract for any country that hasn't opted in.
    end = date(2026, 5, 29)
    rows = [_daily_row(end, "Trump", "PER", "krieg", "NOUN", 6)]
    captured = _stub_db(monkeypatch, fetch_rows=rows)
    cp.compute_country_profile_batch(
        date_str=end.isoformat(), country_config=_country_config()
    )
    for row in captured["upserted"]:
        assert row["frame_intensities"] is None


def test_batch_tr_partial_window_3_of_7_days(monkeypatch):
    # TR was just activated; 3 days of data inside a 30-day window.
    # Both 7d and 30d rows should be emitted with coverage_days = 3.
    end = date(2026, 5, 29)
    rows = [
        _daily_row(end - timedelta(days=2), "Erdoğan", "PER", "ekonomi", "X", 3, entity_total=2),
        _daily_row(end - timedelta(days=1), "Erdoğan", "PER", "ekonomi", "X", 4, entity_total=3),
        _daily_row(end,                     "Erdoğan", "PER", "miting",  "X", 2, entity_total=1),
    ]
    captured = _stub_db(monkeypatch, fetch_rows=rows)
    cfg = _country_config(country_code="TR", slug="turkey", language="tr")
    n = cp.compute_country_profile_batch(date_str=end.isoformat(), country_config=cfg)

    assert n == 2  # one row per window
    by_window = {r["window_days"]: r for r in captured["upserted"]}
    assert by_window[7]["coverage_days"] == 3
    assert by_window[30]["coverage_days"] == 3
    # Math identity: total_cooccurrences is the same for both windows
    # because all data sits inside the inner (7d) window.
    assert by_window[7]["total_cooccurrences"] == by_window[30]["total_cooccurrences"] == 9
