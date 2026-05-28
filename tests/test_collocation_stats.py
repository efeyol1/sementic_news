"""Unit tests for src.analysis.collocation_stats (Sprint 6).

Strategy mirrors tests/test_collocations.py — monkeypatched DB
fetch/update helpers, hand-built input rows. No real Postgres or
spaCy required.

Coverage:
  * Math helpers compute_pmi / compute_llr (sign, edges, regression anchors)
  * Marginal aggregation (R(e), T(l), N) and Option-B entity_total
    (distinct mention count from entity_mentions, separate from R(e))
  * Idempotent recompute and below-threshold filtering
  * TR end-to-end smoke (surface lemmatizer path)
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from src.analysis import collocation_stats as cs

# ---------------------------------------------------------------------------
# Math: compute_pmi
# ---------------------------------------------------------------------------


def test_compute_pmi_positive_for_strong_association():
    # c11=10, R(e)=20, T(l)=20, N=1000
    # log(10 * 1000 / (20*20)) = log(25) ≈ 3.2189
    assert cs.compute_pmi(10, 20, 20, 1000) == pytest.approx(math.log(25))


def test_compute_pmi_negative_for_anti_correlation():
    # c11=1, R(e)=100, T(l)=100, N=1000
    # log(1 * 1000 / (100*100)) = log(0.1) ≈ -2.3026
    assert cs.compute_pmi(1, 100, 100, 1000) == pytest.approx(math.log(0.1))


def test_compute_pmi_zero_at_independence():
    # When P(e,l) == P(e)*P(l), PMI is exactly 0.
    # c11=10, R(e)=100, T(l)=100, N=1000 → expected = 10 = c11 → PMI=0
    assert cs.compute_pmi(10, 100, 100, 1000) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Math: compute_llr
# ---------------------------------------------------------------------------


def test_compute_llr_nonnegative_across_random_inputs():
    # LLR is a sum of squared deviations weighted by log; never negative.
    cases = [
        (1, 5, 5, 100),
        (10, 20, 20, 1000),
        (1, 100, 100, 1000),
        (50, 100, 200, 10000),
        (2, 2, 2, 50),
    ]
    for c11, r_e, t_l, n in cases:
        assert cs.compute_llr(c11, r_e, t_l, n) >= 0.0


def test_compute_llr_zero_cell_handling():
    # c22 = N - R(e) - T(l) + c11. Construct a case where c22 = 0.
    # N=20, R(e)=10, T(l)=15, c11=5 → c22 = 20 - 10 - 15 + 5 = 0.
    # Must not raise (we use the 0*log0 = 0 convention).
    llr = cs.compute_llr(5, 10, 15, 20)
    assert llr >= 0.0
    assert math.isfinite(llr)


def test_compute_llr_zero_at_independence():
    # Same setup that gave PMI=0: c11 equals its expected value.
    # G^2 reduces to 0 when observed == expected for every cell.
    assert cs.compute_llr(10, 100, 100, 1000) == pytest.approx(0.0, abs=1e-9)


def test_compute_llr_regression_anchor():
    # Pins the implementation against accidental formula changes.
    # Hand-derived from the closed-form xlogx expansion:
    #   c11=10, R(e)=20, T(l)=20, N=1000  → strong positive association.
    # Computed once with the current implementation and locked here; if
    # this drifts the change was meaningful and the new value should be
    # explained.
    actual = cs.compute_llr(10, 20, 20, 1000)
    # Hand-derived G^2 ≈ 56.7554 (Dunning 1993 §2.1 formulation).
    assert actual == pytest.approx(56.755, abs=0.01)


# ---------------------------------------------------------------------------
# Batch: marginals + Option-B entity_total + DB stubs
# ---------------------------------------------------------------------------


def _country_config(
    *,
    enabled: bool = True,
    min_cooccurrence_count: int = 2,
    language: str = "en",
    country_code: str = "DE",
    slug: str = "germany",
) -> dict[str, Any]:
    return {
        "country_code": country_code,
        "country_slug": slug,
        "language": language,
        "entity_narrative": {
            "enabled": True,
            "collocations": {
                "enabled": enabled,
                "min_cooccurrence_count": min_cooccurrence_count,
            },
        },
    }


def _coll_row(
    row_id: int,
    canonical: str,
    entity_type: str,
    lemma: str,
    pos: str,
    cooccurrence_count: int,
) -> dict[str, Any]:
    return {
        "id": row_id,
        "canonical": canonical,
        "entity_type": entity_type,
        "lemma": lemma,
        "pos": pos,
        "cooccurrence_count": cooccurrence_count,
    }


def _stub_db(
    monkeypatch: pytest.MonkeyPatch,
    *,
    rows: list[dict[str, Any]],
    mention_counts: dict[tuple[str, str], int],
) -> dict[str, Any]:
    captured: dict[str, Any] = {"updated": None}

    monkeypatch.setattr(
        cs, "fetch_collocations_for_stats",
        lambda date_str, country_code: list(rows),
    )
    monkeypatch.setattr(
        cs, "fetch_entity_mention_counts",
        lambda date_str, country_code: dict(mention_counts),
    )

    def fake_update(updates, country_code, date_str):
        captured["updated"] = list(updates)
        captured["country_code"] = country_code
        captured["date_str"] = date_str
        return len(updates)

    monkeypatch.setattr(cs, "bulk_update_collocation_stats", fake_update)
    return captured


def test_batch_skips_when_disabled(monkeypatch):
    cfg = _country_config(enabled=False)
    monkeypatch.setattr(
        cs, "fetch_collocations_for_stats",
        lambda *a, **k: pytest.fail("must not fetch when disabled"),
    )
    monkeypatch.setattr(
        cs, "bulk_update_collocation_stats",
        lambda *a, **k: pytest.fail("must not update when disabled"),
    )
    assert cs.compute_collocation_stats_batch(
        date_str="2026-05-27", country_config=cfg
    ) == 0


def test_batch_totals_marginal_consistency(monkeypatch):
    # Two entities, three lemmas. Expected marginals:
    #   R(Trump,PER) = 5 + 3 = 8
    #   R(Merkel,PER) = 2
    #   T(tariff,NOUN) = 5
    #   T(meet,VERB) = 3 + 2 = 5
    #   N = 5 + 3 + 2 = 10
    rows = [
        _coll_row(1, "Donald Trump", "PER", "tariff", "NOUN", 5),
        _coll_row(2, "Donald Trump", "PER", "meet",   "VERB", 3),
        _coll_row(3, "Angela Merkel", "PER", "meet",  "VERB", 2),
    ]
    mention_counts = {
        ("Donald Trump", "PER"): 4,
        ("Angela Merkel", "PER"): 1,
    }
    cfg = _country_config(min_cooccurrence_count=1)
    captured = _stub_db(monkeypatch, rows=rows, mention_counts=mention_counts)

    n = cs.compute_collocation_stats_batch(
        date_str="2026-05-27", country_config=cfg
    )
    assert n == 3
    updates = captured["updated"]
    by_id = {u["id"]: u for u in updates}

    # Row 1: Trump-tariff
    assert by_id[1]["cooccurrence_with_entity_total"] == 8
    assert by_id[1]["entity_total"] == 4  # distinct mentions of Trump, NOT 8
    assert by_id[1]["token_total"] == 5
    assert by_id[1]["window_total"] == 10
    assert by_id[1]["pmi"] == pytest.approx(math.log(5 * 10 / (8 * 5)))

    # Row 3: Merkel-meet
    assert by_id[3]["cooccurrence_with_entity_total"] == 2
    assert by_id[3]["entity_total"] == 1
    assert by_id[3]["token_total"] == 5
    assert by_id[3]["pmi"] == pytest.approx(math.log(2 * 10 / (2 * 5)))


def test_batch_entity_total_distinct_from_r_e(monkeypatch):
    # Option B contract: entity_total (distinct mention count) is allowed
    # to differ from cooccurrence_with_entity_total (R(e)). This test
    # locks the separation in: same row, two different non-equal values.
    rows = [_coll_row(1, "Donald Trump", "PER", "tariff", "NOUN", 5)]
    mention_counts = {("Donald Trump", "PER"): 2}  # mentioned in 2 articles
    cfg = _country_config(min_cooccurrence_count=1)
    captured = _stub_db(monkeypatch, rows=rows, mention_counts=mention_counts)

    cs.compute_collocation_stats_batch(date_str="2026-05-27", country_config=cfg)
    u = captured["updated"][0]
    assert u["cooccurrence_with_entity_total"] == 5  # R(e) = sum c11
    assert u["entity_total"] == 2                     # distinct mention count
    assert u["entity_total"] != u["cooccurrence_with_entity_total"]


def test_batch_filters_below_min_cooccurrence(monkeypatch):
    # Sprint 5 should have filtered these at write time, but Sprint 6
    # keeps a defensive check: rows with c11 < threshold are excluded
    # from the UPDATE payload (their stats columns stay NULL).
    rows = [
        _coll_row(1, "Donald Trump", "PER", "tariff", "NOUN", 5),
        _coll_row(2, "Donald Trump", "PER", "single", "NOUN", 1),  # below threshold
        _coll_row(3, "Donald Trump", "PER", "talk",   "VERB", 3),
    ]
    mention_counts = {("Donald Trump", "PER"): 3}
    cfg = _country_config(min_cooccurrence_count=2)
    captured = _stub_db(monkeypatch, rows=rows, mention_counts=mention_counts)

    cs.compute_collocation_stats_batch(date_str="2026-05-27", country_config=cfg)
    update_ids = {u["id"] for u in captured["updated"]}
    assert update_ids == {1, 3}  # id=2 (c11=1 < 2) excluded


def test_batch_idempotent_recompute(monkeypatch):
    rows = [
        _coll_row(1, "Donald Trump", "PER", "tariff", "NOUN", 5),
        _coll_row(2, "Donald Trump", "PER", "meet",   "VERB", 3),
    ]
    mention_counts = {("Donald Trump", "PER"): 4}
    cfg = _country_config(min_cooccurrence_count=2)

    first = _stub_db(monkeypatch, rows=rows, mention_counts=mention_counts)
    cs.compute_collocation_stats_batch(date_str="2026-05-27", country_config=cfg)
    first_updates = first["updated"]

    second = _stub_db(monkeypatch, rows=rows, mention_counts=mention_counts)
    cs.compute_collocation_stats_batch(date_str="2026-05-27", country_config=cfg)
    second_updates = second["updated"]

    # Same inputs → same outputs, byte-for-byte (modulo dict ordering,
    # which Python preserves under CPython since 3.7).
    assert first_updates == second_updates


def test_batch_empty_fetch_is_noop(monkeypatch):
    cfg = _country_config(min_cooccurrence_count=2)
    captured = _stub_db(monkeypatch, rows=[], mention_counts={})
    n = cs.compute_collocation_stats_batch(
        date_str="2026-05-27", country_config=cfg
    )
    assert n == 0
    # The update helper must NOT be called when there is nothing to write
    # — fetch_collocations_for_stats already returned empty, the batch
    # short-circuits before computing marginals.
    assert captured["updated"] is None


# ---------------------------------------------------------------------------
# TR-specific path (surface lemmatizer emits POS='X')
# ---------------------------------------------------------------------------


def test_batch_tr_surface_lemmatizer_path(monkeypatch):
    # TurkishSurfaceLemmatizer always emits POS='X'. The PMI math is
    # POS-agnostic; this test verifies the same code path runs cleanly
    # on TR-shaped rows and that totals/PMI are filled for every row.
    rows = [
        _coll_row(1, "Recep Tayyip Erdoğan", "PER", "ekonomi", "X", 4),
        _coll_row(2, "Recep Tayyip Erdoğan", "PER", "miting",  "X", 2),
        _coll_row(3, "Ekrem İmamoğlu",        "PER", "miting",  "X", 3),
    ]
    mention_counts = {
        ("Recep Tayyip Erdoğan", "PER"): 5,
        ("Ekrem İmamoğlu", "PER"): 2,
    }
    cfg = _country_config(
        min_cooccurrence_count=2, country_code="TR", slug="turkey", language="tr",
    )
    captured = _stub_db(monkeypatch, rows=rows, mention_counts=mention_counts)

    n = cs.compute_collocation_stats_batch(
        date_str="2026-05-27", country_config=cfg
    )
    assert n == 3
    for u in captured["updated"]:
        # Every Sprint-6 column non-NULL (acceptance criterion #4).
        for col in (
            "cooccurrence_with_entity_total",
            "entity_total",
            "token_total",
            "window_total",
            "pmi",
            "log_likelihood",
        ):
            assert u[col] is not None
        # LLR always >= 0 per Dunning math.
        assert u["log_likelihood"] >= 0.0
