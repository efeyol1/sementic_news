"""Tests for the Sprint 10 frame-bridge QA scripts (no DB).

``scripts/`` is on sys.path via conftest.py.
"""

from __future__ import annotations

import csv

import audit_frame_bridge as audit
import evaluate_frame_review as evaluate
import export_frame_review_sample as export
import pytest


def _profile(canonical, fi, collocates):
    return {
        "country_code": "DE",
        "window_days": 30,
        "end_date": "2026-05-29",
        "canonical": canonical,
        "wikidata_qid": None,
        "entity_type": "PER",
        "top_collocates": [{"lemma": lem, "llr": llr} for lem, llr in collocates],
        "frame_intensities": fi,
    }


# --- auditor ---------------------------------------------------------------


def test_dominant_frame_picks_max():
    assert audit.dominant_frame({"economic": 0.2, "conflict": 0.8}) == "conflict"


def test_dominant_frame_none_for_empty_or_zero():
    assert audit.dominant_frame(None) is None
    assert audit.dominant_frame({"economic": 0.0, "conflict": 0.0}) is None


def test_compute_frame_audit_coverage_and_unassigned():
    lexicon = {"economic": {"wirtschaft"}, "conflict": {"krieg"}}
    rows = [
        _profile("A", {"conflict": 1.0}, [("krieg", 10.0), ("foobar", 5.0)]),
        _profile("B", None, [("xyz", 2.0)]),
    ]
    report = audit.compute_frame_audit(rows, lexicon)
    assert report["profiles"] == 2
    assert report["profiles_with_frames"] == 1
    assert report["frame_coverage_pct"] == 50.0
    # 3 collocates total; "krieg" assigned, "foobar"+"xyz" unassigned → 2/3.
    assert report["top_collocates_total"] == 3
    assert report["unassigned_collocates"] == 2
    assert report["dominant_frame_distribution"] == {"conflict": 1}


# --- exporter --------------------------------------------------------------


def test_build_sample_stratifies_by_predicted_frame():
    rows = [
        _profile("A", {"conflict": 1.0}, [("krieg", 9.0)]),
        _profile("B", {"conflict": 0.9, "economic": 0.1}, [("krieg", 8.0)]),
        _profile("C", {"economic": 1.0}, [("wirtschaft", 7.0)]),
    ]
    sample = export.build_sample(rows, per_frame=1)
    frames = {r["predicted_frame"] for r in sample}
    # One per predicted frame: conflict (capped at 1 of 2) + economic.
    assert frames == {"conflict", "economic"}
    assert len(sample) == 2


# --- evaluator -------------------------------------------------------------


def test_evaluate_rows_accuracy_and_gate(tmp_path):
    rows = [
        {"predicted_frame": "conflict", "reviewed_frame": "conflict"},
        {"predicted_frame": "economic", "reviewed_frame": "economic"},
        {"predicted_frame": "security", "reviewed_frame": "conflict"},  # wrong
    ]
    report = evaluate.evaluate_rows(rows)
    assert report["evaluated"] == 3
    assert report["correct"] == 2
    assert report["accuracy"] == pytest.approx(2 / 3, abs=1e-4)


def test_load_reviewed_rows_skips_blanks_and_none(tmp_path):
    path = tmp_path / "frame_review.csv"
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["predicted_frame", "reviewed_frame"]
        )
        writer.writeheader()
        writer.writerow({"predicted_frame": "conflict", "reviewed_frame": "conflict"})
        writer.writerow({"predicted_frame": "economic", "reviewed_frame": ""})
        writer.writerow({"predicted_frame": "security", "reviewed_frame": "none"})
    usable, skipped = evaluate.load_reviewed_rows(path)
    assert len(usable) == 1
    assert skipped == 2
