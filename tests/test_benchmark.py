"""Unit tests for the inference benchmark helpers in scripts/benchmark_inference.py.

``scripts/`` is added to sys.path by ``tests/conftest.py``. Heavy
dependencies (``optimum``) are lazy-imported inside the predictor
wrappers, so importing the module here does not require the ``serving``
extra.
"""

from __future__ import annotations

import pytest
from benchmark_inference import _agreement, _markdown_table

# ---------------------------------------------------------------------------
# _agreement
# ---------------------------------------------------------------------------


def test_agreement_all_match():
    assert _agreement(["pos", "neg", "neu"], ["pos", "neg", "neu"]) == 1.0


def test_agreement_none_match():
    assert _agreement(["pos", "neg"], ["neg", "pos"]) == 0.0


def test_agreement_partial():
    # 2 of 4 match → 0.5
    ref = ["pos", "neg", "neu", "pos"]
    cand = ["pos", "neg", "pos", "neg"]
    assert _agreement(ref, cand) == 0.5


def test_agreement_empty_lists():
    # No predictions → defined as 0.0 to keep the gate well-behaved on no-op runs.
    assert _agreement([], []) == 0.0


def test_agreement_mismatched_length_raises():
    with pytest.raises(ValueError):
        _agreement(["a", "b"], ["a"])


# ---------------------------------------------------------------------------
# _markdown_table
# ---------------------------------------------------------------------------


def _row(name: str, p95: float, *, agreement: float | None = None) -> dict:
    # QPS is precomputed (1000/p95 when meaningful) so a degenerate p95=0
    # row stays constructible and exercises the divide-by-zero guard in
    # ``_markdown_table``.
    base = {
        "name": name,
        "p50_ms": max(p95 - 0.5, 0.0),
        "p95_ms": p95,
        "p99_ms": p95 + 0.5,
        "throughput_qps": round(1000.0 / p95, 1) if p95 else 0.0,
    }
    if agreement is not None:
        base["agreement"] = agreement
    return base


def test_markdown_table_baseline_label():
    rows = [_row("PyTorch", 10.0)]
    out = _markdown_table(rows, baseline_p95=10.0)
    assert "_baseline_" in out
    assert "1.00×" in out
    assert "PyTorch" in out


def test_markdown_table_speedup_and_agreement():
    rows = [
        _row("PyTorch", 10.0),
        _row("ONNX int8", 2.5, agreement=1.0),
    ]
    out = _markdown_table(rows, baseline_p95=10.0)
    # 10.0 / 2.5 = 4.00× speedup
    assert "4.00×" in out
    # Agreement renders as percentage with 1 decimal
    assert "100.0%" in out


def test_markdown_table_zero_p95_safe():
    # Defensive: a degenerate row shouldn't divide by zero.
    rows = [_row("Broken", 0.0, agreement=0.99)]
    out = _markdown_table(rows, baseline_p95=10.0)
    assert "0.00×" in out
