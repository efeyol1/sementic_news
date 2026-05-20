from __future__ import annotations

import benchmark_french_ner as bm

# ---------------------------------------------------------------------------
# Canned gold + fake predictions — NO model download anywhere in this module.
# ---------------------------------------------------------------------------

GOLD = [
    {"text": "s1", "expected": ["Macron", "Élysée"]},
    {"text": "s2", "expected": ["OTAN", "Bruxelles"]},
    {"text": "s3", "expected": ["Renault", "Nissan"]},
]


def _loader(model_id: str):
    # The "pipeline" is just the model id; the fake extractor keys off it.
    return model_id


def _make_extractor(predictions: dict[str, dict[str, list[str]]]):
    """predictions: model_id -> {sentence_text -> [found surfaces]}."""

    def _extract(pipe, text: str) -> list[str]:
        return list(predictions.get(pipe, {}).get(text, []))

    return _extract


# ---------------------------------------------------------------------------
# Scoring primitives
# ---------------------------------------------------------------------------


def test_expected_hits_case_insensitive_substring():
    # exact, case-insensitive, and substring-both-ways all count.
    assert bm._expected_hits(["macron", "Union européenne"], ["Macron", "Union"]) == 2
    assert bm._expected_hits(["Le Pen"], ["Le Pen", "Macron"]) == 1
    assert bm._expected_hits([], ["Macron"]) == 0


def test_keep_entity_filters_match_production_gates():
    assert bm._keep_entity("Macron", "PER", 0.9) is True
    assert bm._keep_entity("Macron", "MISC", 0.9) is False  # label not kept
    assert bm._keep_entity("AB", "PER", 0.9) is False  # too short
    assert bm._keep_entity("##ron", "PER", 0.9) is False  # subword artifact
    assert bm._keep_entity("Macron", "PER", 0.3) is False  # below score gate
    assert bm._keep_entity("Macron", "PER", None) is True  # no score → pass


# ---------------------------------------------------------------------------
# run_model recall math
# ---------------------------------------------------------------------------


def test_run_model_recall_math():
    preds = {
        "m1": {
            "s1": ["Macron"],            # 1/2 hits
            "s2": ["OTAN", "Bruxelles"],  # 2/2 hits
            "s3": [],                     # 0/2 hits
        }
    }
    res = bm.run_model(
        "m1", GOLD, loader=_loader, extractor=_make_extractor(preds)
    )
    assert res["available"] is True
    assert res["total_entities"] == 3
    assert res["expected_total"] == 6
    assert res["expected_hits"] == 3
    assert res["recall_pct"] == 50.0


def test_run_model_marks_unavailable_on_load_failure():
    def _boom(model_id: str):
        raise OSError("model not cached, offline")

    res = bm.run_model("nope", GOLD, loader=_boom, extractor=_make_extractor({}))
    assert res["available"] is False
    assert "offline" in res["error"]
    assert res["recall_pct"] == 0.0
    assert res["expected_total"] == 6  # still reports the gold size


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------


def test_verdict_no_override_when_within_tolerance():
    preds = {
        "ml": {"s1": ["Macron", "Élysée"], "s2": ["OTAN"], "s3": ["Renault"]},
        "fr": {
            "s1": ["Macron", "Élysée"],
            "s2": ["OTAN", "Bruxelles"],
            "s3": ["Renault"],
        },
    }
    result = bm.build_result(
        GOLD, "ml", "fr", loader=_loader, extractor=_make_extractor(preds)
    )
    # ml recall = 4/6 = 66.7, fr recall = 5/6 = 83.3 → delta 16.6 > 10 → override
    assert result["multilingual"]["recall_pct"] == 66.7
    assert result["french"]["recall_pct"] == 83.3
    assert "Sprint 3" in result["verdict"]
    assert "materially better" in result["verdict"]


def test_verdict_flags_override_only_above_tolerance():
    # Within tolerance: identical predictions → delta 0 → no override.
    preds = {
        "ml": {"s1": ["Macron", "Élysée"], "s2": ["OTAN", "Bruxelles"], "s3": ["Renault", "Nissan"]},
        "fr": {"s1": ["Macron", "Élysée"], "s2": ["OTAN", "Bruxelles"], "s3": ["Renault", "Nissan"]},
    }
    result = bm.build_result(
        GOLD, "ml", "fr", loader=_loader, extractor=_make_extractor(preds)
    )
    assert result["multilingual"]["recall_pct"] == 100.0
    assert result["french"]["recall_pct"] == 100.0
    assert "no FR override needed for Sprint 2" in result["verdict"]


def test_verdict_inconclusive_when_french_unavailable():
    def _loader_ml_only(model_id: str):
        if model_id == "fr":
            raise OSError("not cached")
        return model_id

    preds = {"ml": {"s1": ["Macron"], "s2": [], "s3": []}}
    result = bm.build_result(
        GOLD, "ml", "fr", loader=_loader_ml_only, extractor=_make_extractor(preds)
    )
    assert result["french"]["available"] is False
    assert "inconclusive" in result["verdict"]
    assert "French model unavailable" in result["verdict"]


def test_verdict_inconclusive_when_both_unavailable():
    def _boom(model_id: str):
        raise OSError("offline")

    result = bm.build_result(
        GOLD, "ml", "fr", loader=_boom, extractor=_make_extractor({})
    )
    assert "inconclusive" in result["verdict"]
    assert "neither model" in result["verdict"]


# ---------------------------------------------------------------------------
# Rendering smoke (no models)
# ---------------------------------------------------------------------------


def test_format_result_renders_summary_and_verdict():
    preds = {
        "ml": {"s1": ["Macron"], "s2": ["OTAN"], "s3": []},
        "fr": {"s1": ["Macron", "Élysée"], "s2": ["OTAN", "Bruxelles"], "s3": ["Renault"]},
    }
    result = bm.build_result(
        GOLD, "ml", "fr", loader=_loader, extractor=_make_extractor(preds)
    )
    text = bm.format_result(result)
    assert "French NER spot benchmark" in text
    assert "Summary:" in text
    assert "Verdict:" in text
    assert "recall=" in text
