"""Tests for src.analysis.entity_resolution.

Pure unit tests — no real DB, no Wikidata network. The DB helpers and
``_resolve_wikidata`` are imported into the module namespace, so we
monkeypatch them on ``src.analysis.entity_resolution`` directly.
"""

from __future__ import annotations

import pytest

from src.analysis import entity_resolution as er
from src.analysis.entity_canonicalization import CanonicalEntity


def _country_config(wikidata_enabled: bool = False) -> dict:
    return {
        "country_code": "FR",
        "country_name": "France",
        "country_slug": "france",
        "language": "fr",
        "entity_narrative": {
            "aliases": {
                "people": {
                    "Emmanuel Macron": {
                        "wikidata_qid": "Q3052772",
                        "aliases": ["macron", "emmanuel macron", "président macron"],
                    },
                    "Donald Trump": {
                        "wikidata_qid": "Q22686",
                        "aliases": ["trump", "donald trump"],
                    },
                },
                "organizations": {},
            },
            "wikidata": {
                "enabled": wikidata_enabled,
                "min_confidence": 0.85,
            },
        },
    }


# ---------------------------------------------------------------------------
# _score_candidate — pure scoring function
# ---------------------------------------------------------------------------


def test_score_candidate_exact_label_beats_partial():
    cfg = _country_config()
    exact = er._score_candidate(
        {"id": "Q3052772", "label": "Emmanuel Macron", "description": "président français"},
        "Emmanuel Macron",
        "PER",
        cfg,
    )
    partial = er._score_candidate(
        {"id": "Q999", "label": "Macron", "description": "village"},
        "Emmanuel Macron",
        "PER",
        cfg,
    )
    assert exact > partial
    assert 0.0 <= exact <= 1.0


def test_score_candidate_type_keyword_bonus():
    cfg = _country_config()
    with_type = er._score_candidate(
        {"id": "Q1", "label": "Trump", "description": "American politician and president"},
        "Trump",
        "PER",
        cfg,
    )
    without_type = er._score_candidate(
        {"id": "Q2", "label": "Trump", "description": "a card game"},
        "Trump",
        "PER",
        cfg,
    )
    assert with_type > without_type


def test_score_candidate_multilingual_per_keywords():
    """FR/DE descriptions should trigger the PER type bonus.

    Sprint 2 regression: English-only keywords used to leave score=0.60 for
    Xi Jinping (FR ``homme politique chinois``), well below ``min_confidence``.
    The ``politi`` stem now catches politician / politique / Politiker(in)."""
    fr = _country_config()
    fr_with_type = er._score_candidate(
        {"id": "Q15031", "label": "Xi Jinping", "description": "homme politique chinois"},
        "Xi Jinping",
        "PER",
        fr,
    )
    # exact label (0.55) + multilingual type stem 'politi' (0.15) + id (0.05)
    assert fr_with_type >= 0.75

    de = {**fr, "country_code": "DE", "country_name": "Germany", "language": "de"}
    de_with_type = er._score_candidate(
        {"id": "Q567", "label": "Angela Merkel", "description": "deutsche politikerin"},
        "Angela Merkel",
        "PER",
        de,
    )
    assert de_with_type >= 0.75


def test_score_candidate_multilingual_org_keywords():
    """DE organisation descriptions (``Unternehmen`` etc.) trigger ORG bonus."""
    de = {**_country_config(), "country_code": "DE", "language": "de"}
    score = er._score_candidate(
        {
            "id": "Q19839259",
            "label": "Uniper",
            "description": "deutsches Unternehmen der Energiewirtschaft",
        },
        "Uniper",
        "ORG",
        de,
    )
    assert score >= 0.75


def test_score_candidate_multilingual_loc_keywords():
    """FR ``région``/``département`` descriptions trigger LOC bonus."""
    fr = _country_config()
    score = er._score_candidate(
        {
            "id": "Q17012",
            "label": "Guadeloupe",
            "description": "région et département français d'outre-mer",
        },
        "Guadeloupe",
        "LOC",
        fr,
    )
    assert score >= 0.75


# ---------------------------------------------------------------------------
# _resolve_one — local alias / cache / wikidata precedence
# ---------------------------------------------------------------------------


def test_resolve_one_local_alias_short_circuits(monkeypatch):
    """A configured alias never hits the cache or the network."""
    calls = {"cache": 0, "wikidata": 0}

    def _fail_cache(*a, **k):
        calls["cache"] += 1
        raise AssertionError("cache must not be queried for local aliases")

    def _fail_wikidata(*a, **k):
        calls["wikidata"] += 1
        raise AssertionError("wikidata must not be called for local aliases")

    monkeypatch.setattr(er, "fetch_entity_resolution_cache", _fail_cache)
    monkeypatch.setattr(er, "_resolve_wikidata", _fail_wikidata)

    result = er._resolve_one(
        {"id": 1, "entity_text": "Macron", "entity_type": "PER"},
        country_config=_country_config(wikidata_enabled=True),
        wikidata_enabled=True,
        min_confidence=0.85,
    )
    assert result.canonical == "Emmanuel Macron"
    assert result.wikidata_qid == "Q3052772"
    assert result.resolver_method == "local_alias"
    assert calls == {"cache": 0, "wikidata": 0}


def test_resolve_one_uses_cache_when_present(monkeypatch):
    monkeypatch.setattr(
        er,
        "fetch_entity_resolution_cache",
        lambda *a, **k: {
            "canonical": "Cached Person",
            "wikidata_qid": "Q42",
            "confidence": 0.91,
            "resolver_method": "wikidata",
        },
    )
    monkeypatch.setattr(
        er,
        "_resolve_wikidata",
        lambda *a, **k: pytest.fail("cache hit must skip wikidata"),
    )

    result = er._resolve_one(
        {"id": 2, "entity_text": "Some Unmapped Name", "entity_type": "PER"},
        country_config=_country_config(wikidata_enabled=True),
        wikidata_enabled=True,
        min_confidence=0.85,
    )
    assert result.wikidata_qid == "Q42"
    assert result.canonical == "Cached Person"
    assert result.confidence == 0.91


def test_resolve_one_wikidata_disabled_returns_local(monkeypatch):
    monkeypatch.setattr(er, "fetch_entity_resolution_cache", lambda *a, **k: None)
    monkeypatch.setattr(
        er,
        "_resolve_wikidata",
        lambda *a, **k: pytest.fail("wikidata disabled must not call the API"),
    )

    result = er._resolve_one(
        {"id": 3, "entity_text": "Unknown Org", "entity_type": "ORG"},
        country_config=_country_config(wikidata_enabled=False),
        wikidata_enabled=False,
        min_confidence=0.85,
    )
    # Falls back to a deterministic display canonical, no QID.
    assert result.wikidata_qid is None
    assert result.canonical


def test_resolve_one_wikidata_failure_is_soft(monkeypatch):
    monkeypatch.setattr(er, "fetch_entity_resolution_cache", lambda *a, **k: None)

    def _boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(er, "_resolve_wikidata", _boom)

    result = er._resolve_one(
        {"id": 4, "entity_text": "Flaky Entity", "entity_type": "PER"},
        country_config=_country_config(wikidata_enabled=True),
        wikidata_enabled=True,
        min_confidence=0.85,
    )
    assert result.wikidata_qid is None
    assert result.resolver_method != "wikidata"


def test_resolve_one_wikidata_success(monkeypatch):
    monkeypatch.setattr(er, "fetch_entity_resolution_cache", lambda *a, **k: None)
    monkeypatch.setattr(
        er,
        "_resolve_wikidata",
        lambda *a, **k: CanonicalEntity(
            canonical="Resolved Name",
            wikidata_qid="Q123",
            confidence=0.95,
            resolver_method="wikidata",
        ),
    )

    result = er._resolve_one(
        {"id": 5, "entity_text": "Some New Politician", "entity_type": "PER"},
        country_config=_country_config(wikidata_enabled=True),
        wikidata_enabled=True,
        min_confidence=0.85,
    )
    assert result.wikidata_qid == "Q123"
    assert result.resolver_method == "wikidata"


# ---------------------------------------------------------------------------
# resolve_entities_batch — orchestration
# ---------------------------------------------------------------------------


def test_resolve_batch_requires_country_config():
    with pytest.raises(ValueError):
        er.resolve_entities_batch(date_str="2026-05-18", country_config=None)


def test_resolve_batch_end_to_end(monkeypatch):
    captured: dict = {}

    monkeypatch.setattr(
        er,
        "fetch_unresolved_entity_mentions",
        lambda *a, **k: [
            {"id": 10, "entity_text": "Macron", "entity_type": "PER"},
            {"id": 11, "entity_text": "Unknown Body", "entity_type": "ORG"},
        ],
    )
    monkeypatch.setattr(er, "fetch_entity_resolution_cache", lambda *a, **k: None)
    monkeypatch.setattr(
        er,
        "_resolve_wikidata",
        lambda *a, **k: None,  # second mention stays unresolved (fail-soft local)
    )
    monkeypatch.setattr(
        er,
        "bulk_update_entity_resolution",
        lambda updates: captured.setdefault("updates", updates),
    )
    monkeypatch.setattr(
        er,
        "upsert_entity_resolution_cache",
        lambda rows: captured.setdefault("cache", rows),
    )

    n = er.resolve_entities_batch(
        date_str="2026-05-18",
        country_config=_country_config(wikidata_enabled=True),
        limit=100,
    )

    assert n == 2
    updates = {u["id"]: u for u in captured["updates"]}
    # Local alias resolved with QID.
    assert updates[10]["wikidata_qid"] == "Q3052772"
    assert updates[10]["resolver_method"] == "local_alias"
    # Unresolved one still gets a row (canonical present, QID null).
    assert updates[11]["wikidata_qid"] is None
    assert updates[11]["canonical"]
    # Cache rows mirror the updates and carry a normalized key.
    assert len(captured["cache"]) == 2
    assert all("normalized_text" in row for row in captured["cache"])


# ---------------------------------------------------------------------------
# Wikidata API rate limiting + 429 retry (added 2026-05-20 after first FR/DE
# live runs hit the rate limit with 156 sequential unthrottled requests).
# ---------------------------------------------------------------------------


def test_wikidata_throttle_enforces_min_interval(monkeypatch):
    """Second call within ``_WIKIDATA_MIN_INTERVAL`` must sleep the difference."""
    fake_clock = {"t": 0.0}
    slept: list[float] = []

    monkeypatch.setattr(er.time, "monotonic", lambda: fake_clock["t"])
    monkeypatch.setattr(er.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(er, "_last_wikidata_call", float("-inf"))

    er._wikidata_throttle()        # first call: no sleep, sets last=0
    fake_clock["t"] = 0.3          # 0.3s later
    er._wikidata_throttle()        # must sleep min_interval - 0.3

    assert len(slept) == 1
    assert slept[0] == pytest.approx(er._WIKIDATA_MIN_INTERVAL - 0.3, rel=1e-6)


def test_wikidata_throttle_skips_when_interval_elapsed(monkeypatch):
    fake_clock = {"t": 0.0}
    slept: list[float] = []

    monkeypatch.setattr(er.time, "monotonic", lambda: fake_clock["t"])
    monkeypatch.setattr(er.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(er, "_last_wikidata_call", float("-inf"))

    er._wikidata_throttle()
    fake_clock["t"] = er._WIKIDATA_MIN_INTERVAL + 0.5  # plenty of time passed
    er._wikidata_throttle()

    assert slept == []  # no sleep when interval already elapsed


def test_wikidata_request_retries_on_429_then_succeeds(monkeypatch):
    """429 → backoff → retry → success returns parsed JSON."""
    import io
    import urllib.error

    monkeypatch.setattr(er, "_wikidata_throttle", lambda: None)
    sleeps: list[float] = []
    monkeypatch.setattr(er.time, "sleep", lambda s: sleeps.append(s))

    attempts = {"n": 0}

    def _fake_urlopen(req, timeout):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise urllib.error.HTTPError(
                req.full_url, 429, "Too Many Requests", hdrs=None, fp=None
            )

        class _Resp:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

            def read(self_inner):
                return b'{"search": [{"id": "Q1", "label": "Ok"}]}'

        return _Resp()

    monkeypatch.setattr(er.urllib.request, "urlopen", _fake_urlopen)

    payload = er._wikidata_request("http://example/api", timeout=5.0)

    assert payload == {"search": [{"id": "Q1", "label": "Ok"}]}
    assert attempts["n"] == 2
    # One backoff sleep before retry (first attempt's backoff is base).
    assert sleeps == [er._WIKIDATA_BACKOFF_BASE]


def test_wikidata_request_gives_up_after_max_retries(monkeypatch):
    """Persistent 429s raise the HTTPError so the caller can fail-soft."""
    import urllib.error

    monkeypatch.setattr(er, "_wikidata_throttle", lambda: None)
    monkeypatch.setattr(er.time, "sleep", lambda s: None)

    def _always_429(req, timeout):
        raise urllib.error.HTTPError(
            req.full_url, 429, "Too Many Requests", hdrs=None, fp=None
        )

    monkeypatch.setattr(er.urllib.request, "urlopen", _always_429)

    with pytest.raises(urllib.error.HTTPError) as excinfo:
        er._wikidata_request("http://example/api", timeout=5.0)
    assert excinfo.value.code == 429


def test_wikidata_request_propagates_non_429_immediately(monkeypatch):
    """A non-rate-limit error fails on the first attempt — no retry."""
    import urllib.error

    monkeypatch.setattr(er, "_wikidata_throttle", lambda: None)
    sleeps: list[float] = []
    monkeypatch.setattr(er.time, "sleep", lambda s: sleeps.append(s))

    attempts = {"n": 0}

    def _server_error(req, timeout):
        attempts["n"] += 1
        raise urllib.error.HTTPError(
            req.full_url, 500, "Internal Server Error", hdrs=None, fp=None
        )

    monkeypatch.setattr(er.urllib.request, "urlopen", _server_error)

    with pytest.raises(urllib.error.HTTPError) as excinfo:
        er._wikidata_request("http://example/api", timeout=5.0)
    assert excinfo.value.code == 500
    assert attempts["n"] == 1
    assert sleeps == []  # no backoff for non-429
