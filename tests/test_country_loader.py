"""Tests for src.config.country_loader.

Covers happy-path loading of the bundled turkey.yaml and the validation
edges (missing file, missing identity field, malformed source) using
``tmp_path`` to write throwaway YAMLs that we point the loader at by
monkeypatching the configs directory.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.config import country_loader

# ---------------------------------------------------------------------------
# Happy path against the real bundled YAML
# ---------------------------------------------------------------------------


def test_load_turkey_config_returns_required_fields():
    cfg = country_loader.load_country_config("turkey")
    assert cfg["country_code"] == "TR"
    assert cfg["country_slug"] == "turkey"
    assert cfg["language"] == "tr"
    assert cfg["country_name"] == "Turkey"


def test_load_turkey_config_normalizes_sources():
    cfg = country_loader.load_country_config("turkey")
    # 10 outlets but multi-type sources (RSS + Google News sitemap for some)
    # mean total entries >= 10.
    assert isinstance(cfg["sources"], list) and len(cfg["sources"]) >= 10
    for src in cfg["sources"]:
        # Every source should have a normalized urls list (not 'url').
        assert "urls" in src and isinstance(src["urls"], list) and src["urls"]
        assert "url" not in src
        assert "name" in src
        # Currently supported types: rss (default), googlenews_sitemap,
        # html_sitemap (plain sitemap + per-article HTML scrape).
        assert src.get("type", "rss") in {"rss", "googlenews_sitemap", "html_sitemap"}


def test_load_by_country_code():
    """Lookup should accept ISO code in addition to slug."""
    cfg_by_code = country_loader.load_country_config("TR")
    cfg_by_slug = country_loader.load_country_config("turkey")
    assert cfg_by_code["country_code"] == cfg_by_slug["country_code"]


def test_load_is_case_insensitive():
    cfg_upper = country_loader.load_country_config("Turkey")
    cfg_lower = country_loader.load_country_config("turkey")
    assert cfg_upper["country_slug"] == cfg_lower["country_slug"]


def test_list_available_countries_includes_turkey():
    countries = country_loader.list_available_countries()
    codes = [c["code"] for c in countries]
    assert "TR" in codes
    tr = next(c for c in countries if c["code"] == "TR")
    assert tr == {
        "code": "TR",
        "slug": "turkey",
        "name": "Turkey",
        "language": "tr",
        "status": "active",
    }


# ---------------------------------------------------------------------------
# Error paths via temporary configs dir
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_configs(monkeypatch, tmp_path: Path):
    """Point the loader at a tmp directory so we can write malformed YAMLs."""
    monkeypatch.setattr(country_loader, "_CONFIGS_DIR", tmp_path)
    return tmp_path


def _write_yaml(dir_: Path, name: str, body: str) -> Path:
    path = dir_ / name
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_missing_country_raises_with_helpful_message(tmp_configs: Path):
    # Drop a single working config so the error can list "available".
    _write_yaml(
        tmp_configs,
        "germany.yaml",
        """
        country_code: DE
        country_name: Germany
        country_slug: germany
        language: de
        sources:
          - {name: X, urls: ["https://example.com/rss"]}
        """,
    )
    with pytest.raises(FileNotFoundError) as excinfo:
        country_loader.load_country_config("atlantis")
    assert "atlantis" in str(excinfo.value)
    assert "germany" in str(excinfo.value)


def test_missing_required_field_raises(tmp_configs: Path):
    _write_yaml(
        tmp_configs,
        "broken.yaml",
        """
        country_code: BR
        # country_name missing on purpose
        country_slug: broken
        language: br
        sources: []
        """,
    )
    with pytest.raises(ValueError) as excinfo:
        country_loader.load_country_config("broken")
    assert "country_name" in str(excinfo.value)


def test_source_without_urls_raises(tmp_configs: Path):
    # Note: ``country_code: NO`` and ``language: no`` would silently
    # become Python booleans under YAML 1.1 — quote / use other strings.
    _write_yaml(
        tmp_configs,
        "noisy.yaml",
        """
        country_code: NS
        country_name: Noisy
        country_slug: noisy
        language: nx
        sources:
          - name: NoUrls
        """,
    )
    with pytest.raises(ValueError) as excinfo:
        country_loader.load_country_config("noisy")
    assert "NoUrls" in str(excinfo.value)


def test_single_url_string_is_normalized_to_list(tmp_configs: Path):
    _write_yaml(
        tmp_configs,
        "simple.yaml",
        """
        country_code: SX
        country_name: Simple
        country_slug: simple
        language: sx
        sources:
          - name: One
            url: "https://example.com/rss"
        """,
    )
    cfg = country_loader.load_country_config("simple")
    src = cfg["sources"][0]
    assert src["urls"] == ["https://example.com/rss"]
    assert "url" not in src


def test_list_available_skips_broken_configs(tmp_configs: Path):
    # Working
    _write_yaml(
        tmp_configs,
        "alpha.yaml",
        """
        country_code: AL
        country_name: Alpha
        country_slug: alpha
        language: al
        sources: []
        """,
    )
    # Broken — missing country_name
    _write_yaml(
        tmp_configs,
        "beta.yaml",
        """
        country_code: BE
        country_slug: beta
        language: be
        sources: []
        """,
    )
    countries = country_loader.list_available_countries()
    codes = [c["code"] for c in countries]
    assert "AL" in codes
    assert "BE" not in codes
