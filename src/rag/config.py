"""Per-country RAG gate + knobs (mirrors collocations/frame_bridge gates).

Config lives under ``entity_narrative.rag`` in ``configs/countries/*.yaml``:

    entity_narrative:
      rag:
        enabled: false               # default OFF — generation only where opted in
        model: claude-sonnet-4-6     # cost lever: claude-haiku-4-5
        max_snippets: 8
        langs: [tr, en]

Default ``enabled: false`` so unconfigured countries never spend money.
"""

from __future__ import annotations

from typing import Any

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_SNIPPETS = 8
DEFAULT_LANGS = ("tr", "en")


def rag_cfg(country_config: dict[str, Any]) -> dict[str, Any]:
    return ((country_config.get("entity_narrative") or {}).get("rag") or {})


def is_enabled(country_config: dict[str, Any]) -> bool:
    return bool(rag_cfg(country_config).get("enabled", False))


def model_for(country_config: dict[str, Any]) -> str:
    return rag_cfg(country_config).get("model") or DEFAULT_MODEL


def max_snippets(country_config: dict[str, Any]) -> int:
    return int(rag_cfg(country_config).get("max_snippets", DEFAULT_MAX_SNIPPETS))


def allowed_langs(country_config: dict[str, Any]) -> list[str]:
    langs = rag_cfg(country_config).get("langs") or list(DEFAULT_LANGS)
    return [str(x).lower() for x in langs]
