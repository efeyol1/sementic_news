"""Frame bridge (Sprint 10 of the entity-narrative track).

Turn an entity's top collocates into a six-frame "framing intensity"
signal. This is the lightweight **seed-word** version: each frame owns a
list of lemmatized seed words (``configs/frames/<lang>.yaml``); a
collocate whose lemma is in a frame's list contributes its LLR weight to
that frame. Intensities are L1-normalized so they sum to 1.0 across the
six frames for one (entity, country, window).

Sprint 12 replaces this with a fine-tuned Policy Frames Codebook
classifier; the seed-word path is the honest, dependency-free baseline
whose accuracy the Sprint 10 quality gate measures.

Ethics: this is an ``(entity, country)``-scoped signal, never a
country-level claim. When no collocate matches any frame we return
``None`` rather than fabricate a distribution.

Usage (library):
    from src.analysis.frame_bridge import compute_frame_intensities
    fi = compute_frame_intensities(top_collocates, language="de")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml
from loguru import logger

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FRAMES_DIR = _REPO_ROOT / "configs" / "frames"

# Canonical frame order — fixes the axis order of the dashboard radar and
# makes serialized output deterministic regardless of YAML key order.
FRAME_NAMES: tuple[str, ...] = (
    "economic",
    "security",
    "identity",
    "governance",
    "humanitarian",
    "conflict",
)

_LEXICON_CACHE: dict[str, dict[str, set[str]]] = {}


# ---------------------------------------------------------------------------
# Config gate (mirrors collocations.is_enabled)
# ---------------------------------------------------------------------------


def _frame_bridge_cfg(country_config: dict[str, Any]) -> dict[str, Any]:
    return ((country_config.get("entity_narrative") or {})
            .get("frame_bridge") or {})


def is_enabled(country_config: dict[str, Any]) -> bool:
    """True when ``entity_narrative.frame_bridge.enabled`` is set.

    Defaults to ``False`` so the bridge is opt-in per country.
    """
    return bool(_frame_bridge_cfg(country_config).get("enabled", False))


# ---------------------------------------------------------------------------
# Lexicon loading
# ---------------------------------------------------------------------------


def _frames_path(language: str) -> Path:
    return _FRAMES_DIR / f"{language}.yaml"


def load_frame_lexicon(language: str) -> dict[str, set[str]]:
    """Read ``configs/frames/<language>.yaml`` into ``{frame: {seed, ...}}``.

    Seeds are lowercased to match the lemmas stored in
    ``entity_collocations`` (both the spaCy and TR surface lemmatizers
    emit lowercased lemmas). Process-wide cached. Returns an empty dict
    (→ no frame signal) when the file is missing or malformed.
    """
    cached = _LEXICON_CACHE.get(language)
    if cached is not None:
        return cached

    path = _frames_path(language)
    lexicon: dict[str, set[str]] = {}
    if not path.exists():
        logger.warning(
            f"Frame lexicon not found at {path} — frame bridge is a no-op "
            f"for language {language!r}"
        )
        _LEXICON_CACHE[language] = lexicon
        return lexicon

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    frames = data.get("frames") if isinstance(data, dict) else None
    if not isinstance(frames, dict):
        logger.warning(
            f"Frame lexicon {path} has no 'frames' mapping — returning empty"
        )
        _LEXICON_CACHE[language] = lexicon
        return lexicon

    for name, words in frames.items():
        if not isinstance(words, list):
            continue
        lexicon[str(name)] = {str(w).lower().strip() for w in words if w}

    _LEXICON_CACHE[language] = lexicon
    return lexicon


# ---------------------------------------------------------------------------
# Intensity computation
# ---------------------------------------------------------------------------


def compute_frame_intensities(
    top_collocates: Iterable[dict[str, Any]],
    language: str,
    *,
    lexicon: dict[str, set[str]] | None = None,
) -> dict[str, float] | None:
    """Score top collocates against the frame lexicon → normalized dict.

    Each collocate is ``{lemma, pos, pmi, llr, ...}``. A collocate whose
    lowercased lemma is in a frame's seed set adds its LLR (clamped ≥0)
    to that frame. The six frame scores are L1-normalized to sum to 1.0.

    Returns ``None`` when the lexicon is empty, no collocate matched, or
    the total matched weight is 0 — we never invent a distribution.
    Frames with no match are kept at 0.0 so every axis is present for the
    radar view.
    """
    lex = lexicon if lexicon is not None else load_frame_lexicon(language)
    if not lex:
        return None

    scores: dict[str, float] = {name: 0.0 for name in lex}
    matched = 0
    for collocate in top_collocates:
        lemma = str(collocate.get("lemma", "")).lower().strip()
        if not lemma:
            continue
        raw = collocate.get("llr")
        try:
            weight = float(raw) if raw is not None else 0.0
        except (TypeError, ValueError):
            weight = 0.0
        if weight < 0:
            weight = 0.0
        for name, seeds in lex.items():
            if lemma in seeds:
                scores[name] += weight
                matched += 1
                break

    total = sum(scores.values())
    if matched == 0 or total <= 0:
        return None

    # Deterministic key order: canonical frames first, then any extras.
    ordered = [n for n in FRAME_NAMES if n in scores]
    ordered += [n for n in scores if n not in FRAME_NAMES]
    return {name: scores[name] / total for name in ordered}
