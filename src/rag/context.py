"""Context assembly + shared types for the RAG layer.

Pure functions, no I/O — fully unit-testable. Turns a profile row + a
list of retrieved snippets into the prompt blocks fed to the generator,
and defines the normalized ``Explanation`` shape both backends return.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from src.analysis.frame_bridge import FRAME_NAMES

# A profile needs at least this many collocate-bearing snippets to be
# explained without an "insufficient evidence" hedge.
MIN_EVIDENCE_SNIPPETS = 2

# Turkish + English frame labels for the deterministic extractive summary.
_FRAME_LABELS = {
    "tr": {
        "economic": "ekonomi", "security": "güvenlik", "identity": "kimlik",
        "governance": "yönetim", "humanitarian": "insani", "conflict": "çatışma",
    },
    "en": {
        "economic": "economic", "security": "security", "identity": "identity",
        "governance": "governance", "humanitarian": "humanitarian",
        "conflict": "conflict",
    },
}


@dataclass(frozen=True)
class Snippet:
    """One retrieved, citable piece of evidence."""

    id: str                       # "S1"
    text: str
    source_name: str
    date: str                     # ISO date string
    link: str | None
    article_id: int
    matched_collocates: list[str] = field(default_factory=list)


@dataclass
class PromptContext:
    """Everything a generator needs — structured signal + evidence."""

    canonical: str
    entity_type: str
    wikidata_qid: str | None
    country_code: str
    window_days: int
    end_date: str
    lang: str
    top_collocates: list[dict[str, Any]]
    frame_intensities: dict[str, float] | None
    snippets: list[Snippet]
    insufficient_evidence: bool
    structured_block: str
    evidence_block: str


@dataclass
class Explanation:
    """Normalized output — identical shape from either backend."""

    frame_summary: str
    claims: list[dict[str, Any]]          # {text, citations:[ids], metric_refs:[str]}
    citations: list[dict[str, Any]]       # {id, source_name, date, link, snippet}
    insufficient_evidence: bool
    backend: str                          # "anthropic" | "extractive"
    model: str | None = None

    def to_payload(self) -> dict[str, Any]:
        """JSONB-serializable dict stored in the cache + returned by the API."""
        return {
            "frame_summary": self.frame_summary,
            "claims": self.claims,
            "citations": self.citations,
            "insufficient_evidence": self.insufficient_evidence,
            "backend": self.backend,
            "model": self.model,
        }


# ---------------------------------------------------------------------------
# Profile hash (cache invalidation key)
# ---------------------------------------------------------------------------


def profile_hash(profile: dict[str, Any]) -> str:
    """sha256 over the load-bearing profile fields only.

    Excludes volatile fields like ``profile_computed_at`` so the cache
    invalidates when the *signal* changes, not merely when the daily
    aggregator re-runs.
    """
    payload = {
        "top_collocates": profile.get("top_collocates"),
        "frame_intensities": profile.get("frame_intensities"),
        "avg_pmi": profile.get("avg_pmi"),
        "avg_log_likelihood": profile.get("avg_log_likelihood"),
        "total_mentions": profile.get("total_mentions"),
        "coverage_days": profile.get("coverage_days"),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------


def dominant_frame(frame_intensities: dict[str, float] | None) -> tuple[str, float] | None:
    """(frame, intensity) with the highest value, FRAME_NAMES tie-break."""
    if not frame_intensities:
        return None
    best, best_val = None, 0.0
    for name in FRAME_NAMES:
        v = float(frame_intensities.get(name, 0.0) or 0.0)
        if v > best_val:
            best, best_val = name, v
    return (best, best_val) if best else None


def frame_label(frame: str, lang: str) -> str:
    return _FRAME_LABELS.get(lang, _FRAME_LABELS["en"]).get(frame, frame)


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------


def _structured_block(profile: dict[str, Any], top_k: int = 10) -> str:
    lines: list[str] = []
    cols = (profile.get("top_collocates") or [])[:top_k]
    if cols:
        lines.append("Top collocates (statistically computed):")
        for c in cols:
            pmi = c.get("pmi")
            llr = c.get("llr")
            pmi_s = f"{pmi:.2f}" if isinstance(pmi, (int, float)) else "—"
            llr_s = f"{llr:.1f}" if isinstance(llr, (int, float)) else "—"
            lines.append(f"  - {c.get('lemma')} ({c.get('pos')}): PMI={pmi_s}, LLR={llr_s}")
    fi = profile.get("frame_intensities")
    if fi:
        ordered = sorted(
            ((k, v) for k, v in fi.items() if v), key=lambda kv: -kv[1]
        )
        if ordered:
            lines.append("Frame intensities (sum=1.0):")
            lines.append("  " + ", ".join(f"{k}: {v:.2f}" for k, v in ordered))
    return "\n".join(lines)


def _evidence_block(snippets: list[Snippet]) -> str:
    if not snippets:
        return "(no snippets retrieved)"
    return "\n".join(
        f'[{s.id}] ({s.source_name}, {s.date}): "{s.text}"' for s in snippets
    )


def build_context(
    profile: dict[str, Any],
    snippets: list[Snippet],
    lang: str,
) -> PromptContext:
    """Assemble the structured + evidence blocks for one entity profile."""
    collocate_bearing = sum(1 for s in snippets if s.matched_collocates)
    insufficient = collocate_bearing < MIN_EVIDENCE_SNIPPETS
    return PromptContext(
        canonical=profile["canonical"],
        entity_type=profile["entity_type"],
        wikidata_qid=profile.get("wikidata_qid"),
        country_code=profile["country_code"],
        window_days=profile["window_days"],
        end_date=str(profile["end_date"]),
        lang=lang,
        top_collocates=profile.get("top_collocates") or [],
        frame_intensities=profile.get("frame_intensities"),
        snippets=snippets,
        insufficient_evidence=insufficient,
        structured_block=_structured_block(profile),
        evidence_block=_evidence_block(snippets),
    )


def citations_from_snippets(snippets: list[Snippet]) -> list[dict[str, Any]]:
    """The citation payload (id → source/date/link/snippet) for the response."""
    return [
        {
            "id": s.id,
            "source_name": s.source_name,
            "date": s.date,
            "link": s.link,
            "snippet": s.text,
        }
        for s in snippets
    ]
