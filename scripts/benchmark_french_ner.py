"""French NER spot benchmark — default multilingual vs French-specific model.

Sprint 2 decision tool. The entity-narrative track uses a single multilingual
NER artifact (``Davlan/bert-base-multilingual-cased-ner-hrl``) for all
languages. This script spot-checks French against a French-native model
(``Jean-Baptiste/camembert-ner``) over a small hand-built gold set so we can
decide whether French warrants a per-language NER override (Sprint 3) or the
multilingual model is good enough for Sprint 2.

Mirrors :mod:`src.analysis.entity_extraction` pipeline settings
(``aggregation_strategy="average"``, PER/ORG/LOC keep labels, subword/length
filtering) but stays standalone — no DB, no country config.

This is a dev tool: if a model can't be loaded (offline / not cached) it logs
a warning, marks that model unavailable, and still exits 0 so CI never breaks.

Usage:
    python scripts/benchmark_french_ner.py
    python scripts/benchmark_french_ner.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# Defaults (mirrored from src.analysis.entity_extraction)
# ---------------------------------------------------------------------------

DEFAULT_MULTILINGUAL_MODEL = "Davlan/bert-base-multilingual-cased-ner-hrl"
DEFAULT_FRENCH_MODEL = "Jean-Baptiste/camembert-ner"
KEEP_LABELS = ("PER", "ORG", "LOC")
AGGREGATION_STRATEGY = "average"
MIN_ENTITY_LENGTH = 3
MIN_ENTITY_SCORE = 0.5
_SUBWORD_MARKER = "##"

# Verdict band: if multilingual recall is within this many points of the
# French model, the multilingual artifact is "good enough" for Sprint 2.
VERDICT_TOLERANCE_PCT = 10.0

# ---------------------------------------------------------------------------
# Inline French gold set — representative news sentences with key entities.
# ``expected`` is a lightweight surface-form gold list (case-insensitive
# substring match against detected entities counts as a hit).
# ---------------------------------------------------------------------------

GOLD: list[dict[str, Any]] = [
    {
        "text": "Emmanuel Macron a reçu le chancelier allemand à l'Élysée pour évoquer l'avenir de l'Union européenne.",
        "expected": ["Macron", "Élysée", "Union européenne"],
    },
    {
        "text": "Marine Le Pen et le Rassemblement national contestent la politique migratoire du gouvernement.",
        "expected": ["Le Pen", "Rassemblement national"],
    },
    {
        "text": "L'OTAN a tenu un sommet à Bruxelles en présence des représentants de l'Ukraine.",
        "expected": ["OTAN", "Bruxelles", "Ukraine"],
    },
    {
        "text": "La Banque centrale européenne, basée à Francfort, a relevé ses taux directeurs.",
        "expected": ["Banque centrale européenne", "Francfort"],
    },
    {
        "text": "Le Premier ministre s'est rendu à Marseille pour annoncer un plan d'investissement.",
        "expected": ["Marseille"],
    },
    {
        "text": "Vladimir Poutine a rencontré Xi Jinping à Moscou pour renforcer les liens entre la Russie et la Chine.",
        "expected": ["Poutine", "Xi Jinping", "Moscou", "Russie", "Chine"],
    },
    {
        "text": "Le constructeur Renault a annoncé une alliance avec Nissan pour les véhicules électriques.",
        "expected": ["Renault", "Nissan"],
    },
    {
        "text": "Le président américain Joe Biden s'est entretenu avec les dirigeants du G7 au sujet de l'inflation.",
        "expected": ["Joe Biden", "G7"],
    },
    {
        "text": "La Commission européenne a ouvert une enquête contre Google pour abus de position dominante.",
        "expected": ["Commission européenne", "Google"],
    },
    {
        "text": "Des manifestants se sont rassemblés place de la République à Paris contre la réforme des retraites.",
        "expected": ["Paris", "République"],
    },
]


# ---------------------------------------------------------------------------
# Filtering helpers (mirrored from src.analysis.entity_extraction)
# ---------------------------------------------------------------------------


def _normalize_label(raw_label: str) -> str:
    label = (raw_label or "").upper().strip()
    if "-" in label:
        label = label.split("-", 1)[1]
    return label


def _is_subword_artifact(word: str) -> bool:
    if not word:
        return True
    if word.startswith(_SUBWORD_MARKER):
        return True
    return any(part.startswith(_SUBWORD_MARKER) for part in word.split())


def _keep_entity(word: str, label: str, score: float | None) -> bool:
    """Same accept/reject gates as the production extractor."""
    if label not in KEEP_LABELS:
        return False
    if not word or _is_subword_artifact(word):
        return False
    if len(word) < MIN_ENTITY_LENGTH:
        return False
    if score is not None and score < MIN_ENTITY_SCORE:
        return False
    return True


# ---------------------------------------------------------------------------
# Model loading + inference (monkeypatched in tests; never downloads there)
# ---------------------------------------------------------------------------


def load_ner_pipeline(model_id: str):
    """Build a transformers NER pipeline mirroring production settings.

    Imported lazily so the module (and its tests) don't require ``torch`` /
    ``transformers`` unless an actual benchmark run is attempted.
    """
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    logger.info(
        f"Loading NER model {model_id!r} on "
        f"{'CUDA' if device == 0 else 'CPU'}"
    )
    pipe = pipeline(
        "ner",
        model=model_id,
        aggregation_strategy=AGGREGATION_STRATEGY,
        device=device,
    )
    logger.success(f"NER model {model_id!r} loaded.")
    return pipe


def extract_entities(pipe, text: str) -> list[str]:
    """Run the pipeline on one sentence, return kept PER/ORG/LOC surfaces."""
    raw: list[dict[str, Any]] = pipe(text)
    surfaces: list[str] = []
    for ent in raw:
        label = _normalize_label(ent.get("entity_group") or ent.get("entity") or "")
        word = (ent.get("word") or "").strip()
        score = ent.get("score")
        if _keep_entity(word, label, score):
            surfaces.append(word)
    return surfaces


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _expected_hits(found: list[str], expected: list[str]) -> int:
    """Case-insensitive substring match: an expected term hits if it appears
    inside any detected entity surface (or vice versa)."""
    found_lower = [f.lower() for f in found]
    hits = 0
    for exp in expected:
        e = exp.lower()
        if any(e in f or f in e for f in found_lower):
            hits += 1
    return hits


def run_model(
    model_id: str,
    gold: list[dict[str, Any]],
    *,
    loader=load_ner_pipeline,
    extractor=extract_entities,
) -> dict[str, Any]:
    """Run one model over the gold set. Returns a structured result dict.

    On load failure the result is marked ``available=False`` with the error
    captured — the caller still reports whatever ran.
    """
    try:
        pipe = loader(model_id)
    except Exception as exc:  # noqa: BLE001 — dev tool, must not hard-fail
        logger.warning(
            f"Could not load NER model {model_id!r}: {exc}. "
            f"Marking results unavailable."
        )
        return {
            "model_id": model_id,
            "available": False,
            "error": str(exc),
            "per_sentence": [],
            "total_entities": 0,
            "expected_total": sum(len(g["expected"]) for g in gold),
            "expected_hits": 0,
            "recall_pct": 0.0,
        }

    per_sentence: list[dict[str, Any]] = []
    total_entities = 0
    total_hits = 0
    total_expected = 0
    for g in gold:
        try:
            found = extractor(pipe, g["text"])
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Inference failed on a sentence ({exc}); treating as no entities.")
            found = []
        hits = _expected_hits(found, g["expected"])
        total_entities += len(found)
        total_hits += hits
        total_expected += len(g["expected"])
        per_sentence.append(
            {
                "text": g["text"],
                "expected": g["expected"],
                "found": found,
                "hits": hits,
            }
        )

    recall_pct = (100.0 * total_hits / total_expected) if total_expected else 0.0
    return {
        "model_id": model_id,
        "available": True,
        "error": None,
        "per_sentence": per_sentence,
        "total_entities": total_entities,
        "expected_total": total_expected,
        "expected_hits": total_hits,
        "recall_pct": round(recall_pct, 1),
    }


def build_verdict(multilingual: dict[str, Any], french: dict[str, Any]) -> str:
    """One-line Sprint 2 recommendation comparing the two models."""
    if not multilingual["available"] and not french["available"]:
        return "inconclusive: neither model could be loaded (offline / not cached)"
    if not french["available"]:
        return (
            "inconclusive: French model unavailable — cannot compare; "
            f"multilingual recall={multilingual['recall_pct']}%"
        )
    if not multilingual["available"]:
        return (
            "inconclusive: multilingual model unavailable — cannot compare; "
            f"French recall={french['recall_pct']}%"
        )

    delta = french["recall_pct"] - multilingual["recall_pct"]
    if delta <= VERDICT_TOLERANCE_PCT:
        return (
            f"multilingual within {delta:.1f} pts of french "
            f"(<= {VERDICT_TOLERANCE_PCT:.0f} pt tolerance) "
            f"→ no FR override needed for Sprint 2"
        )
    return (
        f"french model materially better (+{delta:.1f} pts recall) "
        f"→ flag for Sprint 3 FR NER override"
    )


def build_result(
    gold: list[dict[str, Any]],
    multilingual_model: str,
    french_model: str,
    *,
    loader=load_ner_pipeline,
    extractor=extract_entities,
) -> dict[str, Any]:
    multilingual = run_model(
        multilingual_model, gold, loader=loader, extractor=extractor
    )
    french = run_model(french_model, gold, loader=loader, extractor=extractor)
    return {
        "gold_size": len(gold),
        "multilingual": multilingual,
        "french": french,
        "verdict": build_verdict(multilingual, french),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def format_result(result: dict[str, Any]) -> str:
    ml = result["multilingual"]
    fr = result["french"]
    lines = [
        "French NER spot benchmark",
        "",
        f"  multilingual: {ml['model_id']} "
        f"({'ok' if ml['available'] else 'UNAVAILABLE'})",
        f"  french:       {fr['model_id']} "
        f"({'ok' if fr['available'] else 'UNAVAILABLE'})",
        "",
        "Per-sentence entities (multilingual | french):",
    ]

    ml_by_text = {s["text"]: s for s in ml["per_sentence"]}
    fr_by_text = {s["text"]: s for s in fr["per_sentence"]}
    for g in (s["text"] for s in (ml["per_sentence"] or fr["per_sentence"])):
        lines.append("")
        lines.append(f"  • {g}")
        ml_s = ml_by_text.get(g)
        fr_s = fr_by_text.get(g)
        expected = (ml_s or fr_s or {}).get("expected", [])
        lines.append(f"      expected:     {expected}")
        lines.append(
            f"      multilingual: {ml_s['found'] if ml_s else '(n/a)'}"
        )
        lines.append(
            f"      french:       {fr_s['found'] if fr_s else '(n/a)'}"
        )

    lines.extend(["", "Summary:"])
    for label, m in (("multilingual", ml), ("french", fr)):
        if m["available"]:
            lines.append(
                f"  {label:<13} entities_found={m['total_entities']:<4} "
                f"expected_hits={m['expected_hits']}/{m['expected_total']} "
                f"recall={m['recall_pct']}%"
            )
        else:
            lines.append(f"  {label:<13} UNAVAILABLE ({m['error']})")

    lines.extend(["", f"Verdict: {result['verdict']}"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spot-benchmark French NER: multilingual vs French model."
    )
    parser.add_argument("--multilingual-model", default=DEFAULT_MULTILINGUAL_MODEL)
    parser.add_argument("--french-model", default=DEFAULT_FRENCH_MODEL)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Cap the number of gold sentences (0 = all)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    gold = GOLD if args.limit == 0 else GOLD[: args.limit]
    result = build_result(gold, args.multilingual_model, args.french_model)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        print(format_result(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
