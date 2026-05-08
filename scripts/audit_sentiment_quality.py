"""Audit sentiment scoring freshness and obvious label-quality risks.

Usage:
    python scripts/audit_sentiment_quality.py --date 2026-05-06
    python scripts/audit_sentiment_quality.py --date 2026-05-06 --json
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal
from statistics import mean
from typing import Any

from src.db.queries import fetch_sentiment_quality_rows

_LABELS = ("negative", "neutral", "positive")


def _normalize(text: str) -> str:
    # str.lower() alone is locale-broken for Turkish ("İ".lower() yields
    # "i̇" with a combining dot above). NFKD + combining-mark strip
    # folds diacritic and dotted-i variants to a single ASCII form so cues
    # can be written once and still match upper/lower/accented inputs.
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch)).lower()


# Cues match by stem with a trailing ``\w*`` so "öldür" catches "öldürdü" /
# "öldürüldü" without listing each Turkish inflection. Prefix a cue with "="
# to require an exact word-boundary match — needed for short stems whose
# lookalikes would otherwise leak across sentiment (``kaza`` must not match
# ``kazandı``; ``başarı`` must not match ``başarısız``).
_NEGATIVE_CUES_RAW = (
    "deprem",
    "hayatını kaybet",
    "iflas",
    "istifa",
    "=kaza",
    "kan donduran",
    "kriz",
    "kurşunlan",
    "öldür",
    "orman yangını",
    "saldır",
    "soruşturma",
    "tutuklan",
    "uyuşturucu",
    "zarar",
    "vefat",
    "cinayet",
    "patlama",
    "intihar",
    "yaralan",
    "yolsuzluk",
    "rüşvet",
    "gözaltı",
    "iddianame",
    "hapis",
    "mahkum",
    "felaket",
    "skandal",
)
_POSITIVE_CUES_RAW = (
    "=başarı",
    "büyüdü",
    "hizmete açıl",
    "imzalandı",
    "kârını ikiye katladı",
    "kazandı",
    "rekor",
    "madalya",
    "müjde",
    "şampiyon",
    "ödül",
    "zafer",
)


def _prepare_cues(raw: tuple[str, ...]) -> tuple[tuple[str, bool], ...]:
    """Normalize cues and tag the ``=`` exact-match entries."""
    prepared: list[tuple[str, bool]] = []
    for cue in raw:
        is_exact = cue.startswith("=")
        body = cue[1:] if is_exact else cue
        prepared.append((_normalize(body), is_exact))
    return tuple(prepared)


_NEGATIVE_CUES = _prepare_cues(_NEGATIVE_CUES_RAW)
_POSITIVE_CUES = _prepare_cues(_POSITIVE_CUES_RAW)


def _pct(part: int, whole: int) -> float:
    return round(part / whole * 100, 1) if whole else 0.0


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def _text(row: dict[str, Any]) -> str:
    return _normalize(" ".join(
        str(row.get(field) or "")
        for field in ("cleaned_title", "cleaned_summary", "cleaned_article_text")
    ))


def _cue_text(row: dict[str, Any]) -> str:
    return _normalize(" ".join(
        str(row.get(field) or "")
        for field in ("cleaned_title", "cleaned_summary")
    ))


def _title(row: dict[str, Any]) -> str:
    return str(row.get("cleaned_title") or "").strip()


def _has_any(text: str, cues: tuple[tuple[str, bool], ...]) -> bool:
    for cue, is_exact in cues:
        if is_exact:
            pattern = rf"(?<!\w){re.escape(cue)}(?!\w)"
        else:
            pattern = rf"(?<!\w){re.escape(cue)}\w*"
        if re.search(pattern, text):
            return True
    return False


def _is_stale_after_body(row: dict[str, Any]) -> bool:
    analyzed_at = row.get("analyzed_at")
    article_fetched_at = row.get("article_fetched_at")
    return bool(row.get("sentiment_label") and analyzed_at and article_fetched_at and analyzed_at < article_fetched_at)


def _is_scored_after_body(row: dict[str, Any]) -> bool:
    analyzed_at = row.get("analyzed_at")
    article_fetched_at = row.get("article_fetched_at")
    return bool(row.get("sentiment_label") and analyzed_at and article_fetched_at and analyzed_at >= article_fetched_at)


def _label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("sentiment_label")) for row in rows if row.get("sentiment_label"))
    return {label: counts.get(label, 0) for label in _LABELS}


def _avg_scores(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    by_label: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        label = row.get("sentiment_label")
        score = _as_float(row.get("sentiment_score"))
        if label in _LABELS and score is not None:
            by_label[str(label)].append(score)
    return {
        label: round(mean(scores), 4) if scores else None
        for label, scores in ((label, by_label[label]) for label in _LABELS)
    }


def _suspicious_rows(
    rows: list[dict[str, Any]],
    min_score: float,
    max_examples: int,
) -> tuple[Counter[str], list[dict[str, Any]]]:
    counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    for row in rows:
        label = row.get("sentiment_label")
        score = _as_float(row.get("sentiment_score"))
        if score is None or score < min_score:
            continue
        text = _cue_text(row)
        reason = None
        if label == "positive" and _has_any(text, _NEGATIVE_CUES):
            reason = "positive_with_negative_cue"
        elif label == "negative" and _has_any(text, _POSITIVE_CUES):
            reason = "negative_with_positive_cue"

        if not reason:
            continue
        counts[reason] += 1
        examples.append(
            {
                "id": row.get("id"),
                "source": row.get("source_name") or "unknown",
                "category": row.get("category") or "unknown",
                "label": label,
                "score": round(score, 4),
                "reason": reason,
                "title": _title(row),
            }
        )

    examples.sort(key=lambda row: row["score"], reverse=True)
    return counts, examples[:max_examples]


def summarize_rows(
    rows: list[dict[str, Any]],
    min_neutral_pct: float = 5.0,
    high_confidence: float = 0.85,
    max_examples: int = 20,
) -> dict[str, Any]:
    """Build sentiment freshness and label-risk summary from DB rows."""
    total = len(rows)
    scored_rows = [row for row in rows if row.get("sentiment_label")]
    scored = len(scored_rows)
    with_body = sum(1 for row in rows if row.get("cleaned_article_text"))
    scored_with_body = sum(1 for row in scored_rows if row.get("cleaned_article_text"))
    stale_after_body = sum(1 for row in scored_rows if _is_stale_after_body(row))
    scored_after_body = sum(1 for row in scored_rows if _is_scored_after_body(row))
    label_counts = _label_counts(scored_rows)
    label_pct = {label: _pct(count, scored) for label, count in label_counts.items()}
    suspicious_counts, suspicious_examples = _suspicious_rows(
        scored_rows,
        min_score=high_confidence,
        max_examples=max_examples,
    )

    warnings: list[str] = []
    if stale_after_body:
        warnings.append("stale_sentiment_after_body_fetch")
    if scored and label_pct["neutral"] < min_neutral_pct:
        warnings.append("low_neutral_share")
    if suspicious_counts:
        warnings.append("high_confidence_label_cue_mismatch")

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row.get("source_name") or "unknown")].append(row)

    sources = []
    for source, source_rows in by_source.items():
        source_scored_rows = [row for row in source_rows if row.get("sentiment_label")]
        source_scored = len(source_scored_rows)
        sources.append(
            {
                "source": source,
                "total": len(source_rows),
                "scored": source_scored,
                "scored_pct": _pct(source_scored, len(source_rows)),
                "stale_after_body": sum(1 for row in source_scored_rows if _is_stale_after_body(row)),
                "label_counts": _label_counts(source_scored_rows),
            }
        )
    sources.sort(key=lambda row: (row["stale_after_body"], row["scored"], row["total"]), reverse=True)

    return {
        "totals": {
            "total": total,
            "scored": scored,
            "unscored": total - scored,
            "with_body": with_body,
            "scored_with_body": scored_with_body,
            "scored_after_body": scored_after_body,
            "stale_after_body": stale_after_body,
            "label_counts": label_counts,
            "label_pct": label_pct,
            "avg_scores": _avg_scores(scored_rows),
        },
        "warnings": warnings,
        "sources": sources,
        "suspicious": {
            "counts": dict(suspicious_counts),
            "examples": suspicious_examples,
        },
        "thresholds": {
            "min_neutral_pct": min_neutral_pct,
            "high_confidence": high_confidence,
        },
    }


def format_summary(summary: dict[str, Any], date_str: str) -> str:
    totals = summary["totals"]
    lines = [
        f"Sentiment quality audit: {date_str}",
        "",
        "Totals:",
        f"  total rows: {totals['total']}",
        f"  scored rows: {totals['scored']} ({_pct(totals['scored'], totals['total'])}%)",
        f"  unscored rows: {totals['unscored']}",
        f"  rows with body: {totals['with_body']}",
        f"  scored with body present: {totals['scored_with_body']}",
        f"  scored after body fetch: {totals['scored_after_body']}",
        f"  stale after body fetch: {totals['stale_after_body']}",
        f"  label counts: {totals['label_counts']}",
        f"  label pct: {totals['label_pct']}",
        f"  avg scores: {totals['avg_scores']}",
    ]

    if summary["warnings"]:
        lines.extend(["", "Warnings:"])
        for warning in summary["warnings"]:
            lines.append(f"  {warning}")

    lines.extend(["", "Sources:"])
    for row in summary["sources"]:
        lines.append(
            "  "
            f"{row['source']}: scored={row['scored']}/{row['total']} "
            f"scored_pct={row['scored_pct']}% stale_after_body={row['stale_after_body']} "
            f"labels={row['label_counts']}"
        )

    suspicious = summary["suspicious"]
    lines.extend(["", f"Suspicious high-confidence cue mismatches: {suspicious['counts']}"])
    for row in suspicious["examples"]:
        lines.append(
            "  "
            f"{row['id']} | {row['source']} | {row['label']} {row['score']} | "
            f"{row['reason']} | {row['title']}"
        )

    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit sentiment freshness and label-quality risks.")
    parser.add_argument("--date", default=date.today().isoformat(), metavar="YYYY-MM-DD")
    parser.add_argument("--min-neutral-pct", type=float, default=5.0)
    parser.add_argument("--high-confidence", type=float, default=0.85)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--json", action="store_true", help="Print machine-readable summary JSON.")
    args = parser.parse_args()
    if not 0 <= args.min_neutral_pct <= 100:
        parser.error("--min-neutral-pct must be between 0 and 100")
    if not 0 <= args.high_confidence <= 1:
        parser.error("--high-confidence must be between 0 and 1")
    if args.max_examples < 1:
        parser.error("--max-examples must be positive")
    return args


def main() -> int:
    args = _parse_args()
    rows = fetch_sentiment_quality_rows(args.date)
    summary = summarize_rows(
        rows,
        min_neutral_pct=args.min_neutral_pct,
        high_confidence=args.high_confidence,
        max_examples=args.max_examples,
    )
    if args.json:
        print(json.dumps({"date": args.date, **summary}, ensure_ascii=False, indent=2, default=str))
    else:
        print(format_summary(summary, args.date))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
