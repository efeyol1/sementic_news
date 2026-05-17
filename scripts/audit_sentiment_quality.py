"""Audit sentiment scoring freshness and obvious label-quality risks.

Usage:
    python scripts/audit_sentiment_quality.py --date 2026-05-06
    python scripts/audit_sentiment_quality.py --country germany --date 2026-05-06
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

from src.config import load_country_config
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


def _avg_scores(rows: list[dict[str, Any]], field: str = "sentiment_score") -> dict[str, float | None]:
    by_label: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        label = row.get("sentiment_label")
        score = _as_float(row.get(field))
        if label in _LABELS and score is not None:
            by_label[str(label)].append(score)
    return {
        label: round(mean(scores), 4) if scores else None
        for label, scores in ((label, by_label[label]) for label in _LABELS)
    }


def _margin(row: dict[str, Any]) -> float | None:
    scores = row.get("sentiment_scores")
    if not isinstance(scores, dict) or len(scores) < 2:
        return None
    values = sorted((_as_float(v) for v in scores.values() if v is not None), reverse=True)
    if len(values) < 2 or values[0] is None or values[1] is None:
        return None
    return values[0] - values[1]


def _confidence_summary(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    raw_scores = [_as_float(row.get("sentiment_score")) for row in rows]
    calibrated_scores = [
        _as_float(row.get("calibrated_sentiment_score"))
        for row in rows
        if row.get("calibrated_sentiment_score") is not None
    ]
    raw_scores = [score for score in raw_scores if score is not None]
    return {
        "raw_avg_confidence": round(mean(raw_scores), 4) if raw_scores else None,
        "calibrated_avg_confidence": (
            round(mean(calibrated_scores), 4) if calibrated_scores else None
        ),
        "raw_high_confidence_count": sum(1 for score in raw_scores if score >= threshold),
        "calibrated_high_confidence_count": sum(
            1 for score in calibrated_scores if score >= threshold
        ),
    }


def _low_margin_examples(rows: list[dict[str, Any]], max_examples: int) -> list[dict[str, Any]]:
    examples = []
    for row in rows:
        margin = _margin(row)
        if margin is None or margin > 0.15:
            continue
        examples.append(
            {
                "id": row.get("id"),
                "source": row.get("source_name") or "unknown",
                "label": row.get("sentiment_label"),
                "raw_score": _as_float(row.get("sentiment_score")),
                "calibrated_score": _as_float(row.get("calibrated_sentiment_score")),
                "margin": round(margin, 4),
                "title": _title(row),
            }
        )
    examples.sort(key=lambda row: (row["margin"], -(row["raw_score"] or 0.0)))
    return examples[:max_examples]


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
    max_positive_pct: float = 45.0,
    max_mismatch_pct: float = 2.0,
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
    suspicious_total = sum(suspicious_counts.values())
    suspicious_pct = _pct(suspicious_total, scored)
    confidence = _confidence_summary(scored_rows, threshold=0.90)
    low_margin_examples = _low_margin_examples(scored_rows, max_examples=max_examples)

    warnings: list[str] = []
    if stale_after_body:
        warnings.append("stale_sentiment_after_body_fetch")
    if scored and label_pct["neutral"] < min_neutral_pct:
        warnings.append("low_neutral_share")
    if scored and label_pct["positive"] > max_positive_pct:
        warnings.append("high_positive_share")
    if suspicious_counts:
        warnings.append("high_confidence_label_cue_mismatch")
    if scored and suspicious_pct > max_mismatch_pct:
        warnings.append("high_mismatch_rate")

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
            "avg_raw_scores": _avg_scores(scored_rows, field="sentiment_score"),
            "avg_calibrated_scores": _avg_scores(scored_rows, field="calibrated_sentiment_score"),
            **confidence,
        },
        "warnings": warnings,
        "sources": sources,
        "suspicious": {
            "counts": dict(suspicious_counts),
            "total": suspicious_total,
            "pct_scored": suspicious_pct,
            "examples": suspicious_examples,
        },
        "low_margin_examples": low_margin_examples,
        "thresholds": {
            "min_neutral_pct": min_neutral_pct,
            "max_positive_pct": max_positive_pct,
            "max_mismatch_pct": max_mismatch_pct,
            "high_confidence": high_confidence,
        },
    }


def format_summary(summary: dict[str, Any], date_str: str, country_code: str | None = None) -> str:
    totals = summary["totals"]
    heading = f"Sentiment quality audit: {date_str}"
    if country_code:
        heading = f"{heading} [{country_code}]"
    lines = [
        heading,
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
        f"  avg raw scores: {totals['avg_raw_scores']}",
        f"  avg calibrated scores: {totals['avg_calibrated_scores']}",
        f"  raw avg confidence: {totals['raw_avg_confidence']}",
        f"  calibrated avg confidence: {totals['calibrated_avg_confidence']}",
        f"  raw >=0.90 count: {totals['raw_high_confidence_count']}",
        f"  calibrated >=0.90 count: {totals['calibrated_high_confidence_count']}",
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
    lines.extend([
        "",
        "Suspicious high-confidence cue mismatches: "
        f"{suspicious['counts']} ({suspicious['pct_scored']}% of scored)",
    ])
    for row in suspicious["examples"]:
        lines.append(
            "  "
            f"{row['id']} | {row['source']} | {row['label']} {row['score']} | "
            f"{row['reason']} | {row['title']}"
        )

    if summary["low_margin_examples"]:
        lines.extend(["", "Low-margin scored examples:"])
        for row in summary["low_margin_examples"]:
            lines.append(
                "  "
                f"{row['id']} | {row['source']} | {row['label']} "
                f"raw={row['raw_score']} calibrated={row['calibrated_score']} "
                f"margin={row['margin']} | {row['title']}"
            )

    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit sentiment freshness and label-quality risks.")
    parser.add_argument("--date", default=date.today().isoformat(), metavar="YYYY-MM-DD")
    parser.add_argument("--country", default="turkey", help="Country slug or ISO code (default: turkey)")
    parser.add_argument("--min-neutral-pct", type=float, default=5.0)
    parser.add_argument("--max-positive-pct", type=float, default=45.0)
    parser.add_argument("--max-mismatch-pct", type=float, default=2.0)
    parser.add_argument("--high-confidence", type=float, default=0.85)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--json", action="store_true", help="Print machine-readable summary JSON.")
    args = parser.parse_args()
    if not 0 <= args.min_neutral_pct <= 100:
        parser.error("--min-neutral-pct must be between 0 and 100")
    if not 0 <= args.max_positive_pct <= 100:
        parser.error("--max-positive-pct must be between 0 and 100")
    if not 0 <= args.max_mismatch_pct <= 100:
        parser.error("--max-mismatch-pct must be between 0 and 100")
    if not 0 <= args.high_confidence <= 1:
        parser.error("--high-confidence must be between 0 and 1")
    if args.max_examples < 1:
        parser.error("--max-examples must be positive")
    return args


def main() -> int:
    args = _parse_args()
    country_config = load_country_config(args.country)
    country_code = country_config["country_code"]
    rows = fetch_sentiment_quality_rows(args.date, country_code=country_code)
    summary = summarize_rows(
        rows,
        min_neutral_pct=args.min_neutral_pct,
        max_positive_pct=args.max_positive_pct,
        max_mismatch_pct=args.max_mismatch_pct,
        high_confidence=args.high_confidence,
        max_examples=args.max_examples,
    )
    if args.json:
        print(json.dumps(
            {"date": args.date, "country_code": country_code, **summary},
            ensure_ascii=False,
            indent=2,
            default=str,
        ))
    else:
        print(format_summary(summary, args.date, country_code=country_code))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
