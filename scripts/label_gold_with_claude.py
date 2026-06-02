"""Label Turkish sentiment gold seed rows with Claude as an annotator.

Reads the input review CSV without modifying it, labels rows from title +
summary only, and writes a resumable labeled copy.

Usage:
    python scripts/label_gold_with_claude.py
    python scripts/label_gold_with_claude.py --model claude-opus-4-8
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv

INPUT_PATH = Path("data/qa/sentiment_review_TR_goldseed_2026-06-01.csv")
OUTPUT_PATH = Path("data/qa/sentiment_review_TR_goldseed_2026-06-01.labeled.csv")
VALID_LABELS = {"negative", "neutral", "positive"}
DEFAULT_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """Sen Türkçe haber sentiment'i için bağımsız gold-label annotator'sın.

Görev: Her haberin sentiment etiketini SADECE verilen title + summary metnine göre seç.
Geçerli etiketler tam olarak: negative, neutral, positive.

Rubric:
- negative (olumsuz): ölüm, kaza, şiddet, savaş, terör, suç, afet, kriz, ekonomik kötüleşme (zam/işsizlik/iflas), çatışma, kınama, başarısızlık, yolsuzluk, mağduriyet; bir aktöre/kuruma zarar veya suçlama.
- positive (olumlu): başarı, zafer, kurtarma, iyileşme, anlaşma/barış, yatırım, büyüme, ödül, kutlama, rekor, yardım, çözüm, terfi, kazanç.
- neutral (nötr): net olumlu/olumsuz valans taşımayan olgusal/dengeli haber; rutin açıklama/atama, takvim-program duyurusu, istatistik aktarımı, hem olumlu hem olumsuz dengeli, yorumsuz bilgi aktarımı.

Kurallar:
* Sadece verilen metne dayan; güncel olaylara dair kendi bilgini KATMA.
* Olayın valansını değerlendir (haberin öznesi açısından), gazetecinin üslubunu değil.
* Spor/magazin: bildirilen sonucun valansı (galibiyet=positive, sakatlık/mağlubiyet=negative, transfer dedikodusu/açıklama=neutral).
* Başlık ile özet çelişirse başlığa ağırlık ver.
* Net olumlu/olumsuz sinyal yoksa neutral.
* Mevcut model tahmini verilmez; bağımsız etiketle.

Few-shot:
Input: "Depremde can kaybı arttı" + "Yetkililer arama kurtarma çalışmalarının sürdüğünü açıkladı."
Output: {"label":"negative","rationale":"can kaybı ve afet"}

Input: "Milli takım finalde kazandı" + "Takım tarihi zaferle kupaya ulaştı."
Output: {"label":"positive","rationale":"zafer ve başarı"}

Input: "Bakan bugün yeni takvimi açıklayacak" + "Toplantı saat 14.00'te yapılacak."
Output: {"label":"neutral","rationale":"rutin program duyurusu"}

Yanıt formatı kesin JSON array olmalı:
[{"id": 123, "label": "negative", "rationale": "<=12 kelime"}]
JSON dışında hiçbir metin yazma."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate Turkish sentiment review CSV with Claude."
    )
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional delay between successful batches.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for smoke tests/resume chunks.")
    return parser.parse_args()


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    required = {"id", "title", "summary", "reviewed_label", "review_notes", "label_source", "review_status"}
    missing = sorted(required - set(fieldnames))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    return rows, fieldnames


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _merge_existing_labels(rows: list[dict[str, str]], output: Path) -> int:
    if not output.exists():
        return 0
    with output.open(encoding="utf-8", newline="") as fh:
        existing = {
            str(row.get("id", "")): row
            for row in csv.DictReader(fh)
            if row.get("reviewed_label") in VALID_LABELS
        }
    merged = 0
    for row in rows:
        prior = existing.get(str(row.get("id", "")))
        if not prior:
            continue
        row["reviewed_label"] = prior.get("reviewed_label", "")
        row["label_source"] = prior.get("label_source", "")
        row["review_status"] = prior.get("review_status", "")
        row["review_notes"] = prior.get("review_notes", "")
        merged += 1
    return merged


def _chunks(rows: list[dict[str, str]], size: int) -> list[list[dict[str, str]]]:
    return [rows[i : i + size] for i in range(0, len(rows), size)]


def _clean_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _user_payload(batch: list[dict[str, str]]) -> str:
    items = [
        {
            "id": int(row["id"]),
            "title": _clean_text(row.get("title")),
            "summary": _clean_text(row.get("summary")),
        }
        for row in batch
    ]
    return json.dumps({"items": items}, ensure_ascii=False)


def _extract_json_array(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def _shorten_rationale(value: Any) -> str:
    words = str(value or "").strip().split()
    return " ".join(words[:12])


def _validate_annotations(payload: Any, expected_ids: set[int]) -> dict[int, dict[str, str]]:
    if not isinstance(payload, list):
        raise ValueError("response is not a JSON array")
    annotations: dict[int, dict[str, str]] = {}
    bad: list[str] = []
    for item in payload:
        if not isinstance(item, dict):
            bad.append("non-object item")
            continue
        try:
            row_id = int(item.get("id"))
        except (TypeError, ValueError):
            bad.append(f"bad id: {item.get('id')!r}")
            continue
        label = str(item.get("label", "")).strip().lower()
        if row_id not in expected_ids:
            bad.append(f"unexpected id: {row_id}")
            continue
        if label not in VALID_LABELS:
            bad.append(f"bad label for {row_id}: {label!r}")
            continue
        annotations[row_id] = {
            "label": label,
            "rationale": _shorten_rationale(item.get("rationale")),
        }
    missing = expected_ids - set(annotations)
    if bad or missing:
        raise ValueError(f"invalid annotations; bad={bad[:5]} missing={sorted(missing)[:10]}")
    return annotations


def _temperature_kwarg(model: str) -> dict[str, float]:
    """Opus 4.x deprecates the ``temperature`` param (400 invalid_request);
    Sonnet still honours ``temperature=0`` for deterministic labels."""
    if "opus-4" in model:
        return {}
    return {"temperature": 0}


def _call_claude(
    client: Anthropic,
    *,
    model: str,
    batch: list[dict[str, str]],
    max_retries: int,
) -> dict[int, dict[str, str]]:
    expected_ids = {int(row["id"]) for row in batch}
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=1200,
                **_temperature_kwarg(model),
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": _user_payload(batch),
                            }
                        ],
                    }
                ],
            )
            text_parts = [
                block.text
                for block in response.content
                if getattr(block, "type", None) == "text"
            ]
            annotations = _validate_annotations(
                _extract_json_array("\n".join(text_parts)),
                expected_ids,
            )
            return annotations
        except Exception as exc:  # noqa: BLE001 - external API/parse retries are deliberate.
            last_error = exc
            if attempt == max_retries:
                break
            delay = (2 ** (attempt - 1)) + random.uniform(0.0, 0.5)
            print(
                f"Batch ids {min(expected_ids)}-{max(expected_ids)} failed "
                f"(attempt {attempt}/{max_retries}): {exc}; retrying in {delay:.1f}s",
                file=sys.stderr,
            )
            time.sleep(delay)
    raise RuntimeError(f"Claude batch failed after {max_retries} attempts: {last_error}")


def main() -> int:
    args = _parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.max_retries < 1:
        raise ValueError("--max-retries must be positive")

    load_dotenv()
    rows, fieldnames = _read_csv(args.input)
    merged = _merge_existing_labels(rows, args.output)
    pending = [row for row in rows if row.get("reviewed_label") not in VALID_LABELS]
    if args.limit is not None:
        pending = pending[: args.limit]

    print(
        f"Loaded {len(rows)} rows from {args.input}; "
        f"resumed {merged}; pending this run {len(pending)}"
    )
    if not pending:
        _write_csv(args.output, rows, fieldnames)
        print(f"No pending rows. Wrote {args.output}")
        return 0

    client = Anthropic()
    label_source = args.model
    processed = 0
    failed_rows = 0

    for batch in _chunks(pending, args.batch_size):
        ids = [int(row["id"]) for row in batch]
        try:
            annotations = _call_claude(
                client,
                model=args.model,
                batch=batch,
                max_retries=args.max_retries,
            )
        except Exception as exc:  # noqa: BLE001 - leave labels blank after retries.
            failed_rows += len(batch)
            print(
                f"FAILED batch {ids[0]}-{ids[-1]} after retries; leaving blank: {exc}",
                file=sys.stderr,
            )
            _write_csv(args.output, rows, fieldnames)
            continue

        for row in batch:
            ann = annotations[int(row["id"])]
            row["reviewed_label"] = ann["label"]
            row["label_source"] = label_source
            row["review_status"] = "reviewed"
            row["review_notes"] = ann["rationale"]
        processed += len(batch)
        _write_csv(args.output, rows, fieldnames)

        done = sum(1 for row in rows if row.get("reviewed_label") in VALID_LABELS)
        counts = Counter(row.get("reviewed_label") for row in rows if row.get("reviewed_label") in VALID_LABELS)
        print(
            f"Labeled batch {ids[0]}-{ids[-1]} ({len(batch)} rows); "
            f"done={done}/{len(rows)} counts={dict(counts)}"
        )
        if args.sleep:
            time.sleep(args.sleep)

    valid = sum(1 for row in rows if row.get("reviewed_label") in VALID_LABELS)
    blank = len(rows) - valid
    counts = Counter(row.get("reviewed_label") for row in rows if row.get("reviewed_label") in VALID_LABELS)
    print(f"Final: valid={valid}/{len(rows)} blank={blank} failed_this_run={failed_rows}")
    print(f"Label distribution: {dict(counts)}")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
