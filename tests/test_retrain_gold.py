from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from src.training.retrain import (
    _balance_gold,
    _load_gold_dataset,
    _load_human_eval,
    _row_text,
)

_GOLD_FIELDS = ("id", "reviewed_label", "title", "summary")


def _write_gold_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_GOLD_FIELDS)
        w.writeheader()
        w.writerows(rows)


def test_row_text_joins_title_and_summary():
    assert _row_text("Başlık", "Özet metni") == "Başlık. Özet metni"
    assert _row_text(" a ", "") == "a."


def test_load_gold_dataset_excludes_ids_and_filters(tmp_path: Path):
    csv_path = tmp_path / "gold.csv"
    _write_gold_csv(csv_path, [
        {"id": "1", "reviewed_label": "negative", "title": "ekonomi küçüldü enflasyon", "summary": "faiz arttı"},
        {"id": "2", "reviewed_label": "positive", "title": "borsa yükseldi rekor kırdı", "summary": "yatırım"},
        {"id": "3", "reviewed_label": "neutral", "title": "bakan atama kararı yayımlandı", "summary": "resmi gazete"},
        {"id": "4", "reviewed_label": "", "title": "etiketsiz uzun bir baslik metni", "summary": "x"},     # no label
        {"id": "5", "reviewed_label": "negative", "title": "kısa", "summary": ""},                          # too short
    ])

    ds = _load_gold_dataset([csv_path], exclude_ids={"2"})

    # id 2 excluded, id 4 unlabeled, id 5 too short → only 1 and 3 survive.
    assert len(ds) == 2
    assert sorted(ds["label"]) == [0, 1]  # negative(0), neutral(1)


def test_balance_gold_equalizes_classes(tmp_path: Path):
    csv_path = tmp_path / "gold.csv"
    _write_gold_csv(csv_path, [
        {"id": "1", "reviewed_label": "neutral", "title": "bakan atama karari yayimlandi bugun", "summary": "resmi gazete"},
        {"id": "2", "reviewed_label": "neutral", "title": "meclis bugun toplandi gundem maddesi", "summary": "oturum"},
        {"id": "3", "reviewed_label": "neutral", "title": "belediye yeni hizmet binasi acti", "summary": "tanitim"},
        {"id": "4", "reviewed_label": "positive", "title": "borsa yukseldi rekor kirdi piyasa", "summary": "yatirim"},
        {"id": "5", "reviewed_label": "positive", "title": "ihracat artti buyume hizlandi haber", "summary": "rekor"},
        {"id": "6", "reviewed_label": "negative", "title": "ekonomi kuculdu enflasyon yukseldi", "summary": "faiz"},
    ])
    gold = _load_gold_dataset([csv_path], exclude_ids=set())
    assert Counter(gold["label"]) == {1: 3, 2: 2, 0: 1}  # pre: imbalanced (neu 3 / pos 2 / neg 1)

    balanced = _balance_gold(gold)

    # every class up-sampled to the largest (neutral=3); minority rows reused.
    assert Counter(balanced["label"]) == {0: 3, 1: 3, 2: 3}
    assert len(balanced) == 9


def test_load_human_eval_baseline_macro_f1(tmp_path: Path):
    seed = tmp_path / "seed.csv"
    with open(seed, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=("id", "current_label", "title", "summary"))
        w.writeheader()
        w.writerows([
            {"id": "1", "current_label": "negative", "title": "ekonomi küçüldü enflasyon", "summary": "faiz"},
            {"id": "2", "current_label": "negative", "title": "borsa yükseldi rekor kırdı", "summary": "yatırım"},
            {"id": "3", "current_label": "neutral", "title": "atlanmali satir cok kisa", "summary": "x"},
        ])
    j = tmp_path / "human.json"
    j.write_text(json.dumps({"items": [
        {"id": "1", "human_label": "negative"},   # model correct
        {"id": "2", "human_label": "positive"},   # model wrong (said negative)
        {"id": "3", "human_label": ""},            # unlabeled → skipped
    ]}, ensure_ascii=False), encoding="utf-8")

    ds, baseline, ids = _load_human_eval(j, seed)

    assert len(ds) == 2
    assert ids == {"1", "2"}
    # gold=[neg,pos], pred=[neg,neg]: neg F1=0.667, neu=0, pos=0 → macro≈0.222
    assert abs(baseline - 0.2222) < 1e-3
