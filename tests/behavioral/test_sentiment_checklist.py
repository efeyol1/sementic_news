"""Behavioral / CheckList tests for the production Turkish sentiment model.

Three categories:

1. **must-pass** — currently green; regression-blocking. A failure here means
   the model got *worse* than the version when these tests were written.

2. **regression watchlist** — `@pytest.mark.xfail(strict=True)`. These cases
   currently *fail* on the production model, exposing a known neutral-bias
   on news-domain positive sentences (validation F1 ≈ 0.95 on `winvoker`
   does NOT generalize: 0/8 explicit-positive cases predict `positive`).
   Strict xfail means the day a retrain fixes them, pytest alerts via XPASS
   and forces a manual review — they should be promoted into must-pass.

3. **aspirational (should-pass)** — `@pytest.mark.xfail(strict=False)`.
   Sarcasm, implicit polarity. Reported but never blocking.

Override the model under test via ``BEHAVIORAL_MODEL_ID`` env var (defaults
to the production HF Hub model). For a weekly-retrain CI that wants to gate
on a candidate, point this at the freshly-trained checkpoint *before* push.
"""

from __future__ import annotations

import os

import pytest
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL_ID = "efeyol11/bert-turkish-sentiment"
MODEL_ID = os.environ.get("BEHAVIORAL_MODEL_ID", DEFAULT_MODEL_ID)


@pytest.fixture(scope="session")
def predictor():
    """Load model once for the entire session — model load dominates wall time."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()

    @torch.no_grad()
    def predict(text: str) -> str:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        logits = model(**inputs).logits
        return model.config.id2label[int(logits.argmax(dim=-1).item())]

    return predict


def _ids(prefix, items):
    return [f"{prefix}-{i:02d}" for i in range(len(items))]


# ===========================================================================
# MUST-PASS  (regression-blocking)
# ===========================================================================

NEUTRAL_CASES: list[str] = [
    "Bakanlık yarın bir basın toplantısı düzenleyecek.",
    "Toplantı saat 14.00'te başlayacak.",
    "Cumhurbaşkanı bugün Ankara'da bir programa katılacak.",
    "Yeni yönetmelik resmi gazetede yayımlandı.",
    "Hava sıcaklığı mevsim normalleri civarında seyredecek.",
    "Borsa kapanışta yatay seyretti.",
]


@pytest.mark.behavioral
@pytest.mark.parametrize("text", NEUTRAL_CASES, ids=_ids("neu", NEUTRAL_CASES))
def test_must_pass_neutral(predictor, text):
    pred = predictor(text)
    assert pred == "neutral", f"{text!r} → expected 'neutral', got {pred!r}"


# Invariance — fire / disaster template across cities. Model correctly
# detects strong negative cues; this validates the negative class behaves
# the same regardless of which city is named.
INVARIANCE_NEGATIVE_CITIES = ["Ankara", "İstanbul", "İzmir", "Bursa"]
INVARIANCE_NEGATIVE_TEMPLATE = "{city}'da büyük bir yangın çıktı, hasar büyük."


@pytest.mark.behavioral
@pytest.mark.parametrize("city", INVARIANCE_NEGATIVE_CITIES, ids=_ids("fire", INVARIANCE_NEGATIVE_CITIES))
def test_must_pass_invariance_negative(predictor, city):
    text = INVARIANCE_NEGATIVE_TEMPLATE.format(city=city)
    pred = predictor(text)
    assert pred == "negative", f"{text!r} → expected 'negative', got {pred!r}"


# Invariance — neutral announcement template across actor names.
INVARIANCE_NEUTRAL_ACTORS = ["Bakan", "Genel müdür", "Sözcü"]
INVARIANCE_NEUTRAL_TEMPLATE = "{person} bugün açıklama yaptı."


@pytest.mark.behavioral
@pytest.mark.parametrize("person", INVARIANCE_NEUTRAL_ACTORS, ids=_ids("actor", INVARIANCE_NEUTRAL_ACTORS))
def test_must_pass_invariance_neutral(predictor, person):
    text = INVARIANCE_NEUTRAL_TEMPLATE.format(person=person)
    pred = predictor(text)
    assert pred == "neutral", f"{text!r} → expected 'neutral', got {pred!r}"


# ===========================================================================
# REGRESSION WATCHLIST  (xfail strict=True)
#
# These currently fail. When the next retrain raises positive-class recall
# they will start passing → pytest XPASS → forces a manual promotion review.
# ===========================================================================

WATCHLIST_REASON = (
    "Known fail: production model has neutral bias on news-domain positive cues "
    "(winvoker→news domain shift). Promote to must-pass once retrained."
)


POSITIVE_CASES_WATCHLIST: list[str] = [
    "Ekonomi büyüdü ve enflasyon geriledi.",
    "Milli takım maçı 3-0 kazandı.",
    "İhracat geçen yılın aynı dönemine göre rekor kırdı.",
    "Anlaşma başarıyla imzalandı, taraflar memnun ayrıldı.",
    "Yeni hastane hizmete açıldı, halk teşekkür etti.",
    "Şirket çeyrek karını ikiye katladı.",
    "Bilim insanları kanser tedavisinde önemli bir ilerleme kaydetti.",
    "Öğrenci uluslararası yarışmada altın madalya kazandı.",
]


@pytest.mark.behavioral
@pytest.mark.xfail(strict=True, reason=WATCHLIST_REASON)
@pytest.mark.parametrize("text", POSITIVE_CASES_WATCHLIST, ids=_ids("pos", POSITIVE_CASES_WATCHLIST))
def test_watchlist_positive(predictor, text):
    pred = predictor(text)
    assert pred == "positive", f"{text!r} → expected 'positive', got {pred!r}"


# Negative cases beyond strong disaster cues. Most still get smoothed to
# neutral — same domain shift, different polarity.
NEGATIVE_CASES_WATCHLIST: list[str] = [
    "Ekonomi küçüldü ve işsizlik arttı.",
    "Milli takım maçı 0-3 kaybetti.",
    "Şirket büyük zarar açıkladı, hisseler düştü.",
    "Deprem nedeniyle çok sayıda bina yıkıldı.",
    "Hastane kapatıldı, hastalar mağdur oldu.",
    "Saldırıda çok sayıda kişi hayatını kaybetti.",
    "Anlaşma çöktü, görüşmeler askıya alındı.",
    "Kriz büyüdü, fiyatlar kontrolden çıktı.",
]


@pytest.mark.behavioral
@pytest.mark.xfail(strict=True, reason=WATCHLIST_REASON)
@pytest.mark.parametrize("text", NEGATIVE_CASES_WATCHLIST, ids=_ids("neg", NEGATIVE_CASES_WATCHLIST))
def test_watchlist_negative(predictor, text):
    pred = predictor(text)
    assert pred == "negative", f"{text!r} → expected 'negative', got {pred!r}"


# Negation flips. Both halves must hold; failing the positive half currently
# trips every pair (model returns neutral for the un-negated form).
NEGATION_PAIRS: list[tuple[str, str, str, str]] = [
    (
        "Şirket bu çeyrekte kar açıkladı.",
        "positive",
        "Şirket bu çeyrekte kar açıklamadı.",
        "negative",
    ),
    (
        "Anlaşma sağlandı, görüşmeler olumlu sonuçlandı.",
        "positive",
        "Anlaşma sağlanamadı, görüşmeler olumsuz sonuçlandı.",
        "negative",
    ),
    (
        "Ekonomi büyüdü.",
        "positive",
        "Ekonomi büyümedi, küçüldü.",
        "negative",
    ),
    (
        "Hastalar tedaviye yanıt verdi.",
        "positive",
        "Hastalar tedaviye yanıt vermedi.",
        "negative",
    ),
]


@pytest.mark.behavioral
@pytest.mark.xfail(strict=True, reason=WATCHLIST_REASON)
@pytest.mark.parametrize(
    "pos_text,pos_label,neg_text,neg_label",
    NEGATION_PAIRS,
    ids=_ids("neg-pair", NEGATION_PAIRS),
)
def test_watchlist_negation(predictor, pos_text, pos_label, neg_text, neg_label):
    pred_pos = predictor(pos_text)
    pred_neg = predictor(neg_text)
    assert pred_pos == pos_label, f"{pos_text!r} → expected {pos_label!r}, got {pred_pos!r}"
    assert pred_neg == neg_label, f"{neg_text!r} → expected {neg_label!r}, got {pred_neg!r}"


# Positive invariance — hospital opening across cities. Same shape as the
# negative-fire template but model under-predicts positive class.
INVARIANCE_POSITIVE_CITIES = ["Ankara", "İstanbul", "İzmir", "Bursa"]
INVARIANCE_POSITIVE_TEMPLATE = "{city}'da yeni hastane hizmete açıldı."


@pytest.mark.behavioral
@pytest.mark.xfail(strict=True, reason=WATCHLIST_REASON)
@pytest.mark.parametrize(
    "city", INVARIANCE_POSITIVE_CITIES, ids=_ids("hosp", INVARIANCE_POSITIVE_CITIES)
)
def test_watchlist_invariance_positive(predictor, city):
    text = INVARIANCE_POSITIVE_TEMPLATE.format(city=city)
    pred = predictor(text)
    assert pred == "positive", f"{text!r} → expected 'positive', got {pred!r}"


# ===========================================================================
# ASPIRATIONAL  (xfail strict=False — non-blocking, no XPASS surprise)
# ===========================================================================

SHOULD_PASS: list[tuple[str, str]] = [
    ("Tabii ki harika bir karar, herkes çok memnun!", "negative"),
    ("Vay be, ne büyük başarı...", "negative"),
    ("Ekonomi büyüdü ama halkın alım gücü dibe vurdu.", "negative"),
    ("Bakan istifa etti.", "negative"),
    ("Tutuklu gazeteciler tahliye edildi.", "positive"),
]


@pytest.mark.behavioral
@pytest.mark.xfail(
    strict=False,
    reason="Aspirational — sarcasm / implicit polarity. Reported, non-blocking.",
)
@pytest.mark.parametrize("text,expected", SHOULD_PASS, ids=_ids("aspire", SHOULD_PASS))
def test_should_pass(predictor, text, expected):
    pred = predictor(text)
    assert pred == expected, f"{text!r} → expected {expected!r}, got {pred!r}"
