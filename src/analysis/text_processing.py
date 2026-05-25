"""Per-language tokenization + lemmatization for the entity-narrative track.

Sprint 5: produce a stream of ``Token`` objects from a string so the
collocation extractor (``src.analysis.collocations``) can take ±N-token
windows around an entity mention. The position of each token is a
character offset into the *same* string the lemmatizer received — when
the collocation step rebuilds the input via
:func:`src.analysis.text_inputs.build_ner_text`, the offsets line up
with ``entity_mentions.position_in_article`` (which is the HuggingFace
NER ``start`` offset into that same concatenation).

Two backends:

  * ``SpacyLemmatizer`` — used by EN/DE/FR/IT/ES (loaded lazily via
    ``spacy.load(model_name)``). Sentence boundaries come for free from
    the model's parser/sentencizer.
  * ``TurkishSurfaceLemmatizer`` — TR's spaCy support is weak (no
    ``tr_core_news_*`` model), so we ship a deterministic suffix-strip
    + lowercase fallback. Sentence boundaries are detected by a regex
    on ``[.!?]+\\s+``. Sprint 5 keeps TR collocations off
    (``turkey.yaml::entity_narrative.collocations.enabled: false``);
    this implementation exists so Sprint 6 can flip the flag.

Factory caching: ``get_lemmatizer(language)`` returns a process-wide
cached instance. spaCy model load is ~0.5–1 s, so re-using the same
lemmatizer across articles is mandatory for pipeline runtime budget.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

import yaml
from loguru import logger

# ---------------------------------------------------------------------------
# Token model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Token:
    """One token's surface form + lemma + POS + char span + sentence id.

    ``start`` / ``end`` are character offsets *into the string fed to
    the lemmatizer* — caller is responsible for using the exact same
    string when interpreting them (see ``build_ner_text``).
    """

    text: str
    lemma: str
    pos: str
    start: int
    end: int
    sent_id: int


# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------


def load_stopwords(path: str | Path) -> set[str]:
    """Read a YAML stopword list. Accepts either a bare list or a
    ``{stopwords: [...]}`` document.
    """
    p = Path(path)
    if not p.exists():
        logger.warning(f"Stopwords file not found at {p} — returning empty set")
        return set()
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("stopwords") or []
    if not isinstance(data, list):
        logger.warning(
            f"Stopwords file {p} is not a list/{{stopwords: list}} — "
            f"returning empty set"
        )
        return set()
    return {str(w).lower() for w in data if w}


# ---------------------------------------------------------------------------
# Lemmatizer protocol
# ---------------------------------------------------------------------------


class Lemmatizer(Protocol):
    """Implementations turn a string into ``Token`` objects."""

    language: str

    def __call__(self, text: str) -> list[Token]:  # pragma: no cover - protocol
        ...


# ---------------------------------------------------------------------------
# spaCy backend
# ---------------------------------------------------------------------------


# Default per-language spaCy model. ``country_config`` may override via
# ``entity_narrative.collocations.spacy_model``.
SPACY_MODEL_DEFAULTS: dict[str, str] = {
    "en": "en_core_web_sm",
    "de": "de_core_news_sm",
    "fr": "fr_core_news_sm",
    "it": "it_core_news_sm",
    "es": "es_core_news_sm",
}


class SpacyLemmatizer:
    """spaCy-backed lemmatizer for EN/DE/FR/IT/ES."""

    def __init__(self, language: str, model_name: str | None = None):
        self.language = language
        self.model_name = model_name or SPACY_MODEL_DEFAULTS.get(language)
        if not self.model_name:
            raise ValueError(
                f"No default spaCy model for language {language!r}; "
                f"pass model_name explicitly"
            )
        self._nlp = _load_spacy_model(self.model_name)

    def __call__(self, text: str) -> list[Token]:
        if not text:
            return []
        doc = self._nlp(text)
        tokens: list[Token] = []
        # spaCy assigns each Token a ``.sent`` reference; we number
        # sentences sequentially so the collocation extractor can avoid
        # crossing sentence boundaries with simple ``sent_id`` equality.
        sent_index: dict[int, int] = {}
        next_sent_id = 0
        for tok in doc:
            if tok.is_space:
                continue
            sent_start = tok.sent.start_char
            sent_id = sent_index.get(sent_start)
            if sent_id is None:
                sent_id = next_sent_id
                sent_index[sent_start] = sent_id
                next_sent_id += 1
            lemma = (tok.lemma_ or tok.text).lower().strip()
            if not lemma:
                continue
            tokens.append(
                Token(
                    text=tok.text,
                    lemma=lemma,
                    pos=tok.pos_ or "X",
                    start=tok.idx,
                    end=tok.idx + len(tok.text),
                    sent_id=sent_id,
                )
            )
        return tokens


_SPACY_CACHE: dict[str, Any] = {}


def _load_spacy_model(model_name: str) -> Any:
    """Load a spaCy model once per process and cache it."""
    cached = _SPACY_CACHE.get(model_name)
    if cached is not None:
        return cached
    try:
        import spacy
    except ImportError as e:  # pragma: no cover - dep gate
        raise RuntimeError(
            "spaCy is required for collocation extraction in non-TR "
            "languages. Install with: pip install 'spacy>=3.7' and "
            f"python -m spacy download {model_name}"
        ) from e
    logger.info(f"Loading spaCy model {model_name!r}")
    nlp = spacy.load(model_name, disable=["ner"])
    _SPACY_CACHE[model_name] = nlp
    logger.success(f"spaCy model loaded: {model_name}")
    return nlp


# ---------------------------------------------------------------------------
# Turkish surface-form backend (Sprint 5: off in turkey.yaml; Sprint 6
# will flip the flag — until then this is exercised only by tests.)
# ---------------------------------------------------------------------------


# Strip-from-longest-first. Strictly nominal/case suffixes; no verb
# morphology. Designed to collapse the common variants seen in TR news
# text ("vergiyi", "vergileri", "vergilerin") to a shared stem without
# over-stripping short stems. Sprint 6 may replace this with zeyrek.
_TR_SUFFIXES = (
    "lerimizden", "larımızdan", "lerinizden", "larınızdan",
    "lerinden", "larından", "leriyle", "larıyla",
    "lerine", "larına", "lerini", "larını",
    "leriniz", "larınız", "lerimiz", "larımız",
    "lerin", "ların", "lerde", "larda",
    "leri", "ları", "lere", "lara",
    "ler", "lar",
    "den", "dan", "ten", "tan",
    "nin", "nın", "nun", "nün",
    "yle", "yla",
    "de", "da", "te", "ta",
    "ye", "ya", "ne", "na",
)
_TR_MIN_STEM = 3
_TR_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_TR_SENT_RE = re.compile(r"[.!?]+\s+")


def _tr_strip(word: str) -> str:
    w = word.lower()
    for suf in _TR_SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= _TR_MIN_STEM:
            return w[: -len(suf)]
    return w


class TurkishSurfaceLemmatizer:
    """Lowercase + suffix-strip TR lemmatizer.

    POS is always ``"X"`` (we don't tag); the collocation step's POS
    filter is therefore effectively a no-op for TR. That's intentional
    until Sprint 6 swaps in a real morphology library.
    """

    language = "tr"

    def __call__(self, text: str) -> list[Token]:
        if not text:
            return []
        sent_id = 0
        cursor = 0
        sentence_breaks = _split_sentence_offsets(text, _TR_SENT_RE)
        tokens: list[Token] = []
        for match in _TR_TOKEN_RE.finditer(text):
            start, end = match.start(), match.end()
            while cursor < len(sentence_breaks) and sentence_breaks[cursor] <= start:
                sent_id += 1
                cursor += 1
            surface = match.group(0)
            lemma = _tr_strip(surface)
            if not lemma:
                continue
            tokens.append(
                Token(
                    text=surface,
                    lemma=lemma,
                    pos="X",
                    start=start,
                    end=end,
                    sent_id=sent_id,
                )
            )
        return tokens


def _split_sentence_offsets(text: str, sentence_re: re.Pattern[str]) -> list[int]:
    """Char offsets at which a new sentence *begins* (i.e. after a
    sentence-terminating run of punctuation + whitespace)."""
    return [m.end() for m in sentence_re.finditer(text)]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


_LEMMATIZER_CACHE: dict[tuple[str, str], Lemmatizer] = {}


def get_lemmatizer(language: str, model_name: str | None = None) -> Lemmatizer:
    """Return a process-cached lemmatizer for the language."""
    lang = (language or "").lower()
    cache_key = (lang, model_name or "")
    cached = _LEMMATIZER_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if lang == "tr":
        impl: Lemmatizer = TurkishSurfaceLemmatizer()
    elif lang in SPACY_MODEL_DEFAULTS or model_name:
        impl = SpacyLemmatizer(lang, model_name=model_name)
    else:
        raise ValueError(
            f"Unsupported language for lemmatization: {language!r}. "
            f"Supported: {sorted(SPACY_MODEL_DEFAULTS) + ['tr']}"
        )
    _LEMMATIZER_CACHE[cache_key] = impl
    return impl


def reset_lemmatizer_cache() -> None:
    """Test hook: drop cached lemmatizers between scenarios."""
    _LEMMATIZER_CACHE.clear()
    _SPACY_CACHE.clear()


# ---------------------------------------------------------------------------
# Convenience: filter a token stream by stopwords + POS + length
# ---------------------------------------------------------------------------


def filter_tokens(
    tokens: Iterable[Token],
    *,
    stopwords: set[str],
    keep_pos: set[str] | None,
    min_lemma_length: int,
) -> list[Token]:
    """Apply the standard collocation filters in one pass.

    ``keep_pos=None`` disables the POS filter (TR surface lemmatizer
    produces only ``"X"``, so the filter is bypassed there).
    """
    out: list[Token] = []
    for tok in tokens:
        if keep_pos is not None and tok.pos not in keep_pos:
            continue
        if len(tok.lemma) < min_lemma_length:
            continue
        if tok.lemma in stopwords:
            continue
        out.append(tok)
    return out
