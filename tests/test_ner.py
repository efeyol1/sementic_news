from __future__ import annotations


class _FakePipe:
    """Mimics a HuggingFace ``aggregation_strategy='simple'`` NER pipeline."""

    def __init__(self, ents: list[dict]):
        self._ents = ents

    def __call__(self, text: str) -> list[dict]:
        return list(self._ents)


def test_is_subword_artifact():
    from src.analysis.ner import _is_subword_artifact

    assert _is_subword_artifact("##mgrup") is True
    assert _is_subword_artifact("##i jinping") is True  # leak after a space
    assert _is_subword_artifact("") is True
    assert _is_subword_artifact("Rusya") is False
    assert _is_subword_artifact("Vladimir Putin") is False


def test_extract_entities_drops_wordpiece_fragments():
    """Regression: ``aggregation_strategy='simple'`` leaks ``##USYA``/``##mgrup``
    fragments alongside the full word; they must not reach the entities dict."""
    from src.analysis.ner import _extract_entities

    pipe = _FakePipe([
        {"entity_group": "LOC", "word": "Rusya"},
        {"entity_group": "LOC", "word": "##USYA"},
        {"entity_group": "ORG", "word": "##mgrup"},
        {"entity_group": "PER", "word": "Vladimir Putin"},
    ])

    entities = _extract_entities("x", pipe)

    assert entities["LOC"] == ["Rusya"]
    assert entities["ORG"] == []
    assert entities["PER"] == ["Vladimir Putin"]
