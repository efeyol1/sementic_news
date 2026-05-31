"""RAG explainability layer (Sprint 11 of the entity-narrative track).

Explains an entity's pre-computed framing signals (top collocates with
PMI/LLR + frame_intensities from ``entity_country_profile``) in grounded
natural language. Retriever (pgvector/structured) → context assembly →
generator (Claude API, or a deterministic extractive fallback) → DB cache.

Every claim is grounded in either a computed metric or a retrieved real
article snippet; the layer never makes country-level claims and refuses
to speculate when evidence is thin.
"""

from src.rag.explain import explain_entity

__all__ = ["explain_entity"]
