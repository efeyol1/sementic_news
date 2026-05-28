# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to a sprint-based versioning scheme (`v2-phase-N`,
`v2-entity-sprint-N`, `v2-sprint-N`).

## [Unreleased]

### Added
- Sprint 5 (in PR review): entity collocations module (`src/analysis/collocations.py`),
  language-keyed stopwords config (`configs/stopwords/`), text-processing helper
  (`src/analysis/text_processing.py`), Alembic migration `0007_entity_collocations`.
- Sprint 5.5: public release hardening — `LICENSE` (MIT), `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CHANGELOG.md`, `MODEL_CARD.md`,
  `DATA_PROVENANCE.md`, `DATA_LICENSE.md`, `PRIVACY.md`. Country YAML schema
  gains `license` field per source. New `/about` API endpoint with ethics
  disclaimer.
- Sprint 6: PMI + log-likelihood (Dunning 1993) backfill module
  (`src/analysis/collocation_stats.py`) — fills the six NULL columns Sprint 5
  reserved on `entity_collocations`. New DB helpers in `src/db/queries.py`
  (`fetch_collocations_for_stats`, `fetch_entity_mention_counts`,
  `bulk_update_collocation_stats`). New pipeline soft step `entity_pmi` runs
  immediately after `entity_collocations`. `entity_total` column carries the
  distinct mention count from `entity_mentions` (separate from the
  lemma-weighted `cooccurrence_with_entity_total` marginal), giving Sprint 7/8
  an "entity prominence" signal alongside PMI.
- Sprint 6: TR collocations activated (`turkey.yaml::entity_narrative.collocations.enabled`
  flipped to `true`) using the surface-form lemmatizer as the agreed baseline.

### Changed
- `README.md` rewritten to reflect V2 multi-country reality (6 countries: TR,
  DE, FR, IT, ES, UK), fine-tuned sentiment as production default with
  zero-shot fallback for non-TR languages.
- `SERVICES.md` updated to V2 service inventory (was V1 Streamlit/JSON
  description, now reflects Next.js + Postgres + entity-narrative).
- `ARCHITECTURE.md`: daily corpus updated `~400-500 articles` → `~3,853 items`
  after sitemap handlers; entity-narrative pipeline and multi-country layer
  added.
- `pyproject.toml`: `license = {text = "MIT"}` declared.

### Removed
- DVC remnants (`dvc.lock`, `.dvcignore`) — DVC was never wired into the
  pipeline. `.gitignore` now prevents reintroduction.
- Build artifact `dashboard/tsconfig.tsbuildinfo` untracked.

## [Sprint 4] — 2026-05-25

- UK onboarding (`configs/countries/uk.yaml`) — 6th country.
- TR entity-narrative activation: `entity_narrative.enabled: true`,
  Wikidata linking ON with `min_confidence: 0.65`. Legacy `ner` path
  remains in parallel for backward compatibility.
- Type-agnostic alias fallback + surname disambiguation reject in
  `entity_resolution.py`.
- Tag: (none yet, post-merge)

## [v2-sprint-3] — 2026-05-XX

- Italy + Spain onboarding (5 countries total).
- Wikidata `min_confidence: 0.65` calibration for IT/ES.

## [v2-entity-sprint-2] — 2026-05-18

- France onboarding + cross-country Wikidata Q-ID linking.
- `src/analysis/entity_resolution.py` (Wikidata API client) + Alembic
  `0006_entity_resolution_cache`.
- `src/analysis/entity_canonicalization.py`.
- Scripts: `build_wikidata_cache.py`, `benchmark_french_ner.py`,
  `audit_entity_resolution.py`.

## [v2-entity-sprint-1] — 2026-05-14

- Multilingual NER (`Davlan/bert-base-multilingual-cased-ner-hrl`).
- New `entity_mentions` table (Alembic `0005`).
- Sentiment QA infrastructure: `audit_sentiment_quality.py`,
  `export_sentiment_review_sample.py`, `evaluate_sentiment_review.py`,
  scoped re-score flags (`--only-missing`, `--only-stale-after-body`,
  `--only-with-body`).
- Sentiment `max_length` 128 → 256 for body coverage.
- Germany pilot activation (`germany.yaml`).

## [v2-phase-4] — earlier

- `country_code` and `language` columns added to `news_items`.
- Country-aware pipeline (`--country` CLI arg).
- Country-aware FastAPI endpoints (`?country=TR`, `/api/countries`).
- Dashboard country selector + `?country` URL propagation.

## [v2-phase-2.1] — earlier

- Sitemap handlers: `googlenews_sitemap`, `html_sitemap` source types
  in `rss_collector.py`. Daily corpus grew 475 → 3,853 items (8.1×).
- Discovery health analyzer (`scripts/analyze_discovery_health.py`).
- Article body ingestion (`src/data/article_fetcher.py`,
  `article_parser.py`), opt-in via `--fetch-articles --article-limit N`.

## [v2-phase-2] — earlier

- Country configuration system (`configs/countries/turkey.yaml`,
  `src/config/country_loader.py`).
- First country: Turkey.

## [v2-phase-1] — earlier

- Turkey V1 baseline stabilization. Pgvector replaces ChromaDB. DVC
  removed (was never used).

## V1 (pre-V2 transition)

- Initial pipeline: RSS → sentiment (zero-shot) → NER → KMeans clustering
  → pgvector → FastAPI + Next.js dashboard.
- MLflow experiment tracking, Prometheus/Grafana monitoring.
- Behavioral CheckList test suite (`tests/behavioral/test_sentiment_checklist.py`).
- Fine-tuned `efeyol11/bert-turkish-sentiment` released on HuggingFace Hub.
