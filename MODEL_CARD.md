# Model Card — Semantic News

This card covers the ML components used by the Semantic News pipeline. The
project is a research/engineering platform; per-component cards live below.
Outputs are descriptive signals from news content, **not** investment advice,
country risk ratings, or factual guarantees.

---

## 1. Turkish Sentiment Classifier (production default)

**Model ID:** [`efeyol11/bert-turkish-sentiment`](https://huggingface.co/efeyol11/bert-turkish-sentiment)
(HuggingFace Hub)

**Architecture:** Fine-tuned BERT-base over a Turkish news corpus.

**Task:** 3-way sentiment classification — `positive`, `neutral`, `negative` —
on Turkish news text (title + summary, optionally + article body).

**Inputs:** Turkish text, `max_length = 256` tokens (up from 128 after body
ingestion landed in 2026-05).

**Outputs:** `sentiment_label` (string), `sentiment_score` (0-1 confidence),
`sentiment_scores` (per-class softmax dict).

**Backends:** PyTorch (default), ONNX fp32, ONNX int8 — see README benchmark
table. CI gate requires ≥99% backend agreement with PyTorch baseline.

**Used by:**
- `src/analysis/sentiment.py` (daily pipeline)
- `src/api/main.py` reads precomputed labels from Postgres; the API does
  not run inference at request time

**Training procedure:** Weekly retrain (`src/training/retrain.py`,
`weekly_retrain.yml` cron Sunday 02:00 UTC) gated by the behavioral
CheckList suite (`tests/behavioral/test_sentiment_checklist.py`). A model
that fails a `must-pass` capability is NOT pushed to the Hub.

**Known limitations:**
- TR-only fine-tuning; do not use on other languages.
- Short headlines lose context; the model is more reliable when summary or
  body is present.
- Sarcasm, implicit polarity, and finance-specific irony are systematically
  harder — flagged in the `watchlist` and `aspirational` behavioral tiers.
- Politically charged headlines can polarize neutral content; neutral recall
  is the main eval focus.

**Appropriate use:**
- News stream monitoring at the publisher / topic / day level.
- Time-series analysis of theme-level sentiment trends.
- Input to clustering, framing analysis, and drift detection.

**NOT appropriate for:**
- Judging individual persons mentioned in articles.
- Producing country-level "risk" or "stability" scores from sentiment alone.
- Standalone decision-making for finance, security, hiring, or any
  high-impact use case.

---

## 2. Multilingual Zero-shot Sentiment (fallback / non-TR)

**Model ID:** `joeddav/xlm-roberta-large-xnli` (HuggingFace Hub)

**Used when:** `SENTIMENT_MODEL_ID` env var is unset, or the country config
has `sentiment.finetuned_model: null` (currently: DE, FR, IT, ES, UK).

**Approach:** NLI-based zero-shot with per-language candidate labels declared
in `configs/countries/<slug>.yaml` (`sentiment.candidate_labels`).

**Known limitations:** Domain mismatch is the dominant error source — XNLI
training data is not news. Zero-shot accuracy is **not** comparable to the
fine-tuned TR model. Use of zero-shot outputs for cross-country comparison
requires an explicit caveat in any presentation.

**Gold-seed status:**
- TR: production gold reviewed via `data/qa/sentiment_review_*.csv`.
- DE: partial gold seed in progress (Phase 11).
- FR/IT/ES/UK: no gold seed yet; treat sentiment outputs as exploratory.

---

## 3. Multilingual NER (entity-narrative track)

**Model ID:** `Davlan/bert-base-multilingual-cased-ner-hrl`

**Task:** PER / ORG / LOC / MISC entity recognition across all 6 countries.

**Used by:** `src/analysis/entity_extraction.py`, populates `entity_mentions`
table (Alembic migration `0005`).

**Note:** TR also runs the legacy `savasy/bert-base-turkish-ner-cased` in
parallel (populating `news_items.entities`) until Sprint 6+ retirement.

**Known limitations:**
- News-specific organizations and shortened acronyms (party names, agencies)
  are systematically under-detected.
- Person disambiguation is handled downstream by `entity_resolution.py`
  (Wikidata API, throttled 1.1s/req, 429 backoff). Cache lives in the
  `entity_resolution_cache` table (Alembic `0006`).

---

## 4. Sentence Embeddings

**Model ID:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

**Output:** 384-dim dense vectors stored in `news_items.embedding` (pgvector
HNSW index, cosine distance).

**Used by:** `src/analysis/clustering.py` (KMeans, k=15),
`src/analysis/vector_store.py` (semantic search → `/api/similar` endpoint).

**Known limitations:**
- Multilingual model is convenient but not domain-tuned. Clustering quality
  is the main known weak spot — see `project_clustering_plan.md` for
  candidate replacements (UMAP+HDBSCAN, BERTopic).
- 384-dim is a quality/cost trade; larger models would likely improve
  silhouette but at infra cost.

---

## Evaluation

| Component | Eval Method | Where to Find Results |
|---|---|---|
| TR sentiment | Manual review (gold seed); accuracy / macro-F1 / neutral recall | `scripts/evaluate_sentiment_review.py` outputs |
| TR sentiment | Behavioral CheckList suite (must-pass / watchlist / aspirational) | `mlflow ui` → experiment `news-sentiment-behavioral` |
| Multilingual NER | Spot-checked via `scripts/audit_entity_resolution.py` | Per-country audit JSON in `data/qa/` |
| Embeddings/clustering | Silhouette score (logged to MLflow) | `clustering.py` step output |
| Drift | PSI (Population Stability Index) on sentiment distribution | `data/drift_reports/<country>_<date>.json` + Grafana |

Full evaluation reports per release are tracked in `MLflow`; a public
benchmark summary is planned for Phase 12.

---

## Ethical Considerations

This pipeline produces signals over **media framing patterns**, not facts about
the world. Per-country sentiment time series describe what news sources
publish, not what is true. Specifically:

- Source selection bias dominates: 10 Turkish sources is broad but not
  representative of "Turkish public opinion". The same caveat applies to all
  six countries.
- Model bias: zero-shot XLM-R was not trained on news; the TR fine-tuned model
  is not transferable.
- Embedding-based clustering imposes structure (KMeans k=15) that may not
  match human topic taxonomies.

Outputs must be presented as "news-derived descriptive indicators", not
"country reality" or "market direction". The framing analysis layer
(Phase 10, future) will only output observable frame intensities — never
claims like "country X is Y".

---

## Card Maintenance

This card describes the production state at the time of writing. Major model
or training changes are recorded in `CHANGELOG.md`. Per-component model
versions can be queried at runtime via the `/about` endpoint.

Contact: efeyol11@gmail.com — open an issue for corrections.
