# Semantic News — Architecture

V2 multi-country media intelligence pipeline. Single Postgres backend serves
both the daily ingestion+analysis chain and the API layer.

## Big Picture

```
┌────────────────────────────────────────────────────────────────────┐
│           DAILY PIPELINE (per country, 04:00 UTC cron)             │
│                                                                    │
│  configs/countries/<slug>.yaml                                     │
│         │                                                          │
│         │  source registry (RSS / Google News sitemap / HTML)      │
│         ▼                                                          │
│   rss_collector  ──► PostgreSQL (news_items, country_code, lang)   │
│         │                                                          │
│   preprocessor   (HTML strip, langdetect, char filter)             │
│         │                                                          │
│   article_fetcher + article_parser  (opt-in --fetch-articles)      │
│         │                                                          │
│   sentiment      (TR: fine-tuned BERT; others: zero-shot XLM-R)    │
│         │                                                          │
│   ner            (legacy savasy for TR; Davlan multilingual all)   │
│         │                                                          │
│   entity_resolution  (Wikidata Q-ID linking, throttled + cached)   │
│         │                                                          │
│   clustering     (MiniLM embeddings + KMeans, per-country stopwords)│
│         │                                                          │
│   vector_store   (384-dim → pgvector HNSW)                         │
│         │                                                          │
│   drift          (PSI on sentiment distribution, MLflow + reports) │
└────────────────────────┬───────────────────────────────────────────┘
                         │ PostgreSQL (Neon, pgvector enabled)
              ┌──────────▼──────────┐
              │  FastAPI (Render)   │
              │  port 8000          │
              │  ── reads only ──   │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │   Next.js (Vercel)  │
              │   port 3000         │
              │   country selector  │
              └─────────────────────┘
```

Six countries are wired today: Turkey, Germany, France, Italy, Spain,
United Kingdom — see `configs/countries/`. Daily corpus is ~3,853 items
across all countries (measured 2026-05-02, post-sitemap handler rollout;
up from ~475 items/day under the pre-V2 RSS-only setup).

## Data Layer

### `news_items` table

One row per article. Each pipeline step mutates a subset of columns; rows are
identified by `(link)` (unique) and tagged with `country_code` / `language`.

| Column | Type | Step that populates it |
|---|---|---|
| `country_code`, `language` | TEXT | rss_collector (from country config) |
| `collected_date`, `published_at` | DATE / TIMESTAMP | rss_collector |
| `title`, `summary`, `source_name`, `link` | TEXT | rss_collector |
| `cleaned_title`, `cleaned_summary` | TEXT | preprocessor |
| `cleaned_article_text` | TEXT | article_parser (opt-in via `--fetch-articles`) |
| `article_fetched_at` | TIMESTAMP | article_fetcher |
| `is_turkish`, `char_count` | BOOL / INT | preprocessor |
| `sentiment_label`, `sentiment_score`, `sentiment_scores` | TEXT / FLOAT / JSONB | sentiment |
| `entities`, `entity_count` | JSONB / INT | ner (legacy path, still on for TR) |
| `cluster_id`, `cluster_keywords`, `cluster_title` | INT / JSONB / TEXT | clustering |
| `embedding` | vector(384) | vector_store |

### `cluster_summaries` table

Per-day aggregates: `(country_code, date, cluster_id)` unique. Stores
keywords, sentiment distribution, source distribution.

### `entity_mentions` table (Alembic `0005`)

Multilingual NER output. One row per (article, entity) pair. Includes
canonical name, raw mention text, entity type, position, and (if resolved)
`wikidata_qid`.

### `entity_resolution_cache` table (Alembic `0006`)

Wikidata API response cache to avoid re-querying. Throttled at 1.1 s/request
with 429 backoff in `entity_resolution.py`.

### `entity_collocations` table (Alembic `0007`, Sprint 5)

Per-day co-occurrence pairs between resolved entities, used for narrative
graph features.

### pgvector

`embedding vector(384)` + HNSW index. `/api/similar`:

```sql
SELECT title, 1 - (embedding <=> $1::vector) AS similarity
FROM news_items
WHERE country_code = $2
ORDER BY embedding <=> $1::vector
LIMIT $3
```

## Pipeline Steps

### 1. `rss_collector.py`

Source types supported (declared per-source in country YAML):

- `rss` — feedparser over RSS / Atom; most sources
- `googlenews_sitemap` — XML sitemap with Google News namespace
- `html_sitemap` — plain XML sitemap of URLs; scrape `<title>` +
  `og:description` via BeautifulSoup (15-thread pool)

Deduplication: `INSERT ... ON CONFLICT (link) DO NOTHING`.

### 2. `preprocessor.py`

HTML strip, whitespace normalize, langdetect (writes `is_turkish` flag).
Articles below a length threshold get dropped (`char_count` filter).

### 3. `article_fetcher.py` + `article_parser.py` (opt-in)

Only runs when pipeline is invoked with `--fetch-articles --article-limit N`.
Downloads article HTML, parses body text into `cleaned_article_text` and
records `article_fetched_at`. Default daily cron does NOT fetch bodies.

### 4. `sentiment.py`

Per-country backend selection driven by `configs/countries/<slug>.yaml`:

- **TR (default)**: `efeyol11/bert-turkish-sentiment` (fine-tuned, production).
  Behavioral CheckList gate blocks retraining if `must-pass` capabilities
  regress. ONNX int8 backend available for ~3.8× speedup vs PyTorch (see
  README benchmark).
- **DE / FR / IT / ES / UK**: `joeddav/xlm-roberta-large-xnli` (multilingual
  zero-shot, NLI-based). Per-language candidate labels declared in YAML.

Scoped re-score flags: `--only-missing`, `--only-stale-after-body`,
`--only-with-body`. `max_length = 256`.

### 5. `ner.py` + `entity_extraction.py`

Two NER paths run today:

- **Legacy TR-only** (`ner.py`): `savasy/bert-base-turkish-ner-cased`,
  populates `news_items.entities` for the legacy top-entities dashboard.
- **Multilingual** (`entity_extraction.py`):
  `Davlan/bert-base-multilingual-cased-ner-hrl`, populates `entity_mentions`.

Both are kept in parallel until Sprint 6+ retirement of the legacy path.

### 6. `entity_resolution.py` + `entity_canonicalization.py`

Each `entity_mentions` row goes through resolution:

1. Local alias lookup (country YAML `entity_narrative.aliases`)
2. `entity_resolution_cache` lookup
3. Wikidata API call (`uselang=<country lang>`, throttled 1.1 s, 429 backoff)
4. Confidence threshold (per-country, e.g., TR/IT/ES use 0.65)

The same canonical Wikidata Q-ID (e.g., Trump → Q22686, Erdoğan → Q39259)
appears in all country YAMLs that mention the person — this is the
cross-country invariant.

### 7. `clustering.py`

Sentence-transformer embeddings (`paraphrase-multilingual-MiniLM-L12-v2`),
KMeans (k=15 default, per-country override allowed). Cluster keywords via
TF-IDF over the per-language stopword list (`configs/stopwords/<lang>.txt`).
Silhouette score logged to MLflow.

### 8. `vector_store.py`

Same MiniLM model (singleton, cached). Writes `embedding` column. Indexed
with pgvector HNSW for similarity search.

### 9. `drift.py`

PSI (Population Stability Index) on sentiment distribution, per country,
day-over-day. Thresholds: `<0.10` stable, `<0.25` moderate, `≥0.25` significant.
Writes:

- JSON to `data/drift_reports/<country_code>_<date>.json`
- Postgres drift table
- MLflow run (`news-drift` experiment)
- Prometheus gauge (`sentiment_psi`)

## API Layer

```
FastAPI (Render)
├── GET /health                          — liveness
├── GET /about                           — version + ethics disclaimer + governance links
├── GET /api/countries                   — list of configured countries
├── GET /api/today?country=&date=        — daily summary
├── GET /api/topic/{id}?country=&date=   — cluster detail
├── GET /api/source-comparison?country=  — per-source sentiment
├── GET /api/similar?q=&country=&n=      — pgvector cosine search
└── GET /metrics                         — Prometheus
```

Startup runs `init_db()` — creates tables if absent. The API installs only
the base deps (no torch/transformers) to fit Render's 512MB free tier.

## Dashboard Layer

```
Next.js 14 App Router (Vercel)
├── /                — daily analytics
├── /topic/[id]      — cluster detail + related articles
└── /sources         — source × sentiment heatmap
```

The country selector is URL-driven (`?country=DE`) and propagates across
navigations. Server components fetch from the API; revalidate window is
~5 minutes.

## CI / CD

```
git push main
  ├── ci.yml          → ruff + pytest (unit + behavioral)
  ├── deploy.yml      → Render deploy hook (API)
  └── Vercel          → automatic Next.js deploy

cron daily_pipeline.yml   04:00 UTC daily
  └── per-country matrix: python -m src.pipeline --country <slug>

cron weekly_retrain.yml   Sun 02:00 UTC
  └── python -m src.training.retrain
      → behavioral gate
      → if pass: push to HF Hub (efeyol11/bert-turkish-sentiment)
```

## Technology Choices

| Decision | Alternative | Why |
|---|---|---|
| Fine-tuned BERT (TR) + zero-shot XLM-R (others) | All zero-shot or all fine-tuned | TR has the gold seed to support a fine-tune; non-TR countries don't yet — see `project_v2_phases.md` Phase 11 |
| pgvector | ChromaDB | One database to operate, not two; Neon supports it natively |
| Neon Postgres | Supabase / Render PG | Serverless, generous free tier, pgvector built-in |
| Davlan multilingual NER | Per-country NER models | Cross-country canonical Q-ID joining is the V2 backbone; single model = single failure mode |
| KMeans clustering | UMAP+HDBSCAN, BERTopic | Open question — see `project_clustering_plan.md`; Phase 10 decision |
| Next.js App Router | CRA, Vite | Server components fetch API server-side, no CORS, no client API keys |
| psycopg2 | SQLAlchemy ORM | Queries are simple; ORM overhead unnecessary; direct SQL more transparent |
| Render API + Vercel Dashboard | Single host | Different tier shapes (heavy API vs static dashboard); zero-config CD on push |

## Open Architectural Questions

See `README.md` "V2 Roadmap" section and the related Claude memory entries
for the live decision queue. Major open items:

- **Phase 8**: Multi-country GitHub Actions matrix (currently single-country
  daily cron).
- **Phase 9**: Per-country run reports (`data/reports/{slug}/{date}_run_report.json`).
- **Phase 10**: Framing analysis foundation — observable frame intensities,
  not "country X is Y" claims.
- **Phase 11**: Country-aware sentiment QA cue lists (currently TR-only).
- **Clustering replacement**: KMeans → UMAP+HDBSCAN or BERTopic. Decision
  due before Phase 10.
- **Sprint 5.5** (this branch): Public release hardening — governance docs,
  LICENSE, data provenance.

See also: [`SERVICES.md`](SERVICES.md), [`MODEL_CARD.md`](MODEL_CARD.md),
[`DATA_PROVENANCE.md`](DATA_PROVENANCE.md), [`CHANGELOG.md`](CHANGELOG.md).
