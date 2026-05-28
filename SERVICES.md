# Services & Runtime Topology

A description of the live services that make up the Semantic News stack and
how to run each one locally or in production.

> **V2 status (2026-05):** Multi-country pipeline (TR, DE, FR, IT, ES, UK)
> backed by a single Postgres + entity-narrative track. This file describes
> the V2 layout; earlier V1 (Streamlit dashboard, JSON-file reads) is
> retired.

## Topology

```
                  ┌───────────────────────────────┐
                  │  GitHub Actions (cron)        │
                  │  daily_pipeline.yml 04:00 UTC │
                  │  weekly_retrain.yml Sun 02:00 │
                  └──────────────┬────────────────┘
                                 │
              python -m src.pipeline --country <slug>
                                 │
                                 ▼
            ┌───────────────────────────────────────┐
            │  Pipeline (per country)               │
            │  RSS/Sitemap → preprocess → sentiment │
            │  → NER (legacy + multilingual)        │
            │  → entity resolution (Wikidata)       │
            │  → clustering → embeddings → drift    │
            └──────────────┬────────────────────────┘
                           │ writes
                           ▼
                ┌─────────────────────┐
                │  PostgreSQL (Neon)  │
                │  + pgvector HNSW    │
                └──────────┬──────────┘
                           │ reads
                           ▼
                ┌─────────────────────┐
                │  FastAPI (Render)   │  ←  Prometheus + Grafana
                │  port 8000          │     (monitoring/)
                └──────────┬──────────┘
                           │ HTTPS
                           ▼
                ┌─────────────────────┐
                │  Next.js (Vercel)   │
                │  port 3000          │
                └─────────────────────┘
```

## Services

### 1. Pipeline (`src/pipeline.py`)

**What it does:** Per-country daily ingestion + analysis. Each step writes
back to Postgres so downstream steps read consistent state.

**How to run locally:**

```bash
source .venv/bin/activate

# Full chain, one country, today
python -m src.pipeline --country turkey

# Specific date
python -m src.pipeline --country germany --date 2026-05-25

# Skip RSS collection (already ingested)
python -m src.pipeline --country france --skip-collect

# Article body fetch (opt-in, slower)
python -m src.pipeline --country uk --fetch-articles --article-limit 100

# Individual step (debug)
python -m src.analysis.sentiment --country italy --date 2026-05-25 --only-missing
```

**Where it runs in production:** GitHub Actions cron
(`.github/workflows/daily_pipeline.yml`) at 04:00 UTC. The runner installs
the `pipeline` extras and uses `DATABASE_URL` + `SENTIMENT_MODEL_ID` from
repo secrets.

### 2. FastAPI (`src/api/main.py`)

**What it does:** Read-only HTTP layer over the precomputed pipeline results
in Postgres. The API never runs inference — all heavy ML happens upstream.

**Why no ML in the API:** Torch + transformers + sentence-transformers is
~400MB and would blow past Render's 512MB free tier on import. The API
only pulls `fastapi`, `psycopg2`, `pydantic`, and `prometheus` — see
`pyproject.toml` base `dependencies`.

**How to run locally:**

```bash
source .venv/bin/activate
uvicorn src.api.main:app --reload --port 8000
# → http://localhost:8000/docs (OpenAPI UI)
```

**Endpoints:**

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness probe |
| `GET /about` | Service name, version, ethics disclaimer, governance links |
| `GET /api/countries` | List of configured countries |
| `GET /api/today?country=TR&date=…` | Daily summary |
| `GET /api/topic/{id}?country=TR&date=…` | Cluster detail |
| `GET /api/source-comparison?country=TR` | Per-source sentiment |
| `GET /api/similar?q=…&country=TR&n=…` | pgvector cosine search |
| `GET /metrics` | Prometheus metrics |

**Where it runs in production:** Render (`render.yaml`). Auto-deploy on
`git push main`.

### 3. Next.js Dashboard (`dashboard/`)

**What it does:** Server-rendered UI that calls the FastAPI. The country
selector is URL-driven (`?country=DE`) and propagates across navigations.

**How to run locally:**

```bash
cd dashboard
npm install
npm run dev
# → http://localhost:3000
```

**Pages:**

| Path | Purpose |
|---|---|
| `/` | Daily analytics: sentiment, entities, clusters |
| `/sources` | Source × sentiment heatmap |
| `/topic/[id]` | Cluster detail with related articles |

**Where it runs in production:** Vercel. Auto-deploy on `git push main`.
Environment: `NEXT_PUBLIC_API_URL` points to the Render API.

### 4. PostgreSQL (Neon)

**What it does:** Single source of truth for everything — raw items,
analysis outputs, entity mentions, entity resolution cache, drift reports.

**Tables (Alembic-managed):**
- `news_items` — one row per article, all pipeline steps mutate the same row
- `cluster_summaries` — per-day cluster aggregates
- `entity_mentions` (migration `0005`) — multilingual NER output
- `entity_resolution_cache` (`0006`) — Wikidata API cache
- `entity_collocations` (`0007`, Sprint 5) — co-occurrence pairs

**Migrations:** `alembic upgrade head`.

**Where it runs:** Neon serverless Postgres (managed). Connection string in
`DATABASE_URL` env var. pgvector extension required (Neon supports it
natively).

### 5. MLflow

**What it does:** Experiment tracking for the weekly retrain, plus the
behavioral CheckList outcome log. Drift reports also get logged here.

**Local:** `mlflow ui` reads `mlflow.db` (SQLite, gitignored).

**Production:** GitHub Actions retrain job logs to an MLflow tracking URI
configured via `MLFLOW_TRACKING_URI`. Behavioral test outcomes appear under
the `news-sentiment-behavioral` experiment.

### 6. Prometheus + Grafana (monitoring stack)

**What it does:** Scrapes `GET /metrics` from the API every 15 seconds and
visualizes request latency, error rates, plus custom drift gauges (per-country
PSI values).

**Local:**

```bash
docker compose up
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000  (admin / admin)
```

**Dashboards** (provisioned in `monitoring/grafana/dashboards/`):
- `api_dashboard.json` — request volume, latency, errors
- `drift_dashboard.json` — per-country sentiment PSI over time

**Not deployed in production today** — this is a self-hosted operator
concern; the SaaS deployment (Render + Vercel) does not include
Grafana/Prometheus by default. The drift gauges are exposed on `/metrics`
either way and any Prometheus-compatible collector can read them.

## What to run when

| Scenario | Commands |
|---|---|
| Dashboard dev (read-only) | `uvicorn src.api.main:app` + `cd dashboard && npm run dev` |
| One-country pipeline smoke | `python -m src.pipeline --country <slug>` |
| API endpoint debug | `uvicorn` + `curl http://localhost:8000/docs` |
| Drift / monitoring | `docker compose up` → Grafana |
| Behavioral test suite | `pytest tests/behavioral/` |
| Weekly retrain (manual) | `python -m src.training.retrain` (requires `[training]` extras) |

## Configuration files

| File | Owns |
|---|---|
| `configs/countries/*.yaml` | All country-specific data (RSS sources, sentiment model, NER, clustering, entity_narrative). Six files: turkey, germany, france, italy, spain, uk. |
| `configs/stopwords/*.txt` | Per-language clustering stopwords (Sprint 5). |
| `pyproject.toml` | Python deps + extras (`pipeline`, `training`, `serving`, `dev`). |
| `dashboard/package.json` | Next.js deps. |
| `render.yaml` | Render API service definition. |
| `docker-compose.yml` | Local monitoring stack. |
| `.env.example` | Required env vars (no real values committed). |
| `alembic.ini` + `alembic/versions/` | DB migrations (run with `alembic upgrade head`). |

See also: [`README.md`](README.md), [`ARCHITECTURE.md`](ARCHITECTURE.md),
[`MODEL_CARD.md`](MODEL_CARD.md), [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md).
