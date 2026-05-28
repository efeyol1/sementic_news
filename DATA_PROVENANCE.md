# Data Provenance

This document tracks the origin, retrieval method, and licensing status of the
news content ingested by the Semantic News pipeline. It is a companion to
[`DATA_LICENSE.md`](DATA_LICENSE.md) and the source-of-truth is each country's
config in `configs/countries/<slug>.yaml`.

## What gets stored

The pipeline persists the following fields per article into the `news_items`
Postgres table:

| Field | Source | Notes |
|---|---|---|
| `title`, `summary`, `link`, `published_at` | RSS / sitemap feed | Always stored |
| `source_name`, `country_code`, `language` | From country YAML | Always stored |
| `cleaned_title`, `cleaned_summary` | Pipeline (HTML strip) | Always stored |
| `is_turkish`, `char_count` | langdetect | Always stored |
| `cleaned_article_text` | Fetched from article URL (opt-in) | Only when pipeline invoked with `--fetch-articles`; retention 90 days |
| `article_fetched_at` | Timestamp | Coupled with above |
| `sentiment_label`, `sentiment_score`, `sentiment_scores` | Pipeline (`sentiment.py`) | Derived; MIT |
| `entities` (legacy TR), `entity_count` | Pipeline (`ner.py`) | Derived; MIT |
| `cluster_id`, `cluster_keywords`, `cluster_title` | Pipeline (`clustering.py`) | Derived; MIT |
| `embedding` | Pipeline (`vector_store.py`, 384-dim) | Derived; MIT |

Additional tables:
- `entity_mentions` (Alembic `0005`) — multilingual NER output with Wikidata
  Q-IDs
- `entity_resolution_cache` (Alembic `0006`) — Wikidata API response cache
- `entity_collocations` (Alembic `0007`, Sprint 5) — co-occurrence pairs
- `cluster_summaries` — per-day cluster aggregates

## What is NOT stored

- HTML markup of article pages (stripped during ingestion)
- Comments, social-engagement counts, or any user-generated content
- User session data, IP addresses, or analytics traces

## Source inventory (per country)

License status legend:
- `unknown` — we have not reviewed publisher ToS or `robots.txt`
- `robots-allowed` — `robots.txt` explicitly permits crawling the path
- `restricted` — publisher ToS suggests limits; review case-by-case
- `feed-public` — endpoint is a documented public RSS / news sitemap

> **Current honest status: nearly every source below is `unknown`.** We
> document the matrix here so future per-source review can fill it in; until
> then, please treat the data as collected under the same conditions you would
> a manual reader visiting the public RSS endpoint of each publisher.

### Turkey (TR, 10 publishers)

| Publisher | Source Type | License | Notes |
|---|---|---|---|
| Habertürk | rss + googlenews_sitemap | unknown | 11 RSS categories + Google News sitemap → ~600 unique/day |
| Hürriyet | rss | unknown | 9 category feeds → ~680 unique/day |
| NTV | rss | unknown | 9 categories, 20-cap each → ~177 unique/day |
| CNN Türk | rss | unknown | 11 categories → ~350 unique/day (occasional future-dated entries) |
| Sözcü | rss + googlenews_sitemap | unknown | Sitemap primary (~536), RSS as fallback |
| Milliyet | rss | unknown | 8 categories → ~335 unique/day |
| Sabah | rss | unknown | 13 categories → ~148 unique/day |
| TRT Haber | rss | unknown | Public broadcaster; sondakika + manset → ~92/day |
| Cumhuriyet | rss | unknown | 9 categories → ~860 unique/day (largest TR source) |
| Yeni Şafak | rss + html_sitemap | unknown | RSS hard-capped at 15; HTML sitemap scrape → ~200/day |

### Germany (DE, 6 publishers)

| Publisher | Source Type | License | Notes |
|---|---|---|---|
| tagesschau | rss | unknown | Public broadcaster (ARD); 5 RSS topics |
| Deutsche Welle | rss | unknown | Public international broadcaster |
| DER SPIEGEL | rss | unknown | 5 RSS topics |
| ZEIT ONLINE | rss | unknown | 4 RSS topics |
| ZDFheute | rss | unknown | Public broadcaster (ZDF) |
| FAZ.NET | rss | unknown | Frankfurter Allgemeine |

### France (FR, 6 publishers)

| Publisher | Source Type | License | Notes |
|---|---|---|---|
| Le Monde | rss | unknown | 5 RSS topics |
| Le Figaro | rss | unknown | 4 RSS topics |
| Libération | rss | unknown | |
| Le Parisien | rss | unknown | 2 RSS topics |
| France 24 | rss | unknown | International broadcaster; 2 RSS topics |
| RTL | googlenews_sitemap | unknown | Sitemap-based |

### Italy (IT, 8 publishers)

| Publisher | Source Type | License | Notes |
|---|---|---|---|
| Corriere della Sera | rss | unknown | 4 RSS topics |
| La Repubblica | rss | unknown | 4 RSS topics |
| ANSA | rss | unknown | News agency; 4 RSS topics |
| Il Sole 24 Ore | rss | unknown | Financial daily; 3 RSS topics |
| Il Fatto Quotidiano | rss | unknown | |
| La Stampa | rss | unknown | |
| Il Post | rss | unknown | |
| RAI News | rss | unknown | Public broadcaster |

### Spain (ES, 7 publishers)

| Publisher | Source Type | License | Notes |
|---|---|---|---|
| El País | rss | unknown | 4 RSS topics |
| El Mundo | rss | unknown | 4 RSS topics |
| ABC | rss | unknown | 2 RSS topics |
| La Vanguardia | rss | unknown | 3 RSS topics |
| El Confidencial | rss | unknown | 2 RSS topics |
| 20 Minutos | rss | unknown | |
| RTVE | rss | unknown | Public broadcaster |

### United Kingdom (GB, 7+ publishers)

| Publisher | Source Type | License | Notes |
|---|---|---|---|
| BBC News | rss | unknown | 6 RSS topics; public broadcaster |
| The Guardian | rss | unknown | 4 RSS topics |
| Sky News | rss | unknown | 4 RSS topics |
| The Independent | rss | unknown | 2 RSS topics |
| Channel 4 News | rss | unknown | |
| Financial Times | rss | unknown | Paywalled site; only RSS-exposed snippets |
| Reuters UK | rss | unknown | News agency |

Total: **~44 publishers across 6 countries**, daily corpus ~3,853 items
(measured 2026-05-02 after sitemap handler rollout).

## Retrieval methods

The pipeline supports three source types, all declared in country YAML:

1. **`rss`** — standard RSS / Atom feeds via `feedparser`. Most sources.
2. **`googlenews_sitemap`** — Google News–compatible XML sitemaps; richer
   per-day coverage than RSS for some publishers (Habertürk, Sözcü, RTL).
3. **`html_sitemap`** — plain XML sitemap of URLs; the pipeline scrapes
   `<title>` and `og:description` from each article page using
   BeautifulSoup4 + a 15-thread pool. Used for Yeni Şafak.

Article body fetching (`--fetch-articles`) is **opt-in**: only when invoked
does the pipeline call `newspaper3k` / custom parsing logic to populate
`cleaned_article_text`. Default daily pipeline runs without it.

## Third-party processors

When run in the default deployment topology, ingested data passes through:

| Processor | Role | Region |
|---|---|---|
| Neon | Managed Postgres database | EU / US (per project) |
| HuggingFace Hub | Model checkpoints | US |
| Render | API service hosting | US / EU |
| Vercel | Dashboard hosting | Global edge |
| GitHub Actions | Cron pipeline + retrain runners | US |
| MLflow | Local SQLite or self-hosted server | (operator's choice) |

See [`PRIVACY.md`](PRIVACY.md) for legal-basis analysis.

## Robots.txt and ToS posture

We do not currently fetch or honor `robots.txt` programmatically. Operators
running this pipeline at scale should:

1. Review each publisher's ToS for the specific endpoints they intend to use.
2. Implement `robots.txt` fetch + parse before scaling beyond the default
   1-fetch-per-source-per-day rhythm.
3. Lower the parallel fetch count in `html_sitemap` handler if a publisher
   complains (currently 15 threads).

The default daily cron runs once per day at 04:00 UTC and fetches each
publisher's feed exactly once — a load comparable to a single human refreshing
the homepage. This is what we consider polite-but-uncoordinated; it is not a
substitute for proper ToS review.

## Takedown process

See [`DATA_LICENSE.md`](DATA_LICENSE.md) §5. Summary: email
`efeyol11@gmail.com` with `[takedown] <publisher>`; we remove the source from
`configs/countries/*.yaml` and tag a release within 7 days.

## Auto-generation note

This document is currently maintained by hand. A future script
(`scripts/generate_data_provenance.py`, planned for Sprint 6+) will render the
per-country tables from `configs/countries/*.yaml` automatically, so adding
or removing a source updates this document via the same PR.

## How to update

When a source changes status (e.g., publisher confirms a license, or a
takedown is processed):

1. Update the `license` field in the relevant `configs/countries/<slug>.yaml`
   source entry.
2. Update the corresponding row in this document.
3. Note the change in `CHANGELOG.md` under `[Unreleased]`.

See also: [`DATA_LICENSE.md`](DATA_LICENSE.md), [`PRIVACY.md`](PRIVACY.md),
[`MODEL_CARD.md`](MODEL_CARD.md).
