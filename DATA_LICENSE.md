# Data License & Rights Notice

This project distinguishes three categories of artifact, each governed by a
different licensing regime.

## 1. Source code

Everything under version control in this repository — Python sources, YAML
configs, dashboard TypeScript, Alembic migrations, GitHub Actions workflows,
scripts, tests, and docs — is licensed under the [**MIT License**](LICENSE).

Use, modify, and redistribute freely, subject to the MIT terms.

## 2. Collected raw text (news titles, summaries, article bodies)

**Not covered by the MIT license.**

The pipeline ingests RSS feeds, Google News sitemaps, and HTML sitemaps from
publishers in six countries (see `configs/countries/*.yaml`). The titles,
summaries, and article bodies returned by those endpoints are the
**copyrighted property of the originating publishers**.

- This repository does **NOT** ship raw collected text. `.gitignore` excludes
  `data/raw/*.json`, `data/processed/*.json`, and similar paths.
- A user running the pipeline collects this content into their own database
  for **analysis purposes only**.
- Redistribution of collected raw text (e.g., publishing a dump on
  HuggingFace Datasets, mirroring articles on a website) is **not authorized**
  by this project and would require per-publisher licensing.

If you want to use this codebase to build a redistributable dataset, you must
clear rights with each publisher individually. The `DATA_PROVENANCE.md`
document lists current sources and what we know about their licensing terms
(largely "unknown" today — see that file for details).

## 3. Derived artifacts (embeddings, sentiment labels, entity mentions, cluster summaries)

This category includes:
- 384-dim sentence embeddings stored in `news_items.embedding`
- Sentiment labels and per-class scores (`sentiment_label`, `sentiment_scores`)
- Entity mention rows (`entity_mentions` table, including Wikidata Q-IDs)
- Cluster summaries (`cluster_summaries` table) — keywords, sentiment
  distributions, source distributions
- Drift reports (`data/drift_reports/<country>_<date>.json`)

These derived artifacts, produced by the MIT-licensed code in this repository,
are released under the **MIT License** when a user chooses to publish them.

**Caveat:** Even though the artifacts themselves are MIT, they reference
specific articles by URL and entity. Republishing them at scale may, in some
jurisdictions, constitute a derivative work of the underlying news content
(especially for keyword and entity tables that closely track article titles).
We have not had legal review of this question. If you plan to publish derived
artifacts at meaningful scale, get advice.

## 4. Models

- **`efeyol11/bert-turkish-sentiment`** (HuggingFace Hub): owned by the project
  author; released under the model card's stated license.
- **`joeddav/xlm-roberta-large-xnli`**, **`savasy/bert-base-turkish-ner-cased`**,
  **`Davlan/bert-base-multilingual-cased-ner-hrl`**, **`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`**:
  upstream models with their own licenses on HuggingFace Hub. This repository
  does not redistribute them; the pipeline downloads them at runtime.

## 5. Takedown requests

If you are a publisher and would like your sources removed from this project's
default configuration:

1. Email **efeyol11@gmail.com** with the subject `[takedown] <publisher>` and
   the URL pattern you want excluded.
2. We will remove the source from `configs/countries/*.yaml` and tag a new
   release within 7 days.
3. Users of this codebase are responsible for re-syncing their local config
   from `main` after a takedown.

Note: this affects the **default configuration** only; downstream users may
have their own forks. We cannot guarantee removal from third-party
deployments.

## 6. Disclaimer

This project is provided "as is". The maintainers make no representation that
the default RSS feeds, sitemap endpoints, or scraping behavior comply with
any specific publisher's Terms of Service. Users are responsible for their
own compliance review before running the pipeline in production.

See also: [`LICENSE`](LICENSE), [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md),
[`PRIVACY.md`](PRIVACY.md).
