# Privacy Notice (KVKK + GDPR)

This notice covers personal-data handling by the Semantic News project. It is
written to satisfy both Turkish KVKK and EU GDPR transparency expectations.
This is not legal advice; for production deployments, get jurisdiction-specific
review.

## 1. Who is the data controller?

For the default deployment operated by the project maintainer:

- **Controller:** Efe Yolartıran (project author)
- **Contact:** efeyol11@gmail.com
- **Repository:** https://github.com/efeyol1/sementic_news

Users who fork and self-host this codebase become their own controllers and
should publish their own privacy notice.

## 2. What personal data is processed?

The pipeline analyzes **publicly published news content**. The only personal
data that flows into the system is **the names of individuals mentioned in
those articles** (entity_mentions rows, type=PER), plus any names appearing in
titles, summaries, and article bodies.

We do **not** collect, process, or store:
- User accounts, logins, or session data (the dashboard is read-only and
  anonymous)
- IP addresses or analytics beyond what the host platforms (Render, Vercel)
  log by default
- User-submitted content
- Cookies for tracking (only what Next.js sets by default)
- Children's data (the system is not directed at minors)

## 3. Legal basis (GDPR Art. 6)

Processing of names mentioned in published news content relies on
**legitimate interests** (Art. 6(1)(f)) — namely, academic and research
interest in media framing patterns, balanced against the modest expectation
of privacy of public figures named in published journalism.

For KVKK: same reasoning applies under Art. 5(2)(f) — legitimate interest of
the controller, provided fundamental rights of the data subject are not
violated. We do not infer special-category data (political opinion, religion,
health, etc.) about named individuals; sentiment labels apply to **articles**,
not to the people mentioned in them.

## 4. Retention

| Data | Retention |
|------|-----------|
| Raw RSS / sitemap items (`data/raw/`, `news_items` table) | Indefinite (research archive) |
| Cleaned article text (`cleaned_article_text` column) | 90 days then nulled on rolling basis (planned; not yet automated) |
| Sentiment labels, entity mentions, embeddings | Indefinite (derived artifacts) |
| Drift reports (`data/drift_reports/`) | Indefinite (small JSON files) |
| Wikidata resolution cache (`entity_resolution_cache`) | Indefinite (public reference data) |
| Behavioral test MLflow runs | Indefinite (eval history) |

Cleaned article text retention is set short because it duplicates information
already covered by upstream publishers; titles/summaries are retained as
metadata.

## 5. Your rights

Whether you live in the EU (GDPR) or Türkiye (KVKK), if you believe your
personal data has been processed by this system you have the right to:

- **Access**: ask what we hold about you
- **Rectification**: correct inaccurate data
- **Erasure** ("right to be forgotten"): request removal of `entity_mentions`
  rows naming you
- **Restriction**: ask us to limit how we process your data
- **Objection**: object to processing based on legitimate interests
- **Lodge a complaint**: with the relevant authority — KVKK
  (kvkk.gov.tr) in Türkiye, or your national DPA in the EU

To exercise any of these rights, email **efeyol11@gmail.com** with the subject
`[privacy] <request type> <your name>`. We aim to respond within 30 days.

For **erasure requests for a named person**, we will:
1. Delete `entity_mentions` rows where canonical name or any alias matches.
2. Add the name (or canonical Wikidata Q-ID, if applicable) to an exclusion
   list in `configs/privacy_exclusions.yaml` (planned; not yet implemented)
   so future runs do not re-extract the entity.
3. **Not** edit underlying news articles (which are owned by publishers,
   not us).

## 6. Automated decision-making

There is **no automated decision-making about identified individuals** in
this system. Sentiment labels are attached to news articles, not to people.
Cluster IDs group articles, not people. No score is output that ranks,
profiles, or makes judgments about a named person.

The framing analysis layer (Phase 10, future) will follow the same rule:
frame intensities are computed per article and aggregated per source /
country / topic, never per named individual.

## 7. Third-party processors

When you run the default deployment, your data may pass through:

| Processor | Role | Region |
|-----------|------|--------|
| Neon | Managed Postgres (database) | EU / US (configurable per project) |
| HuggingFace Hub | Model downloads (inference happens locally) | US-based |
| Render | API hosting | US/EU (configurable) |
| Vercel | Dashboard hosting | Global edge |
| GitHub Actions | Daily pipeline cron + retrain | US-based |

Each has its own privacy policy. If you self-host, choose regions and
processors to match your jurisdiction.

## 8. Changes to this notice

Material changes are noted in `CHANGELOG.md`. The current version of this
file is canonical; check the `main` branch on GitHub for the latest.

## 9. KVKK-specific notice

> Bu sistem, kamuya açıklanmış haber içeriklerinde geçen kişi adlarını
> entity_mentions tablosunda saklar. Hukuki dayanak: 6698 sayılı KVKK
> Madde 5(2)(f) (meşru menfaat). Veri sahibi olarak verinizin işlenip
> işlenmediğini öğrenme, düzeltme, silme ve itiraz haklarınız vardır.
> Talepler: efeyol11@gmail.com.

See also: [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md),
[`DATA_LICENSE.md`](DATA_LICENSE.md), [`MODEL_CARD.md`](MODEL_CARD.md).
