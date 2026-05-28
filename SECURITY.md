# Security Policy

## Supported Versions

Active development is on the `main` branch. Older tags (`v2-phase-*`,
`v2-entity-sprint-*`) are historical and do not receive security updates.

| Version | Supported |
|---------|-----------|
| `main`  | ✅        |
| Tags    | ❌        |

## Reporting a Vulnerability

Please report suspected security vulnerabilities **privately** so we can fix
them before they're public:

1. **Preferred**: open a [GitHub Security Advisory](https://github.com/efeyol1/sementic_news/security/advisories/new)
   on this repository.
2. **Alternative**: email **efeyol11@gmail.com** with the subject line
   `[security] <short summary>`.

Please include:
- Affected component (API endpoint, pipeline script, dashboard route, etc.)
- Reproduction steps or proof of concept
- Impact assessment (data exposure, RCE, DoS, etc.)
- Suggested mitigation if you have one

You should receive an initial acknowledgment within 5 business days. We aim
to resolve confirmed issues within 90 days of disclosure (coordinated
disclosure). If the fix takes longer, we will keep you updated.

## Scope

**In scope:**
- FastAPI service (`src/api/`) — auth bypass, injection, SSRF, etc.
- Pipeline (`src/pipeline.py`, `src/data/`, `src/analysis/`) — RCE via crafted
  RSS / sitemap / HTML, dependency confusion
- Dashboard (`dashboard/`) — XSS, CSRF, prototype pollution, unsafe redirects
- Database schema (`src/db/`, `alembic/`) — SQL injection, privilege issues
- Build & deploy (`render.yaml`, `.github/workflows/`) — secret leakage,
  privilege escalation in CI

**Out of scope** (please don't report these as security issues):
- Sentiment / NER / clustering model **behavior bugs** (e.g., wrong sentiment
  on a specific headline) — file as a regular issue
- Upstream HuggingFace model vulnerabilities (`efeyol11/bert-turkish-sentiment`,
  Davlan multilingual NER, etc.) — report to the model author or HF
- Third-party platform issues (Neon, Render, Vercel, HF Hub) — report to the
  platform directly
- Theoretical "what if you collected sensitive PII" concerns absent a specific
  exploit — for governance questions see `PRIVACY.md` and `DATA_PROVENANCE.md`

## Secrets

The repo does not contain any secrets. `.env` is gitignored. If you find
committed credentials, please report it as a security issue immediately.

Active environment variables required for operation:
- `DATABASE_URL` (Neon connection string)
- `SENTIMENT_MODEL_ID`, `SENTIMENT_BACKEND`, `SENTIMENT_ONNX_DIR`
- `HF_TOKEN` (only for weekly retrain push to Hub)
- `RENDER_DEPLOY_HOOK_URL`
- `MLFLOW_TRACKING_URI`

## Public Disclosure

After a fix is released, we will credit reporters in the security advisory
unless they prefer to remain anonymous.
