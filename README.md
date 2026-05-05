# LiveOps Agent

[![tests](https://github.com/sathwikarr/liveops-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/sathwikarr/liveops-agent/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![live](https://img.shields.io/badge/live-liveops--agent.onrender.com-10b981)](https://liveops-agent.onrender.com)

> **An AI co-pilot for retail ops.** Upload a CSV of orders, ask the agent in plain
> English, watch it pick from 10 analytic tools, and get the answer + plan + an
> observation trace. Backed by a 55-case eval harness with an honest dual-baseline.

**🔗 Live demo: [liveops-agent.onrender.com](https://liveops-agent.onrender.com)** *(~20 sec cold start on Render free tier)*

---

## What's interesting about this

Most portfolio AI projects flex a single eval number. This one pins **three**, and the gap between them is the story:

| Corpus | Backend | Pass rate | What it tells you |
|---|---|---:|---|
| **Main** (55 cases) | heuristic | **100.0%** | Regression-catcher. Same regex, same questions — moves only if something breaks. |
| **Holdout** (14 cases) | heuristic | **7.1%** | Paraphrases the heuristic was never tuned on. Falls off a cliff. |
| **Holdout** (14 cases) | Gemini | **28.6%** | Same paraphrases through an LLM. ~4× the heuristic — meaningful generalization, with room to grow on prompt tuning. |

The **7.1%** is the most honest number. It's the cliff that tells you whether the agent generalizes or just memorizes. Both holdout numbers are pinned in `tests/fixtures/`, regression-checked in CI, and surfaced on the live `/evals` page.

---

## Try it in 30 seconds

```bash
git clone https://github.com/sathwikarr/liveops-agent.git
cd liveops-agent
pip install -r requirements.txt
uvicorn web.server:app --reload
# → http://127.0.0.1:8000
```

The bundled retail dataset (3,878 orders, 200 customers, 30 SKUs, 18 months) is loaded automatically. No signup required — `/demo`, `/workbench`, and `/evals` all work anonymously.

To exercise the full multi-user surface (saved datasets, question history, encrypted Slack/SMTP connectors), sign up at `/signup`.

---

## What you can do on the live site

| Page | What's there |
|---|---|
| **`/`** | Honest dual-baseline ribbon (100% / 7.1% / 28.6%) right on the landing |
| **`/demo`** | KPI tiles + 3 server-rendered charts (revenue line, top products bar, churn donut) over the bundled dataset |
| **`/workbench`** | Drop a CSV → ask plain-English questions → see plan + tools + answer + 6 charts inline. Loading skeletons, observability badges, copy-as-Markdown |
| **`/evals`** | Pass-rate timeline, per-tool breakdown, per-case failure log, Run-evals button |
| **`/dashboard`** *(auth)* | Active dataset KPIs + charts, saved-datasets list, recent + pinned questions, connector status |
| **`/history`** *(auth)* | Chronological feed of every upload + question, click-to-expand to see past answers |
| **`/settings`** *(auth)* | Slack webhook + SMTP credentials, Fernet-encrypted at rest, "Send test ping" |

---

## The 10 analytic tools

The agent picks from these. Each is independently tested, eval-graded, and chartable:

| Tool | What it returns |
|---|---|
| `revenue_by_period` | Daily / weekly / monthly revenue trend |
| `top_products` / `top_customers` | Pareto rankings by total revenue |
| `segment_customers` | RFM bucket counts (Champions, Loyal, At-Risk, Lost…) |
| `product_quadrants` | BCG quadrant counts (Star, Cash Cow, Question Mark, Dog) |
| `co_purchases` | Frequent product pairs by lift |
| `price_elasticity` | Per-SKU log-log regression slope |
| `churn_risk` | Distribution of Active / Cooling / At-Risk / Churned |
| `cohort_retention` | Signup-cohort retention matrix |
| `describe_columns` | Schema + dtypes (fallback for unrecognised questions) |

The tool registry lives in `analyst/agent.py`. Each tool is a `ToolSpec` with name, description, params, and a callable.

---

## Architecture

```mermaid
flowchart LR
    subgraph In["Input"]
        U[User CSV upload]
        B[Bundled retail dataset]
    end
    subgraph Plan["Planner"]
        H[Heuristic regex]
        L[Gemini]
    end
    subgraph Exec["Executor"]
        T[10 tool registry]
    end
    subgraph Out["Surfaces"]
        WB[/workbench]
        DSH[/dashboard]
        HIST[/history]
    end
    subgraph Store["Storage"]
        DB[(SQLite WAL)]
        UF[user_data/<br/>per-user uploads]
    end
    subgraph Eval["Eval harness"]
        M[Main corpus<br/>55 cases]
        HO[Holdout<br/>14 paraphrases]
    end

    U & B --> Plan --> T --> Out
    H & L -.fallback.-> T
    Out --> DB
    U --> UF
    M & HO --> Plan
```

---

## Stack

- **Backend** — FastAPI + Starlette sessions + bcrypt + Fernet
- **Data** — pandas, scikit-learn, Prophet (per-segment forecasting), networkx (basket lift)
- **LLM** — Gemini 2.5 Flash via the `google-genai` SDK, with a deterministic regex-routing fallback
- **Frontend** — Jinja2 + Tailwind CDN + Alpine.js + Chart.js
- **Storage** — SQLite (WAL), schema in `agent/db.py`
- **Auth** — bcrypt, signed-cookie sessions, per-IP sliding-window rate limit on signup + login
- **Tests** — 325 pytest cases, run on 3.11 + 3.12 in CI
- **Deploy** — Docker + GitHub Actions + Render free tier

---

## What makes this different from other portfolio AI projects

1. **Honest about generalization.** Most demos show one number. This one shows the cliff between memorized and novel input — and explains why the LLM number is 28.6%, not 80%, with the next iteration named explicitly.
2. **The eval harness is the product.** Not an afterthought. Pass rates, per-tool scoreboards, regression checks, and a 14-case holdout are all surfaced in the UI, persisted to SQLite, and charted as a timeline.
3. **Multi-user from the start.** Encrypted connector storage, isolated uploads, isolated history, isolated questions — every test asserts the isolation explicitly.
4. **Real production plumbing.** Auth, rate limiting, mobile-responsive layout, loading skeletons, accessibility (`aria-busy`, `aria-controls`, focus-visible), CI on every PR.

---

## Project layout

```
agent/                       # SQLite + auth + secrets
  auth.py        bcrypt signup + login
  db.py          users · anomalies · actions · datasets · questions · connectors · eval_runs
  secret.py      Fernet encrypt/decrypt with LIVEOPS_FERNET_KEY
  bandit.py      Thompson-sampling action picker
  detect.py      per-segment z-score + IsolationForest
  explain.py     Gemini wrapper (deterministic-fallback)
  forecast.py    Prophet per-segment + walk-forward CV
  notify.py      Slack + SMTP unified notifier

analyst/                     # Analytic surface
  agent.py       Tool registry, heuristic planner, LLM planner, executor, ask()
  evals/         55-case main corpus + 14-case holdout + scorer + runner
  analysis.py    RFM, cohorts, market basket, elasticity
  predict.py     churn, stockout, demand
  charts.py      8 chart builders
  sample_data/   bundled retail_orders.csv

web/                         # FastAPI app
  server.py      Routes, /api/* endpoints, dataset/profile/chart helpers
  templates/     landing, demo, workbench, evals, dashboard, history,
                 run_agent, settings, login (Jinja2 + Alpine)
  static/site.css Premium-vein motion polish

tests/                       # 325 pytest cases
  test_web.py            60 web tests
  test_analyst_agent.py  41 agent tests (incl. why-string regression)
  test_analyst_evals.py  17 eval harness tests
  …                      auth, DB, forecast, notify, charts, pinboard
  fixtures/              pinned baselines (3 JSON files)
```

---

## Configuration (.env)

See `.env.example` for the full list. Most are optional — the app degrades gracefully.

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | LLM backend; without it, heuristic still works (degrades on holdout) |
| `GEMINI_MODEL` | Defaults to `gemini-2.5-flash` |
| `SESSION_SECRET` | Signs the session cookie. Generate with `python -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `LIVEOPS_FERNET_KEY` | Encrypts saved Slack webhooks + SMTP creds. Generate with `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` |
| `LIVEOPS_DB` | SQLite path. Defaults to `data/liveops.sqlite3` |
| `LIVEOPS_UPLOAD_DIR` | Per-user upload directory. Defaults to `user_data/workbench_uploads/` |
| `LIVEOPS_LLM_DEBUG` | Set to `1` to log LLM-fallback reasons to stderr (network errors, parse failures, etc.) |

---

## Tests

325 pytest cases cover auth, DB, every analytic tool, the LLM agent loop, the heuristic planner's friendly-prose contract (regex literals are explicitly forbidden in plan `why` strings), the 55-case eval harness, the held-out 14-case corpus, the FastAPI routes (signup → login → dashboard → upload → ask → history → settings), per-user isolation across datasets / questions / connectors, encryption-at-rest verification, rate-limiter behavior, and frontend HTML scaffolding.

```bash
pip install pytest
SLACK_WEBHOOK="" SMTP_HOST="" GEMINI_API_KEY="" pytest -v
```

CI runs on every PR — see `.github/workflows/tests.yml`.

---

## Deployment

### Docker

```bash
docker build -t liveops-agent .
docker run --rm -p 8000:8000 \
  --env-file .env \
  -v "$PWD/data:/app/data" \
  -v "$PWD/user_data:/app/user_data" \
  liveops-agent
```

### Render (current live deploy)

1. Connect the GitHub repo on render.com → New Web Service.
2. Build command: `pip install -r requirements.txt`
3. Start command: `uvicorn web.server:app --host 0.0.0.0 --port $PORT`
4. Set `GEMINI_API_KEY`, `SESSION_SECRET`, `LIVEOPS_FERNET_KEY` in the Environment tab.

The app auto-deploys on every push to `main`.

---

## Security

Sensitive data handling:

- **Passwords** — bcrypt-hashed, never stored plaintext.
- **Sessions** — signed with `SESSION_SECRET` (Starlette's `SessionMiddleware`).
- **Connector secrets** (Slack webhooks, SMTP credentials) — Fernet-encrypted at rest, derivation key from `LIVEOPS_FERNET_KEY`. The encryption is verified by tests: stored blob never contains the plaintext substring.
- **Per-user isolation** — datasets, questions, and connectors are all keyed by username and tested for cross-user access (404 on read, 401 on mutate).
- **Rate limiting** — 5 attempts / 60s sliding window per IP on `/login` and `/signup`.
- `.env` is git-ignored. See `SECURITY.md` for the secret-rotation playbook.

---

## Roadmap (what I'd build next)

1. **Tighten the LLM routing prompt.** The 28.6% holdout pass rate is the single biggest lever — most failures are the LLM returning the right tool name in prose but not in the JSON shape the parser wants. ~30 minutes of prompt iteration could push this past 60%.
2. **Token + per-tool latency in `obs_meta`.** Currently `obs_meta` reports wall-clock latency and tool count; surfacing token cost (LLM) and per-tool timing would close the per-question observability story.
3. **Self-serve password reset.** Today the login page has an honest "Forgot password?" stub — proper email-driven reset needs a one-shot token table and SMTP-from-server.
4. **Bandit feedback loop.** Capturing thumbs-up/down on each agent answer to learn which tool selections lead to satisfying responses.

---

## License

MIT — see [LICENSE](LICENSE).
