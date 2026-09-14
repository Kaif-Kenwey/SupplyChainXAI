# SupplyChainXAI

**Explainable Demand Forecasting & Procurement Intelligence** — an end-to-end,
MLOps-instrumented decision-support system for procurement teams, built around
one idea: *every number the system produces must be able to explain itself —
and every model must be traceable, monitored and honestly evaluated.*

Procurement decisions are still made on historical averages and gut feel.
SupplyChainXAI replaces that with a transparent pipeline that answers the six
questions a buyer actually asks:

| Question | Component |
|---|---|
| What will demand be? | **Forecasting** — 5 model families compared per SKU (MAE / RMSE / MAPE / WAPE / bias) |
| How do we know the model is good? | **Walk-forward backtesting** — rolling-origin evaluation across multiple folds, per-SKU stability analysis |
| What could go wrong? | **Risk engine** — stockout probability, overstock, demand spikes, supplier lead-time drift, price anomalies |
| What should I buy, when, from whom? | **Recommendation engine** — order quantity + supplier selection with drift-aware scoring |
| Why is it recommended? | **Explainability** — SHAP attributions for the forecast + exact additive decomposition of every BUY |
| What if things change? | **What-If Simulator** — re-plans the entire portfolio under stress scenarios |

…plus a **grounded GenAI Procurement Copilot** (refuses to invent numbers),
a full **MLOps layer** (experiment tracking, model registry, promotion gates,
retraining policy, drift & coverage monitoring, business KPIs), served through
a **FastAPI backend and an executive dashboard**.

![Dashboard](docs/screenshots/dashboard-overview.png)

---

## What it looks like

| | |
|---|---|
| ![Forecast](docs/screenshots/dashboard-forecast-p104.png) | ![Recommendations](docs/screenshots/dashboard-recommendations.png) |
| *Actual vs forecast with 80% PI — P104's demand spike is visible and explained* | *Every BUY decomposes into demand / safety stock / inventory / rounding* |
| ![Simulator](docs/screenshots/dashboard-simulator.png) | ![Copilot](docs/screenshots/dashboard-copilot.png) |
| *What-If Simulator: demand +20%, lead +30% → full re-plan with per-SKU deltas* | *Copilot answers cite the exact SQL / artifacts they came from* |

More screenshots in [`docs/screenshots/`](docs/screenshots).

---

## Problem → solution

A distribution company buys 12 SKUs from 6 suppliers. Buyers must decide
daily, per SKU: order now or wait, how much, from which supplier — under
demand uncertainty, promotion effects, stockout-censored sales history and
suppliers whose real lead times drift away from their quoted ones.

The system turns raw transactional CSVs (sales, inventory, POs, supplier
terms) into:

1. a per-SKU demand forecast with prediction intervals from the best of five
   model families, selected by rolling-origin backtesting;
2. risk alerts with the evidence that triggered them;
3. pack-rounded, supplier-assigned BUY recommendations whose arithmetic is
   exactly explainable;
4. a monitoring loop that detects when the model, the data or the intervals
   degrade — and a policy that decides when to retrain and when a new model
   may replace the incumbent.

## ML methodology

- **Features** are leakage-safe by construction: demand lags (1/7/14/28),
  shifted rolling mean/std (7/28), calendar features, promo flag, price,
  stockout-censoring indicator. Future information can never enter a
  training row.
- **Model zoo**: Moving Average (28d), Holt-Winters ES, SARIMA(1,1,1)x(1,0,1)7,
  Random Forest, Gradient Boosting, XGBoost (auto-detected). Tree models
  forecast recursively (predicted values feed back as lags; promos are
  assumed 0 ahead — they are unplannable).
- **Evaluation is walk-forward**: 4 expanding-window folds of 30 days each
  (configurable), metrics per fold/SKU, mean and σ across folds, and an
  explicit *unstable* flag when a model's fold-to-fold WAPE variability is
  high. A single hold-out can crown a lucky model; this cannot.
- **Selection**: per SKU, lowest composite score of min–max-normalised
  WAPE/MAE/RMSE (weights 0.5/0.3/0.2, configurable), ties broken
  deterministically.
- **Metrics honesty**: MAPE excludes zero-actual days (stockouts censor true
  demand); WAPE is the primary aggregate for the same reason; forecast bias
  is tracked per model.

## Results — old single hold-out vs new walk-forward

All numbers are produced by `python scripts/train.py` on the committed
dataset (seeded generator — fully reproducible; see `artifacts/`).

**OLD — original single 90-day hold-out** (the evaluation this project
shipped with first):

| Model | WAPE |
|---|---|
| **XGBoost** | **15.57%** |
| Random Forest | 15.96% |
| Gradient Boosting | 16.29% |
| Holt-Winters ES | 18.97% |
| SARIMA | 19.16% |
| Moving Average (28d) | 35.73% |

**NEW — walk-forward, 4 folds × 30 days, expanding window** (mean of per-SKU
mean WAPE across folds; σ is the standard deviation of per-SKU means):

| Model | Mean WAPE ↓ | Worst SKU | σ across SKUs | Mean MAE | Mean bias | SKU wins |
|---|---|---|---|---|---|---|
| **XGBoost** | **12.15%** | 16.29% | 2.73 | 8.13 | −3.2 u | 4 |
| Random Forest | 12.24% | 16.82% | 2.65 | 8.07 | −3.8 u | 4 |
| Gradient Boosting | 12.42% | 17.45% | 2.59 | 8.36 | −2.7 u | 4 |
| SARIMA (1,1,1)x(1,0,1)7 | 14.33% | 17.55% | 4.01 | 9.41 | −2.3 u | 0 |
| Holt-Winters ES | 14.66% | 19.12% | 4.79 | 9.49 | −1.3 u | 0 |
| Moving Average (28d) — baseline | 33.75% | 38.45% | 2.64 | 22.21 | −3.4 u | 0 |

Honest observations:

- the walk-forward numbers are *better* than the old hold-out (12.15% vs
  15.57%) because every fold trains on more history than the single 90-day
  window allowed — both tables are kept so the difference is visible, not
  cherry-picked;
- the 4/4/4 win split between XGBoost / Random Forest / Gradient Boosting is
  the strongest argument for per-SKU bake-offs: no single family dominates a
  12-SKU portfolio;
- all models carry a negative bias (−1.3 to −3.8 units/day) — systematic
  under-forecasting on stockout-censored days. That is a real, visible
  weakness, not a hidden one;
- 2 model/SKU combinations are flagged unstable (WAPE CV > 0.5) and
  surfaced in the dashboard instead of being averaged away;
- the *most recent* fold is harder (17.5% portfolio WAPE vs 16.3% in
  previous folds) because P104's +36% demand spike sits there — the
  monitoring layer detects exactly this instead of hiding it inside an
  average.

Full per-fold and per-SKU tables: [`docs/results.md`](docs/results.md),
`artifacts/evaluation/`.

## Risk engine (runs on every pipeline pass)

| Family | Method |
|---|---|
| stockout_risk | P(demand over lead time > position) under Normal(mu, σ); σ includes forecast-PI width **and supplier lead-time variability** |
| overstock | days-of-cover ≥ 90 → capital-tied estimate |
| demand_spike | Welch's t-test level shift (last 21d vs prior 60d, p ≤ 0.05) |
| supplier_lead_drift | recent-90d vs baseline lead times per (supplier, SKU) |
| abnormal_procurement | unit-price z-score ≥ 2.5σ vs the supplier-SKU PO history |
| supplier_reliability | on-time + contract-OTD blend, small-sample guard |

On the shipped snapshot (2025-12-31): 11 alerts — including `P104` demand
+36% (p = 0.005), two SKUs below safety stock inside their lead time, and
supplier `S2` running +37% late — and the optimizer reacts automatically
(drift-widened lead times, de-ranked suppliers, paused replenishment).

## Procurement optimizer

Periodic-review policy, fully transparent:

```
gross requirement = Σ forecast over (effective lead time + review period)
safety stock      = z(service level) × σ(gross)      # PI width + lead-time spread
net requirement   = gross + safety stock − (on hand + on order)
order quantity    = ceil(net / pack) × pack
supplier          = argmin 0.45·price + 0.35·(1−OTD) + 0.20·lead
                    + drift penalty + reliability-floor penalty + unproven penalty
order by          = projected stockout date − lead time − review period
```

Actions: **BUY**, **EXPEDITE** (stock breaks before arrival), **HOLD**.

## Explainability — two questions, kept separate

- *"Why did the model forecast this demand?"* → SHAP. Global mean |SHAP|
  per feature for tree winners, plus per-day local attributions (top
  positive / negative drivers in units of demand) for the horizon's inputs.
  Without SHAP installed, permutation importance is used — same surface.
- *"Why did the system recommend buying this quantity?"* → the exact
  additive decomposition `forecast demand + safety stock − inventory
  position (+ pack rounding)`, rendered in units and percentages, with a
  unit test asserting the components reconcile to the shipped BUY number.

## Grounded GenAI Copilot

RAG-shaped, but the retrieval source is the analytics warehouse itself:
question → intent rules + SKU regex → SQL/artifact context pack → (optional)
LLM re-wording under a strict grounding prompt → answer **plus the exact
sources used**. Unknown intents are refused, never hallucinated. Works with
any OpenAI-compatible endpoint; without a key it runs the deterministic
narrator with identical numbers. Tests never require an API key.

## MLOps

| Capability | Implementation |
|---|---|
| Experiment tracking | `ExperimentTracker` — MLflow when available (`MLFLOW_TRACKING_URI`), JSON-lines fallback otherwise; params, fold metrics, artifacts, tags per (SKU × model) run |
| Model registry | versioned metadata per SKU: model, dataset/feature/code versions, metrics, selection reason, status lifecycle `candidate → production → archived` |
| Promotion gates | validation passed · no DQ errors · artifacts generated · candidate WAPE ≤ production × (1 + 5%) — configurable; failed gates keep the candidate, success archives the incumbent |
| Retraining policy | triggers on WAPE threshold/degradation, drift, coverage loss, stale data, new-data volume, schedule; returns `retrain_required` + reasons; never auto-deploys |
| Drift detection | PSI (5-bin, small-sample-safe) + KS on demand-level features, last 90d vs previous 90d |
| PI coverage monitoring | empirical 80% coverage per SKU + portfolio, with a documented (never silently applied) recalibration hint |
| Data-quality monitoring | ERROR/WARNING sweep incl. schema drift, staleness, domain rules → `data_quality_report.json` |
| Business KPIs | every metric labelled **observed / estimated / simulated** — no invented savings |
| Model health | one status object (health, drift, coverage, DQ) served at `/api/model-health` and visualised in the dashboard |
| Provenance | dataset manifest with content-hash version, embedded in every run/model/report |

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` · `GET /ready` | liveness · readiness (store, artifacts, registry) |
| `GET /api/kpis` | Executive KPIs + selection narrative |
| `GET /api/inventory` | Days of cover, value, status per SKU |
| `GET /api/forecast/{sku}` | History + 30-day forecast + per-model back-test |
| `GET /api/models` | Bake-off results |
| `GET /api/risk` | Active alerts with evidence |
| `GET /api/suppliers` | OTD, lead-time drift, spend |
| `GET /api/recommendations` | BUY/HOLD/EXPEDITE + explanations |
| `GET /api/recommendations/{sku}/explain` | BUY decomposition + SHAP forecast drivers |
| `POST /api/simulate` | What-If scenario re-plan |
| `POST /api/copilot` | Grounded Q&A |
| `GET /api/model-health` | Model health + retraining recommendation |
| `GET /api/data-quality` | DQ report + dataset manifest |
| `GET /api/business-kpis` | Observed / estimated / simulated KPIs |
| `GET /api/experiments` | Walk-forward evaluation summary + registry |

Full request/response documentation: [`docs/api.md`](docs/api.md) ·
OpenAPI at `/docs`.

## Dashboard

Nine views: Overview, Forecast, Inventory Health, Suppliers, Risk Monitor,
Recommendations, What-If Simulator, **Model Ops** (model health, data
health, walk-forward experiment table, business impact with honesty badges)
and AI Copilot. Dependency-light vanilla JS + vendored ECharts; responsive
down to 390 px.

## Testing

```
pytest                # unit + integration
```

81 tests across: metrics (incl. bias), cleaning rules, validation gates,
leakage-free features, walk-forward splits (temporal ordering, no leakage,
min-train enforcement, zero-actual handling, missing dates, determinism),
composite scoring (tie handling), experiment tracking (incl. graceful
degradation), registry lifecycle + promotion gates, retraining triggers,
data quality (ERROR/WARNING), drift (PSI/KS math), PI coverage, model
health status logic, business-KPI labelling, SHAP integration, optimizer
math, simulator physics, explanation reconciliation, copilot grounding,
and the API. CI runs them on SQLite and PostgreSQL with no API keys.

## Docker

```bash
docker compose up --build            # API (with PostgreSQL) on :8000
docker compose --profile mlflow up   # + MLflow tracking server on :5000
```

The API container initialises the analytics store on boot
(`scripts/init_store.py`) and serves dashboard + API. Healthchecks on
`/health`; non-root user; no secrets baked in — see `.env.example`.

## Local setup

```bash
pip install -r requirements.txt

# 1. regenerate the raw dataset (optional — data/ is committed)
python scripts/generate_data.py

# 2. full training workflow (~4 min): data → validation → data quality →
#    walk-forward bake-off → tracking → selection → registry → forecasts →
#    SHAP → risk → recommendations → business KPIs → monitoring
python scripts/train.py

# 3. serve API + dashboard
uvicorn supplychainxai.api.app:app --port 8000
# -> http://localhost:8000        dashboard
# -> http://localhost:8000/docs   OpenAPI

# standalone evaluation / monitoring
python scripts/evaluate.py
python scripts/monitor.py
```

SQLite + JSON tracking work with zero configuration. Production-like mode:

```bash
export DATABASE_URL=postgresql://user:pass@host:5432/scx   # or use docker compose
export MLFLOW_TRACKING_URI=http://localhost:5000
cp .env.example .env   # optional: LLM copilot + tuning knobs
```

## Project structure

```
supplychainxai/
├── config/          # scenario constants + env-driven settings
├── data/            # ingest → clean → validate → features → store (SQLite/PG)
├── evaluation/      # WalkForwardBacktester + composite scoring
├── forecasting/     # model zoo + metrics + walk-forward selection
├── risk/            # 6-family alert engine
├── optimization/    # order quantities, supplier scoring, what-if simulator
├── explain/         # additive BUY decomposition + SHAP bridge + NLG
├── copilot/         # intent → SQL retrieval → grounded answer (LLM optional)
├── monitoring/      # data quality, drift, PI coverage, model health, business KPIs
├── mlops/           # experiment tracking, registry, retraining policy, provenance
└── api/             # FastAPI service + executive dashboard
scripts/             # generate_data, train, run_pipeline, evaluate, monitor, init_store
tests/               # unit/ + integration/ (81 tests)
artifacts/           # committed pipeline outputs (metrics, forecasts, alerts, registry…)
data/                # committed raw + processed dataset + analytics.db + manifest
docs/                # architecture, api, results, screenshots
```

## Design decisions worth noting

- **Censored demand**: stockout days under-report true demand; MAPE excludes
  zero-actual days and WAPE is the selection metric for this reason.
- **Drift-aware lead times**: planning on *quoted* lead times while POs run
  40% late is self-deception — effective lead times widen with observed
  drift, and drifting suppliers get de-ranked with an explicit penalty.
- **Grounding by construction**: the copilot's context pack is built from
  the same SQL/tables the dashboard renders; unknown intents are refused.
- **Two explanations, two questions**: SHAP explains the forecast; the
  additive decomposition explains the BUY. They never blur together.
- **Adjacent-window drift**: comparing the last 90 days against the
  *previous* 90 days catches actionable level shifts without flagging
  ordinary year-on-year growth as drift.
- **Honesty labels on money**: the dataset is synthetic, so every
  financial figure is labelled observed / estimated / simulated.

## Limitations

- **Synthetic data** (documented, seeded generator) — swap
  `scripts/generate_data.py` for a real extract; the pipeline needs the same
  six tables. All financial values are synthetic-data units, not dollars.
- PIs are residual-Gaussian; empirical coverage is currently ~49% against
  the 80% target on the hardest window — measured, reported and flagged by
  the monitor rather than assumed, with a recalibration hint surfaced.
- Models under-forecast censored demand slightly (negative bias, visible in
  every results table).
- The MLflow Model Registry mirror is best-effort; the JSON registry is the
  source of truth.
- Single-echelon inventory; no budget constraints.

## Roadmap

- [ ] Conformal / empirically calibrated prediction intervals (the monitoring hint shows the needed σ scale)
- [ ] Bias correction for stockout-censored demand
- [ ] Hierarchical forecasting (category → SKU reconciliation)
- [ ] Multi-echelon inventory (warehouse + store)
- [ ] scheduled retraining orchestration (cron / Airflow) around `scripts/train.py`
- [ ] Champion–challenger shadow scoring before promotion

## Author

**Mohammad Kaif** — B.Tech CSE (Data Science)
[GitHub](https://github.com/Kaif-Kenwey) ·
[LinkedIn](https://linkedin.com/in/kaif-kenwey) ·
mdkaifmaharaja@gmail.com

*Stack: Python · Pandas · NumPy · SQL (SQLite/PostgreSQL) · Scikit-learn ·
Statsmodels · XGBoost · SHAP · MLflow · FastAPI · Docker · ECharts · pytest ·
GitHub Actions*

License: [MIT](LICENSE)
