# Architecture

## 1. Data pipeline (phases 1–2)

```
raw CSVs ──> ingest ──> clean ──> validate ──> features ──> SQLite store
```

- **ingest** (`supplychainxai/data/ingest.py`): schema-enforced loading —
  every column coerced to its expected dtype, dates parsed tolerantly.
- **clean** (`clean.py`): rule-based repair with a full audit trail
  (`cleaning_report.json`): duplicate removal, negative-quantity handling,
  SKU normalisation, PO date-swap correction, open-PO status derivation.
- **validate** (`validate.py`): 15 business gates (referential integrity,
  domain rules, unique keys, one-primary-supplier-per-SKU…). CRITICAL
  failures abort the pipeline; the report is persisted.
- **features** (`features.py`): leakage-safe supervised frame per SKU/day —
  lags (1/7/14/28), shifted rolling mean/std (7/28), calendar features,
  promo flag, price, and a stockout-censoring indicator.
- **store** (`store.py`): dim/fact star-ish schema in SQLite
  (`data/processed/analytics.db`), queried with real SQL by the copilot.

## 2. Forecasting (phase 3)

- **Model zoo** (`forecasting/models.py`): Moving Average (28d),
  Holt-Winters ES (additive trend + weekly seasonality), SARIMA(1,1,1)x(1,0,1)7,
  Random Forest, Gradient Boosting, XGBoost (auto-detected if installed).
  Tree models forecast recursively: predicted values feed back as lags;
  promos are assumed 0 and price held constant on the horizon.
- **Selector** (`selector.py`): back-test on the final 90 days; per-SKU
  winner by WAPE; portfolio narrative generated from the actual results
  table. Final refit on full history produces the 30-day forecast with
  80% prediction intervals (empirical residual sigma).
- **Feature importance**: permutation importance on the hold-out window
  for the winning tabular model per SKU.

## 3. Risk engine (phase 4)

Six alert families (`risk/engine.py`), each alert carrying its evidence:

| Family | Method |
|---|---|
| stockout_risk | P(demand over lead time > position) under Normal(mu_LT, sigma_LT); sigma includes forecast-PI width **and supplier lead-time variability**; severity by probability and days-of-cover |
| overstock | days-of-cover >= 90 → capital-tied estimate |
| demand_spike | Welch's t-test level shift: last 21d vs prior 60d, p <= 0.05 and lift >= 12% |
| supplier_lead_drift | recent-90d vs baseline lead times per (supplier, SKU); needs >= 2 recent POs and >= 1.5 days absolute delta |
| abnormal_procurement | unit price z-score vs the supplier-SKU PO history (>= 2.5σ) |
| supplier_reliability | on-time rate + contract OTD blend, floor 0.70, small-sample guard (>= 5 delivered POs) |

`lead_time_intelligence()` publishes per-(SKU, supplier) drift; the risk
engine and the optimizer both plan on **effective (drift-adjusted) lead
times** rather than quoted ones.

## 4. Procurement optimizer (phase 5)

Periodic-review policy per SKU (`optimization/engine.py`):

```
gross requirement = sum(forecast over lead time + review period)
safety stock      = z(service level) × sigma(gross)   # PI-decomposed + lead-time spread
net requirement   = gross + safety − (on hand + on order)
order quantity    = ceil(net / pack) × pack
supplier          = argmin  0.45·price + 0.35·(1−OTD) + 0.20·lead
                    with drift penalty (+0.15), reliability-floor penalty (+0.10),
                    unproven-supplier penalty (+0.05 when no PO history)
order-by          = projected stockout date − lead time − review period
```

Actions: **BUY** (normal replenishment), **EXPEDITE** (stockout precedes
arrival), **HOLD**.

## 5. Explainability (phase 6)

Two surfaces (`explain/engine.py`):

1. **Forecast explanations** — permutation-importance shares plus a
   counterfactual read: forecast level vs trailing-14/28-day momentum,
   promo assumption, weekly shape.
2. **Recommendation explanations** — the BUY quantity decomposed into its
   exact additive components (demand / safety stock / inventory position /
   rounding), each with units and % impact, then rendered as natural
   language. A unit test asserts the components reconcile to the shipped
   quantity, so explanations can never drift from the math.

## 6. Copilot (phase 7)

RAG-shaped, but the retrieval source is the analytics warehouse:

```
question → intent rules + SKU regex → SQL / artifact retrieval → context pack
         → LLM re-wording (optional, strict system prompt) | deterministic narrator
         → answer + sources
```

Grounding rules: the LLM may only use context numbers; unknown intents get
an honest refusal; every answer cites the queries/tables used. Works with
any OpenAI-compatible endpoint; falls back to the narrator without one.

## 7. Service & dashboard (phase 8)

- **FastAPI** (`api/app.py`): loads committed artifacts at startup (< 2 s),
  serves the dashboard and REST API, CORS-open for local tooling.
- **Dashboard** (`api/static/`): dependency-light vanilla JS + vendored
  ECharts; 8 views (Overview, Forecast, Inventory, Suppliers, Risk,
  Recommendations, What-If Simulator, Copilot); responsive down to 390 px.
- **What-If Simulator**: `simulate_scenario()` re-runs the optimizer under
  demand / lead-time / inventory / service-level shifts and returns
  baseline vs scenario stockout probabilities, coverage, cost and supplier
  switches per SKU.

## Deliberate limitations

- Synthetic data (documented generator) — swap `scripts/generate_data.py`
  for a real extract; the pipeline only needs the same six tables.
- Single-echelon inventory, no budget constraints yet.
- PIs are residual-Gaussian, not bootstrapped.
