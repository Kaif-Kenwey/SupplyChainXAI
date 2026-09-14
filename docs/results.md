# Results

All numbers below are produced by `python scripts/train.py` on the committed
dataset (seeded generator — fully reproducible). Raw tables:
`artifacts/evaluation/`, `artifacts/*.json`.

## Dataset

| | |
|---|---|
| Window | 2023-01-01 → 2025-12-31 (1,095 days) |
| SKUs | 12 industrial components, 5 categories |
| Suppliers | 6 (2–3 competing sources per SKU) |
| Raw sales rows | 13,162 (13,133 after cleaning) |
| Inventory snapshots | 13,152 |
| Purchase orders | 938 (10 open at snapshot) |
| Injected data-quality issues | ~800 cells (duplicates, negatives, NaNs, casing, swapped PO dates) — all caught by cleaning + validation |
| Validation | 15 checks — PASS |
| Data-quality sweep | 1 warning (10 open POs legitimately lack a delivered date); 0 errors |
| Dataset version | `2ab9ece20309` (content hash; recorded in every run/model/report) |

## Evaluation methodology

- **OLD** (v1.x): one hold-out — the final 90 days, single train/predict
  pass per model.
- **NEW** (v2.x): walk-forward, rolling-origin — 4 expanding-window folds ×
  30-day horizon (min training window 365 days, configured via
  `SCX_EVAL__*`). Every fold's metrics are stored per model and per SKU;
  selection uses the mean across all folds with a composite of normalised
  WAPE/MAE/RMSE (0.5/0.3/0.2).

## Model bake-off — walk-forward (mean of per-SKU means across 4 folds)

| Model | Mean WAPE ↓ | Worst SKU | σ (per-SKU means) | Mean MAE | Mean bias | SKU wins | Unstable SKUs |
|---|---|---|---|---|---|---|---|
| **XGBoost** | **12.15%** | 16.29% | 2.73 | 8.13 | −3.22 u | 4 | 0 |
| Random Forest | 12.24% | 16.82% | 2.65 | 8.07 | −3.82 u | 4 | 0 |
| Gradient Boosting | 12.42% | 17.45% | 2.59 | 8.36 | −2.71 u | 4 | 0 |
| SARIMA (1,1,1)x(1,0,1)7 | 14.33% | 17.55% | 4.01 | 9.41 | −2.29 u | 0 | 0 |
| Holt-Winters ES | 14.66% | 19.12% | 4.79 | 9.49 | −1.31 u | 0 | 2 |
| Moving Average (28d) — baseline | 33.75% | 38.45% | 2.64 | 22.21 | −3.36 u | 0 | 0 |

**Interpretation.**

- The win split is now **4 / 4 / 4** between XGBoost, Random Forest and
  Gradient Boosting — no single family owns the portfolio. Under the old
  single hold-out XGBoost won 5 of 12; under honest multi-fold evaluation
  the "one best model" story disappears, which is exactly the failure mode
  rolling-origin evaluation exists to expose.
- All models show a **negative bias** (−1.3 to −3.8 units/day): systematic
  under-forecasting on stockout-censored days. Reported, not hidden; a
  bias-correction layer is on the roadmap.
- Two Holt-Winters/SKU combinations are flagged **unstable** (fold-to-fold
  WAPE coefficient of variation > 0.5) — visible in
  `walk_forward_per_sku.csv`, not averaged away.

## Old hold-out vs new walk-forward (same models, same data)

| Model | Old 90-day hold-out WAPE | New walk-forward mean WAPE |
|---|---|---|
| XGBoost | 15.57% | 12.15% |
| Random Forest | 15.96% | 12.24% |
| Gradient Boosting | 16.29% | 12.42% |
| Holt-Winters ES | 18.97% | 14.66% |
| SARIMA | 19.16% | 14.33% |
| Moving Average (28d) | 35.73% | 33.75% |

The walk-forward numbers are *better* because each fold trains on more
history than the single 90-day window allowed (folds start after a
365-day minimum window and expand). Both tables are kept deliberately so
the effect of the evaluation protocol itself is visible. Per-fold spread
(σ ≈ 2.6–4.8 WAPE points across SKUs) shows why single-window results
should not be quoted to two decimal places and treated as gospel.

## Per-SKU winners (walk-forward, composite score)

| SKU | Selected model | Mean WAPE | σ | Note |
|---|---|---|---|---|
| P101 | Random Forest | 9.51% | 3.48 | stable |
| P102 | XGBoost | 11.14% | 2.88 | stable |
| P103 | Random Forest | 11.42% | 3.74 | stable |
| P104 | XGBoost | 14.59% | 3.92 | demand-spike SKU; hardest window sits in the last fold |
| P105 | Gradient Boosting | 9.77% | 1.56 | stable |
| P106 | XGBoost | 16.29% | 2.28 | overstock SKU (121 days cover) |
| P107 | Gradient Boosting | 11.65% | 3.17 | stable |
| P108 | XGBoost | 11.22% | 2.10 | stable |
| P109 | Random Forest | 11.19% | 0.26 | most stable model/SKU pair in the portfolio |
| P110 | Gradient Boosting | 12.94% | 3.03 | supplier S2 lead drift +37% |
| P111 | Random Forest | 11.53% | 3.58 | stable |
| P112 | Gradient Boosting | 11.05% | 2.58 | stable |

(exact values in `artifacts/evaluation/walk_forward_per_sku.csv`, including
per-fold metrics, medians and per-metric standard deviations)

## Model health on the shipped snapshot

`artifacts/monitoring/model_health.json`, served at `GET /api/model-health`:

| Signal | Value | Reading |
|---|---|---|
| WAPE — most recent fold | 17.54% | hardest window: P104's +36% spike sits there |
| WAPE — previous folds | 16.27% | degradation +7.8% — inside the 25% trigger, no error alarm |
| WAPE — MA baseline | 33.75% | the model still halves the naive baseline |
| PI coverage (80% nominal) | 49.3% | **under-covering** — flagged, with σ-scale hint ≈ 1.9× |
| Drift (lag features, PSI/KS) | critical | genuine demand level shifts (P104 spike; P105/P106 up-trend); P103 healthy |
| Data quality | WARN | 1 warning (open POs without delivery date), 0 errors |
| Overall status | CRITICAL | honest: drift + miscalibration are real; recommendation = "Retrain: drift; coverage" |

This is the monitoring layer working as designed: the shipped snapshot
*genuinely contains a demand regime shift and under-covering intervals*,
and the system says so instead of reporting a green dashboard. Note the
interval-miscalibration contribution is capped at *warning* severity by
policy — it calls for recalibration, not model replacement (the drift and
coverage triggers are what drive the retrain recommendation).

## Snapshot KPIs (2025-12-31)

| KPI | Value |
|---|---|
| Forecast demand (30d, all SKUs) | 26,196 units |
| Inventory value (on-hand at cost) | $664,208.50 — *observed* |
| Open POs | 10 ($818,713.45 committed) — *observed* |
| Supplier OTD (delivered POs) | 65.8% — *observed* |
| In-stock rate (trailing 90d) | 93.2% SKU-days (73 stockout SKU-days) — *observed* |
| Buy actions | 4 — 1,560 units, $19,603.20 |
| Excess inventory (above 90d cover) | $41,667.74 — *estimated* |
| Est. stockout units avoided (active buys) | 1,427 units — *simulated* |
| Procurement savings opportunity (runner-up prices) | $278.60 — *simulated* |

Every business metric carries its basis label (observed / estimated /
simulated) in the API and dashboard. Currency values are synthetic-data
units.

## Risk findings on the shipped snapshot

11 alerts (3 CRITICAL, 8 WARNING) across all six families:

- **Demand spike (CRITICAL)** — `P104` running +36% vs baseline
  (145/d vs 107/d, Welch p = 0.005).
- **Stockout risk (CRITICAL ×2)** — `P101` (~9–10 days of cover vs 5-day
  effective lead), `P102`.
- **Supplier lead drift (WARNING ×3)** — `S2`/`P110` +37% (17d→24d),
  `S4`/`P105` +25%, `S5`/`P103` +24%.
- **Abnormal procurement (WARNING ×2)** — two `P104` POs from `S4` priced
  2.7σ and 3.7σ above their 78-PO mean.
- **Overstock (WARNING)** — `P106` at 121 days of cover, ~$163,379 capital
  tied up; replenishment paused.
- **Supplier reliability (WARNING)** — `S2` at 48% on-time across 166 POs
  (floor 70%).

Downstream consequences (all automatic): the optimizer de-ranks drifting
suppliers, `P104`'s plan uses the drift-widened 15-day effective lead time
instead of the quoted 12, and `P106` gets HOLD.

## What-If simulator (Q4 stress: demand +20%, lead +30%)

| Metric | Baseline → Scenario |
|---|---|
| Procurement required | → recomputed per SKU (pack-rounded) |
| Avg stockout probability | rises sharply across the portfolio |
| Supplier switches | flagged per SKU ("was Sx") |
| Coverage | recomputed post-replenishment |

(Sim values are live — hit `POST /api/simulate` or move the sliders.)

## Experiment tracking & registry

- 72 tracked runs per training pass (12 SKUs × 6 models): parameters,
  fold metrics, tags (`dataset_version`, `git_commit`, `environment`) and
  forecast artifacts — to MLflow (sqlite backend, `mlflow.db`) when
  available, JSON-lines (`artifacts/experiments/runs.jsonl`) otherwise.
- 12 registered model versions (one per SKU), each with training
  timestamp, dataset/feature/code versions, selection reason and gates;
  12 promoted to production on this pass (all gates passed).

## Test suite

```
81 passed — metrics (incl. bias) · cleaning · validation · features (leakage)
            walk-forward splits · leakage guards · fold determinism
            composite scoring & tie handling · tracker degradation
            registry lifecycle · promotion gates · retraining triggers
            data-quality gates · drift (PSI/KS) · PI coverage
            model-health status logic · business-KPI labelling · SHAP
            risk triggers · optimizer math · simulator physics
            explanation reconciliation · copilot grounding · API
            (+ PostgreSQL store round-trip when a server is provided)
```

## Reproducibility

- fixed-seed dataset generator; content-hash dataset version
  (`2ab9ece20309`) recorded in every artifact;
- deterministic model training (fixed `random_state`);
- deterministic selection (alphabetical tie-breaking);
- every run logs `run_id`, `git_commit`, dataset version, timestamps;
- re-running `python scripts/train.py` reproduces the same winners,
  forecasts and BUY quantities.
