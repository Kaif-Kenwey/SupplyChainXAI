# API Reference

Base URL (local): `http://localhost:8000` — start with
`uvicorn supplychainxai.api.app:app --port 8000`.
Interactive OpenAPI docs: `/docs` · ReDoc: `/redoc`.

All analytics endpoints are read-only `GET`s served from artifacts loaded at
startup; the two write-shaped endpoints (`/api/simulate`, `/api/copilot`)
accept JSON bodies validated by Pydantic. The service never exposes secrets;
`DATABASE_URL` and API keys stay server-side.

Conventions:

* dates are `YYYY-MM-DD`;
* money values are synthetic-data units (never real currency);
* `404` with `{"detail": ...}` when a resource is unknown or artifacts are
  not yet generated;
* `422` with FastAPI validation detail for invalid request bodies.

---

## Service

### `GET /health`

Liveness. Process is up and answering.

```json
{"status": "ok", "version": "2.0.0", "llm_mode": "grounded-narrator"}
```

`llm_mode` is `llm` when an OpenAI-compatible key is configured, otherwise
`grounded-narrator` (deterministic copilot).

### `GET /ready`

Readiness — verifies the dependencies needed to answer analytical questions.

```json
{
  "status": "ready",
  "checks": {"analytics_store": "ok", "artifacts": "ok", "model_registry": "ok"},
  "version": "2.0.0"
}
```

Check values are `ok` / `empty` / `error: <ExceptionType>`; connection
strings are never echoed.

---

## Analytics (existing, unchanged contracts)

### `GET /api/kpis`

Executive KPIs + the model-selection narrative.

```json
{
  "snapshot_date": "2025-12-31",
  "inventory_value": 664208.5,
  "open_po_count": 10,
  "open_po_spend": 818713.45,
  "supplier_on_time_pct": 65.8,
  "skus_tracked": 12,
  "buy_actions": 4,
  "buy_units": 1560,
  "buy_cost": 19603.2,
  "forecast_30d_demand": 26196.0,
  "model_selection": {"portfolio_best_model": "XGBoost", "mean_wape_by_model": {...},
                      "wins_by_model": {...}, "improvement_vs_moving_average_pct": 64.0,
                      "evaluation": {"type": "walk_forward_rolling_origin", "folds": 4},
                      "provenance": {"dataset_version": "2ab9ece20309", "git_commit": "..."},
                      "narrative": "..."}
}
```

### `GET /api/inventory`

Inventory health per SKU (days of cover vs forecast burn-rate, value,
status `OK|LOW|CRITICAL|OVERSTOCK`).

### `GET /api/forecast/{sku}?history_days=120`

History + 30-day forecast + per-model back-test table.

| Param | Range | Default |
|---|---|---|
| `history_days` | 30–365 | 120 |

```json
{
  "sku": "P104", "model": "XGBoost",
  "history": [{"date": "2025-09-03", "units": 96}, ...],
  "forecast": [{"date": "2026-01-01", "prediction": 141.2,
                "lower_80": 120.3, "upper_80": 162.1}, ...],
  "model_comparison": [{"model": "XGBoost", "MAE": 8.9, "RMSE": 14.2,
                        "MAPE": 11.4, "WAPE": 10.2, "WAPE_std": 2.9, ...}, ...]
}
```

Errors: `404` unknown SKU.

### `GET /api/models`

Full bake-off results (per-model portfolio aggregates, per-SKU table,
narrative). Under walk-forward evaluation the per-SKU numbers are
*means across folds*; `WAPE_std`, `mean_bias`, `n_folds` and `unstable`
columns expose fold-to-fold variability.

### `GET /api/risk`

Active risk alerts with evidence payloads (stockout probability, drift %,
z-scores…).

### `GET /api/suppliers`

On-time %, average and baseline lead times, drift %, spend per supplier.

### `GET /api/recommendations`

`BUY / EXPEDITE / HOLD` per SKU with quantity, supplier, cost, dates and
the additive explanation (`components`, `narrative`).

### `GET /api/recommendations/{sku}/explain`

Full explanation pack for one SKU. `forecast_explanation` answers *why the
model forecast this demand* (SHAP attributions when SHAP is installed,
permutation importance otherwise); `explanation` answers *why this BUY
quantity* (exact additive decomposition — unchanged by SHAP).

```json
{
  "sku": "P104",
  "explanation": {"components": [...], "drivers": [...], "narrative": "..."},
  "forecast_explanation": {"top_features": [...], "attribution_method": "shap",
                           "local_drivers": {"date": "2026-01-01",
                             "top_positive": [{"feature": "lag_1", "shap_units": 18.2,
                                               "feature_value": 96.0}], "top_negative": [...]}},
  "feature_importance": [...],
  "shap_importance": [{"sku": "P104", "feature": "lag_1", "importance": 4.21, "method": "shap"}]
}
```

Errors: `404` no explanation for the SKU.

---

## Simulator & Copilot

### `POST /api/simulate`

Re-plans the entire portfolio under a scenario.

```json
{"demand_pct": 0.2, "lead_time_pct": 0.3, "inventory_pct": -0.15, "service_level": 0.95}
```

Ranges: `demand_pct` −0.6…1.5 · `lead_time_pct` −0.5…2.0 ·
`inventory_pct` −0.9…1.0 · `service_level` 0.80…0.99. Response: scenario,
summary (procurement units/cost, buy actions, avg stockout probability,
avg coverage) and per-SKU deltas vs the baseline plan.

Errors: `422` body out of range.

### `POST /api/copilot`

```json
{"question": "Which products will stock out next month?"}
```

Response: `{"question", "intent", "answer", "sources": [...], "context": {...},
"engine": "narrator"|"llm"}`. Unknown intents get an honest refusal; the
LLM (when configured) may only re-word retrieved numbers. Tests never
require an API key.

Errors: `422` question shorter than 3 chars.

---

## MLOps

### `GET /api/model-health`

Production-model health assembled from the latest walk-forward fold
(observed performance), drift monitors, interval coverage, data quality
and the retraining policy.

```json
{
  "snapshot_date": "2025-12-31",
  "generated_at": "2026-09-14T15:26:19+00:00",
  "status": "warning",
  "model_version": "2ab9ece20309",
  "model_name": "XGBoost",
  "models_by_sku": {"P101": "Random Forest", "...": "..."},
  "last_trained": "2026-09-14T15:22:01+00:00",
  "metrics": {
    "wape_current_fold": 17.54,
    "wape_reference": 12.15,
    "wape_reference_previous_folds": 16.27,
    "wape_baseline": 33.75,
    "per_sku_wape_current_fold": {"P101": 14.96, "...": 0}
  },
  "drift": {"status": "critical", "by_sku": {...}, "detail": {...}},
  "interval_coverage": {"target_coverage": 0.8, "observed_coverage": 0.493,
                        "calibration_status": "warning", "sigma_scale_for_target": 1.929},
  "data_quality_status": "WARN",
  "retraining": {"retrain_required": true, "reasons": ["..."]},
  "recommendation": "Retrain: data/feature drift status = critical; ..."
}
```

Errors: `404` before the first `python scripts/train.py` run.

### `GET /api/data-quality`

Validation-style sweep with ERROR/WARNING separation, statistics and the
dataset manifest (provenance statement, version, row counts).

### `GET /api/business-kpis`

Business KPIs, each labelled with its honesty basis:

| Basis | Meaning |
|---|---|
| `observed` | computed from transactions only |
| `estimated` | depends on a model output or documented approximation |
| `simulated` | additionally assumes a cost heuristic on synthetic data |

```json
{"snapshot_date": "2025-12-31", "kpis": [
  {"key": "inventory_carrying_value", "value": 664208.5, "basis": "observed", ...},
  {"key": "excess_inventory_value", "value": 141983.1, "basis": "estimated", ...},
  {"key": "procurement_savings_opportunity", "value": 1545.0, "basis": "simulated", ...}],
 "index": {...}, "basis_legend": {...}}
```

### `GET /api/experiments`

Walk-forward evaluation summary: evaluation configuration, portfolio table
(mean/worst WAPE, σ across folds, bias, unstable-SKU counts), per-SKU
winners, registry summary (production versions per SKU).

### `POST /api/admin/recompute`

Intentionally `501` — the training pipeline is a CLI job
(`python scripts/train.py`), not an in-process request; artifacts are read
at startup.
