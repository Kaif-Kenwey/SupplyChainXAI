# Results

All numbers below are produced by `python scripts/run_pipeline.py` on the
committed dataset (seeded generator — fully reproducible).

## Dataset

| | |
|---|---|
| Window | 2023-01-01 → 2025-12-31 (1,095 days) |
| SKUs | 12 industrial components, 5 categories |
| Suppliers | 6 (2–3 competing sources per SKU) |
| Raw sales rows | 13,162 (13,133 after cleaning) |
| Inventory snapshots | 13,152 |
| Purchase orders | 923 (10 open at snapshot) |
| Injected data-quality issues | ~800 cells (duplicates, negatives, NaNs, casing, swapped PO dates) — all caught by cleaning + validation |
| Validation | 15 checks — PASS |

## Model bake-off

Hold-out: final 90 days per SKU; aggregated over 12 SKUs.

| Model | MAE ↓ | RMSE ↓ | MAPE ↓ | WAPE ↓ | SKU wins |
|---|---|---|---|---|---|
| **XGBoost** | **10.25** | **16.99** | **15.94%** | **15.57%** | 5 |
| Random Forest | 10.51 | 17.30 | 15.85% | 15.96% | 2 |
| Gradient Boosting | 10.83 | 17.19 | 17.12% | 16.29% | 4 |
| Holt-Winters ES | 12.31 | 18.36 | 25.02% | 18.97% | 0 |
| SARIMA (1,1,1)x(1,0,1)7 | 12.59 | 19.04 | 23.50% | 19.16% | 1 |
| Moving Average (28d) | 23.97 | 29.20 | 61.13% | 35.73% | 0 |

**Interpretation.** The best model cuts the naive baseline's error by 56%
(WAPE). Tree ensembles dominate because demand carries promo/price/calendar
interactions that linear/statistical baselines cannot express; SARIMA still
wins one SKU with clean weekly structure — evidence that running a bake-off
per SKU (instead of assuming one global model) pays for itself. The spike
window raises every model's error on `P104` (WAPE ~27% there vs ~15%
portfolio-wide); that is honest difficulty, not model failure.

## Snapshot KPIs (2025-12-31)

| KPI | Value |
|---|---|
| Forecast demand (30d, all SKUs) | 26,239 units |
| Inventory value (on-hand at cost) | $664,208.50 |
| Open POs | 10 ($818,713.45 committed) |
| Supplier OTD (delivered POs) | 65.8% |
| Buy actions | 4 — 1,720 units, $33,180.80 |

## Risk findings on the shipped snapshot

11 alerts (3 CRITICAL, 8 WARNING) across all six families:

- **Demand spike (CRITICAL)** — `P104` running +36% vs baseline
  (145/d vs 107/d, Welch p = 0.005).
- **Stockout risk (CRITICAL ×2)** — `P101` (~9–10 days of cover vs 5-day
  effective lead), `P102`.
- **Supplier lead drift (WARNING ×3)** — `S2`/`P110` +37% (17d→24d),
  `S4`/`P105` +25%, `S5`/`P103` +24%.
- **Abnormal procurement (WARNING ×2)** — two `P104` POs from `S4` priced
  2.7σ and 3.7σ above their 78-PO mean ($65.36 / $67.95 vs $58.52).
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

## Test suite

```
36 passed — metrics · cleaning · validation · features (leakage)
            risk triggers · optimizer math · simulator physics
            explanation reconciliation · copilot grounding · API
```
