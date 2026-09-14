"""Business KPIs — the metrics a head of procurement actually asks about.

Honesty contract: every metric carries a `basis` field.

  observed   — computed directly from transactional data (inventory, POs,
               sales). No assumptions.
  estimated  — depends on a model output (forecast, probability) or a
               documented approximation.
  simulated  — additionally assumes a cost/price heuristic; treat the number
               as directional, not financial truth.

No invented savings: synthetic-data costs make any currency figure
simulated by definition, and the label says so.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from supplychainxai import config
from supplychainxai.mlops.runlog import utc_now
from supplychainxai.optimization.engine import Recommendation
from supplychainxai.risk.engine import effective_lead, lead_time_intelligence

EXCESS_DOH = 90.0  # days of cover above which stock counts as excess
CARRYING_COST_RATE = 0.20  # annualised holding cost assumption (SIMULATED)


def _m(key: str, value, basis: str, unit: str, definition: str) -> dict:
    return {"key": key, "value": value, "basis": basis, "unit": unit, "definition": definition}


def observed_kpis(data) -> list[dict]:
    """From transactions only: inventory value, POs, OTD, stockout days,
    in-stock rate, working capital tied in inventory."""
    products = data.products.set_index("sku")
    inv = data.inventory.sort_values("date")
    latest = inv.groupby("sku").tail(1)

    carrying_value = float((latest["on_hand"] * latest["sku"].map(products["unit_cost"])).sum())
    open_pos = data.purchase_orders[data.purchase_orders["status"] == "OPEN"]
    open_spend = float((open_pos["quantity"] * open_pos["unit_price"]).sum())

    po = data.purchase_orders[data.purchase_orders["status"] == "DELIVERED"].copy()
    po["on_time"] = po["delivered_date"] <= po["expected_date"]
    otd = float(po["on_time"].mean()) if len(po) else None

    # stockout days / in-stock rate over the trailing 90 days (OBSERVED)
    cutoff = inv["date"].max() - pd.Timedelta(days=90)
    recent = inv[inv["date"] > cutoff]
    stockout_days = int((recent["on_hand"] <= 0).sum())
    total_rows = len(recent)
    in_stock_rate = 1.0 - stockout_days / total_rows if total_rows else None

    return [
        _m(
            "inventory_carrying_value",
            round(carrying_value, 2),
            "observed",
            "currency_units",
            "on-hand units × unit cost at snapshot",
        ),
        _m(
            "working_capital_tied_in_inventory",
            round(carrying_value, 2),
            "observed",
            "currency_units",
            "same asset base as carrying value, viewed as working capital",
        ),
        _m("open_po_count", len(open_pos), "observed", "count", ""),
        _m(
            "open_po_committed_spend",
            round(open_spend, 2),
            "observed",
            "currency_units",
            "open PO quantity × unit price",
        ),
        _m(
            "supplier_on_time_delivery",
            round(otd, 4) if otd is not None else None,
            "observed",
            "fraction",
            f"delivered POs on or before expected date (n={len(po)})",
        ),
        _m(
            "stockout_days_trailing_90d",
            stockout_days,
            "observed",
            "sku-days",
            "snapshot days with on_hand = 0 across all SKUs",
        ),
        _m(
            "in_stock_rate_trailing_90d",
            round(in_stock_rate, 4) if in_stock_rate is not None else None,
            "observed",
            "fraction",
            "1 − stockout sku-days / observed sku-days (demand-censored proxy)",
        ),
    ]


def estimated_kpis(data, forecasts: pd.DataFrame) -> list[dict]:
    """Depend on model outputs: days of cover, excess inventory, forecast bias."""
    products = data.products.set_index("sku")
    latest = data.inventory.sort_values("date").groupby("sku").tail(1).set_index("sku")
    daily = forecasts.groupby("sku")["prediction"].mean()

    doh_portfolio, excess_value = [], 0.0
    for sku, row in latest.iterrows():
        d = float(daily.get(sku, 0.0))
        if d <= 0:
            continue
        doh = float(row["on_hand"]) / d
        doh_portfolio.append(doh)
        if doh > EXCESS_DOH:
            excess_value += (doh - EXCESS_DOH) * d * float(products.loc[sku, "unit_cost"])

    return [
        _m(
            "inventory_days_of_cover",
            round(float(np.mean(doh_portfolio)), 1) if doh_portfolio else None,
            "estimated",
            "days",
            f"portfolio mean on-hand ÷ forecast daily demand (excess threshold {EXCESS_DOH:.0f}d)",
        ),
        _m(
            "excess_inventory_value",
            round(excess_value, 2),
            "estimated",
            "currency_units",
            f"value of stock beyond {EXCESS_DOH:.0f} days of forecast cover",
        ),
    ]


def simulated_kpis(
    data, forecasts: pd.DataFrame, recommendations: list[Recommendation]
) -> list[dict]:
    """Assume heuristics: stockouts avoided, procurement savings opportunity,
    carrying-cost impact. Directional only."""
    lt_intel = lead_time_intelligence(data.purchase_orders)
    daily = forecasts.groupby("sku")["prediction"].mean()

    stockouts_avoided = 0.0
    savings_opportunity = 0.0
    for rec in recommendations:
        if rec.action not in ("BUY", "EXPEDITE"):
            continue
        lead, _, _ = effective_lead(rec.sku, data.supply_terms, lt_intel)
        d = float(daily.get(rec.sku, 0.0))
        if d > 0:
            # units of demand protected by replenishing before the stockout date
            stockouts_avoided += min(d * lead, rec.quantity)
        if rec.runner_up and rec.supplier_id:
            alt_price = float(rec.runner_up["unit_price"])
            if alt_price < rec.unit_price:
                savings_opportunity += (rec.unit_price - alt_price) * rec.quantity

    carrying_impact = (
        sum(
            r.quantity * float(r.unit_price)
            for r in recommendations
            if r.action in ("BUY", "EXPEDITE")
        )
        * CARRYING_COST_RATE
    )

    return [
        _m(
            "estimated_stockout_units_avoided",
            round(stockouts_avoided, 1),
            "simulated",
            "units",
            "Σ min(daily demand × effective lead, order qty) over active BUY/EXPEDITE "
            "— assumes orders arrive before the projected stockout date",
        ),
        _m(
            "procurement_savings_opportunity",
            round(savings_opportunity, 2),
            "simulated",
            "currency_units",
            "Σ (chosen price − runner-up price) × qty where the runner-up is cheaper — "
            "ignores switching cost, capacity and risk",
        ),
        _m(
            "new_order_carrying_cost_pa",
            round(carrying_impact, 2),
            "simulated",
            "currency_units",
            f"new order value × {CARRYING_COST_RATE:.0%} annual holding-cost assumption",
        ),
        _m(
            "supplier_lead_time_reliability_note",
            "lead-time drift alerts feed supplier choice automatically",
            "observed",
            "text",
            "cross-reference: risk engine R4/R6 alerts",
        ),
    ]


def compute_business_kpis(
    data, forecasts: pd.DataFrame, recommendations: list[Recommendation]
) -> dict:
    kpis = (
        observed_kpis(data)
        + estimated_kpis(data, forecasts)
        + simulated_kpis(data, forecasts, recommendations)
    )
    out = {
        "generated_at": utc_now(),
        "snapshot_date": str(data.sales["date"].max().date()),
        "basis_legend": {
            "observed": "computed from transactional data — no assumptions",
            "estimated": "depends on model output or documented approximation",
            "simulated": "assumes a cost heuristic on synthetic data — "
            "directional, not financial truth",
        },
        "kpis": kpis,
        "index": {k["key"]: k["value"] for k in kpis},
    }
    (config.ARTIFACTS / "business_kpis.json").write_text(json.dumps(out, indent=2))
    return out
