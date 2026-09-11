"""Phase 5 — Procurement recommendation engine.

For every SKU the engine answers: BUY or HOLD? How much? From which supplier?
By when? At what cost?

Method (classic periodic-review inventory policy, fully transparent):
    gross requirement  = forecast demand over the protection horizon
                         (supplier lead time + review period)
    safety stock       = z * sigma(demand over horizon), sigma from forecast PIs
                         widened by supplier lead-time variability
    net requirement    = max(0, gross + safety stock - inventory position)
    order quantity     = net requirement rounded UP to the supplier pack size
    supplier choice    = min effective cost score:
                             0.45 * unit price + 0.35 * (1 - reliability)
                             + 0.20 * lead time (all min-max normalised),
                         with a reliability floor and drift penalties
    order-by date      = expected stock-out date - supplier lead time
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from supplychainxai import config
from supplychainxai.risk.engine import lead_time_intelligence, sps_norm_cdf


@dataclass
class Recommendation:
    sku: str
    product_name: str
    action: str                          # BUY | HOLD | EXPEDITE
    quantity: int
    supplier_id: str | None
    supplier_name: str
    unit_price: float
    total_cost: float
    order_by: str | None
    expected_stockout_date: str | None
    coverage_days: float
    runner_up: dict | None = None
    drivers: list[dict] = field(default_factory=list)   # filled by explain layer


def _delivered(po: pd.DataFrame) -> pd.DataFrame:
    """Defensively filter delivered POs (status column added by the cleaner)."""
    po = po.copy()
    if "status" in po.columns:
        return po[po["status"] == "DELIVERED"]
    return po


def _supplier_stats(purchase_orders: pd.DataFrame) -> pd.DataFrame:
    po = _delivered(purchase_orders)
    if po.empty:
        return pd.DataFrame(columns=["supplier_id", "on_time_rate", "avg_lead"])
    po["lead"] = (po["delivered_date"] - po["order_date"]).dt.days
    po["on_time"] = po["delivered_date"] <= po["expected_date"]
    return po.groupby("supplier_id").agg(on_time_rate=("on_time", "mean"),
                                         avg_lead=("lead", "mean")).reset_index()


def choose_supplier(sku: str, data, sup_stats: pd.DataFrame,
                    reliability_target: float = config.SUPPLIER_RELIABILITY_FLOOR) -> tuple[dict | None, dict | None]:
    """Score all suppliers eligible for the SKU; return (best, runner-up)."""
    terms = data.supply_terms[data.supply_terms["sku"] == sku].copy()
    if terms.empty:
        return None, None
    names = data.suppliers.set_index("supplier_id")["name"].to_dict()

    drift = {}
    po = _delivered(data.purchase_orders)
    po = po[po["sku"] == sku].copy()
    if not po.empty:
        po["lead"] = (po["delivered_date"] - po["order_date"]).dt.days
        cutoff = po["order_date"].max() - pd.Timedelta(days=90)
        for sup, g in po.groupby("supplier_id"):
            recent = g[g["order_date"] >= cutoff]
            base = g[g["order_date"] < cutoff]
            if len(recent) >= 2 and len(base) >= 3 and base["lead"].mean() > 0:
                drift[sup] = float(recent["lead"].mean() / base["lead"].mean() - 1)

    rows = []
    for t in terms.itertuples():
        proven = t.supplier_id in set(sup_stats["supplier_id"])
        if proven:
            ot_observed = float(sup_stats.loc[sup_stats["supplier_id"] == t.supplier_id,
                                              "on_time_rate"].iloc[0])
            ot_target = float(data.suppliers.set_index("supplier_id")
                              .loc[t.supplier_id, "target_otd"])
            ot = 0.5 * ot_observed + 0.5 * ot_target        # observed + contract SLA
        else:
            ot = float(data.suppliers.set_index("supplier_id")
                       .loc[t.supplier_id, "target_otd"])    # SLA only, unproven
        lt = float(t.quoted_lead_days) * (1.0 + max(drift.get(t.supplier_id, 0.0), 0.0))
        price_score = t.unit_price / terms["unit_price"].max()
        rows.append({
            "supplier_id": t.supplier_id, "supplier_name": names.get(t.supplier_id, t.supplier_id),
            "unit_price": float(t.unit_price), "lead_days": round(lt, 1),
            "quoted_lead_days": int(t.quoted_lead_days), "on_time_rate": round(ot, 3),
            "lead_drift": round(drift.get(t.supplier_id, 0.0), 3),
            "is_primary": bool(t.is_primary),
            "proven": proven,
            "score": round(config.W_PRICE * price_score
                           + config.W_RELIABILITY * (1 - ot)
                           + config.W_LEAD_TIME * min(lt / 30.0, 1.0)
                           + (0.15 if drift.get(t.supplier_id, 0.0) >= config.LEADTIME_DRIFT_PCT else 0)
                           + (0.10 if ot < reliability_target else 0)
                           + (0.0 if proven else 0.05), 4),
        })
    rows.sort(key=lambda r: r["score"])
    return (rows[0] if rows else None), (rows[1] if len(rows) > 1 else None)


def recommend_sku(sku: str, data, forecasts: pd.DataFrame, sup_stats: pd.DataFrame,
                  inventory_override: dict | None = None,
                  demand_multiplier: float = 1.0,
                  service_level: float = config.SERVICE_LEVEL,
                  today: pd.Timestamp | None = None) -> Recommendation:
    state = data.inventory.sort_values("date").groupby("sku").tail(1).set_index("sku")
    products = data.products.set_index("sku")
    fc = forecasts[forecasts["sku"] == sku].sort_values("date")
    if fc.empty or sku not in state.index:
        return Recommendation(sku, products.loc[sku, "name"], "HOLD", 0, None, "",
                              0.0, 0.0, None, None, 0.0)

    best, runner = choose_supplier(sku, data, sup_stats)
    lead = int(best["lead_days"]) if best else 7
    horizon = min(len(fc), lead + config.REVIEW_PERIOD_DAYS)

    demand = fc["prediction"].iloc[:horizon].to_numpy() * demand_multiplier
    gross = float(demand.sum())
    pi_width = float(np.mean(fc["upper_80"].iloc[:horizon] - fc["prediction"].iloc[:horizon]))
    sigma_h = pi_width / 1.2816 * np.sqrt(horizon)                      # de-normalise PI
    sigma_h = sigma_h + (lead - (best["quoted_lead_days"] if best else lead)) * float(demand.mean()) * 0.5 if best else sigma_h
    z = {0.90: 1.2816, 0.95: 1.6449, 0.98: 2.0537, 0.99: 2.3263}.get(round(service_level, 2), 1.6449)
    safety = z * max(sigma_h, 0.0)

    row = state.loc[sku]
    on_hand = int(row["on_hand"]) * (inventory_override or {}).get("on_hand_pct", 1.0)
    on_order = int(row["on_order"])
    if inventory_override and "on_hand" in inventory_override:
        on_hand = inventory_override["on_hand"]
    position = on_hand + on_order

    net = gross + safety - position
    pack = int(products.loc[sku, "pack_size"])
    qty = int(np.ceil(max(net, 0) / pack) * pack) if net > 0 else 0

    daily = float(demand.mean()) or 1.0
    days_to_stockout = position / daily
    today = today or fc["date"].iloc[0]
    stockout_date = today + pd.Timedelta(days=float(days_to_stockout))
    order_by = stockout_date - pd.Timedelta(days=lead) - pd.Timedelta(days=config.REVIEW_PERIOD_DAYS)

    if qty == 0:
        action = "HOLD"
    elif days_to_stockout < lead:
        action = "EXPEDITE"                              # will break before arrival
    else:
        action = "BUY"

    total_cost = qty * float(best["unit_price"]) if best else 0.0
    return Recommendation(
        sku=sku, product_name=str(products.loc[sku, "name"]), action=action,
        quantity=qty,
        supplier_id=best["supplier_id"] if best and qty > 0 else None,
        supplier_name=best["supplier_name"] if best and qty > 0 else "",
        unit_price=float(best["unit_price"]) if best else 0.0,
        total_cost=round(total_cost, 2),
        order_by=str(order_by.date()) if qty > 0 else None,
        expected_stockout_date=str(stockout_date.date()),
        coverage_days=round((position + qty) / daily, 1),
        runner_up={k: runner[k] for k in ("supplier_id", "supplier_name", "unit_price",
                                          "lead_days", "on_time_rate", "score")} if runner and qty > 0 else None,
    )


def run_recommender(data, forecasts: pd.DataFrame) -> list[Recommendation]:
    sup_stats = _supplier_stats(data.purchase_orders)
    out = []
    for sku in data.products["sku"]:
        rec = recommend_sku(sku, data, forecasts, sup_stats)
        out.append(rec)
    order = {"EXPEDITE": 0, "BUY": 1, "HOLD": 2}
    out.sort(key=lambda r: (order[r.action], -r.quantity))
    return out


def stockout_probability(rec_state: dict, mu_lt: float, sigma_lt: float) -> float:
    """P(demand over lead time exceeds available position)."""
    if sigma_lt <= 0:
        return 1.0 if rec_state["position"] < mu_lt else 0.0
    return float(1.0 - sps_norm_cdf((rec_state["position"] - mu_lt) / sigma_lt))


# ------------------------------------------------------- what-if simulator
def simulate_scenario(data, forecasts: pd.DataFrame,
                      demand_pct: float = 0.0,
                      lead_time_pct: float = 0.0,
                      inventory_pct: float = 0.0,
                      service_level: float = config.SERVICE_LEVEL,
                      baseline: list[Recommendation] | None = None) -> dict:
    """Recompute the whole procurement plan under a scenario.

    Question the simulator answers: "what if demand rises 20%, our supplier
    slips 30%, inventory drops 15%?" — recomputing stockout probability,
    expected inventory, procurement requirement, cost and supplier per SKU.
    """
    sup_stats = _supplier_stats(data.purchase_orders)
    state = data.inventory.sort_values("date").groupby("sku").tail(1).set_index("sku")
    products = data.products.set_index("sku")
    lt_intel = lead_time_intelligence(data.purchase_orders)

    baseline = baseline or run_recommender(data, forecasts)
    per_sku, tot_cost, tot_units, p_sum, cov_sum = [], 0.0, 0, 0.0, 0.0
    buys = 0

    for sku in data.products["sku"]:
        fc = forecasts[forecasts["sku"] == sku].sort_values("date")
        if fc.empty:
            continue
        base_rec = next((r for r in baseline if r.sku == sku), None)

        best, runner = choose_supplier(sku, data, sup_stats)
        if best is None:
            continue
        lead = max(1, int(round(best["lead_days"] * (1.0 + lead_time_pct))))
        horizon = min(len(fc), lead + config.REVIEW_PERIOD_DAYS)

        demand = fc["prediction"].iloc[:horizon].to_numpy() * (1.0 + demand_pct)
        gross = float(demand.sum())
        pi_w = float(np.mean(fc["upper_80"].iloc[:horizon] - fc["prediction"].iloc[:horizon]))
        sigma_daily = pi_w / 1.2816
        daily_mean = float(demand.mean()) or 1.0
        sigma_lt = float(np.sqrt((sigma_daily * np.sqrt(horizon)) ** 2
                                 + (daily_mean * lead * 0.25) ** 2)) + 1.0

        row = state.loc[sku]
        on_hand = int(row["on_hand"]) * (1.0 + inventory_pct)
        position = on_hand + int(row["on_order"])

        z = {0.90: 1.2816, 0.95: 1.6449, 0.98: 2.0537, 0.99: 2.3263}.get(
            round(service_level, 2), 1.6449)
        safety = z * sigma_lt
        net = gross + safety - position
        pack = int(products.loc[sku, "pack_size"])
        qty = int(np.ceil(max(net, 0) / pack) * pack) if net > 0 else 0

        p_stockout = float(1.0 - sps_norm_cdf((position - gross) / sigma_lt))
        cost = qty * float(best["unit_price"])
        coverage = (position + qty) / daily_mean

        per_sku.append({
            "sku": sku, "name": str(products.loc[sku, "name"]),
            "action": "BUY" if qty > 0 else "HOLD",
            "quantity": qty, "cost": round(cost, 2),
            "supplier_id": best["supplier_id"] if qty > 0 else None,
            "supplier_id_baseline": base_rec.supplier_id if base_rec else None,
            "stockout_probability": round(p_stockout, 3),
            "stockout_probability_baseline": _baseline_stockout_prob(data, fc, sku, state, lt_intel),
            "coverage_days": round(coverage, 1),
            "coverage_days_baseline": base_rec.coverage_days if base_rec else None,
        })
        if qty > 0:
            buys += 1
            tot_cost += cost
            tot_units += qty
        p_sum += p_stockout
        cov_sum += coverage

    n = max(len(per_sku), 1)
    return {
        "scenario": {"demand_pct": demand_pct, "lead_time_pct": lead_time_pct,
                     "inventory_pct": inventory_pct, "service_level": service_level},
        "summary": {
            "procurement_units": tot_units,
            "procurement_cost": round(tot_cost, 2),
            "buy_actions": buys,
            "avg_stockout_probability": round(p_sum / n, 3),
            "avg_coverage_days": round(cov_sum / n, 1),
        },
        "per_sku": per_sku,
    }


def _baseline_stockout_prob(data, fc: pd.DataFrame, sku: str, state, lt_intel) -> float:
    from supplychainxai.risk.engine import effective_lead
    lead, lead_std, _ = effective_lead(sku, data.supply_terms, lt_intel)
    horizon = min(len(fc), lead)
    daily_fc = fc["prediction"].iloc[:horizon]
    mu_lt = float(daily_fc.sum())
    sigma_daily = float(np.mean(fc["upper_80"].iloc[:horizon] - daily_fc)) / 1.2816
    sigma_lt = float(np.sqrt((sigma_daily * np.sqrt(horizon)) ** 2
                             + (mu_lt / max(lead, 1) * lead_std) ** 2)) + 1.0
    row = state.loc[sku]
    position = int(row["on_hand"]) + int(row["on_order"])
    return float(round(1.0 - sps_norm_cdf((position - mu_lt) / sigma_lt), 3))
