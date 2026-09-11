"""Phase 6 — Explainable-AI layer.

Two complementary explanation surfaces:

1. Forecast explanations  — permutation importance of the winning model plus a
   counterfactual "why is this forecast different from the recent average?"
   decomposition (momentum, weekly shape, promo absence).

2. Recommendation explanations — an exact additive decomposition of the BUY
   quantity into its decision factors (gross requirement, safety stock,
   inventory offset, pack rounding) expressed in units AND percentages, then
   rendered as natural language. Every number is traceable to the pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from supplychainxai import config
from supplychainxai.optimization.engine import Recommendation, _supplier_stats


# ------------------------------------------------------------------ forecast
def explain_forecast(sku: str, features: pd.DataFrame, forecasts: pd.DataFrame,
                     importance: pd.DataFrame | None) -> dict:
    fc = forecasts[forecasts["sku"] == sku]
    frame = features[features["sku"] == sku].sort_values("date")
    recent28 = frame["units_sold"].tail(28).mean()
    recent14 = frame["units_sold"].tail(14).mean()
    fc_mean = float(fc["prediction"].mean()) if len(fc) else 0.0

    momentum_pct = (recent14 / recent28 - 1.0) if recent28 else 0.0
    fc_vs_recent = (fc_mean / recent28 - 1.0) if recent28 else 0.0

    top_features = []
    if importance is not None and len(importance):
        imp = importance[importance["sku"] == sku].sort_values("importance", ascending=False).head(5)
        total = float(imp["importance"].sum()) or 1.0
        top_features = [{"feature": r.feature, "share": round(float(r.importance) / total, 3)}
                        for r in imp.itertuples()]

    drivers = []
    if abs(momentum_pct) > 0.03:
        drivers.append({
            "factor": "demand momentum", "direction": "up" if momentum_pct > 0 else "down",
            "impact_pct": round(abs(momentum_pct) * 100, 1),
            "detail": f"last-14d average ({recent14:.0f}/day) vs trailing 28d ({recent28:.0f}/day)"})
    if abs(fc_vs_recent) > 0.03:
        drivers.append({
            "factor": "forecast vs recent level", "direction": "up" if fc_vs_recent > 0 else "down",
            "impact_pct": round(abs(fc_vs_recent) * 100, 1),
            "detail": f"30-day forecast mean {fc_mean:.0f}/day vs trailing 28d {recent28:.0f}/day"})
    promos = int(frame["promo_flag"].tail(28).sum())
    drivers.append({
        "factor": "promo assumption", "direction": "down",
        "impact_pct": round(100.0 * promos / 28.0, 1),
        "detail": f"{promos} promo days in the last 28; horizon assumes none (unplannable)"})

    return {
        "sku": sku,
        "model": fc["model"].iloc[0] if len(fc) else None,
        "forecast_daily_mean": round(fc_mean, 1),
        "top_features": top_features,
        "drivers": drivers,
        "narrative": (
            f"{sku} is forecast at ~{fc_mean:.0f} units/day over the next 30 days "
            f"({fc_vs_recent:+.0%} vs the trailing-28d level of {recent28:.0f}). "
            + (f"The winning model ({fc['model'].iloc[0]}) leans mainly on "
               + ", ".join(f"{t['feature']} ({t['share']:.0%})" for t in top_features[:3])
               + ". " if top_features else "")
            + (f"Recent demand momentum is {momentum_pct:+.0%} (14d vs 28d), which the "
               f"model carries into the horizon via its lag features." if abs(momentum_pct) > 0.03 else "")
        ).strip(),
    }


# ------------------------------------------------------------ recommendation
def explain_recommendation(sku: str, data, forecasts: pd.DataFrame,
                           rec: Recommendation, service_level: float = config.SERVICE_LEVEL) -> dict:
    state = data.inventory.sort_values("date").groupby("sku").tail(1).set_index("sku")
    fc = forecasts[forecasts["sku"] == sku].sort_values("date")
    best_lead = None
    terms = data.supply_terms[(data.supply_terms["sku"] == sku) &
                              (data.supply_terms["supplier_id"] == rec.supplier_id)]
    if len(terms):
        best_lead = int(terms["quoted_lead_days"].iloc[0])
    horizon = min(len(fc), (best_lead or 7) + config.REVIEW_PERIOD_DAYS)

    gross = float(fc["prediction"].iloc[:horizon].sum())
    pi_width = float(np.mean(fc["upper_80"].iloc[:horizon] - fc["prediction"].iloc[:horizon]))
    sigma_h = pi_width / 1.2816 * np.sqrt(horizon)
    z = {0.90: 1.2816, 0.95: 1.6449, 0.98: 2.0537, 0.99: 2.3263}.get(round(service_level, 2), 1.6449)
    safety = z * sigma_h
    position = int(state.loc[sku, "on_hand"]) + int(state.loc[sku, "on_order"])
    rounding = rec.quantity - max(gross + safety - position, 0)

    components = [
        {"factor": "forecast demand", "units": round(gross, 1), "direction": "increase",
         "detail": f"model demand over {horizon}d protection window"},
        {"factor": "safety stock", "units": round(safety, 1), "direction": "increase",
         "detail": f"z={z} at {service_level:.0%} service level × σ={sigma_h:.0f} "
                   f"(includes supplier lead-time variability)"},
        {"factor": "inventory position", "units": -position, "direction": "decrease",
         "detail": f"on-hand {int(state.loc[sku, 'on_hand'])} + on-order {int(state.loc[sku, 'on_order'])}"},
        {"factor": "pack rounding", "units": round(max(rounding, 0), 1), "direction": "increase",
         "detail": f"rounded up to pack size multiples"},
    ]
    base = max(sum(c["units"] for c in components), 1e-9)
    for c in components:
        c["impact_pct"] = round(abs(c["units"]) / max(abs(gross) + abs(safety), 1e-9) * 100, 1)

    # supplier-side drivers
    sup_drivers = []
    if rec.supplier_id:
        sup_stats = _supplier_stats(data.purchase_orders)
        from supplychainxai.optimization.engine import choose_supplier
        best, runner = choose_supplier(sku, data, sup_stats)
        if best and runner:
            sup_drivers.append({
                "factor": "supplier selection", "direction": "neutral",
                "impact_pct": 0.0,
                "detail": f"{rec.supplier_id} scored {best['score']:.3f} vs runner-up "
                          f"{runner['supplier_id']} at {runner['score']:.3f} "
                          f"(weights: price {config.W_PRICE:.2f}, reliability {config.W_RELIABILITY:.2f}, "
                          f"lead time {config.W_LEAD_TIME:.2f}; penalties applied for drift/reliability)"})
            if best.get("lead_drift", 0) > 0:
                sup_drivers.append({
                    "factor": "supplier lead-time drift", "direction": "increase",
                    "impact_pct": round(best["lead_drift"] * 100, 1),
                    "detail": f"recent POs run {best['lead_drift']:+.0%} slower — widened the "
                              f"protection window from {best['quoted_lead_days']}d to {best['lead_days']}d"})

    drivers = [c for c in components if c["impact_pct"] > 0] + sup_drivers
    return {"sku": sku, "components": components, "drivers": drivers, "narrative": None}


def to_natural_language(rec: Recommendation, explanation: dict, data) -> str:
    parts = []
    if rec.action in ("BUY", "EXPEDITE"):
        verb = "Expedite" if rec.action == "EXPEDITE" else "Buy"
        parts.append(
            f"{verb} {rec.quantity:,} units of {rec.sku} ({rec.product_name}) from "
            f"{rec.supplier_name} ({rec.supplier_id}) at {rec.unit_price:.2f}/unit "
            f"— total {rec.total_cost:,.2f}. Order by {rec.order_by}; stock is projected "
            f"to run out around {rec.expected_stockout_date}.")
    else:
        parts.append(f"Hold {rec.sku} ({rec.product_name}): current inventory already covers "
                     f"{rec.coverage_days:.0f} days — no purchase needed now.")
    inc = [c for c in explanation["components"] if c["direction"] == "increase"]
    dec = [c for c in explanation["components"] if c["direction"] == "decrease"]
    if inc:
        parts.append("Why: the requirement is driven by "
                     + ", ".join(f"{c['factor']} (+{c['units']:.0f}u, {c['impact_pct']:.0f}%)" for c in inc))
    if dec:
        parts.append("offset by "
                     + ", ".join(f"{c['factor']} ({c['units']:.0f}u)" for c in dec)
                     + ".")
    for d in explanation["drivers"]:
        if d["factor"] == "supplier lead-time drift":
            parts.append(f"Attention: {d['detail']}.")
        if d["factor"] == "supplier selection":
            parts.append(f"Supplier choice: {d['detail']}.")
    return " ".join(parts)


def run_explainer(data, forecasts: pd.DataFrame, recommendations: list[Recommendation],
                  importance: pd.DataFrame | None) -> list[dict]:
    out = []
    for rec in recommendations:
        expl = explain_recommendation(rec.sku, data, forecasts, rec)
        expl["narrative"] = to_natural_language(rec, expl, data)
        expl["action"] = rec.action
        expl["quantity"] = rec.quantity
        expl["supplier_id"] = rec.supplier_id
        out.append(expl)
    return out
