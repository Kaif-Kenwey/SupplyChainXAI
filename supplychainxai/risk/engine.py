"""Phase 4 — Inventory risk & anomaly detection engine.

Detects, per SKU / supplier, with every alert carrying the numbers it is
derived from (no opaque scores):

  R1 stockout risk        P(demand over lead time > stock position) + days-of-cover
  R2 overstock            capital tied in slow-moving stock
  R3 demand spike         recent 14d vs 90d baseline z-score
  R4 supplier lead drift  recent PO lead times vs historical baseline
  R5 abnormal procurement unit-price outliers on recent PO lines
  R6 supplier reliability on-time delivery / quantity fill / composite score
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from supplychainxai import config


@dataclass
class Alert:
    alert_id: str
    type: str
    severity: str            # CRITICAL | WARNING | INFO
    sku: str | None
    supplier_id: str | None
    message: str
    data: dict


def _delivered(po: pd.DataFrame) -> pd.DataFrame:
    """Defensively filter delivered POs (status column is added by the cleaner)."""
    po = po.copy()
    if "status" in po.columns:
        return po[po["status"] == "DELIVERED"]
    return po


def _latest_state(inventory: pd.DataFrame) -> pd.DataFrame:
    return inventory.sort_values("date").groupby("sku").tail(1).set_index("sku")


def lead_time_intelligence(purchase_orders: pd.DataFrame) -> pd.DataFrame:
    """Per (sku, supplier) delivered-PO lead-time stats incl. recent drift."""
    po = _delivered(purchase_orders)
    if po.empty:
        return pd.DataFrame(columns=["sku", "supplier_id", "baseline_lead", "recent_lead",
                                     "lead_std", "drift_pct", "n_recent"])
    po["lead"] = (po["delivered_date"] - po["order_date"]).dt.days
    cutoff = po["order_date"].max() - pd.Timedelta(days=90)
    rows = []
    for (sku, sup), g in po.groupby(["sku", "supplier_id"]):
        recent = g[g["order_date"] >= cutoff]
        baseline = g[g["order_date"] < cutoff]
        base_mean = float(baseline["lead"].mean()) if len(baseline) else float(g["lead"].mean())
        recent_mean = float(recent["lead"].mean()) if len(recent) else base_mean
        rows.append({
            "sku": sku, "supplier_id": sup,
            "baseline_lead": round(base_mean, 1),
            "recent_lead": round(recent_mean, 1),
            "lead_std": round(float(g["lead"].std() or 1.0), 1),
            "drift_pct": round(recent_mean / max(base_mean, 1e-9) - 1.0, 3),
            "n_recent": len(recent),
        })
    return pd.DataFrame(rows)


def effective_lead(sku: str, supply_terms: pd.DataFrame,
                   lt_intel: pd.DataFrame) -> tuple[int, float, str | None]:
    """Drift-adjusted lead time for the SKU's primary supplier.

    Returns (effective_days, std_days, supplier_id). Uses the quoted lead time
    widened by observed recent drift — if a supplier's POs are running 40%
    late, planning on the quoted number would be a lie.
    """
    terms = supply_terms[supply_terms["sku"] == sku]
    primary = terms[terms["is_primary"] == True]  # noqa: E712
    if len(primary) == 0 and len(terms):
        primary = terms
    quoted = int(primary["quoted_lead_days"].iloc[0]) if len(primary) else 7
    std = float(primary["lead_time_std"].iloc[0]) if len(primary) else 2.0
    sup = primary["supplier_id"].iloc[0] if len(primary) else None
    intel = lt_intel[(lt_intel["sku"] == sku) & (lt_intel["supplier_id"] == sup)] if sup else None
    if intel is not None and len(intel):
        drift = float(intel["drift_pct"].iloc[0])
        if drift > 0:
            quoted = int(round(quoted * (1.0 + drift)))
            std = max(std, float(intel["lead_std"].iloc[0]))
    return max(quoted, 1), std, sup


def stockout_risk(inventory: pd.DataFrame, forecasts: pd.DataFrame,
                  data, z: float = config.SERVICE_LEVEL_Z) -> list[Alert]:
    alerts = []
    state = _latest_state(inventory)
    lt_intel = lead_time_intelligence(data.purchase_orders)

    for sku, st in state.iterrows():
        fc = forecasts[forecasts["sku"] == sku].sort_values("date")
        if fc.empty:
            continue
        lead, lead_std, sup_id = effective_lead(sku, data.supply_terms, lt_intel)

        on_hand, on_order = int(st["on_hand"]), int(st["on_order"])
        horizon = min(len(fc), lead)
        daily_fc = fc["prediction"].iloc[:horizon]
        mu_lt = float(daily_fc.sum())
        sigma_daily = float(np.mean(fc["upper_80"].iloc[:horizon] - daily_fc)) / 1.2816
        sigma_lt = float(np.sqrt((sigma_daily * np.sqrt(horizon)) ** 2
                                 + (mu_lt / max(lead, 1) * lead_std) ** 2)) + 1.0
        position = on_hand + on_order

        p_stockout = float(1.0 - sps_norm_cdf((position - mu_lt) / sigma_lt))
        daily = float(fc["prediction"].mean()) or 1.0
        cover_days = position / daily

        if p_stockout >= 0.60 or cover_days < config.CRITICAL_COVER_DAYS:
            sev = "CRITICAL"
        elif p_stockout >= 0.30:
            sev = "WARNING"
        else:
            continue
        when = int(max(0, round(position / daily))) if daily > 0 else 999
        sup_note = f" (effective lead {lead}d incl. drift)" if sup_id else f" (lead {lead}d)"
        alerts.append(Alert(
            f"R1-{sku}", "stockout_risk", sev, sku, sup_id,
            f"{sku} forecast to breach safety stock in ~{when} days "
            f"({cover_days:.0f} days of cover{sup_note}); "
            f"P(stockout before replenishment) = {p_stockout:.0%}.",
            {"on_hand": on_hand, "on_order": on_order, "lead_time_days": lead,
             "demand_over_lead_time": round(mu_lt, 1),
             "demand_sigma": round(sigma_lt, 1),
             "days_of_cover": round(cover_days, 1),
             "stockout_probability": round(p_stockout, 3)}))
    return alerts


def sps_norm_cdf(x: float) -> float:
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def overstock(inventory: pd.DataFrame, forecasts: pd.DataFrame,
              products: pd.DataFrame) -> list[Alert]:
    alerts = []
    state = _latest_state(inventory)
    cost = products.set_index("sku")["unit_cost"]
    for sku, st in state.iterrows():
        fc = forecasts[forecasts["sku"] == sku]
        daily = float(fc["prediction"].mean()) if len(fc) else 1.0
        doh = int(st["on_hand"]) / max(daily, 1e-9)
        if doh >= config.OVERSTOCK_DOH_DAYS:
            tied = int(st["on_hand"]) * float(cost.get(sku, 0.0))
            alerts.append(Alert(
                f"R2-{sku}", "overstock", "WARNING", sku, None,
                f"{sku} holds {doh:.0f} days of cover (threshold {config.OVERSTOCK_DOH_DAYS}d) "
                f"— {tied:,.0f} in capital tied up. Pause replenishment.",
                {"days_of_cover": round(doh, 1), "on_hand": int(st["on_hand"]),
                 "capital_tied": round(tied, 2)}))
    return alerts


def demand_spikes(sales: pd.DataFrame, baseline_days: int = 60,
                  recent_days: int = 21, min_lift: float = 0.12) -> list[Alert]:
    """Level-shift detection: Welch's t-test of last 21d vs prior 60d mean."""
    alerts = []
    for sku, g in sales.sort_values("date").groupby("sku"):
        y = g.set_index("date")["units_sold"].asfreq("D").fillna(0.0)
        if len(y) < baseline_days + recent_days:
            continue
        baseline = y.iloc[-(baseline_days + recent_days):-recent_days]
        recent = y.iloc[-recent_days:]
        mu, sd_b = float(baseline.mean()), float(baseline.std() or 1.0)
        mu_r, sd_r = float(recent.mean()), float(recent.std() or 1.0)
        # Welch's t statistic (normal approximation for the p-value)
        se = np.sqrt(sd_b ** 2 / len(baseline) + sd_r ** 2 / len(recent)) or 1e-9
        t_stat = (mu_r - mu) / se
        p_value = 2.0 * (1.0 - sps_norm_cdf(abs(t_stat)))
        lift = mu_r / max(mu, 1e-9) - 1.0
        if p_value <= 0.05 and lift >= min_lift:
            alerts.append(Alert(
                f"R3-{sku}", "demand_spike", "CRITICAL" if lift >= 0.30 else "WARNING",
                sku, None,
                f"{sku} demand running {lift:+.0%} vs baseline "
                f"({mu_r:.0f}/d vs {mu:.0f}/d, Welch p={p_value:.3f}). "
                f"Re-check forecast inputs.",
                {"recent_mean": round(mu_r, 1), "baseline_mean": round(mu, 1),
                 "lift_pct": round(lift, 2), "t_stat": round(float(t_stat), 2),
                 "p_value": round(p_value, 4)}))
    return alerts


def supplier_leadtime_drift(purchase_orders: pd.DataFrame) -> list[Alert]:
    alerts = []
    po = _delivered(purchase_orders)
    po["actual_lead"] = (po["delivered_date"] - po["order_date"]).dt.days
    cutoff = po["order_date"].max() - pd.Timedelta(days=90)
    for (sup, sku), g in po.groupby(["supplier_id", "sku"]):
        recent = g[g["order_date"] >= cutoff]
        baseline = g[g["order_date"] < cutoff]
        if len(recent) < 2 or len(baseline) < 3:
            continue
        drift = recent["actual_lead"].mean() / max(baseline["actual_lead"].mean(), 1e-9) - 1
        abs_delta = recent["actual_lead"].mean() - baseline["actual_lead"].mean()
        if drift >= config.LEADTIME_DRIFT_PCT and abs_delta >= 1.5:
            alerts.append(Alert(
                f"R4-{sup}-{sku}", "supplier_lead_drift", "CRITICAL" if drift >= 0.35 else "WARNING",
                sku, sup,
                f"Supplier {sup} lead time on {sku} up {drift:+.0%} "
                f"({baseline['actual_lead'].mean():.0f}d -> {recent['actual_lead'].mean():.0f}d "
                f"over last 90d, {len(recent)} POs). De-rank or split volume.",
                {"baseline_lead_days": round(float(baseline["actual_lead"].mean()), 1),
                 "recent_lead_days": round(float(recent["actual_lead"].mean()), 1),
                 "drift_pct": round(drift, 2), "recent_pos": len(recent)}))
    return alerts


def price_anomalies(purchase_orders: pd.DataFrame,
                    z_threshold: float = config.PRICE_ANOMALY_Z) -> list[Alert]:
    alerts = []
    for (sup, sku), g in purchase_orders.groupby(["supplier_id", "sku"]):
        if len(g) < 5:
            continue
        mu, sd = g["unit_price"].mean(), g["unit_price"].std() or 1.0
        recent = g.sort_values("order_date").tail(2)
        zscores = (recent["unit_price"] - mu) / sd
        for po_id, z_ in zscores.items():
            if z_ >= z_threshold:
                row = g.loc[po_id]
                alerts.append(Alert(
                    f"R5-{po_id}", "abnormal_procurement", "WARNING", sku, sup,
                    f"PO {po_id} priced {row['unit_price']:.2f} "
                    f"({z_:.1f}σ above the {len(g)}-PO mean {mu:.2f}) for {sku} "
                    f"from {sup}. Verify contract / invoice.",
                    {"unit_price": float(row["unit_price"]), "mean_price": round(float(mu), 2),
                     "z_score": round(float(z_), 2)}))
    return alerts


def supplier_reliability(purchase_orders: pd.DataFrame,
                         suppliers: pd.DataFrame) -> list[Alert]:
    alerts = []
    delivered = _delivered(purchase_orders).copy()
    delivered["lead"] = (delivered["delivered_date"] - delivered["order_date"]).dt.days
    delivered["on_time"] = delivered["delivered_date"] <= delivered["expected_date"]
    for sup, g in delivered.groupby("supplier_id"):
        n_delivered = len(g)
        if n_delivered < 5:      # small-sample guard: never judge on 1-2 POs
            continue
        on_time = float(g["on_time"].mean())
        n_pos = len(purchase_orders[purchase_orders["supplier_id"] == sup])
        score = 0.6 * on_time + 0.4 * float(suppliers.set_index("supplier_id")
                                             .loc[sup, "target_otd"])
        if score < config.SUPPLIER_RELIABILITY_FLOOR or on_time < 0.65:
            alerts.append(Alert(
                f"R6-{sup}", "supplier_reliability", "WARNING", None, sup,
                f"Supplier {sup} on-time rate {on_time:.0%} across {n_pos} POs — "
                f"reliability score {score:.2f} (floor {config.SUPPLIER_RELIABILITY_FLOOR}).",
                {"on_time_rate": round(on_time, 3), "reliability_score": round(score, 3),
                 "pos_total": n_pos}))
    return alerts


def run_risk_engine(data, forecasts: pd.DataFrame) -> list[Alert]:
    alerts: list[Alert] = []
    alerts += stockout_risk(data.inventory, forecasts, data)
    alerts += overstock(data.inventory, forecasts, data.products)
    alerts += demand_spikes(data.sales)
    alerts += supplier_leadtime_drift(data.purchase_orders)
    alerts += price_anomalies(data.purchase_orders)
    alerts += supplier_reliability(data.purchase_orders, data.suppliers)

    order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
    alerts.sort(key=lambda a: (order[a.severity], a.type))
    return alerts
