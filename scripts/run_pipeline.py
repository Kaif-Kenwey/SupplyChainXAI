#!/usr/bin/env python3
"""Run the full SupplyChainXAI analytics pipeline (phases 1-7).

    raw CSVs -> clean -> validate -> features -> SQLite warehouse
             -> model bake-off -> forecasts -> risk engine
             -> procurement recommendations -> explanations
             -> artifacts/ (csv + json) + data/processed/analytics.db

Usage:
    python scripts/run_pipeline.py
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from supplychainxai import config                                    # noqa: E402
from supplychainxai.data import store                                # noqa: E402
from supplychainxai.data.clean import clean_all                      # noqa: E402
from supplychainxai.data.features import build_features              # noqa: E402
from supplychainxai.data.ingest import load_all                      # noqa: E402
from supplychainxai.data.validate import DataValidationError, validate_all  # noqa: E402
from supplychainxai.explain.engine import run_explainer              # noqa: E402
from supplychainxai.forecasting.selector import run as run_forecast  # noqa: E402
from supplychainxai.optimization.engine import (                     # noqa: E402
    Recommendation, _supplier_stats, run_recommender)
from supplychainxai.risk.engine import run_risk_engine               # noqa: E402


@dataclass
class PipelineContext:
    """Everything the API / copilot needs, loaded once at startup."""
    data: object = None
    features: pd.DataFrame | None = None
    forecasts: pd.DataFrame | None = None
    comparison: pd.DataFrame | None = None
    importance: pd.DataFrame | None = None
    narrative: dict = field(default_factory=dict)
    winners: dict = field(default_factory=dict)
    risk_alerts: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)
    explanations: list = field(default_factory=list)
    snapshot_date: str = ""
    kpis: dict = field(default_factory=dict)


def compute_kpis(data, forecasts: pd.DataFrame, recommendations: list[Recommendation]) -> dict:
    inv = data.inventory
    latest = inv.sort_values("date").groupby("sku").tail(1)
    products = data.products.set_index("sku")

    inventory_value = float((latest["on_hand"] * latest["sku"].map(products["unit_cost"])).sum())
    open_pos = data.purchase_orders[data.purchase_orders["status"] == "OPEN"]
    po_spend = float((open_pos["quantity"] * open_pos["unit_price"]).sum())

    po = data.purchase_orders[data.purchase_orders["status"] == "DELIVERED"].copy()
    po["lead"] = (po["delivered_date"] - po["order_date"]).dt.days
    on_time = float((po["delivered_date"] <= po["expected_date"]).mean()) if len(po) else None

    buy = [r for r in recommendations if r.action in ("BUY", "EXPEDITE")]
    total_demand = float(forecasts.groupby("sku")["prediction"].sum().sum())
    stockout_skus = {a.sku for a in []}  # filled by caller below

    return {
        "inventory_value": round(inventory_value, 2),
        "open_po_count": int(len(open_pos)),
        "open_po_spend": round(po_spend, 2),
        "supplier_on_time_pct": round(on_time * 100, 1) if on_time is not None else None,
        "skus_tracked": int(len(data.products)),
        "buy_actions": len(buy),
        "buy_units": int(sum(r.quantity for r in buy)),
        "buy_cost": round(sum(r.total_cost for r in buy), 2),
        "forecast_30d_demand": round(total_demand, 0),
    }


def main() -> PipelineContext:
    t0 = time.time()
    ctx = PipelineContext()

    print("[1/8] ingesting raw data ...")
    raw = load_all()
    print(f"      sales={len(raw.sales):,}  pos={len(raw.purchase_orders):,}")

    print("[2/8] cleaning ...")
    cleaned, clean_report = clean_all(raw)
    cleaned.sales.to_csv(config.PROCESSED_FILES["sales"], index=False)
    cleaned.inventory.to_csv(config.PROCESSED_FILES["inventory"], index=False)
    cleaned.purchase_orders.to_csv(config.PROCESSED_FILES["purchase_orders"], index=False)
    print(f"      actions={len(clean_report.to_list())}  cells fixed={clean_report.total_fixed()}")

    print("[3/8] validating ...")
    vrep = validate_all(cleaned)
    vdict = vrep.to_dict()
    (config.DATA_PROCESSED / "validation_report.json").write_text(json.dumps(vdict, indent=2))
    (config.DATA_PROCESSED / "cleaning_report.json").write_text(
        json.dumps({"steps": clean_report.to_list()}, indent=2))
    print(f"      status={vdict['status']}  checks={vdict['summary']['total']}")
    if not vrep.passed:
        raise DataValidationError("CRITICAL validation rules failed — aborting pipeline")

    print("[4/8] feature engineering ...")
    ctx.features = build_features(cleaned.sales, cleaned.inventory)
    ctx.features.to_csv(config.PROCESSED_FILES["features"], index=False)
    print(f"      rows={len(ctx.features):,}")

    print("[5/8] building SQLite analytics warehouse ...")
    db = store.build_db(cleaned)
    print(f"      {db}")

    print("[6/8] forecasting (model bake-off) ...")
    skus = cleaned.products["sku"].tolist()
    fc = run_forecast(ctx.features, skus)
    ctx.forecasts, ctx.comparison = fc["forecasts"], fc["comparison"]
    ctx.importance, ctx.narrative, ctx.winners = fc["importance"], fc["narrative"], fc["winners"]
    print(f"      portfolio best: {ctx.narrative['portfolio_best_model']}")

    print("[7/8] risk engine ...")
    ctx.risk_alerts = run_risk_engine(cleaned, ctx.forecasts)
    sev = {"CRITICAL": 0, "WARNING": 0}
    for a in ctx.risk_alerts:
        if a.severity in sev:
            sev[a.severity] += 1
    print(f"      alerts={len(ctx.risk_alerts)}  CRITICAL={sev['CRITICAL']}  WARNING={sev['WARNING']}")

    print("[8/8] recommendations + explanations ...")
    ctx.recommendations = run_recommender(cleaned, ctx.forecasts)
    ctx.explanations = run_explainer(cleaned, ctx.forecasts, ctx.recommendations,
                                     ctx.importance)
    ctx.kpis = compute_kpis(cleaned, ctx.forecasts, ctx.recommendations)
    ctx.data = cleaned
    ctx.snapshot_date = str(cleaned.sales["date"].max().date())

    # ---------------- persist artifacts
    sup_stats = _supplier_stats(cleaned.purchase_orders)
    with open(config.ARTIFACTS / "risk_alerts.json", "w") as fh:
        json.dump([a.__dict__ for a in ctx.risk_alerts], fh, indent=2, default=str)
    with open(config.ARTIFACTS / "recommendations.json", "w") as fh:
        json.dump([r.__dict__ for r in ctx.recommendations], fh, indent=2)
    with open(config.ARTIFACTS / "explanations.json", "w") as fh:
        json.dump(ctx.explanations, fh, indent=2, default=str)
    with open(config.ARTIFACTS / "kpis.json", "w") as fh:
        json.dump({"snapshot_date": ctx.snapshot_date, **ctx.kpis}, fh, indent=2)
    sup_stats.to_csv(config.ARTIFACTS / "supplier_stats.csv", index=False)

    print(f"\nPIPELINE OK in {time.time() - t0:.1f}s — snapshot {ctx.snapshot_date}")
    print(f"  kpis: {ctx.kpis}")
    return ctx


if __name__ == "__main__":
    main()
