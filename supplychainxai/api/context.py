"""Loads pipeline artifacts from disk into a context object for the API.

The API boots in ~1s from committed artifacts; to refresh them, re-run
`python scripts/run_pipeline.py` (or POST /api/admin/recompute).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import pandas as pd

from supplychainxai import config
from supplychainxai.data.ingest import RawData
from supplychainxai.optimization.engine import Recommendation, run_recommender, simulate_scenario
from supplychainxai.risk.engine import Alert


@dataclass
class AppContext:
    data: RawData
    forecasts: pd.DataFrame
    comparison: pd.DataFrame
    importance: pd.DataFrame | None
    narrative: dict
    kpis: dict
    risk_alerts: list[Alert]
    recommendations: list[Recommendation]
    explanations: list[dict]
    supplier_stats: pd.DataFrame
    features: pd.DataFrame | None = None
    snapshot_date: str = ""
    scenario_cache: dict = field(default_factory=dict)


def load_context() -> AppContext:
    products = pd.read_csv(config.RAW_FILES["products"])
    suppliers = pd.read_csv(config.RAW_FILES["suppliers"])
    terms = pd.read_csv(config.RAW_FILES["supply_terms"])
    sales = pd.read_csv(config.PROCESSED_FILES["sales"], parse_dates=["date"])
    inventory = pd.read_csv(config.PROCESSED_FILES["inventory"], parse_dates=["date"])
    pos = pd.read_csv(config.PROCESSED_FILES["purchase_orders"],
                      parse_dates=["order_date", "expected_date", "delivered_date"])
    data = RawData(products=products, suppliers=suppliers, supply_terms=terms,
                   sales=sales, inventory=inventory, purchase_orders=pos)

    forecasts = pd.read_csv(config.ARTIFACTS / "forecasts.csv", parse_dates=["date"])
    comparison = pd.read_csv(config.ARTIFACTS / "model_comparison.csv")
    imp_path = config.ARTIFACTS / "feature_importance.csv"
    importance = pd.read_csv(imp_path) if imp_path.exists() else None
    narrative = json.loads((config.ARTIFACTS / "model_selection.json").read_text())
    kpis = json.loads((config.ARTIFACTS / "kpis.json").read_text())
    alerts = [Alert(**a) for a in json.loads((config.ARTIFACTS / "risk_alerts.json").read_text())]
    recs = [Recommendation(**r) for r in json.loads(
        (config.ARTIFACTS / "recommendations.json").read_text())]
    explanations = json.loads((config.ARTIFACTS / "explanations.json").read_text())
    sup_stats = pd.read_csv(config.ARTIFACTS / "supplier_stats.csv")
    feat_path = config.PROCESSED_FILES["features"]
    features = pd.read_csv(feat_path, parse_dates=["date"]) if feat_path.exists() else None

    return AppContext(data=data, forecasts=forecasts, comparison=comparison,
                      importance=importance, narrative=narrative, kpis=kpis,
                      risk_alerts=alerts, recommendations=recs,
                      explanations=explanations, supplier_stats=sup_stats,
                      features=features, snapshot_date=str(sales["date"].max().date()))


def refresh_recommendations(ctx: AppContext) -> None:
    ctx.recommendations = run_recommender(ctx.data, ctx.forecasts)


def run_what_if(ctx: AppContext, demand_pct: float, lead_time_pct: float,
                inventory_pct: float, service_level: float) -> dict:
    return simulate_scenario(ctx.data, ctx.forecasts,
                             demand_pct=demand_pct, lead_time_pct=lead_time_pct,
                             inventory_pct=inventory_pct, service_level=service_level,
                             baseline=ctx.recommendations)
