#!/usr/bin/env python3
"""SupplyChainXAI training pipeline (the reproducible retraining workflow).

    python scripts/train.py          # recommended entrypoint (same as below)
    python scripts/run_pipeline.py

Workflow — every stage writes traceable artifacts:

    [01] run context            run_id, git commit, environment
    [02] dataset manifest       dataset_version (content hash), provenance
    [03] ingest                 schema-enforced load
    [04] clean                  rule-based repair + audit report
    [05] validate               15 integrity gates (CRITICAL aborts)
    [06] data-quality report    ERROR/WARNING monitoring sweep
    [07] features               leakage-safe supervised frame
    [08] analytics store        SQLite (default) or DATABASE_URL
    [09] walk-forward bake-off  rolling-origin evaluation + experiment tracking
    [10] model selection        composite score + registry + promotion gates
    [11] forecasts              final refit + 80% PIs
    [12] risk engine            6 alert families
    [13] recommendations        BUY/EXPEDITE/HOLD + explanations
    [14] business KPIs          observed / estimated / simulated
    [15] monitoring             PI coverage, drift, model health, retrain decision
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

from supplychainxai import config
from supplychainxai.config.settings import get_settings
from supplychainxai.data import store
from supplychainxai.data.clean import clean_all
from supplychainxai.data.features import build_features
from supplychainxai.data.ingest import load_all
from supplychainxai.data.validate import DataValidationError, validate_all
from supplychainxai.explain.engine import explain_forecast, run_explainer
from supplychainxai.explain.shap_bridge import (
    local_forecast_drivers,
    portfolio_shap_importance,
)
from supplychainxai.forecasting.selector import run as run_forecast
from supplychainxai.mlops.provenance import build_manifest, write_manifest
from supplychainxai.mlops.registry import (
    ModelRegistry,
    ModelVersion,
    feature_version,
)
from supplychainxai.mlops.retraining import promotion_gates
from supplychainxai.mlops.runlog import new_run_context
from supplychainxai.mlops.tracking import ExperimentTracker
from supplychainxai.monitoring.business_kpis import compute_business_kpis
from supplychainxai.monitoring.coverage import portfolio_coverage
from supplychainxai.monitoring.data_quality import run_data_quality
from supplychainxai.monitoring.drift import (
    drift_report,
    reference_recent_split,
)
from supplychainxai.monitoring.model_health import build_model_health
from supplychainxai.optimization.engine import (
    Recommendation,
    _supplier_stats,
    run_recommender,
)
from supplychainxai.risk.engine import run_risk_engine


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
    forecast_explanations: dict = field(default_factory=dict)
    snapshot_date: str = ""
    kpis: dict = field(default_factory=dict)
    fold_results: pd.DataFrame | None = None
    model_health: dict = field(default_factory=dict)
    business_kpis: dict = field(default_factory=dict)
    run_id: str = ""


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

    return {
        "inventory_value": round(inventory_value, 2),
        "open_po_count": len(open_pos),
        "open_po_spend": round(po_spend, 2),
        "supplier_on_time_pct": round(on_time * 100, 1) if on_time is not None else None,
        "skus_tracked": len(data.products),
        "buy_actions": len(buy),
        "buy_units": int(sum(r.quantity for r in buy)),
        "buy_cost": round(sum(r.total_cost for r in buy), 2),
        "forecast_30d_demand": round(total_demand, 0),
    }


def main() -> PipelineContext:
    t0 = time.time()
    settings = get_settings()
    ctx = PipelineContext()
    tracker = ExperimentTracker(
        tracking_uri=settings.infra.mlflow_tracking_uri, experiment=settings.infra.mlflow_experiment
    )
    manifest = write_manifest(build_manifest())
    run_ctx = new_run_context(
        dataset_version=manifest["dataset_version"],
        environment=settings.infra.environment,
        pipeline_version="2.0.0",
    )
    ctx.run_id = run_ctx.run_id
    log = run_ctx.log
    log(
        "pipeline",
        "pipeline start",
        dataset_version=manifest["dataset_version"],
        backend=tracker.backend_name,
    )
    registry = ModelRegistry()

    # ------------------------------------------------------------- data
    log("ingest", "loading raw tables")
    raw = load_all()
    log("ingest", "raw loaded", sales_rows=len(raw.sales), po_rows=len(raw.purchase_orders))

    cleaned, clean_report = clean_all(raw)
    cleaned.sales.to_csv(config.PROCESSED_FILES["sales"], index=False)
    cleaned.inventory.to_csv(config.PROCESSED_FILES["inventory"], index=False)
    cleaned.purchase_orders.to_csv(config.PROCESSED_FILES["purchase_orders"], index=False)
    (config.DATA_PROCESSED / "cleaning_report.json").write_text(
        json.dumps({"steps": clean_report.to_list()}, indent=2)
    )
    log(
        "clean",
        "cleaning done",
        actions=len(clean_report.to_list()),
        cells_fixed=clean_report.total_fixed(),
    )

    vrep = validate_all(cleaned)
    vdict = vrep.to_dict()
    (config.DATA_PROCESSED / "validation_report.json").write_text(json.dumps(vdict, indent=2))
    log("validate", "validation done", status=vdict["status"], checks=vdict["summary"]["total"])
    if not vrep.passed:
        raise DataValidationError("CRITICAL validation rules failed — aborting pipeline")

    snapshot_now = pd.Timestamp(cleaned.sales["date"].max())
    dq = run_data_quality(cleaned, now=snapshot_now)
    log(
        "data_quality",
        "data-quality sweep",
        status=dq["status"],
        errors=len(dq["errors"]),
        warnings=len(dq["warnings"]),
    )

    log("features", "engineering supervised frame")
    ctx.features = build_features(cleaned.sales, cleaned.inventory)
    ctx.features.to_csv(config.PROCESSED_FILES["features"], index=False)
    log("features", "features built", rows=len(ctx.features))

    log(
        "store",
        "building analytics warehouse",
        target=settings.infra.database_url.split("://")[0] or "sqlite",
    )
    db = store.build_db(cleaned)
    log("store", "warehouse ready", path=str(db))

    # ------------------------------------------------------------- models
    skus = cleaned.products["sku"].tolist()
    log(
        "backtesting",
        "walk-forward bake-off start",
        horizon=settings.eval.horizon,
        folds=settings.eval.folds,
    )
    fc = run_forecast(ctx.features, skus, tracker=tracker, run_ctx=run_ctx)
    ctx.forecasts, ctx.comparison = fc["forecasts"], fc["comparison"]
    ctx.importance, ctx.narrative, ctx.winners = fc["importance"], fc["narrative"], fc["winners"]
    ctx.fold_results = fc["fold_results"]
    log(
        "backtesting",
        "bake-off complete",
        portfolio_best=ctx.narrative["portfolio_best_model"],
        mean_wape=ctx.narrative["mean_wape_by_model"].get(ctx.narrative["portfolio_best_model"]),
    )

    # ------------------------------------------------------------- SHAP layer
    # SHAP answers "why did the model forecast this demand?" — the BUY
    # decomposition in the explain layer stays a separate surface. Optional
    # dependency: absent SHAP falls back to permutation importance untouched.
    fitted_models = fc.get("fitted_models", {})
    shap_df = portfolio_shap_importance(fitted_models, ctx.features, skus)
    if shap_df is not None and len(shap_df):
        shap_df.to_csv(config.ARTIFACTS / "feature_importance_shap.csv", index=False)
        log("explain", "shap attributions computed", skus=int(shap_df["sku"].nunique()))
    else:
        log("explain", "shap unavailable — permutation importance only")
    local_src = {}
    for sku, fitted in fitted_models.items():
        rows = getattr(fitted, "forecast_rows_", None)
        if rows is None or not len(rows):
            continue
        drivers = local_forecast_drivers(fitted, rows)
        if drivers:
            local_src[sku] = {"date": str(rows["date"].iloc[0].date()), **drivers}
    ctx.forecast_explanations = {
        sku: explain_forecast(
            sku,
            ctx.features,
            ctx.forecasts,
            ctx.importance,
            shap_importance=shap_df,
            local_drivers=local_src.get(sku),
        )
        for sku in skus
    }
    (config.ARTIFACTS / "forecast_explanations.json").write_text(
        json.dumps(ctx.forecast_explanations, indent=2, default=str)
    )
    log("explain", "forecast explanations written", skus=len(ctx.forecast_explanations))

    # registry + promotion gates ------------------------------------------------
    last_trained = None
    prod_entries = {}
    for sku in skus:
        win = ctx.winners.get(sku)
        if not win:
            continue
        prod = registry.production_model(sku)
        prod_wape = prod["metrics"].get("mean_WAPE") if prod else None
        gates = promotion_gates(
            candidate_wape=float(win["mean_WAPE"]),
            production_wape=float(prod_wape) if prod_wape else None,
            data_quality_ok=(dq["status"] != "FAIL"),
            artifacts_ok=True,
            validation_ok=vrep.passed,
        )
        mv = ModelVersion(
            model_id=f"{sku}-{win['model'].lower().replace(' ', '-')}-{manifest['dataset_version'][:6]}",
            model_name=win["model"],
            model_family=win["model"],
            sku=sku,
            version=manifest["dataset_version"],
            training_timestamp=run_ctx.started_at,
            dataset_version=manifest["dataset_version"],
            feature_version=feature_version(),
            code_version=run_ctx.git_commit,
            metrics={
                "mean_WAPE": round(win["mean_WAPE"], 4),
                "std_WAPE": round(win["std_WAPE"], 4),
                "composite": round(win["score"], 4),
            },
            selection_reason=f"lowest {settings.eval.selection_metric} across "
            f"{settings.eval.folds} walk-forward folds",
            artifact_path="",
            evaluation={"folds": win["folds"], "unstable": win["unstable"]},
            promotion_gates=gates,
        )
        fitted = fitted_models.get(sku)
        if fitted is not None:
            path = registry.save_model_artifact(fitted, sku, win["model"])
            mv.artifact_path = str(path.relative_to(config.PROJECT_ROOT))
        registry.register(mv)
        try:
            registry.promote(mv.model_id, gates)
            status = "production"
        except PermissionError:
            status = "candidate"  # gates failed — stays a candidate
        prod_entries[sku] = {
            "model_version": mv.version,
            "model_name": mv.model_name,
            "status": status,
        }
        if last_trained is None:
            last_trained = mv.training_timestamp
    log(
        "registry",
        "models registered",
        promoted=sum(1 for v in prod_entries.values() if v["status"] == "production"),
    )

    # ------------------------------------------------------------- decisions
    log("risk", "risk engine")
    ctx.risk_alerts = run_risk_engine(cleaned, ctx.forecasts)
    sev = {"CRITICAL": 0, "WARNING": 0}
    for a in ctx.risk_alerts:
        if a.severity in sev:
            sev[a.severity] += 1
    log("risk", "alerts ready", alerts=len(ctx.risk_alerts), **sev)

    ctx.recommendations = run_recommender(cleaned, ctx.forecasts)
    ctx.explanations = run_explainer(cleaned, ctx.forecasts, ctx.recommendations, ctx.importance)
    ctx.kpis = compute_kpis(cleaned, ctx.forecasts, ctx.recommendations)
    ctx.data = cleaned
    ctx.snapshot_date = str(cleaned.sales["date"].max().date())

    ctx.business_kpis = compute_business_kpis(cleaned, ctx.forecasts, ctx.recommendations)
    log("business_kpis", "business KPIs computed")

    # ------------------------------------------------------------- monitoring
    coverage = portfolio_coverage(ctx.features, ctx.winners)
    log("monitoring", "interval coverage", observed=coverage.get("observed_coverage"))

    # Demand-level features only: rolling means are derived series (their
    # 'drift' is just the level shift the lags already capture) and unit_price
    # follows contract escalation, so including them produced permanent
    # critical alerts with no actionable content.
    drift_features = ["lag_1", "lag_7", "lag_28"]
    drifts = {}
    for sku in skus[:6]:  # representative subset, documented
        ref, rec = reference_recent_split(ctx.features, sku, recent_days=90)
        if len(ref) and len(rec):
            drifts[sku] = drift_report(ref, rec, drift_features, categorical={"promo_flag"})
    drift_overall = max(
        (d["status"] for d in drifts.values()),
        key=lambda s: {"healthy": 0, "warning": 1, "critical": 2}.get(s, 3),
        default="unknown",
    )

    portfolio = fc["summary"]["portfolio"]
    model_by_sku = {sku: w["model"] for sku, w in ctx.winners.items()}
    ctx.model_health = build_model_health(
        fold_results=ctx.fold_results,
        model_by_sku=model_by_sku,
        reference_wape=float(portfolio.iloc[0]["mean_WAPE"]) if len(portfolio) else None,
        baseline_wape=float(
            portfolio.loc[portfolio["model"] == "Moving Average (28d)", "mean_WAPE"].iloc[0]
        )
        if len(portfolio)
        else None,
        drift={
            "status": drift_overall,
            "by_sku": {k: v["status"] for k, v in drifts.items()},
            "detail": {k: v["drifted_columns"] for k, v in drifts.items()},
        },
        coverage=coverage,
        dq_status=dq["status"],
        last_trained=last_trained,
        data_max_date=ctx.snapshot_date,
        monitor_now=snapshot_now.to_pydatetime(),
        production_meta={
            "model_version": manifest["dataset_version"],
            "model_name": ctx.narrative["portfolio_best_model"],
        },
    )
    log(
        "monitoring",
        "model health",
        status=ctx.model_health["status"],
        recommendation=ctx.model_health["recommendation"],
    )

    # ------------------------------------------------------------- artifacts
    sup_stats = _supplier_stats(cleaned.purchase_orders)
    with open(config.ARTIFACTS / "risk_alerts.json", "w") as fh:
        json.dump([a.__dict__ for a in ctx.risk_alerts], fh, indent=2, default=str)
    with open(config.ARTIFACTS / "recommendations.json", "w") as fh:
        json.dump([r.__dict__ for r in ctx.recommendations], fh, indent=2)
    with open(config.ARTIFACTS / "explanations.json", "w") as fh:
        json.dump(ctx.explanations, fh, indent=2, default=str)
    with open(config.ARTIFACTS / "kpis.json", "w") as fh:
        json.dump(
            {"snapshot_date": ctx.snapshot_date, "run_id": ctx.run_id, **ctx.kpis}, fh, indent=2
        )
    sup_stats.to_csv(config.ARTIFACTS / "supplier_stats.csv", index=False)
    (config.ARTIFACTS / "run_manifest.json").write_text(
        json.dumps({**run_ctx.to_dict(), "events": run_ctx.events[-1:]}, indent=2)
    )
    if settings.infra.mlflow_tracking_uri:
        synced = registry.sync_mlflow(settings.infra.mlflow_tracking_uri)
        log("registry", "mlflow mirror", versions_synced=synced)

    log(
        "pipeline",
        "pipeline complete",
        elapsed_s=round(time.time() - t0, 1),
        snapshot=ctx.snapshot_date,
    )
    print(f"\nPIPELINE OK in {time.time() - t0:.1f}s — snapshot {ctx.snapshot_date}")
    print(f"  run_id: {ctx.run_id}  | tracking: {tracker.backend_name}")
    print(f"  model health: {ctx.model_health['status']} — {ctx.model_health['recommendation']}")
    print(f"  kpis: {ctx.kpis}")
    return ctx


if __name__ == "__main__":
    main()
