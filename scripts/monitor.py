#!/usr/bin/env python3
"""Standalone monitoring sweep over the latest committed artifacts.

    python scripts/monitor.py

Recomputes data quality, feature drift, interval coverage and model health
from the current data + artifacts and (re)writes:
    artifacts/data_quality_report.json
    artifacts/monitoring/model_health.json

Useful for running monitoring WITHOUT retraining (e.g. on a schedule between
training runs). Requires that forecasts/evaluation artifacts already exist.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from supplychainxai import config
from supplychainxai.data.clean import clean_all
from supplychainxai.data.features import build_features
from supplychainxai.data.ingest import load_all
from supplychainxai.mlops.provenance import load_manifest
from supplychainxai.monitoring.coverage import portfolio_coverage
from supplychainxai.monitoring.data_quality import run_data_quality
from supplychainxai.monitoring.drift import (
    drift_report,
    reference_recent_split,
)
from supplychainxai.monitoring.model_health import build_model_health


def main() -> None:
    raw = load_all()
    cleaned, _ = clean_all(raw)
    manifest = load_manifest()
    print(f"dataset {manifest['dataset_version']} — running monitors ...")

    snapshot_now = pd.Timestamp(cleaned.sales["date"].max()).to_pydatetime()
    dq = run_data_quality(cleaned, now=snapshot_now)
    print(
        f"data quality: {dq['status']} ({len(dq['errors'])} errors, {len(dq['warnings'])} warnings)"
    )

    selection = json.loads((config.ARTIFACTS / "model_selection.json").read_text())
    winners = {
        sku: {
            "model": v["model"],
            "score": v["score"],
            "mean_WAPE": v["mean_WAPE"],
            "std_WAPE": v["std_WAPE"],
            "unstable": v.get("unstable", False),
            "folds": v.get("folds", 0),
            "competing": {},
        }
        for sku, v in selection.get("per_sku_selection", {}).items()
    }
    fold_path = config.ARTIFACTS_EVALUATION / "walk_forward_results.csv"
    fold_results = pd.read_csv(fold_path) if fold_path.exists() else None
    features = build_features(cleaned.sales, cleaned.inventory)

    coverage = portfolio_coverage(features, winners)
    print(
        f"PI coverage: {coverage.get('observed_coverage')} (target {coverage['target_coverage']})"
    )

    # Demand-level features only: rolling means are derived series (their
    # 'drift' is just the level shift the lags already capture) and unit_price
    # follows contract escalation, so including them produced permanent
    # critical alerts with no actionable content.
    drift_features = ["lag_1", "lag_7", "lag_28"]
    drifts = {}
    for sku in list(winners)[:6]:
        ref, rec = reference_recent_split(features, sku, recent_days=90)
        if len(ref) and len(rec):
            drifts[sku] = drift_report(ref, rec, drift_features, categorical={"promo_flag"})
    drift_overall = max(
        (d["status"] for d in drifts.values()),
        key=lambda s: {"healthy": 0, "warning": 1, "critical": 2}.get(s, 3),
        default="unknown",
    )

    portfolio = (
        pd.read_csv(config.ARTIFACTS_EVALUATION / "walk_forward_portfolio.csv")
        if (config.ARTIFACTS_EVALUATION / "walk_forward_portfolio.csv").exists()
        else None
    )
    baseline_wape = None
    reference_wape = None
    if portfolio is not None and len(portfolio):
        reference_wape = float(portfolio.iloc[0]["mean_WAPE"])
        base_row = portfolio[portfolio["model"] == "Moving Average (28d)"]
        baseline_wape = float(base_row["mean_WAPE"].iloc[0]) if len(base_row) else None

    health = build_model_health(
        fold_results=fold_results,
        model_by_sku={s: w["model"] for s, w in winners.items()},
        reference_wape=reference_wape,
        baseline_wape=baseline_wape,
        drift={"status": drift_overall, "by_sku": {k: v["status"] for k, v in drifts.items()}},
        coverage=coverage,
        dq_status=dq["status"],
        last_trained=selection.get("provenance", {}).get("generated_at"),
        data_max_date=str(cleaned.sales["date"].max().date()),
        monitor_now=snapshot_now,
        production_meta={
            "model_version": manifest["dataset_version"],
            "model_name": selection.get("portfolio_best_model"),
        },
    )
    print(f"model health: {health['status']} — {health['recommendation']}")


if __name__ == "__main__":
    main()
