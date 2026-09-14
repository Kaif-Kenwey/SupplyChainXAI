"""Monitoring tests: data quality gates, drift, PI coverage, model health,
business KPI labelling."""

import json

import numpy as np
import pandas as pd
import pytest

from supplychainxai import config
from supplychainxai.data.ingest import RawData
from supplychainxai.monitoring.business_kpis import compute_business_kpis
from supplychainxai.monitoring.coverage import coverage_from_frames, portfolio_coverage
from supplychainxai.monitoring.data_quality import run_data_quality
from supplychainxai.monitoring.drift import (
    categorical_psi,
    drift_report,
    psi,
    reference_recent_split,
)
from supplychainxai.monitoring.model_health import build_model_health


# ------------------------------------------------------------------ fixtures
def make_raw() -> RawData:
    n = 60
    dates = pd.date_range("2025-11-01", periods=n, freq="D")
    sales = pd.DataFrame(
        {
            "date": dates.tolist() * 2,
            "sku": ["P101"] * n + ["P102"] * n,
            "units_sold": [10.0] * (2 * n),
            "unit_price": [5.0] * (2 * n),
            "promo_flag": [0] * (2 * n),
        }
    )
    inv = pd.DataFrame(
        {
            "date": dates.tolist() * 2,
            "sku": ["P101"] * n + ["P102"] * n,
            "on_hand": [50.0] * (2 * n),
            "on_order": [0.0] * (2 * n),
        }
    )
    products = pd.DataFrame(
        {
            "sku": ["P101", "P102"],
            "name": ["A", "B"],
            "category": ["F", "F"],
            "unit_cost": [4.0, 6.0],
            "base_demand": [140.0, 140.0],
            "trend_per_year": [0.06, 0.06],
            "pack_size": [10, 10],
        }
    )
    suppliers = pd.DataFrame(
        {"supplier_id": ["S1"], "name": ["Acme"], "country": ["X"], "target_otd": [0.9]}
    )
    terms = pd.DataFrame(
        {
            "sku": ["P101", "P102"],
            "supplier_id": ["S1", "S1"],
            "unit_price": [4.0, 6.0],
            "quoted_lead_days": [5, 5],
            "lead_time_std": [1.0, 1.0],
            "is_primary": [True, True],
        }
    )
    pos = pd.DataFrame(
        {
            "po_id": [f"PO-{i}" for i in range(6)],
            "sku": ["P101"] * 6,
            "supplier_id": ["S1"] * 6,
            "order_date": [pd.Timestamp("2025-11-01") + pd.Timedelta(days=5 * i) for i in range(6)],
            "expected_date": [
                pd.Timestamp("2025-11-06") + pd.Timedelta(days=5 * i) for i in range(6)
            ],
            "delivered_date": [
                pd.Timestamp("2025-11-06") + pd.Timedelta(days=5 * i) for i in range(6)
            ],
            "quantity": [100.0] * 6,
            "unit_price": [4.0] * 6,
            "status": ["DELIVERED"] * 6,
        }
    )
    return RawData(
        products=products,
        suppliers=suppliers,
        supply_terms=terms,
        sales=sales,
        inventory=inv,
        purchase_orders=pos,
    )


# ------------------------------------------------------------------ DQ
def test_dq_clean_data_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACTS", tmp_path)
    report = run_data_quality(make_raw(), stale_days=100000)
    assert report["status"] == "PASS"
    assert report["errors"] == []
    assert (tmp_path / "data_quality_report.json").exists()


def test_dq_errors_and_warnings(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACTS", tmp_path)
    data = make_raw()
    data.sales = pd.concat([data.sales, data.sales.iloc[[0]]])  # duplicate
    data.sales.loc[data.sales.index[0], "units_sold"] = -3  # negative
    data.sales.loc[data.sales.index[1], "sku"] = "P999"  # unknown SKU
    data.inventory.loc[data.inventory.index[0], "date"] = pd.Timestamp("2024-01-01")
    report = run_data_quality(data, stale_days=7)
    assert report["status"] == "FAIL"
    rules = {c["rule"] for c in report["errors"]}
    assert "DQ03 duplicate sku-date" in rules
    assert "DQ06 negative units" in rules
    assert any("DQ10" in r for r in rules)
    # warning-level finding also recorded separately
    assert all(c["severity"] in ("ERROR", "WARNING") for c in report["checks"])
    assert isinstance(report["statistics"]["rows"]["sales"], int)


def test_dq_missing_values_are_warnings(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACTS", tmp_path)
    data = make_raw()
    data.sales.loc[data.sales.index[0], "promo_flag"] = np.nan
    report = run_data_quality(data)
    warn_rules = {c["rule"] for c in report["warnings"]}
    assert "DQ02 missing values" in warn_rules
    assert report["status"] == "WARN"  # warning only, not FAIL


def test_dq_schema_change_flagged(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACTS", tmp_path)
    data = make_raw()
    data.sales = data.sales.drop(columns=["promo_flag"])
    report = run_data_quality(data)
    schema_errors = [c for c in report["errors"] if c["rule"] == "DQ01 schema columns"]
    assert len(schema_errors) == 1
    assert "promo_flag" in schema_errors[0]["detail"]


def test_dq_stale_data(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACTS", tmp_path)
    report = run_data_quality(make_raw(), stale_days=3)
    stale = [c for c in report["checks"] if c["rule"] == "DQ15 data freshness"]
    assert any(c["failures"] > 0 for c in stale)
    assert report["status"] == "WARN"


# ------------------------------------------------------------------ drift
def test_psi_identical_is_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 2000)
    assert psi(x, x) < 0.01


def test_psi_shift_increases():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 2000)
    small = rng.normal(0.3, 1, 1000)
    big = rng.normal(3.0, 1, 1000)
    assert psi(ref, small) < psi(ref, big)
    assert psi(ref, big) > 0.25


def test_categorical_psi_and_ks():
    ref = pd.Series([0] * 900 + [1] * 100)
    same = pd.Series([0] * 90 + [1] * 10)
    shifted = pd.Series([0] * 50 + [1] * 50)
    assert categorical_psi(ref, same) < 0.05
    assert categorical_psi(ref, shifted) > 0.5


def test_drift_report_statuses():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"lag_1": rng.normal(20, 4, 1000), "promo_flag": [0] * 900 + [1] * 100})
    recent_stable = pd.DataFrame(
        {"lag_1": rng.normal(20, 4, 300), "promo_flag": [0] * 270 + [1] * 30}
    )
    recent_shift = pd.DataFrame(
        {"lag_1": rng.normal(30, 4, 300), "promo_flag": [0] * 100 + [1] * 200}
    )
    stable = drift_report(ref, recent_stable, ["lag_1", "promo_flag"], categorical={"promo_flag"})
    shift = drift_report(ref, recent_shift, ["lag_1", "promo_flag"], categorical={"promo_flag"})
    assert stable["status"] == "healthy"
    assert shift["status"] in ("warning", "critical")
    assert "lag_1" in shift["drifted_columns"]


def test_reference_recent_split():
    n = 150
    frame = pd.DataFrame(
        {
            "sku": "P1",
            "date": pd.date_range("2025-06-01", periods=n),
            "lag_1": np.arange(n, dtype=float),
        }
    )
    ref, rec = reference_recent_split(frame, "P1", recent_days=30)
    assert len(rec) == 30  # strict > cutoff
    assert ref["date"].max() < rec["date"].min()


# ------------------------------------------------------------------ coverage
def test_coverage_from_frames_math():
    actual = pd.Series([10, 10, 10, 10])
    lower = pd.Series([8, 8, 12, 12])  # 2 inside, 2 outside
    upper = pd.Series([12, 12, 14, 14])
    assert coverage_from_frames(actual, lower, upper) == pytest.approx(0.5)


def test_coverage_nan_safe():
    actual = pd.Series([np.nan, 10])
    lower = pd.Series([8, 8])
    upper = pd.Series([12, 12])
    assert coverage_from_frames(actual, lower, upper) == pytest.approx(1.0)
    assert np.isnan(
        coverage_from_frames(
            pd.Series([], dtype=float), pd.Series([], dtype=float), pd.Series([], dtype=float)
        )
    )


def test_portfolio_coverage_flags(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACTS", tmp_path)
    n = 120
    dates = pd.date_range("2025-05-01", periods=n)
    rng = np.random.default_rng(3)
    sales = pd.DataFrame(
        {
            "date": dates,
            "sku": "P101",
            "units_sold": rng.poisson(20, n).astype(float),
            "unit_price": 5.0,
            "promo_flag": 0,
        }
    )
    inv = pd.DataFrame({"date": dates, "sku": "P101", "on_hand": 100.0, "on_order": 0.0})
    from supplychainxai.data.features import build_features

    feats = build_features(sales, inv)
    winners = {
        "P101": {
            "model": "Moving Average (28d)",
            "score": 0.0,
            "mean_WAPE": 10.0,
            "std_WAPE": 1.0,
            "unstable": False,
            "folds": 2,
            "competing": {},
        }
    }
    report = portfolio_coverage(feats, winners, test_days=20)
    assert report["target_coverage"] == 0.80
    assert report["per_sku"]["P101"]["n_days"] > 0
    assert report["observed_coverage"] is not None


# ------------------------------------------------------------------ health
def test_build_model_health_status_mapping(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACTS", tmp_path)
    folds = pd.DataFrame(
        [
            {
                "sku": "P101",
                "fold": 0,
                "model": "X",
                "MAE": 1,
                "RMSE": 1,
                "MAPE": 1,
                "WAPE": 12.0,
                "bias": 0,
            },
            {
                "sku": "P101",
                "fold": 1,
                "model": "X",
                "MAE": 1,
                "RMSE": 1,
                "MAPE": 1,
                "WAPE": 30.0,
                "bias": 0,
            },  # most recent fold degrades
        ]
    )
    health = build_model_health(
        fold_results=folds,
        model_by_sku={"P101": "X"},
        reference_wape=13.0,
        baseline_wape=33.0,
        drift={"status": "healthy"},
        coverage={"calibration_status": "healthy"},
        dq_status="PASS",
        last_trained="2026-01-01T00:00:00+00:00",
    )
    assert health["status"] == "critical"  # 30% >= critical threshold 28
    assert health["metrics"]["wape_current_fold"] == 30.0
    assert health["retraining"]["retrain_required"] is True
    assert "Retrain" in health["recommendation"]

    # warning band: >= warn (20) but < critical (28), plus degradation vs ref
    folds_warn = folds.copy()
    folds_warn.loc[folds_warn["fold"] == 1, "WAPE"] = 22.0
    health = build_model_health(
        fold_results=folds_warn,
        model_by_sku={"P101": "X"},
        reference_wape=13.0,
        baseline_wape=33.0,
        drift={"status": "healthy"},
        coverage={"calibration_status": "healthy"},
        dq_status="PASS",
        last_trained=None,
    )
    assert health["status"] == "warning"
    loaded = json.loads((tmp_path / "monitoring" / "model_health.json").read_text())
    assert loaded["status"] == "warning"


def test_build_model_health_healthy(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACTS", tmp_path)
    folds = pd.DataFrame(
        [
            {
                "sku": "P101",
                "fold": 1,
                "model": "X",
                "MAE": 1,
                "RMSE": 1,
                "MAPE": 1,
                "WAPE": 12.0,
                "bias": 0.5,
            },
        ]
    )
    health = build_model_health(
        fold_results=folds,
        model_by_sku={"P101": "X"},
        reference_wape=12.5,
        baseline_wape=33.0,
        drift={"status": "healthy"},
        coverage={"calibration_status": "healthy", "observed_coverage": 0.79},
        dq_status="PASS",
        last_trained=None,
    )
    assert health["status"] == "healthy"
    assert health["recommendation"] == "No retraining required"
    assert health["interval_coverage"]["observed_coverage"] == 0.79


# ------------------------------------------------------------------ KPIs
def test_business_kpis_labels_and_math(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACTS", tmp_path)
    data = make_raw()
    forecasts = pd.DataFrame(
        {
            "sku": ["P101"] * 5 + ["P102"] * 5,
            "date": pd.date_range("2025-12-25", periods=5).tolist() * 2,
            "prediction": [10.0] * 5 + [8.0] * 5,
            "lower_80": [8.0] * 10,
            "upper_80": [12.0] * 10,
            "model": ["M"] * 10,
        }
    )

    from supplychainxai.optimization.engine import Recommendation

    recs = [
        Recommendation(
            sku="P101",
            product_name="A",
            action="BUY",
            quantity=100,
            supplier_id="S1",
            supplier_name="Acme",
            unit_price=4.0,
            total_cost=400.0,
            order_by="2025-12-20",
            expected_stockout_date="2025-12-30",
            coverage_days=15.0,
            runner_up={
                "supplier_id": "S2",
                "supplier_name": "Beta",
                "unit_price": 3.5,
                "lead_days": 5.0,
                "on_time_rate": 0.9,
                "score": 0.4,
            },
        )
    ]
    out = compute_business_kpis(data, forecasts, recs)
    idx = out["index"]
    # observed: 50 on-hand × (4 + 6) = 500
    assert idx["inventory_carrying_value"] == pytest.approx(500.0)
    assert idx["open_po_count"] == 0
    assert 0 <= idx["supplier_on_time_delivery"] <= 1
    # simulated: savings = (4.0 - 3.5) × 100
    assert idx["procurement_savings_opportunity"] == pytest.approx(50.0)
    bases = {k["key"]: k["basis"] for k in out["kpis"]}
    assert bases["inventory_carrying_value"] == "observed"
    assert bases["excess_inventory_value"] == "estimated"
    assert bases["procurement_savings_opportunity"] == "simulated"
    assert set(bases.values()) <= {"observed", "estimated", "simulated"}
