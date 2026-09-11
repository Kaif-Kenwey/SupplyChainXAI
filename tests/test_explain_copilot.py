"""Explainability + copilot tests (grounding guarantees)."""
import numpy as np
import pandas as pd
import pytest

from supplychainxai.copilot.engine import detect_intent, detect_sku
from supplychainxai.explain.engine import (explain_forecast, explain_recommendation,
                                           to_natural_language)


def test_intent_detection():
    assert detect_intent("Which products will stock out next month?") == "stockout"
    assert detect_intent("What should I buy for P104?") == "reorder"
    assert detect_intent("How reliable is supplier S4?") == "supplier"
    assert detect_intent("What is the demand forecast for P101?") == "forecast"
    assert detect_intent("Why do you recommend 420 units?") == "explain"
    assert detect_intent("tell me a joke") == "unknown"


def test_sku_detection():
    assert detect_sku("what about p104 ?") == "P104"
    assert detect_sku("P-105 outlook") == "P105"
    assert detect_sku("no sku here") is None


def test_explain_components_reconcile_to_quantity():
    from tests.test_optimizer import build_forecast, build_world
    from supplychainxai.optimization.engine import recommend_sku

    data = build_world()
    fc = build_forecast()
    data.inventory.loc[data.inventory["sku"] == "P101", "on_hand"] = 300
    rec = recommend_sku("P101", data, fc,
                        pd.DataFrame(columns=["supplier_id", "on_time_rate", "avg_lead"]))
    expl = explain_recommendation("P101", data, fc, rec)

    # additive identity: gross + safety - position + rounding == quantity
    units = {c["factor"]: c["units"] for c in expl["components"]}
    total = (units["forecast demand"] + units["safety stock"]
             + units["inventory position"] + units["pack rounding"])
    assert total == pytest.approx(rec.quantity, abs=1.5)
    assert all(0 <= c["impact_pct"] <= 100 for c in expl["components"])
    narr = to_natural_language(rec, expl, data)
    assert narr and "P101" in narr


def test_explain_forecast_grounding():
    sales = pd.DataFrame({"date": pd.date_range("2025-10-01", periods=90, freq="D"),
                          "sku": "P101", "units_sold": [100] * 90,
                          "unit_price": 5.0, "promo_flag": 0})
    features = sales.assign(lag_1=100.0)
    fc = pd.DataFrame({"sku": "P101", "date": pd.date_range("2026-01-01", periods=30),
                       "model": "XGBoost", "prediction": [120.0] * 30,
                       "lower_80": [100.0] * 30, "upper_80": [140.0] * 30})
    expl = explain_forecast("P101", features, fc, None)
    assert expl["forecast_daily_mean"] == 120.0
    assert "120" in expl["narrative"]
    # momentum driver: recent flat -> none beyond promo assumption
    assert any(d["factor"] == "promo assumption" for d in expl["drivers"])
