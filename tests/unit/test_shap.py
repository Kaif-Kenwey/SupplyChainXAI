"""SHAP bridge tests — run when shap is installed, skip otherwise."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shap")

from sklearn.ensemble import GradientBoostingRegressor

from supplychainxai.data.features import build_features
from supplychainxai.explain.shap_bridge import (
    global_shap_importance,
    local_forecast_drivers,
    portfolio_shap_importance,
    shap_available,
)
from supplychainxai.forecasting.models import (
    ExponentialSmoothingModel,
    _SklearnModel,
)


def make_frame(n: int = 140) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    dates = pd.date_range("2025-05-01", periods=n)
    units = np.maximum(0, rng.poisson(20 + 4 * np.sin(2 * np.pi * np.arange(n) / 7))).astype(float)
    sales = pd.DataFrame(
        {
            "date": dates,
            "sku": "P101",
            "units_sold": units,
            "unit_price": 5.0,
            "promo_flag": (rng.random(n) < 0.15).astype(float),
        }
    )
    inv = pd.DataFrame({"date": dates, "sku": "P101", "on_hand": 100.0, "on_order": 0.0})
    return build_features(sales, inv).dropna(subset=["lag_28"]).reset_index(drop=True)


def _model():
    return _SklearnModel(
        GradientBoostingRegressor(n_estimators=60, max_depth=3, random_state=0), "GB"
    )


def test_shap_available_and_global_importance():
    assert shap_available() is True
    frame = make_frame()
    model = _model()
    model.fit(frame)
    out = global_shap_importance(model, "P101", frame)
    assert out is not None
    assert set(out["method"]) == {"shap"}
    assert len(out) == 15  # one row per feature
    assert (out["importance"] >= 0).all()  # mean |SHAP| is non-negative
    assert out["importance"].sum() > 0  # real attributions, not zeros


def test_local_drivers_directions():
    frame = make_frame()
    model = _model()
    model.fit(frame)
    model.predict(5)  # builds forecast_rows_
    drivers = local_forecast_drivers(model, model.forecast_rows_)
    assert drivers is not None
    assert set(drivers) == {"top_positive", "top_negative"}
    valid_features = set(model.frame.columns)
    for side in drivers.values():
        assert all(item["feature"] in valid_features for item in side)
        assert all(
            item["shap_units"] * (1 if k == "top_positive" else -1) >= 0
            for k, items in drivers.items()
            for item in items
        )


def test_statistical_models_return_none():
    frame = make_frame()
    hw = ExponentialSmoothingModel().fit(frame)
    assert global_shap_importance(hw, "P101", frame) is None
    assert portfolio_shap_importance({"P101": hw}, frame, ["P101"]) is None


def test_portfolio_importance_skips_statistical_winners():
    frame = make_frame()
    model = _model()
    model.fit(frame)
    out = portfolio_shap_importance(
        {"P101": model, "P999": ExponentialSmoothingModel().fit(frame)}, frame, ["P101", "P999"]
    )
    assert out is not None
    assert set(out["sku"]) == {"P101"}
