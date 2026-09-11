"""Metric sanity tests against hand-computed values."""
import numpy as np
import pytest

from supplychainxai.forecasting.metrics import all_metrics, mae, mape, rmse, smape, wape


def test_mae_rmse_known_values():
    y = np.array([100.0, 200.0, 300.0])
    p = np.array([110.0, 190.0, 270.0])
    assert mae(y, p) == pytest.approx(50.0 / 3)
    assert rmse(y, p) == pytest.approx(np.sqrt(1100.0 / 3))


def test_mape_excludes_zero_actuals():
    y = np.array([0.0, 200.0])
    p = np.array([500.0, 100.0])          # first day is a stockout-censored actual
    assert mape(y, p) == pytest.approx(50.0)  # only |(200-100)/200| counted


def test_wape_volume_weighted():
    y = np.array([100.0, 900.0])
    p = np.array([150.0, 850.0])
    assert wape(y, p) == pytest.approx(100.0 / 1000.0 * 100)


def test_smape_symmetric():
    y = np.array([100.0])
    assert smape(y, np.array([150.0])) == pytest.approx(smape(np.array([150.0]), y))


def test_all_metrics_keys():
    m = all_metrics(np.array([1.0, 2.0]), np.array([1.5, 2.5]))
    assert {"MAE", "RMSE", "MAPE", "WAPE", "SMAPE"} == set(m)
