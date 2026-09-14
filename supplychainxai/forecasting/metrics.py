"""Forecast evaluation metrics."""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(d**2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE in %, excluding zero-actual days (stockouts censor true demand)."""
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = yt > 1e-9
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100)


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted (volume-scaled) error in % — robust to small-actual days."""
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    total = np.abs(yt).sum()
    if total < 1e-9:
        return float("nan")
    return float(np.abs(yt - yp).sum() / total * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denom = (np.abs(yt) + np.abs(yp)) / 2.0
    mask = denom > 1e-9
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(yt[mask] - yp[mask]) / denom[mask]) * 100)


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean error (pred − actual) in units. Positive = systematic over-forecast."""
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if len(yt) == 0:
        return float("nan")
    return float(np.mean(yp - yt))


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "WAPE": wape(y_true, y_pred),
        "SMAPE": smape(y_true, y_pred),
        "bias": bias(y_true, y_pred),
    }
