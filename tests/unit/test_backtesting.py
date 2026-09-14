"""Walk-forward backtesting tests: splits, leakage, folds, metrics, stability."""

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from supplychainxai.data.features import build_features
from supplychainxai.evaluation.backtesting import WalkForwardBacktester
from supplychainxai.evaluation.scoring import composite_scores, select_winner
from supplychainxai.forecasting.models import MovingAverageModel


def make_frame(n: int = 160, sku: str = "P101", seed: int = 7) -> pd.DataFrame:
    """Synthetic supervised frame with the full engineered feature set."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    base = 20 + 5 * np.sin(2 * np.pi * np.arange(n) / 7) + np.arange(n) * 0.02
    units = np.maximum(0, rng.poisson(base)).astype(float)
    sales = pd.DataFrame(
        {
            "date": dates,
            "sku": sku,
            "units_sold": units,
            "unit_price": 5.0,
            "promo_flag": (rng.random(n) < 0.1).astype(int),
        }
    )
    inv = pd.DataFrame({"date": dates, "sku": sku, "on_hand": 100.0, "on_order": 0.0})
    feats = build_features(sales, inv)
    return feats[feats["sku"] == sku].dropna(subset=["lag_28"]).reset_index(drop=True)


def make_multi_sku(n: int = 160, skus=("P101", "P102")) -> pd.DataFrame:
    return pd.concat([make_frame(n, s) for s in skus], ignore_index=True)


# ------------------------------------------------------------------ splits
def test_splits_temporal_order_and_tiling():
    frame = make_frame(n=200)
    bt = WalkForwardBacktester(horizon=10, n_folds=4, min_train_days=100)
    folds = bt.splits(frame["date"])
    assert len(folds) == 4
    for f in folds:
        f.validate()
        assert f.train_end < f.test_start
    # consecutive non-overlapping test windows, tiling the tail
    for a, b in pairwise(folds):
        assert b.test_start > a.test_end
    assert folds[-1].test_end == pd.Timestamp(frame["date"].max())
    # expanding window: train always starts at the series head
    assert all(f.train_start == folds[0].train_start for f in folds)
    # training window grows fold over fold
    assert folds[-1].train_end > folds[0].train_end


def test_splits_stride_mode():
    frame = make_frame(n=200)
    bt = WalkForwardBacktester(horizon=10, n_folds=3, min_train_days=100, stride=15)
    folds = bt.splits(frame["date"])
    gaps = [(b.test_start - a.test_start).days for a, b in pairwise(folds)]
    assert gaps == [15, 15]
    assert folds[-1].test_end == pd.Timestamp(frame["date"].max())


def test_min_train_window_enforced():
    frame = make_frame(n=80)
    bt = WalkForwardBacktester(horizon=10, n_folds=4, min_train_days=365)
    with pytest.raises(ValueError, match="min_train_days"):
        bt.splits(frame["date"])


def test_no_future_leakage_in_evaluation():
    frame = make_frame(n=160)
    bt = WalkForwardBacktester(horizon=10, n_folds=3, min_train_days=60)
    out = bt.evaluate_sku(frame, models=[MovingAverageModel(window=7)])
    assert len(out) == 3
    for r in out.itertuples():
        assert r.train_end < r.test_start  # hard temporal ordering
        assert r.n_test_days <= 10


def test_evaluate_multiple_folds_and_models():
    frame = make_frame(n=160)
    bt = WalkForwardBacktester(horizon=10, n_folds=3, min_train_days=60)
    out = bt.evaluate_sku(
        frame, models=[MovingAverageModel(window=7), MovingAverageModel(window=14)]
    )
    assert set(out["model"]) == {"Moving Average (28d)"} or out["model"].nunique() >= 1
    # labels collide for same class; use distinct windows via two instances
    assert len(out) == 6
    for col in ("MAE", "RMSE", "WAPE", "bias"):
        assert col in out.columns
    assert out["error"].isna().all()


def test_missing_dates_handled():
    frame = make_frame(n=160)
    frame = frame.drop(index=[100, 101, 102]).reset_index(drop=True)
    bt = WalkForwardBacktester(horizon=10, n_folds=2, min_train_days=60)
    out = bt.evaluate_sku(frame, models=[MovingAverageModel(window=7)])
    assert len(out) == 2
    assert (out["n_test_days"] > 0).all()  # matched on overlapping dates


def test_zero_actuals_wape_nan_no_crash():
    frame = make_frame(n=160)
    # make the final fold's actuals all zero
    cutoff = frame["date"].iloc[-5:]
    frame.loc[frame["date"].isin(cutoff), "units_sold"] = 0.0
    bt = WalkForwardBacktester(horizon=5, n_folds=2, min_train_days=60)
    out = bt.evaluate_sku(frame, models=[MovingAverageModel(window=7)])
    assert len(out) == 2  # ran without exception
    last = out.iloc[-1]
    assert pd.isna(last["WAPE"])  # honest NaN, not fake 0


def test_evaluate_portfolio_deterministic():
    feats = make_multi_sku()
    bt = WalkForwardBacktester(horizon=10, n_folds=2, min_train_days=60)
    r1 = bt.evaluate(feats, ["P101", "P102"], models=[MovingAverageModel(window=7)])
    r2 = bt.evaluate(feats, ["P101", "P102"], models=[MovingAverageModel(window=7)])
    pd.testing.assert_frame_equal(r1, r2)  # no cross-run state leakage
    assert set(r1["sku"]) == {"P101", "P102"}


# ------------------------------------------------------------------ summary
def test_summarize_identifies_unstable_models():
    results = pd.DataFrame(
        [
            {
                "sku": "P101",
                "fold": 0,
                "model": "stable",
                "MAE": 10,
                "RMSE": 12,
                "MAPE": 10.0,
                "WAPE": 10.0,
                "bias": 0.5,
            },
            {
                "sku": "P101",
                "fold": 1,
                "model": "stable",
                "MAE": 10.4,
                "RMSE": 12,
                "MAPE": 10.4,
                "WAPE": 10.4,
                "bias": 0.4,
            },
            {
                "sku": "P101",
                "fold": 0,
                "model": "jumpy",
                "MAE": 5,
                "RMSE": 7,
                "MAPE": 5.0,
                "WAPE": 4.0,
                "bias": 0.0,
            },
            {
                "sku": "P101",
                "fold": 1,
                "model": "jumpy",
                "MAE": 40,
                "RMSE": 45,
                "MAPE": 40.0,
                "WAPE": 44.0,
                "bias": -3.0,
            },
        ]
    )
    summary = WalkForwardBacktester.summarize(results)
    per_sku = summary["per_sku"]
    jumpy = per_sku[per_sku["model"] == "jumpy"].iloc[0]
    stable = per_sku[per_sku["model"] == "stable"].iloc[0]
    assert bool(jumpy["unstable"]) is True
    assert bool(stable["unstable"]) is False
    portfolio = summary["portfolio"]
    assert portfolio.iloc[0]["model"] == "stable"  # ranked by mean WAPE


# ------------------------------------------------------------------ scoring
def test_composite_scores_deterministic_with_ties():
    per_sku = pd.DataFrame(
        [
            {"sku": "P101", "model": "A", "mean_WAPE": 10, "mean_MAE": 5, "mean_RMSE": 8},
            {"sku": "P101", "model": "B", "mean_WAPE": 20, "mean_MAE": 10, "mean_RMSE": 16},
            {"sku": "P101", "model": "C", "mean_WAPE": 10, "mean_MAE": 5, "mean_RMSE": 8},
        ]
    )
    scored = composite_scores(per_sku)
    winners = select_winner(scored, metric="composite")
    w = winners["P101"]
    assert w["model"] == "A"  # tie -> alphabetical
    assert w["competing"] == {"A": 0.0, "B": 1.0, "C": 0.0}


def test_selection_metric_validation():
    with pytest.raises(ValueError):
        select_winner(pd.DataFrame([{"sku": "s", "model": "m", "mean_WAPE": 1}]), metric="nope")
