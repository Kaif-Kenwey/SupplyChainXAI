"""Walk-forward (rolling-origin) backtesting.

Replaces reliance on a single hold-out: the model zoo is evaluated on
multiple expanding-window origins tiled across the tail of the series.

    fold 0        fold 1        fold 2        ...
    [train ────────][test h]
    [train ────────────────][test h]
    [train ────────────────────────][test h]

Guarantees (unit-tested):
  * temporal ordering — every train window ends strictly before its test
    window starts (no future leakage);
  * features are leakage-safe by construction (lags/rolling are shifted in
    `features.py`), so slicing by date preserves that property;
  * each fold fits a FRESH model instance (no state reuse across folds);
  * short series that cannot honour `min_train_days` fail loudly with a
    clear error instead of silently producing misleading folds.

The backtester reuses the existing model zoo contract
(fit / predict(horizon)) and the existing metrics — no duplicated
forecasting logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from supplychainxai.data.features import model_frame
from supplychainxai.forecasting.metrics import all_metrics
from supplychainxai.forecasting.models import build_model_zoo

METRIC_COLS = ["MAE", "RMSE", "MAPE", "WAPE", "bias"]


@dataclass(frozen=True)
class FoldSplit:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp  # origin (inclusive)
    test_start: pd.Timestamp
    test_end: pd.Timestamp  # inclusive

    def validate(self) -> None:
        if not (self.train_start <= self.train_end < self.test_start <= self.test_end):
            raise ValueError(f"non-temporal fold: {self}")


def model_label(model) -> str:
    return getattr(model, "label", type(model).__name__)


class WalkForwardBacktester:
    """Expanding-window rolling-origin evaluator.

    Parameters
    ----------
    horizon        : forecast length per fold (days).
    n_folds        : number of consecutive test windows tiling the tail.
    min_train_days : minimum acceptable training length; folds that would
                     violate it raise `ValueError`.
    stride         : days between fold origins. 0 (default) tiles the tail
                     with consecutive non-overlapping `horizon` windows;
                     a positive stride spaces origins `stride` days apart
                     (last fold still ends at the series end).
    """

    def __init__(
        self,
        horizon: int = 30,
        n_folds: int = 4,
        min_train_days: int = 365,
        stride: int = 0,
        expanding: bool = True,
    ):
        if horizon < 1 or n_folds < 1:
            raise ValueError("horizon and n_folds must be >= 1")
        if stride < 0:
            raise ValueError("stride must be >= 0")
        self.horizon = horizon
        self.n_folds = n_folds
        self.min_train_days = min_train_days
        self.stride = stride
        self.expanding = expanding

    # ------------------------------------------------------------ splits
    def splits(self, dates: pd.Series | pd.DatetimeIndex) -> list[FoldSplit]:
        """Generate folds over the sorted unique dates of a series."""
        d = pd.DatetimeIndex(pd.unique(pd.Series(dates).sort_values()))
        n = len(d)
        if self.stride == 0:
            stride = self.horizon  # tile the tail
        else:
            stride = self.stride
        spans: list[FoldSplit] = []
        for i in range(self.n_folds):
            test_end_idx = n - 1 - (self.n_folds - 1 - i) * stride
            test_start_idx = test_end_idx - self.horizon + 1
            train_end_idx = test_start_idx - 1
            if train_end_idx + 1 < self.min_train_days:
                raise ValueError(
                    f"fold {i}: training window would be {train_end_idx + 1} days "
                    f"(< min_train_days={self.min_train_days}). Reduce n_folds / "
                    f"stride / horizon or provide a longer series."
                )
            fold = FoldSplit(
                fold_id=i,
                train_start=d[0],
                train_end=d[train_end_idx],
                test_start=d[test_start_idx],
                test_end=d[test_end_idx],
            )
            fold.validate()
            spans.append(fold)
        return spans

    # ------------------------------------------------------------ evaluate
    def evaluate_sku(self, frame: pd.DataFrame, models: list | None = None) -> pd.DataFrame:
        """Backtest the zoo on one SKU across all folds.

        Returns one row per (fold, model) with fold metrics and, for the
        winning-model convenience, nothing else — selection happens upstream.
        """
        models = models if models is not None else build_model_zoo()
        rows = []
        for fold in self.splits(frame["date"]):
            train = frame[frame["date"] <= fold.train_end]
            test = frame[(frame["date"] >= fold.test_start) & (frame["date"] <= fold.test_end)]
            if train.empty or test.empty:
                continue
            assert train["date"].max() < test["date"].min(), "leakage guard"
            y_true_by_date = test.set_index("date")["units_sold"].astype(float)

            for factory_model in models:
                name = model_label(factory_model)
                try:
                    model = _fresh(factory_model)  # fresh instance per fold
                    model.fit(train)
                    pred = model.predict(self.horizon)
                    merged = pred.assign(actual=pred["date"].map(y_true_by_date)).dropna(
                        subset=["actual"]
                    )
                    if merged.empty:
                        rows.append(self._row(fold, name, np.nan, "no overlapping actuals"))
                        continue
                    m = all_metrics(merged["actual"].to_numpy(), merged["prediction"].to_numpy())
                    rows.append(
                        {
                            "sku": frame["sku"].iloc[0],
                            "fold": fold.fold_id,
                            "train_start": str(fold.train_start.date()),
                            "train_end": str(fold.train_end.date()),
                            "test_start": str(fold.test_start.date()),
                            "test_end": str(fold.test_end.date()),
                            "model": name,
                            "n_test_days": len(merged),
                            **{k: round(float(v), 4) for k, v in m.items()},
                            "error": None,
                        }
                    )
                except Exception as exc:  # a failing model never kills the fold
                    rows.append(self._row(fold, name, np.nan, str(exc)[:120]))
        return pd.DataFrame(rows)

    @staticmethod
    def _row(fold: FoldSplit, name: str, sku, error: str | None) -> dict:
        base = {
            "sku": sku,
            "fold": fold.fold_id,
            "train_start": str(fold.train_start.date()),
            "train_end": str(fold.train_end.date()),
            "test_start": str(fold.test_start.date()),
            "test_end": str(fold.test_end.date()),
            "model": name,
            "n_test_days": 0,
        }
        if error is None:
            return base
        return {**base, **dict.fromkeys(METRIC_COLS, np.nan), "error": error}

    def evaluate(
        self, features: pd.DataFrame, skus: list[str], models: list | None = None
    ) -> pd.DataFrame:
        """Evaluate all SKUs; returns long frame (sku × fold × model)."""
        outs = []
        for sku in skus:
            frame = model_frame(features, sku)
            if frame.empty:
                continue
            outs.append(self.evaluate_sku(frame, models))
        if not outs:
            return pd.DataFrame(columns=["sku", "fold", "model", *METRIC_COLS])
        return pd.concat(outs, ignore_index=True)

    # ------------------------------------------------------------ summary
    @staticmethod
    def summarize(results: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Aggregate fold metrics → per-(sku, model) and portfolio views.

        Stability: a model is flagged `unstable` for a SKU when its WAPE
        coefficient of variation across folds exceeds 0.5 — i.e. its error
        moves by half its own level between origins, so a single good fold
        would be luck rather than signal.
        """
        valid = results.dropna(subset=["WAPE"])
        per_sku = (
            valid.groupby(["sku", "model"])
            .agg(
                folds=("fold", "nunique"),
                mean_MAE=("MAE", "mean"),
                std_MAE=("MAE", "std"),
                mean_RMSE=("RMSE", "mean"),
                std_RMSE=("RMSE", "std"),
                mean_MAPE=("MAPE", "mean"),
                mean_WAPE=("WAPE", "mean"),
                std_WAPE=("WAPE", "std"),
                median_WAPE=("WAPE", "median"),
                mean_bias=("bias", "mean"),
            )
            .reset_index()
        )
        per_sku["wape_cv"] = (per_sku["std_WAPE"] / per_sku["mean_WAPE"]).fillna(0.0)
        per_sku["unstable"] = per_sku["wape_cv"] > 0.5
        per_sku = per_sku.round(4)

        portfolio = (
            per_sku.groupby("model")
            .agg(
                skus=("sku", "nunique"),
                mean_WAPE=("mean_WAPE", "mean"),
                worst_sku_WAPE=("mean_WAPE", "max"),
                mean_WAPE_std=("std_WAPE", "mean"),
                mean_MAE=("mean_MAE", "mean"),
                mean_bias=("mean_bias", "mean"),
                unstable_skus=("unstable", "sum"),
            )
            .round(4)
            .sort_values("mean_WAPE")
            .reset_index()
        )
        return {"per_sku": per_sku, "portfolio": portfolio}


def _fresh(model):
    """Return a fresh, un-fitted copy of a zoo model (fit mutates state).

    The zoo is built once and used as a prototype list; every fold deep-copies
    its prototype so no estimator state can leak between folds or SKUs.
    """
    import copy

    return copy.deepcopy(model)
