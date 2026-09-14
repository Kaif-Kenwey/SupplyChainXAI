"""Stage 4 — Feature engineering for the ML forecast models.

Builds one supervised row per (sku, date) with:
  calendar    : day-of-week, month, week-of-year, trend index
  lag         : demand lags 1 / 7 / 14 / 28
  rolling     : mean & std over 7 / 28 days (shifted to avoid leakage)
  commercial  : promo flag, selling price
  operational : stockout flag (sales censored when on-hand hit zero)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "dow",
    "month",
    "weekofyear",
    "trend_idx",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "roll_mean_7",
    "roll_std_7",
    "roll_mean_28",
    "roll_std_28",
    "promo_flag",
    "unit_price",
    "stockout_flag",
]


def build_features(sales: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    """Return a feature frame sorted by (sku, date) with NaN head rows intact."""
    inv = inventory.set_index(["sku", "date"])[["on_hand"]]
    df = sales.sort_values(["sku", "date"]).copy()

    df = df.merge(inv, on=["sku", "date"], how="left")
    df["stockout_flag"] = ((df["on_hand"] <= 0) & (df["units_sold"] == 0)).astype("float64")

    g = df.groupby("sku", group_keys=False)["units_sold"]
    df["lag_1"] = g.shift(1)
    df["lag_7"] = g.shift(7)
    df["lag_14"] = g.shift(14)
    df["lag_28"] = g.shift(28)
    df["roll_mean_7"] = g.transform(lambda s: s.shift(1).rolling(7).mean())
    df["roll_std_7"] = g.transform(lambda s: s.shift(1).rolling(7).std())
    df["roll_mean_28"] = g.transform(lambda s: s.shift(1).rolling(28).mean())
    df["roll_std_28"] = g.transform(lambda s: s.shift(1).rolling(28).std())

    dt = df["date"]
    df["dow"] = dt.dt.dayofweek.astype("float64")
    df["month"] = dt.dt.month.astype("float64")
    df["weekofyear"] = dt.dt.isocalendar().week.astype("float64")
    df["trend_idx"] = (dt.rank(method="first") / len(dt)).astype("float64")

    return df


def model_frame(df: pd.DataFrame, sku: str) -> pd.DataFrame:
    """Feature frame for one SKU, dropping rows whose lags are not yet available."""
    out = df[df["sku"] == sku].dropna(subset=["lag_28"]).reset_index(drop=True)
    return out


def future_calendar(last_date: pd.Timestamp, horizon: int) -> pd.DataFrame:
    """Known-in-advance exogenous features for the forecast horizon."""
    dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "dow": dates.dayofweek.astype("float64"),
            "month": dates.month.astype("float64"),
            "weekofyear": dates.isocalendar().week.astype("float64").to_numpy(),
            "trend_idx": np.linspace(1.0, 1.0 + horizon / 1_000.0, horizon),
        }
    )
