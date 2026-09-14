"""Forecast model zoo: statistical baselines, classical time series, ML models.

Every model implements the same contract:

    fit(frame)            frame = supervised frame for one SKU (see features.py)
    predict(horizon)      -> pd.DataFrame[date, prediction, lower_80, upper_80]
    residuals()           -> in-sample residuals (sigma used for intervals)

Statistical models use the demand series only; ML models use the engineered
feature matrix and forecast recursively (predicted values are fed back as
lags; promo is assumed 0 and price held constant over the horizon).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from supplychainxai.data.features import FEATURE_COLUMNS, future_calendar

Z80 = 1.2816  # 80% prediction interval


# ------------------------------------------------------------------ baselines
class MovingAverageModel:
    """Simple moving average of the last `window` days."""

    label = "Moving Average (28d)"

    def __init__(self, window: int = 28):
        self.window = window
        self.fitted = False

    def fit(self, frame: pd.DataFrame) -> MovingAverageModel:
        self.y = frame["units_sold"].astype(float)
        self.dates = frame["date"]
        self.sigma = float(self.y.iloc[-self.window :].std() or 0.0)
        self.fitted = True
        return self

    def residuals(self) -> np.ndarray:
        return (self.y - self.y.rolling(self.window).mean()).dropna().to_numpy()

    def predict(self, horizon: int) -> pd.DataFrame:
        level = float(self.y.iloc[-self.window :].mean())
        return self._flat(level, horizon)

    def _flat(self, level: float, horizon: int) -> pd.DataFrame:
        dates = pd.date_range(self.dates.iloc[-1] + pd.Timedelta(days=1), periods=horizon)
        return pd.DataFrame(
            {
                "date": dates,
                "prediction": max(level, 0.0),
                "lower_80": max(level - Z80 * self.sigma, 0.0),
                "upper_80": max(level, 0.0) + Z80 * self.sigma,
            }
        )


class ExponentialSmoothingModel:
    """Holt-Winters additive trend + weekly seasonality (statsmodels)."""

    label = "Holt-Winters ES"

    def __init__(self, seasonal_periods: int = 7):
        self.seasonal_periods = seasonal_periods

    def fit(self, frame: pd.DataFrame) -> ExponentialSmoothingModel:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        self.dates = frame["date"]
        self.y = frame["units_sold"].astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = ExponentialSmoothing(
                self.y,
                trend="add",
                seasonal="add",
                seasonal_periods=self.seasonal_periods,
                initialization_method="estimated",
            ).fit(optimized=True)
        self.sigma = float(np.std(self.model.resid))
        return self

    def residuals(self) -> np.ndarray:
        return np.asarray(self.model.resid)

    def predict(self, horizon: int) -> pd.DataFrame:
        f = np.asarray(self.model.forecast(horizon))
        dates = pd.date_range(self.dates.iloc[-1] + pd.Timedelta(days=1), periods=horizon)
        return pd.DataFrame(
            {
                "date": dates,
                "prediction": np.clip(f, 0, None),
                "lower_80": np.clip(f - Z80 * self.sigma, 0, None),
                "upper_80": f + Z80 * self.sigma,
            }
        )


class SARIMAModel:
    """SARIMA(1,1,1)(1,0,1,7) — weekly-seasonal ARIMA for daily data."""

    label = "SARIMA (1,1,1)x(1,0,1)7"

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7)):
        self.order, self.seasonal_order = order, seasonal_order

    def fit(self, frame: pd.DataFrame) -> SARIMAModel:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        self.dates = frame["date"]
        y = frame["units_sold"].astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = SARIMAX(
                y,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
        self.sigma = float(np.std(self.model.resid))
        return self

    def residuals(self) -> np.ndarray:
        return np.asarray(self.model.resid)

    def predict(self, horizon: int) -> pd.DataFrame:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = self.model.get_forecast(horizon)
        mean = np.clip(np.asarray(f.predicted_mean), 0, None)
        std = np.asarray(f.se_mean)
        dates = pd.date_range(self.dates.iloc[-1] + pd.Timedelta(days=1), periods=horizon)
        return pd.DataFrame(
            {
                "date": dates,
                "prediction": mean,
                "lower_80": np.clip(mean - Z80 * std, 0, None),
                "upper_80": mean + Z80 * std,
            }
        )


# ------------------------------------------------------------------ ML models
class _SklearnModel:
    """Shared recursive-forecast machinery for tabular regressors."""

    def __init__(self, estimator, name: str):
        self.estimator = estimator
        self.name = name
        self.label = name

    def fit(self, frame: pd.DataFrame) -> _SklearnModel:
        self.frame = frame.reset_index(drop=True)
        X = frame[FEATURE_COLUMNS]
        y = frame["units_sold"].astype(float)
        self.estimator.fit(X, y)
        self.sigma = float(np.std(y - self.estimator.predict(X)))
        self.last_price = float(frame["unit_price"].iloc[-1])
        self.n_days = len(frame)
        return self

    def residuals(self) -> np.ndarray:
        return self.frame["units_sold"].to_numpy() - self.estimator.predict(
            self.frame[FEATURE_COLUMNS]
        )

    def predict(self, horizon: int) -> pd.DataFrame:
        hist = self.frame[["date", "units_sold", "promo_flag", "unit_price"]].copy()
        preds = []
        forecast_rows = []  # exact feature rows built for the horizon
        last_date = hist["date"].iloc[-1]
        cal = future_calendar(last_date, horizon)

        for h in range(horizon):
            y = hist["units_sold"].astype(float)
            row = {
                "date": cal["date"].iloc[h],
                "dow": cal["dow"].iloc[h],
                "month": cal["month"].iloc[h],
                "weekofyear": cal["weekofyear"].iloc[h],
                "trend_idx": float(self.frame["trend_idx"].iloc[-1]) * (1 + (h + 1) / 1000.0),
                "lag_1": y.iloc[-1],
                "lag_7": y.iloc[-7] if len(y) >= 7 else y.mean(),
                "lag_14": y.iloc[-14] if len(y) >= 14 else y.mean(),
                "lag_28": y.iloc[-28] if len(y) >= 28 else y.mean(),
                "roll_mean_7": y.iloc[-7:].mean(),
                "roll_std_7": y.iloc[-7:].std() if len(y) >= 8 else 0.0,
                "roll_mean_28": y.iloc[-28:].mean(),
                "roll_std_28": y.iloc[-28:].std() if len(y) >= 29 else 0.0,
                "promo_flag": 0.0,  # promos unknowable ahead
                "unit_price": self.last_price,
                "stockout_flag": 0.0,  # assume healthy stock ahead
            }
            p = float(self.estimator.predict(pd.DataFrame([row])[FEATURE_COLUMNS])[0])
            p = max(p, 0.0)
            preds.append(p)
            forecast_rows.append(row)
            hist = pd.concat(
                [
                    hist,
                    pd.DataFrame(
                        {
                            "date": [cal["date"].iloc[h]],
                            "units_sold": [p],
                            "promo_flag": [0],
                            "unit_price": [self.last_price],
                        }
                    ),
                ],
                ignore_index=True,
            )
        # expose the horizon's input rows so explainers (e.g. SHAP local
        # attributions) can attribute the exact rows that were predicted on
        self.forecast_rows_ = pd.DataFrame(forecast_rows)

        sigma = self.sigma
        dates = cal["date"]
        return pd.DataFrame(
            {
                "date": dates,
                "prediction": preds,
                "lower_80": np.clip(np.array(preds) - Z80 * sigma, 0, None),
                "upper_80": np.array(preds) + Z80 * sigma,
            }
        )


def _tree_models():
    """Return [(name, factory)] of available tree ensembles (xgboost optional)."""
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

    models = [
        (
            "Random Forest",
            lambda: RandomForestRegressor(
                n_estimators=300, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=-1
            ),
        ),
        (
            "Gradient Boosting",
            lambda: GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.9, random_state=42
            ),
        ),
    ]
    try:  # optional accelerator — same interface if xgboost is installed
        from xgboost import XGBRegressor

        models.append(
            (
                "XGBoost",
                lambda: XGBRegressor(
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0,
                ),
            )
        )
    except ImportError:
        pass
    return models


def build_model_zoo() -> list:
    """Instantiate every candidate model with uniform fit/predict contract."""
    zoo: list = [MovingAverageModel(window=28), ExponentialSmoothingModel(), SARIMAModel()]
    for name, factory in _tree_models():
        zoo.append(_SklearnModel(factory(), name))
    return zoo


def model_family(model) -> str:
    return model.__class__.__name__
