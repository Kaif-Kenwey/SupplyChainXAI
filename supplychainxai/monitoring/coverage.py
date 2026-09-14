"""Prediction-interval coverage monitoring.

Measures Prediction Interval Coverage Probability (PICP): how often actual
demand lands inside the predicted 80% interval. Target coverage is the
nominal 80%; the *error* is |observed − target|. Badly calibrated intervals
(under-covering → overconfident planning, over-covering → bloated safety
stock) are flagged per SKU and at portfolio level.

Honesty note: intervals in this system are residual-Gaussian around the
point forecast. Coverage monitoring verifies their EMPIRICAL behaviour; it
does not claim probabilistic calibration beyond that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from supplychainxai import config
from supplychainxai.config.settings import get_settings
from supplychainxai.data.features import model_frame
from supplychainxai.forecasting.models import build_model_zoo


def coverage_from_frames(actual: pd.Series, lower: pd.Series, upper: pd.Series) -> float:
    """Fraction of actuals inside [lower, upper]; NaN-safe."""
    a = np.asarray(actual, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(lo) | np.isnan(hi))
    if not mask.any():
        return float("nan")
    inside = (a[mask] >= lo[mask]) & (a[mask] <= hi[mask])
    return float(inside.mean())


def _winner_model(label: str):
    zoo = {getattr(m, "label", type(m).__name__): m for m in build_model_zoo()}
    return zoo.get(label)


def portfolio_coverage(
    features: pd.DataFrame, winners: dict[str, dict], test_days: int | None = None
) -> dict:
    """Empirical 80%-PI coverage per SKU for the selected models.

    Protocol per SKU: fit the winning model on everything except the final
    `test_days`, forecast that window with intervals, compare against
    actuals. Same protocol as the legacy hold-out — reused, not duplicated.
    """
    cfg = get_settings().monitoring
    test_days = test_days or config.TEST_WINDOW_DAYS
    per_sku = {}
    for sku, win in winners.items():
        frame = model_frame(features, sku)
        model = _winner_model(win["model"])
        if model is None or len(frame) <= test_days:
            continue
        split = len(frame) - test_days
        train, test = frame.iloc[:split], frame.iloc[split:]
        try:
            fresh = _winner_model(win["model"])
            fresh.fit(train)
            pred = fresh.predict(test_days)
        except Exception:
            continue
        merged = pred.assign(actual=pred["date"].map(test.set_index("date")["units_sold"])).dropna(
            subset=["actual"]
        )
        if merged.empty:
            continue
        cov = coverage_from_frames(merged["actual"], merged["lower_80"], merged["upper_80"])
        per_sku[sku] = {
            "observed_coverage": round(cov, 3) if cov == cov else None,
            "n_days": len(merged),
            "flag": _flag(cov, cfg.coverage_target, cfg.coverage_min),
        }
    observed = [
        v["observed_coverage"] for v in per_sku.values() if v["observed_coverage"] is not None
    ]
    portfolio = float(np.mean(observed)) if observed else float("nan")
    # Informational recalibration hint (Gaussian z-ratio): the factor by which
    # residual sigma would need widening to hit the nominal coverage. The
    # pipeline never applies it silently — safety stock and PIs are business
    # decisions; this number just quantifies the miscalibration.
    sigma_scale_hint = None
    if portfolio == portfolio and 0.0 < portfolio < 1.0 and portfolio < cfg.coverage_target:
        from supplychainxai.forecasting.models import Z80

        z_obs = float(_inv_norm((1.0 + portfolio) / 2.0))
        sigma_scale_hint = round(Z80 / z_obs, 3) if z_obs > 0 else None
    return {
        "target_coverage": cfg.coverage_target,
        "nominal_interval": "80%",
        "observed_coverage": round(portfolio, 3) if portfolio == portfolio else None,
        "coverage_error": (
            round(abs(portfolio - cfg.coverage_target), 3) if portfolio == portfolio else None
        ),
        "calibration_status": ("healthy" if portfolio >= cfg.coverage_min else "warning")
        if portfolio == portfolio
        else "unknown",
        "sigma_scale_for_target": sigma_scale_hint,
        "per_sku": per_sku,
    }


def _inv_norm(p: float) -> float:
    """Inverse standard-normal CDF (scipy)."""
    from scipy import stats

    return float(stats.norm.ppf(p))


def _flag(cov: float, target: float, floor: float) -> str:
    if cov != cov:
        return "unknown"
    if cov < floor:
        return "under-covering"
    if cov > target + 0.10:
        return "over-covering"
    return "ok"
