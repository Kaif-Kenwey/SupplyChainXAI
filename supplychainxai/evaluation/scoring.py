"""Model-selection scoring: configurable metric → comparable per-SKU scores.

The composite score min-max normalises each component metric ACROSS CANDIDATE
MODELS within one SKU (so the scale of one metric cannot dominate the others),
then combines them with configurable weights:

    composite = w_wape·norm(WAPE) + w_mae·norm(MAE) + w_rmse·norm(RMSE)

Lower is better. Normalisation is per-SKU because error magnitudes differ by
an order of magnitude across a portfolio; what matters for selection is the
relative ranking inside each SKU.

If a metric does not discriminate (all candidates identical), its normalised
contribution is 0 for every model — the metric simply does not influence that
SKU's choice, instead of injecting random noise.
"""

from __future__ import annotations

import pandas as pd


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi - lo < 1e-12:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def composite_scores(
    per_sku: pd.DataFrame, w_wape: float = 0.5, w_mae: float = 0.3, w_rmse: float = 0.2
) -> pd.DataFrame:
    """Add a `composite` column to a per-(sku, model) summary frame.

    Expects the WalkForwardBacktester.summarize()['per_sku'] frame
    (columns mean_WAPE / mean_MAE / mean_RMSE per sku+model).
    """
    df = per_sku.copy()
    parts = (
        w_wape * df.groupby("sku")["mean_WAPE"].transform(_minmax)
        + w_mae * df.groupby("sku")["mean_MAE"].transform(_minmax)
        + w_rmse * df.groupby("sku")["mean_RMSE"].transform(_minmax)
    )
    df["composite"] = parts.round(4)
    wsum = w_wape + w_mae + w_rmse
    if abs(wsum - 1.0) > 1e-9:
        # keep scores comparable when weights are reconfigured
        df["composite"] = (df["composite"] / wsum).round(4)
    return df


def select_winner(scored: pd.DataFrame, metric: str = "composite") -> dict[str, dict]:
    """Deterministic per-SKU winner selection on a scored summary frame.

    Ties are broken alphabetically by model name so repeated runs select the
    same model — selection must never depend on dict/row ordering.
    """
    metric_col = {
        "composite": "composite",
        "wape": "mean_WAPE",
        "mae": "mean_MAE",
        "rmse": "mean_RMSE",
    }.get(metric.lower())
    if metric_col is None:
        raise ValueError(f"unknown selection metric: {metric}")
    if metric_col not in scored.columns:
        raise ValueError(f"column {metric_col} missing from scored frame")

    winners: dict[str, dict] = {}
    for sku, g in scored.groupby("sku"):
        valid = g.dropna(subset=[metric_col])
        if valid.empty:
            continue
        ordered = valid.sort_values([metric_col, "model"], ascending=[True, True])
        best = ordered.iloc[0]
        winners[sku] = {
            "model": best["model"],
            "score": float(best[metric_col]),
            "metric": metric,
            "mean_WAPE": float(best["mean_WAPE"]),
            "std_WAPE": float(best.get("std_WAPE", float("nan"))),
            "unstable": bool(best.get("unstable", False)),
            "folds": int(best.get("folds", 0)),
            "competing": {r.model: float(getattr(r, metric_col)) for r in ordered.itertuples()},
        }
    return winners
