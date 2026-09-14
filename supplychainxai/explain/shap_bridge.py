"""SHAP explainability bridge for tree-based forecast models.

Answers a DIFFERENT question than the business explanation layer:

  * SHAP        → "Why did the model forecast this demand?"
                  (feature attributions of the tree estimator)
  * explain/    → "Why did the system recommend buying this quantity?"
                  (exact additive decomposition of the BUY number)

The two stay separate by design. SHAP is an OPTIONAL dependency: when it is
missing, or the winning model is not a tree ensemble, every function here
returns None and the existing permutation-importance path is used unchanged
— the pipeline never fails because SHAP is absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from supplychainxai.data.features import FEATURE_COLUMNS

SUPPORTED_ESTIMATOR_TYPES = (
    "XGBRegressor",
    "RandomForestRegressor",
    "GradientBoostingRegressor",
)


def shap_available() -> bool:
    try:
        import shap  # noqa: F401

        return True
    except Exception:
        return False


def _tree_explainer(model):
    """TreeExplainer for a fitted sklearn/xgboost estimator, or None."""
    est_type = type(model).__name__
    if est_type not in SUPPORTED_ESTIMATOR_TYPES:
        return None
    try:
        import shap

        return shap.TreeExplainer(model)
    except Exception:
        return None


def global_shap_importance(fitted_model, sku: str, frame: pd.DataFrame) -> pd.DataFrame | None:
    """Mean |SHAP| per feature for one SKU's fitted tabular model.

    `fitted_model` is the zoo wrapper (has `.estimator` and `.frame`).
    Returns the same row shape as permutation importance plus a
    `method` column so the two can coexist in one artifact lineage.
    """
    if not hasattr(fitted_model, "estimator"):
        return None
    explainer = _tree_explainer(fitted_model.estimator)
    if explainer is None:
        return None
    try:
        X = frame[FEATURE_COLUMNS]
        sv = explainer.shap_values(X)
        # RF wrappers can return a list (one array per output); average it
        if isinstance(sv, list):
            sv = sum(abs(s) for s in sv) / len(sv)
        mean_abs = np.abs(np.asarray(sv, dtype=float)).mean(axis=0)
        return pd.DataFrame(
            {
                "sku": sku,
                "model": getattr(fitted_model, "label", type(fitted_model).__name__),
                "feature": FEATURE_COLUMNS,
                "importance": [round(float(v), 6) for v in mean_abs],
                "method": "shap",
            }
        )
    except Exception:
        return None


def portfolio_shap_importance(
    fitted_models: dict, features: pd.DataFrame, skus: list[str]
) -> pd.DataFrame | None:
    """Global SHAP importance for every SKU whose winner is a tree model.

    `fitted_models` maps sku -> fitted zoo model (from final_forecasts).
    Returns None when nothing could be attributed (no SHAP / no tree models).
    """
    outs = []
    for sku in skus:
        fitted = fitted_models.get(sku)
        if fitted is None:
            continue
        frame = features[features["sku"] == sku]
        df = global_shap_importance(fitted, sku, frame)
        if df is not None:
            outs.append(df)
    return pd.concat(outs, ignore_index=True) if outs else None


def local_forecast_drivers(fitted_model, feature_row: pd.DataFrame, top_n: int = 3) -> dict | None:
    """Per-day SHAP contributions for one forecast input row.

    Returns {"top_positive": [...], "top_negative": [...]} — the features
    pushing that day's prediction up and down — or None when SHAP cannot
    run. Values are in UNITS OF DEMAND (SHAP additive contribution).
    """
    if not hasattr(fitted_model, "estimator"):
        return None
    explainer = _tree_explainer(fitted_model.estimator)
    if explainer is None:
        return None
    try:
        X = feature_row[FEATURE_COLUMNS].iloc[[0]]
        sv = explainer.shap_values(X)[0]
        if isinstance(sv, list):
            sv = sv[0]
        contrib = sorted(zip(FEATURE_COLUMNS, [float(v) for v in sv]), key=lambda t: -abs(t[1]))
        pos = [
            {"feature": f, "shap_units": round(v, 2), "feature_value": float(X.iloc[0][f])}
            for f, v in contrib
            if v > 0
        ][:top_n]
        neg = [
            {"feature": f, "shap_units": round(v, 2), "feature_value": float(X.iloc[0][f])}
            for f, v in contrib
            if v < 0
        ][:top_n]
        return {"top_positive": pos, "top_negative": neg}
    except Exception:
        return None
