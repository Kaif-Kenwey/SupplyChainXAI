"""Drift detection: PSI + KS test over training vs recent feature/actual
distributions — deliberately simple, appropriate per data type.

  * numeric features  → PSI (10 quantile bins from the reference) + KS test
  * categorical flags → PSI over category frequencies

Status per feature: healthy / warning (PSI >= warn or KS p < alpha with a
meaningful shift) / critical (PSI >= crit). The portfolio status is the
worst observed, with a small-N guard: features with too few recent rows are
skipped, never guessed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from supplychainxai.config.settings import get_settings

NUMERIC_EPS = 1e-6


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 5) -> float:
    """Population Stability Index between reference and recent windows.

    NaNs are dropped on both sides; if either side is empty or the reference
    is constant, PSI returns 0.0 (no evidence of drift is not evidence of
    drift).

    `bins=5` (quantile bins from the reference) instead of the textbook 10:
    these windows hold ~90 daily samples each, and thin quantile bins over a
    lumpy discrete demand distribution make PSI explode on noise. With 5
    bins the classic 0.10/0.25 thresholds remain meaningful.
    """
    e = pd.Series(expected).dropna().astype(float)
    a = pd.Series(actual).dropna().astype(float)
    if len(e) < 10 or len(a) < 5:
        return 0.0
    if e.nunique() == 1 and a.nunique() == 1 and e.iloc[0] == a.iloc[0]:
        return 0.0
    try:
        edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1))).astype(float)
        if len(edges) < 3:  # constant reference
            edges = np.array(
                [
                    e.min() - NUMERIC_EPS,
                    (e.max() + a.max()) / 2 + NUMERIC_EPS,
                    a.max() + 2 * NUMERIC_EPS,
                ]
            )
        # Outer bins extend to ±inf: np.histogram would otherwise DROP
        # out-of-range values, silently removing mass and inflating PSI.
        edges[0], edges[-1] = -np.inf, np.inf
        e_hist = np.histogram(e, bins=edges)[0] / len(e)
        a_hist = np.histogram(a, bins=edges)[0] / len(a)
    except (ValueError, IndexError):
        return 0.0
    e_hist = np.clip(e_hist, NUMERIC_EPS, None)
    a_hist = np.clip(a_hist, NUMERIC_EPS, None)
    return float(np.sum((a_hist - e_hist) * np.log(a_hist / e_hist)))


def categorical_psi(expected: pd.Series, actual: pd.Series) -> float:
    """PSI over category frequencies (works for binary flags too)."""
    e = expected.dropna()
    a = actual.dropna()
    if len(e) < 10 or len(a) < 5:
        return 0.0
    levels = sorted(set(e.astype(str)) | set(a.astype(str)))
    e_freq = e.astype(str).value_counts(normalize=True, dropna=False)
    a_freq = a.astype(str).value_counts(normalize=True, dropna=False)
    out = 0.0
    for lvl in levels:
        pe = max(float(e_freq.get(lvl, 0.0)), NUMERIC_EPS)
        pa = max(float(a_freq.get(lvl, 0.0)), NUMERIC_EPS)
        out += (pa - pe) * np.log(pa / pe)
    return float(out)


def ks_test(expected: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    """Two-sample KS statistic + p-value; (0, 1) when undecidable."""
    from scipy import stats

    e = pd.Series(expected).dropna().astype(float)
    a = pd.Series(actual).dropna().astype(float)
    if len(e) < 10 or len(a) < 5:
        return 0.0, 1.0
    try:
        res = stats.ks_2samp(e, a)
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return 0.0, 1.0


def drift_report(
    reference: pd.DataFrame,
    recent: pd.DataFrame,
    columns: list[str],
    categorical: set[str] | None = None,
) -> dict:
    """Per-column drift over the chosen columns + overall status."""
    cfg = get_settings().monitoring
    categorical = categorical or set()
    features = {}
    for col in columns:
        if col not in reference.columns or col not in recent.columns:
            continue
        if col in categorical:
            score = categorical_psi(reference[col], recent[col])
            features[col] = {
                "method": "categorical_psi",
                "psi": round(score, 4),
                "status": _psi_status(score, cfg.psi_warn, cfg.psi_crit),
            }
        else:
            score = psi(reference[col].to_numpy(), recent[col].to_numpy())
            ks_stat, ks_p = ks_test(reference[col].to_numpy(), recent[col].to_numpy())
            status = _psi_status(score, cfg.psi_warn, cfg.psi_crit)
            if status == "healthy" and ks_p < cfg.ks_alpha and ks_stat > 0.08:
                status = "warning"  # distribution shift sans PSI
            features[col] = {
                "method": "psi+ks",
                "psi": round(score, 4),
                "ks_stat": round(ks_stat, 4),
                "ks_p": round(ks_p, 4),
                "status": status,
            }
    worst = "healthy"
    for f in features.values():
        if f["status"] == "critical":
            worst = "critical"
            break
        if f["status"] == "warning":
            worst = "warning"
    drifted = sorted(k for k, v in features.items() if v["status"] != "healthy")
    return {
        "status": worst,
        "features": features,
        "drifted_columns": drifted,
        "n_reference_rows": len(reference),
        "n_recent_rows": len(recent),
    }


def _psi_status(score: float, warn: float, crit: float) -> str:
    if score >= crit:
        return "critical"
    if score >= warn:
        return "warning"
    return "healthy"


def reference_recent_split(
    features: pd.DataFrame, sku: str, recent_days: int = 90
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Temporal split of one SKU's feature frame for shift detection.

    reference = the `recent_days` window immediately BEFORE the last
    `recent_days`; recent = the last `recent_days`. Comparing two same-length
    adjacent windows measures a *level shift* (actionable), while comparing
    against all history would flag ordinary year-on-year growth as drift
    (alert fatigue). Same convention as the risk engine's lead-time drift
    detector.
    """
    frame = features[features["sku"] == sku].sort_values("date")
    cutoff = frame["date"].max() - pd.Timedelta(days=recent_days)
    prev_start = cutoff - pd.Timedelta(days=recent_days)
    reference = frame[(frame["date"] > prev_start) & (frame["date"] <= cutoff)]
    recent = frame[frame["date"] > cutoff]
    return reference, recent
