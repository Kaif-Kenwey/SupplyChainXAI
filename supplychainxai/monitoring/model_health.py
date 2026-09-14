"""Model health: one honest status object assembled from every monitor.

Combines the latest observed forecast performance (most recent walk-forward
fold), the reference performance, drift findings, interval coverage and
data-quality status into the `model_health.json` contract consumed by the
API and the dashboard — and feeds the retraining policy with the same
signals, so the dashboard, the policy and the API can never disagree.
"""

from __future__ import annotations

import json

import pandas as pd

from supplychainxai import config
from supplychainxai.config.settings import get_settings
from supplychainxai.mlops.retraining import evaluate_retraining
from supplychainxai.mlops.runlog import utc_now

STATUS_ORDER = {"healthy": 0, "warning": 1, "critical": 2, "unknown": 3}


def _worse(a: str, b: str) -> str:
    return a if STATUS_ORDER.get(a, 3) >= STATUS_ORDER.get(b, 3) else b


def latest_fold_wape(
    fold_results: pd.DataFrame | None, model_by_sku: dict[str, str] | None
) -> tuple[float | None, dict[str, float], float | None]:
    """(portfolio WAPE on the most recent origin, per-SKU dict, reference WAPE).

    The most recent fold is the closest thing this system has to *current*
    model performance — observed, not extrapolated. The reference is the
    portfolio mean over the PREVIOUS folds only: 'is the model doing worse
    than it used to?', never a comparison that includes the window being
    judged.
    """
    if fold_results is None or fold_results.empty:
        return None, {}, None
    model_by_sku = model_by_sku or {}
    last_fold = int(fold_results["fold"].max())
    recent = fold_results[
        (fold_results["fold"] == last_fold) & fold_results["sku"].isin(model_by_sku)
    ]
    per_sku: dict[str, float] = {}
    vals = []
    for sku, wape in zip(recent["sku"], recent["WAPE"]):
        if wape == wape:
            per_sku[sku] = round(float(wape), 2)
            vals.append(float(wape))
    current = sum(vals) / len(vals) if vals else None

    prior = fold_results[
        (fold_results["fold"] < last_fold) & fold_results["sku"].isin(model_by_sku)
    ]
    if prior.empty:
        return current, per_sku, None
    prior_vals = [float(w) for w in prior["WAPE"] if w == w]
    reference = sum(prior_vals) / len(prior_vals) if prior_vals else None
    return current, per_sku, reference


def build_model_health(
    fold_results,
    model_by_sku: dict[str, str],
    reference_wape: float | None,
    baseline_wape: float | None,
    drift: dict | None,
    coverage: dict | None,
    dq_status: str,
    last_trained: str | None,
    data_max_date: str | None = None,
    production_meta: dict | None = None,
    monitor_now=None,
) -> dict:
    """Assemble the health object.

    `reference_wape` is the walk-forward portfolio mean (the expectation);
    the degradation check itself uses the previous-folds reference from
    `latest_fold_wape`, falling back to `reference_wape` for single-fold
    evaluations.
    """
    cfg = get_settings().monitoring
    current_wape, per_sku, prior_reference = latest_fold_wape(fold_results, model_by_sku)
    degradation_reference = prior_reference if prior_reference is not None else reference_wape

    wape_status = "healthy"
    if current_wape is not None:
        if current_wape >= cfg.wape_crit:
            wape_status = "critical"
        elif current_wape >= cfg.wape_warn or (
            degradation_reference
            and current_wape / degradation_reference - 1 >= cfg.wape_degrade_pct / 100
        ):
            wape_status = "warning"

    drift_status = (drift or {}).get("status", "unknown")
    coverage_status = (coverage or {}).get("calibration_status", "unknown")
    dq = {"PASS": "healthy", "WARN": "warning", "FAIL": "critical"}.get(dq_status, "unknown")

    # Interval miscalibration is a recalibration concern, not a broken model:
    # it contributes at most a WARNING to the overall status (documented).
    coverage_contrib = (
        "warning" if coverage_status not in ("healthy", "unknown") else coverage_status
    )

    overall = "healthy"
    for s in (wape_status, drift_status, coverage_contrib, dq):
        overall = _worse(overall, s)

    decision = evaluate_retraining(
        current_wape=current_wape,
        reference_wape=degradation_reference,
        drift_status=drift_status,
        interval_coverage=(coverage or {}).get("observed_coverage"),
        last_trained=last_trained,
        data_max_date=data_max_date,
        now=monitor_now,
    )

    if overall == "healthy" and not decision["retrain_required"]:
        recommendation = "No retraining required"
    elif decision["reasons"]:
        recommendation = "Retrain: " + "; ".join(decision["reasons"][:3])
    else:
        recommendation = f"Investigate {overall} signal before next cycle"

    health = {
        "generated_at": utc_now(),
        "status": overall,
        "model_version": (production_meta or {}).get("model_version"),
        "model_name": (production_meta or {}).get("model_name"),
        "models_by_sku": model_by_sku,
        "last_trained": last_trained,
        "metrics": {
            "wape_current_fold": round(current_wape, 2) if current_wape is not None else None,
            "wape_reference": round(reference_wape, 2) if reference_wape else None,
            "wape_reference_previous_folds": round(prior_reference, 2) if prior_reference else None,
            "wape_baseline": round(baseline_wape, 2) if baseline_wape else None,
            "per_sku_wape_current_fold": per_sku,
        },
        "drift": drift or {"status": "unknown"},
        "interval_coverage": coverage or {"status": "unknown"},
        "data_quality_status": dq_status,
        "retraining": decision,
        "recommendation": recommendation,
    }
    out = config.ARTIFACTS / "monitoring" / "model_health.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(health, indent=2, default=str))
    return health
