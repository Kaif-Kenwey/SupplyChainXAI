"""Retraining policy & promotion gates.

Two decisions, kept separate on purpose:

1. `evaluate_retraining(...)` — SHOULD we retrain? Fires when any trigger
   trips: error above threshold, drift, degraded interval coverage, stale
   data, scheduled interval reached, or new-data volume. Returns a
   machine-readable decision with the reasons, never mutates anything.

2. `promotion_gates(...)` — MAY the candidate replace production? A new
   model is promoted only if every gate passes; gates are configurable and
   the exact rule (candidate WAPE <= production WAPE * (1 + tolerance))
   is documented in docs/architecture.md. A worse model is never promoted,
   even if retraining fired.
"""

from __future__ import annotations

from datetime import UTC, datetime

from supplychainxai.config.settings import get_settings


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None


def evaluate_retraining(
    current_wape: float | None,
    reference_wape: float | None,
    drift_status: str = "healthy",
    interval_coverage: float | None = None,
    last_trained: str | None = None,
    data_max_date: str | None = None,
    new_data_rows: int = 0,
    now: datetime | None = None,
) -> dict:
    """Retraining decision from monitoring signals. Pure function.

    Parameters map 1:1 to `monitoring/model_health.py` outputs so the API and
    the pipeline share one policy. `now` is injectable for deterministic tests.
    """
    mon = get_settings().monitoring
    retrain = get_settings().retraining
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)  # anchor timestamps are always UTC
    reasons: list[str] = []

    if current_wape is not None and current_wape >= mon.wape_warn:
        reasons.append(f"portfolio WAPE {current_wape:.2f}% >= warn threshold {mon.wape_warn:.2f}%")
    if (
        current_wape is not None
        and reference_wape
        and reference_wape > 0
        and (current_wape / reference_wape - 1.0) * 100 >= mon.wape_degrade_pct
    ):
        reasons.append(
            f"WAPE degraded {((current_wape / reference_wape) - 1) * 100:.1f}% vs "
            f"reference {reference_wape:.2f}% (threshold {mon.wape_degrade_pct:.0f}%)"
        )
    if drift_status in ("warning", "critical"):
        reasons.append(f"data/feature drift status = {drift_status}")
    if interval_coverage is not None and interval_coverage < mon.coverage_min:
        reasons.append(
            f"prediction-interval coverage {interval_coverage:.2f} < minimum {mon.coverage_min:.2f}"
        )
    if data_max_date:
        d = _parse_ts(data_max_date)
        if d is not None:
            d = d if d.tzinfo else d.replace(tzinfo=UTC)
            age = (now - d).days
            if age > mon.stale_data_days:
                reasons.append(f"data is {age} days stale (threshold {mon.stale_data_days})")
    if retrain.new_data_rows_trigger and new_data_rows >= retrain.new_data_rows_trigger:
        reasons.append(
            f"{new_data_rows} new rows >= trigger volume {retrain.new_data_rows_trigger}"
        )
    trained = _parse_ts(last_trained)
    if trained is not None and retrain.max_model_age_days > 0:
        trained = trained if trained.tzinfo else trained.replace(tzinfo=UTC)
        if (now - trained).days >= retrain.max_model_age_days:
            reasons.append(
                f"scheduled interval reached "
                f"({(now - trained).days}d >= {retrain.max_model_age_days}d)"
            )

    return {
        "retrain_required": bool(reasons),
        "reasons": reasons,
        "evaluated_at": now.isoformat(timespec="seconds"),
    }


def promotion_gates(
    candidate_wape: float | None,
    production_wape: float | None,
    data_quality_ok: bool = True,
    artifacts_ok: bool = True,
    validation_ok: bool = True,
    tolerance: float | None = None,
) -> dict:
    """Gate evaluation for promoting a candidate to production.

    WAPE gate (documented rule): candidate_wape <= production_wape * (1 +
    tolerance), default tolerance 0.05 → a candidate may be at most 5% worse
    than the incumbent. When there is no incumbent (first deployment) the
    gate passes on absolute threshold only.
    """
    rec = get_settings().retraining
    tol = rec.wape_tolerance if tolerance is None else tolerance

    if candidate_wape is None:
        wape_gate, detail = False, "candidate has no WAPE — refusing promotion"
    elif production_wape is None or production_wape <= 0:
        wape_gate, detail = True, "no incumbent — candidate becomes first production model"
    else:
        limit = production_wape * (1.0 + tol)
        wape_gate = candidate_wape <= limit
        detail = (
            f"candidate {candidate_wape:.2f}% vs limit {limit:.2f}% "
            f"(production {production_wape:.2f}% × {1 + tol:.2f})"
        )

    return {
        "validation_gate": bool(validation_ok),
        "data_quality_gate": bool(data_quality_ok),
        "artifact_gate": bool(artifacts_ok),
        "wape_gate": wape_gate,
        "wape_detail": detail,
    }
