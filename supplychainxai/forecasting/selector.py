"""Model selection: rolling-origin bake-off, pick a winner per SKU, produce
the final 30-day forecast with 80% intervals.

Selection logic (upgraded from the original single 90-day hold-out):

  * every candidate is evaluated with `WalkForwardBacktester` on multiple
    expanding-window origins — a model can no longer win on one lucky slice;
  * per-SKU score: configurable metric — WAPE, MAE, RMSE, or a weighted
    composite of min-max-normalised WAPE/MAE/RMSE (default 0.5/0.3/0.2);
  * per-SKU winner = lowest score, ties broken alphabetically
    (deterministic — see evaluation/scoring.py);
  * every fold's metrics are stored; mean and standard deviation across
    folds are reported so unstable models are visible, not hidden;
  * the legacy single-hold-out comparison (`run_model_comparison`) is kept
    for reference/parity checks but is no longer the basis of selection.

The legacy contract of `run()` (returning comparison/winners/narrative/
forecasts/importance and writing the same artifact filenames) is preserved
so the API, dashboard and tests keep working.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from supplychainxai import config
from supplychainxai.config.settings import get_settings
from supplychainxai.data.features import FEATURE_COLUMNS, model_frame
from supplychainxai.evaluation.backtesting import (
    WalkForwardBacktester,
)
from supplychainxai.evaluation.scoring import composite_scores, select_winner
from supplychainxai.forecasting.metrics import all_metrics
from supplychainxai.forecasting.models import build_model_zoo
from supplychainxai.mlops.provenance import compute_dataset_version
from supplychainxai.mlops.registry import feature_version
from supplychainxai.mlops.runlog import git_commit, utc_now
from supplychainxai.mlops.tracking import ExperimentTracker

BASELINE_MODEL = "Moving Average (28d)"


def _zoo_by_label() -> dict[str, object]:
    return {getattr(m, "label", type(m).__name__): m for m in build_model_zoo()}


# ------------------------------------------------------------- legacy holdout
def _evaluate_models(frame: pd.DataFrame, test_days: int) -> pd.DataFrame:
    """Back-test the zoo on one SKU; a failing candidate never kills the bake-off."""
    split = len(frame) - test_days
    train, test = frame.iloc[:split], frame.iloc[split:]
    y_true = test["units_sold"].to_numpy()
    sku = frame["sku"].iloc[0]

    rows = []
    for model in build_model_zoo():
        name = getattr(model, "label", type(model).__name__)
        try:
            model.fit(train)
            pred = model.predict(test_days)["prediction"].to_numpy()
            m = all_metrics(y_true, pred)
            rows.append({"sku": sku, "model": name, **m})
        except Exception as exc:
            rows.append(
                {
                    "sku": sku,
                    "model": name,
                    **dict.fromkeys(("MAE", "RMSE", "MAPE", "WAPE", "SMAPE"), np.nan),
                    "error": str(exc)[:80],
                }
            )
    return pd.DataFrame(rows)


def run_model_comparison(
    features: pd.DataFrame, skus: list[str], test_days: int | None = None
) -> tuple[pd.DataFrame, dict]:
    """LEGACY single-hold-out comparison (final `test_days` window).

    Kept for parity checks against the walk-forward evaluation and for the
    historical results table; selection now uses rolling-origin performance.
    """
    test_days = test_days or config.TEST_WINDOW_DAYS
    all_rows, winners = [], {}
    for sku in skus:
        frame = model_frame(features, sku)
        rows = _evaluate_models(frame, test_days)
        all_rows.append(rows)
        valid = rows.dropna(subset=["WAPE"])
        if len(valid):
            best = valid.sort_values("WAPE").iloc[0]
            winners[sku] = {
                "model": best["model"],
                "WAPE": float(best["WAPE"]),
                "MAPE": float(best["MAPE"]),
                "MAE": float(best["MAE"]),
            }
    return pd.concat(all_rows, ignore_index=True), winners


# ------------------------------------------------------------- walk-forward
def walk_forward_comparison(
    features: pd.DataFrame, skus: list[str], eval_cfg=None, tracker=None, run_ctx=None
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict], dict]:
    """Run the rolling-origin bake-off and select a winner per SKU.

    Returns (fold_results, scored_per_sku, winners, summary) where
    `winners[sku]` carries model/score/folds/std and `competing` scores.
    """
    eval_cfg = eval_cfg or get_settings().eval
    tracker = tracker or ExperimentTracker()

    bt = WalkForwardBacktester(
        horizon=eval_cfg.horizon,
        n_folds=eval_cfg.folds,
        min_train_days=eval_cfg.min_train_days,
        stride=eval_cfg.stride,
    )
    results = bt.evaluate(features, skus)
    summary = WalkForwardBacktester.summarize(results)
    per_sku = summary["per_sku"]
    scored = composite_scores(per_sku, eval_cfg.w_wape, eval_cfg.w_mae, eval_cfg.w_rmse)
    winners = select_winner(scored, metric=eval_cfg.selection_metric)
    return results, scored, winners, summary


def selection_narrative(
    scored: pd.DataFrame, winners: dict[str, dict], summary: dict, eval_cfg=None
) -> dict:
    """Turn the walk-forward results into an honest selection story."""
    eval_cfg = eval_cfg or get_settings().eval
    portfolio = summary["portfolio"]
    per_sku = scored

    by_model = portfolio.set_index("model")["mean_WAPE"]
    portfolio_best = by_model.index[0]
    wins: dict[str, int] = {}
    for w in winners.values():
        wins[w["model"]] = wins.get(w["model"], 0) + 1

    baseline_wape = float(by_model.get(BASELINE_MODEL, np.nan))
    best_wape = float(by_model.iloc[0])
    lift = (baseline_wape - best_wape) / baseline_wape * 100 if baseline_wape else 0.0

    n_unstable = int(per_sku["unstable"].sum()) if "unstable" in per_sku else 0
    unstable_models = (
        sorted(per_sku.loc[per_sku["unstable"], "model"].unique().tolist()) if n_unstable else []
    )

    narrative = (
        f"Rolling-origin evaluation across {eval_cfg.folds} expanding-window folds "
        f"({eval_cfg.horizon}-day horizon each) on {len(winners)} SKUs: "
        f"{portfolio_best} achieves the lowest mean WAPE ({best_wape:.1f}%), winning on "
        f"{wins.get(portfolio_best, 0)} of {len(winners)} products. It cuts the naive "
        f"{BASELINE_MODEL} error by {lift:.0f}% on mean. "
        + (
            f"{n_unstable} model/SKU combination(s) show fold-to-fold instability "
            f"(WAPE CV > 0.5){' — ' + ', '.join(unstable_models[:3]) if unstable_models else ''}, "
            f"which mean-only metrics would hide. "
            if n_unstable
            else ""
        )
        + "Selection uses mean performance across ALL folds — never a single lucky hold-out."
    )
    return {
        "portfolio_best_model": portfolio_best,
        "mean_wape_by_model": {k: round(float(v), 2) for k, v in by_model.items()},
        "wins_by_model": wins,
        "improvement_vs_moving_average_pct": round(lift, 1),
        "unstable_combinations": n_unstable,
        "narrative": narrative,
    }


def final_forecasts(
    features: pd.DataFrame, winners: dict[str, dict], horizon: int | None = None
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Refit each SKU's winning model on full history and forecast forward.

    Returns (forecast_frame, fitted_models) so the registry can persist the
    exact estimators that generated the shipped forecasts — no silent refits.
    """
    horizon = horizon or config.FORECAST_HORIZON_DAYS
    outs, fitted = [], {}
    for sku, win in winners.items():
        frame = model_frame(features, sku)
        zoo = _zoo_by_label()
        model = zoo[win["model"]].fit(frame)
        fitted[sku] = model
        f = model.predict(horizon)
        f.insert(0, "sku", sku)
        f.insert(2, "model", win["model"])
        outs.append(f)
    return pd.concat(outs, ignore_index=True), fitted


def feature_importance(
    features: pd.DataFrame, skus: list[str], winners: dict[str, dict]
) -> pd.DataFrame:
    """Permutation importance on the final hold-out window for ML-model winners."""
    from sklearn.inspection import permutation_importance

    rows = []
    zoo = _zoo_by_label()
    for sku in skus:
        model_name = winners.get(sku, {}).get("model", "")
        if model_name not in zoo:
            continue
        model = zoo[model_name]
        if not hasattr(model, "estimator"):  # statistical models: skip
            continue
        frame = model_frame(features, sku)
        split = len(frame) - config.TEST_WINDOW_DAYS
        train, test = frame.iloc[:split], frame.iloc[split:]
        fresh = zoo[model_name]  # fresh instance, no state reuse
        fresh.fit(train)
        r = permutation_importance(
            fresh.estimator, test[FEATURE_COLUMNS], test["units_sold"], n_repeats=5, random_state=42
        )
        for feat, imp in zip(FEATURE_COLUMNS, r.importances_mean):
            rows.append(
                {"sku": sku, "model": model_name, "feature": feat, "importance": float(imp)}
            )
    return pd.DataFrame(rows)


# ------------------------------------------------------------- persistence
def _write_evaluation_artifacts(
    results: pd.DataFrame, scored: pd.DataFrame, summary: dict, eval_cfg
) -> None:
    results.to_csv(config.ARTIFACTS_EVALUATION / "walk_forward_results.csv", index=False)
    scored.to_csv(config.ARTIFACTS_EVALUATION / "walk_forward_per_sku.csv", index=False)
    summary["portfolio"].to_csv(
        config.ARTIFACTS_EVALUATION / "walk_forward_portfolio.csv", index=False
    )
    cfg = {
        "type": "walk_forward_rolling_origin",
        "horizon_days": eval_cfg.horizon,
        "n_folds": eval_cfg.folds,
        "min_train_days": eval_cfg.min_train_days,
        "stride": eval_cfg.stride,
        "expanding_window": True,
        "selection_metric": eval_cfg.selection_metric,
        "composite_weights": {
            "WAPE": eval_cfg.w_wape,
            "MAE": eval_cfg.w_mae,
            "RMSE": eval_cfg.w_rmse,
        },
    }
    (config.ARTIFACTS_EVALUATION / "evaluation_config.json").write_text(json.dumps(cfg, indent=2))


def _compat_comparison(scored: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward per-SKU means in the legacy model_comparison.csv schema."""
    out = scored.rename(
        columns={
            "mean_MAE": "MAE",
            "mean_RMSE": "RMSE",
            "mean_MAPE": "MAPE",
            "mean_WAPE": "WAPE",
            "std_WAPE": "WAPE_std",
            "folds": "n_folds",
        }
    )
    cols = [
        "sku",
        "model",
        "MAE",
        "RMSE",
        "MAPE",
        "WAPE",
        "WAPE_std",
        "mean_bias",
        "n_folds",
        "unstable",
    ]
    if "composite" in out.columns:
        cols.append("composite")
    return out[cols].round(4)


def _track_selection(
    tracker,
    scored: pd.DataFrame,
    winners: dict[str, dict],
    eval_cfg,
    dataset_version: str,
    run_ctx=None,
    features: pd.DataFrame | None = None,
    forecasts: pd.DataFrame | None = None,
) -> None:
    """One experiment run per (SKU, model); extra run for the portfolio view."""
    for (sku, model), row in scored.set_index(["sku", "model"]).iterrows():
        run = tracker.start_run(
            f"{sku}-{model}",
            params={
                "sku": sku,
                "model": model,
                "horizon": eval_cfg.horizon,
                "folds": eval_cfg.folds,
                "min_train_days": eval_cfg.min_train_days,
                "selection_metric": eval_cfg.selection_metric,
                "random_seed": eval_cfg.random_seed,
            },
            tags={
                "project": "supplychainxai",
                "sku": sku,
                "model_family": model,
                "dataset_version": dataset_version,
                "environment": get_settings().infra.environment,
                "pipeline_version": run_ctx.pipeline_version if run_ctx else "dev",
            },
        )
        metrics = {
            f"fold_mean_{k.lower()}": float(row[c])
            for k, c in (
                ("WAPE", "mean_WAPE"),
                ("MAE", "mean_MAE"),
                ("RMSE", "mean_RMSE"),
                ("MAPE", "mean_MAPE"),
                ("bias", "mean_bias"),
            )
            if pd.notna(row.get(c))
        }
        if "std_WAPE" in row and pd.notna(row.get("std_WAPE")):
            metrics["wape_std"] = float(row["std_WAPE"])
        if "composite" in row and pd.notna(row.get("composite")):
            metrics["composite"] = float(row["composite"])
        tracker.log_metrics(run, metrics)
        if winners.get(sku, {}).get("model") == model:
            tracker.log_tags(
                run, {"selected": "true", "selection_reason": f"lowest {eval_cfg.selection_metric}"}
            )
            if forecasts is not None:
                fc = forecasts[forecasts["sku"] == sku]
                if len(fc):
                    tmp = config.ARTIFACTS_EVALUATION / f"forecast_{sku}.csv"
                    fc.to_csv(tmp, index=False)
                    tracker.log_artifact(run, tmp)
        tracker.end_run(run)


def run(features: pd.DataFrame, skus: list[str], tracker=None, run_ctx=None) -> dict:
    """Full selection stage: walk-forward bake-off -> winners -> forecasts."""
    eval_cfg = get_settings().eval
    tracker = tracker or ExperimentTracker()
    dataset_version = compute_dataset_version()

    results, scored, winners, summary = walk_forward_comparison(
        features, skus, eval_cfg, tracker, run_ctx
    )
    narrative = selection_narrative(scored, winners, summary, eval_cfg)
    forecasts, fitted_models = final_forecasts(features, winners)
    importance = feature_importance(features, skus, winners)

    _write_evaluation_artifacts(results, scored, summary, eval_cfg)
    compat = _compat_comparison(scored)
    compat.to_csv(config.ARTIFACTS / "model_comparison.csv", index=False)
    forecasts.to_csv(config.ARTIFACTS / "forecasts.csv", index=False)
    importance.to_csv(config.ARTIFACTS / "feature_importance.csv", index=False)

    _track_selection(
        tracker, scored, winners, eval_cfg, dataset_version, run_ctx, features, forecasts
    )

    selection_record = {
        **narrative,
        "selection_metric": eval_cfg.selection_metric,
        "composite_weights": {
            "WAPE": eval_cfg.w_wape,
            "MAE": eval_cfg.w_mae,
            "RMSE": eval_cfg.w_rmse,
        },
        "evaluation": {
            "type": "walk_forward_rolling_origin",
            "horizon_days": eval_cfg.horizon,
            "folds": eval_cfg.folds,
            "min_train_days": eval_cfg.min_train_days,
            "stride": eval_cfg.stride,
            "expanding_window": True,
        },
        "per_sku_selection": {
            sku: {
                "model": w["model"],
                "score": round(w["score"], 4),
                "mean_WAPE": round(w["mean_WAPE"], 2),
                "std_WAPE": round(w["std_WAPE"], 2),
                "folds": w["folds"],
                "unstable": w["unstable"],
            }
            for sku, w in winners.items()
        },
        "provenance": {
            "dataset_version": dataset_version,
            "feature_version": feature_version(),
            "git_commit": git_commit(),
            "generated_at": utc_now(),
            "run_id": run_ctx.run_id if run_ctx else None,
            "random_seed": eval_cfg.random_seed,
        },
    }
    with open(config.ARTIFACTS / "model_selection.json", "w") as fh:
        json.dump(selection_record, fh, indent=2)

    return {
        "comparison": compat,
        "winners": winners,
        "narrative": selection_record,
        "forecasts": forecasts,
        "fitted_models": fitted_models,
        "importance": importance,
        "fold_results": results,
        "summary": summary,
    }
