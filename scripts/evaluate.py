#!/usr/bin/env python3
"""Standalone walk-forward evaluation: metrics per fold / model / SKU.

    python scripts/evaluate.py                 # settings from env
    SCX_EVAL__FOLDS=2 python scripts/evaluate.py

Writes artifacts/evaluation/* and prints the portfolio summary.
Selection/forecasting continue in run_pipeline.py (which reuses the same
WalkForwardBacktester via forecasting.selector).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from supplychainxai import config
from supplychainxai.config.settings import get_settings
from supplychainxai.data.clean import clean_all
from supplychainxai.data.features import build_features
from supplychainxai.data.ingest import load_all
from supplychainxai.evaluation.backtesting import WalkForwardBacktester
from supplychainxai.evaluation.scoring import composite_scores


def main() -> None:
    t0 = time.time()
    eval_cfg = get_settings().eval
    raw = load_all()
    cleaned, _ = clean_all(raw)
    features = build_features(cleaned.sales, cleaned.inventory)
    skus = cleaned.products["sku"].tolist()
    print(
        f"evaluating {len(skus)} SKUs | horizon={eval_cfg.horizon}d "
        f"folds={eval_cfg.folds} min_train={eval_cfg.min_train_days}d"
    )

    bt = WalkForwardBacktester(
        horizon=eval_cfg.horizon,
        n_folds=eval_cfg.folds,
        min_train_days=eval_cfg.min_train_days,
        stride=eval_cfg.stride,
    )
    results = bt.evaluate(features, skus)
    summary = WalkForwardBacktester.summarize(results)
    scored = composite_scores(summary["per_sku"], eval_cfg.w_wape, eval_cfg.w_mae, eval_cfg.w_rmse)

    config.ARTIFACTS_EVALUATION.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.ARTIFACTS_EVALUATION / "walk_forward_results.csv", index=False)
    scored.to_csv(config.ARTIFACTS_EVALUATION / "walk_forward_per_sku.csv", index=False)
    summary["portfolio"].to_csv(
        config.ARTIFACTS_EVALUATION / "walk_forward_portfolio.csv", index=False
    )

    print(f"\nPortfolio (mean of per-SKU mean WAPE) — {eval_cfg.folds} folds:")
    print(summary["portfolio"].to_string(index=False))
    n_unstable = int(scored["unstable"].sum())
    print(f"\nunstable model/SKU combinations (WAPE CV > 0.5): {n_unstable}")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
