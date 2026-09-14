#!/usr/bin/env python3
"""Train: the documented, reproducible retraining command.

    python scripts/train.py              # full workflow, settings from env
    SCX_EVAL__FOLDS=2 python scripts/train.py   # faster profile (e.g. CI)

Thin wrapper over scripts/run_pipeline.py so the retraining command from the
docs stays stable even if internals are reorganised.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    print(
        "SupplyChainXAI — training pipeline (data → backtesting → selection → "
        "registry → forecasting → monitoring)"
    )
    runpy.run_path(str(PROJECT_ROOT / "scripts" / "run_pipeline.py"), run_name="__main__")
