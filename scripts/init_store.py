#!/usr/bin/env python3
"""Initialise the analytics warehouse from the committed processed CSVs.

    python scripts/init_store.py                      # SQLite (default)
    DATABASE_URL=postgresql://... python scripts/init_store.py

Used by Docker Compose (api entrypoint) and operators to populate the store
without running the full training pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from supplychainxai.data import store


def main() -> None:
    target = "postgresql" if store.is_postgres() else "sqlite"
    print(f"initialising analytics store ({target}) ...")
    out = store.build_db_from_files()
    print(f"ready: {out}")


if __name__ == "__main__":
    main()
