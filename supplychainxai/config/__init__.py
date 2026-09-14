"""Central configuration: paths, scenario constants, and knobs."""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
ARTIFACTS = PROJECT_ROOT / "artifacts"
DB_PATH = DATA_PROCESSED / "analytics.db"

for _p in (DATA_RAW, DATA_PROCESSED, ARTIFACTS):
    _p.mkdir(parents=True, exist_ok=True)

RAW_FILES = {
    "products": DATA_RAW / "products.csv",
    "suppliers": DATA_RAW / "suppliers.csv",
    "supply_terms": DATA_RAW / "supply_terms.csv",
    "sales": DATA_RAW / "sales_history.csv",
    "inventory": DATA_RAW / "inventory_snapshots.csv",
    "purchase_orders": DATA_RAW / "purchase_orders.csv",
}

PROCESSED_FILES = {
    "sales": DATA_PROCESSED / "sales_clean.csv",
    "inventory": DATA_PROCESSED / "inventory_clean.csv",
    "purchase_orders": DATA_PROCESSED / "purchase_orders_clean.csv",
    "features": DATA_PROCESSED / "features_daily.csv",
}

# ---------------------------------------------------------------- scenario
# "today" for the analytical snapshot = last date present in the data
SIM_START = "2023-01-01"
SIM_END = "2025-12-31"
FORECAST_HORIZON_DAYS = 30
TEST_WINDOW_DAYS = 90          # hold-out window for model comparison
REVIEW_PERIOD_DAYS = 7         # procurement review cadence (periodic review)

# ---------------------------------------------------------------- policy
SERVICE_LEVEL = 0.95           # target cycle-service level -> z = 1.645
SERVICE_LEVEL_Z = 1.645
OVERSTOCK_DOH_DAYS = 90        # days-of-cover above which we flag overstock
CRITICAL_COVER_DAYS = 10       # days-of-cover below which stockout risk is critical
SUPPLIER_RELIABILITY_FLOOR = 0.70

# supplier scoring weights (must sum to 1.0)
W_PRICE = 0.45
W_RELIABILITY = 0.35
W_LEAD_TIME = 0.20

# risk thresholds
DEMAND_SPIKE_Z = 2.0
SPIKE_BASELINE_DAYS = 90
SPIKE_RECENT_DAYS = 14
LEADTIME_DRIFT_PCT = 0.20      # >20% recent-vs-baseline lead time = anomaly
PRICE_ANOMALY_Z = 2.5
