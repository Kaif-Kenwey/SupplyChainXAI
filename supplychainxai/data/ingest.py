"""Stage 1 — Ingestion: load raw CSV/Excel-style files with schema enforcement."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from supplychainxai import config

SCHEMAS: dict[str, dict] = {
    "products": {
        "dtypes": {"sku": "string", "name": "string", "category": "string",
                   "unit_cost": "float64", "base_demand": "float64",
                   "trend_per_year": "float64", "pack_size": "int64"},
        "date_cols": [],
    },
    "suppliers": {
        "dtypes": {"supplier_id": "string", "name": "string", "country": "string",
                   "target_otd": "float64"},
        "date_cols": [],
    },
    "supply_terms": {
        "dtypes": {"sku": "string", "supplier_id": "string", "unit_price": "float64",
                   "quoted_lead_days": "int64", "lead_time_std": "float64",
                   "is_primary": "boolean"},
        "date_cols": [],
    },
    "sales": {
        "dtypes": {"date": "datetime64[ns]", "sku": "string", "units_sold": "float64",
                   "unit_price": "float64", "promo_flag": "float64"},
        "date_cols": ["date"],
    },
    "inventory": {
        "dtypes": {"date": "datetime64[ns]", "sku": "string", "on_hand": "float64",
                   "on_order": "float64"},
        "date_cols": ["date"],
    },
    "purchase_orders": {
        "dtypes": {"po_id": "string", "sku": "string", "supplier_id": "string",
                   "order_date": "datetime64[ns]", "expected_date": "datetime64[ns]",
                   "delivered_date": "datetime64[ns]", "quantity": "float64",
                   "unit_price": "float64"},
        "date_cols": ["order_date", "expected_date", "delivered_date"],
    },
}


@dataclass
class RawData:
    products: pd.DataFrame
    suppliers: pd.DataFrame
    supply_terms: pd.DataFrame
    sales: pd.DataFrame
    inventory: pd.DataFrame
    purchase_orders: pd.DataFrame


def load_table(name: str, path: Path | None = None) -> pd.DataFrame:
    """Read one raw table, coercing columns to the expected schema."""
    path = path or config.RAW_FILES[name]
    schema = SCHEMAS[name]
    df = pd.read_csv(path)
    for col in schema["date_cols"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
    for col, dtype in schema["dtypes"].items():
        if col in df.columns and dtype != "datetime64[ns]":
            if "float" in dtype or "int" in dtype:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                if "int" in dtype:
                    df[col] = df[col].astype("float64")  # keep NaN-able until validation
            elif dtype == "boolean":
                df[col] = df[col].map({"True": True, "False": False, True: True, False: False})
                df[col] = df[col].astype("boolean")
            else:
                df[col] = df[col].astype("string")
    return df


def load_all() -> RawData:
    return RawData(
        products=load_table("products"),
        suppliers=load_table("suppliers"),
        supply_terms=load_table("supply_terms"),
        sales=load_table("sales"),
        inventory=load_table("inventory"),
        purchase_orders=load_table("purchase_orders"),
    )


def as_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", format="mixed")


def safe_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
