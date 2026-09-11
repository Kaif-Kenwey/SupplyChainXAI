"""Stage 5 — Analytics store: processed tables persisted to SQLite.

The warehouse is deliberately simple (SQLite, zero-dependency) but is queried
with real SQL — the copilot's retrieval layer runs against these tables, which
keeps it fully grounded in the same numbers the dashboard shows.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from supplychainxai import config

TABLES = {
    "dim_product": ["sku TEXT PRIMARY KEY", "name TEXT", "category TEXT",
                    "unit_cost REAL", "pack_size INTEGER"],
    "dim_supplier": ["supplier_id TEXT PRIMARY KEY", "name TEXT", "country TEXT",
                     "target_otd REAL"],
    "supply_terms": ["sku TEXT", "supplier_id TEXT", "unit_price REAL",
                     "quoted_lead_days INTEGER", "lead_time_std REAL",
                     "is_primary INTEGER"],
    "fact_sales": ["date TEXT", "sku TEXT", "units_sold INTEGER", "unit_price REAL",
                   "promo_flag INTEGER"],
    "fact_inventory": ["date TEXT", "sku TEXT", "on_hand INTEGER", "on_order INTEGER"],
    "fact_purchase_orders": ["po_id TEXT PRIMARY KEY", "sku TEXT", "supplier_id TEXT",
                             "order_date TEXT", "expected_date TEXT",
                             "delivered_date TEXT", "quantity INTEGER",
                             "unit_price REAL", "status TEXT"],
}


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def build_db(cleaned) -> Path:
    """(Re)create analytics.db from cleaned frames. Returns the db path."""
    if config.DB_PATH.exists():
        config.DB_PATH.unlink()
    conn = connect()
    with conn:
        for table, cols in TABLES.items():
            conn.execute(f"CREATE TABLE {table} ({', '.join(cols)})")

    conn.executemany(
        "INSERT INTO dim_product VALUES (?,?,?,?,?)",
        cleaned.products[["sku", "name", "category", "unit_cost", "pack_size"]]
        .itertuples(index=False, name=None))
    conn.executemany(
        "INSERT INTO dim_supplier VALUES (?,?,?,?)",
        cleaned.suppliers[["supplier_id", "name", "country", "target_otd"]]
        .itertuples(index=False, name=None))
    conn.executemany(
        "INSERT INTO supply_terms VALUES (?,?,?,?,?,?)",
        cleaned.supply_terms.assign(is_primary=cleaned.supply_terms["is_primary"].astype(int))
        [["sku", "supplier_id", "unit_price", "quoted_lead_days",
          "lead_time_std", "is_primary"]].itertuples(index=False, name=None))
    conn.executemany(
        "INSERT INTO fact_sales VALUES (?,?,?,?,?)",
        cleaned.sales.assign(date=cleaned.sales["date"].dt.date.astype(str))
        [["date", "sku", "units_sold", "unit_price", "promo_flag"]]
        .itertuples(index=False, name=None))
    conn.executemany(
        "INSERT INTO fact_inventory VALUES (?,?,?,?)",
        cleaned.inventory.assign(date=cleaned.inventory["date"].dt.date.astype(str))
        [["date", "sku", "on_hand", "on_order"]].itertuples(index=False, name=None))
    conn.executemany(
        "INSERT INTO fact_purchase_orders VALUES (?,?,?,?,?,?,?,?,?)",
        cleaned.purchase_orders.assign(
            order_date=cleaned.purchase_orders["order_date"].dt.date.astype(str),
            expected_date=cleaned.purchase_orders["expected_date"].dt.date.astype(str),
            delivered_date=cleaned.purchase_orders["delivered_date"]
            .dt.strftime("%Y-%m-%d"),
        )[["po_id", "sku", "supplier_id", "order_date", "expected_date",
           "delivered_date", "quantity", "unit_price", "status"]]
        .itertuples(index=False, name=None))
    conn.commit()
    conn.close()
    return config.DB_PATH


def query(sql: str, params: tuple = (), db_path: Path | None = None) -> pd.DataFrame:
    """Run a read-only SQL query against the analytics store.

    NULLs are returned as None (not pandas NaN) so results stay JSON-safe.
    """
    with connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn, params=params)
    if df.isna().any().any():
        df = df.astype(object).where(df.notna(), None)
    return df
