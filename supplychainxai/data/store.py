"""Stage 5 — Analytics store: processed tables persisted to a SQL warehouse.

Two modes, one schema, zero duplicated logic:

  * SQLite (default, zero-dependency) — local development, demos, tests.
    File: `data/processed/analytics.db`.
  * PostgreSQL (production-like) — enabled via `DATABASE_URL`
    (e.g. `postgresql://user:pass@host:5432/scx`). The same DDL, the same
    inserts, the same read path; the only translation is SQLite's
    `julianday()` idiom in consumer SQL (see `to_pg_sql`).

The warehouse is deliberately simple but is queried with real SQL — the
copilot's retrieval layer runs against these tables, which keeps it fully
grounded in the same numbers the dashboard shows.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

from supplychainxai import config
from supplychainxai.config.settings import get_settings

TABLES = {
    "dim_product": [
        "sku TEXT PRIMARY KEY",
        "name TEXT",
        "category TEXT",
        "unit_cost REAL",
        "pack_size INTEGER",
    ],
    "dim_supplier": [
        "supplier_id TEXT PRIMARY KEY",
        "name TEXT",
        "country TEXT",
        "target_otd REAL",
    ],
    "supply_terms": [
        "sku TEXT",
        "supplier_id TEXT",
        "unit_price REAL",
        "quoted_lead_days INTEGER",
        "lead_time_std REAL",
        "is_primary INTEGER",
    ],
    "fact_sales": [
        "date TEXT",
        "sku TEXT",
        "units_sold INTEGER",
        "unit_price REAL",
        "promo_flag INTEGER",
    ],
    "fact_inventory": ["date TEXT", "sku TEXT", "on_hand INTEGER", "on_order INTEGER"],
    "fact_purchase_orders": [
        "po_id TEXT PRIMARY KEY",
        "sku TEXT",
        "supplier_id TEXT",
        "order_date TEXT",
        "expected_date TEXT",
        "delivered_date TEXT",
        "quantity INTEGER",
        "unit_price REAL",
        "status TEXT",
    ],
}

# column order per table for positional inserts
INSERT_COLUMNS = {
    "dim_product": ["sku", "name", "category", "unit_cost", "pack_size"],
    "dim_supplier": ["supplier_id", "name", "country", "target_otd"],
    "supply_terms": [
        "sku",
        "supplier_id",
        "unit_price",
        "quoted_lead_days",
        "lead_time_std",
        "is_primary",
    ],
    "fact_sales": ["date", "sku", "units_sold", "unit_price", "promo_flag"],
    "fact_inventory": ["date", "sku", "on_hand", "on_order"],
    "fact_purchase_orders": [
        "po_id",
        "sku",
        "supplier_id",
        "order_date",
        "expected_date",
        "delivered_date",
        "quantity",
        "unit_price",
        "status",
    ],
}


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def is_postgres(url: str | None = None) -> bool:
    url = url if url is not None else get_settings().infra.database_url
    return bool(url) and url.startswith(("postgresql://", "postgres://", "postgresql+psycopg2://"))


# ------------------------------------------------------------ frame prep
def _prepare_frames(cleaned) -> dict[str, pd.DataFrame]:
    """Row frames for every table, shared by both backends."""
    return {
        "dim_product": cleaned.products[["sku", "name", "category", "unit_cost", "pack_size"]],
        "dim_supplier": cleaned.suppliers[["supplier_id", "name", "country", "target_otd"]],
        "supply_terms": cleaned.supply_terms.assign(
            is_primary=cleaned.supply_terms["is_primary"].astype(int)
        )[["sku", "supplier_id", "unit_price", "quoted_lead_days", "lead_time_std", "is_primary"]],
        "fact_sales": cleaned.sales.assign(date=cleaned.sales["date"].dt.date.astype(str))[
            ["date", "sku", "units_sold", "unit_price", "promo_flag"]
        ],
        "fact_inventory": cleaned.inventory.assign(
            date=cleaned.inventory["date"].dt.date.astype(str)
        )[["date", "sku", "on_hand", "on_order"]],
        "fact_purchase_orders": cleaned.purchase_orders.assign(
            order_date=cleaned.purchase_orders["order_date"].dt.date.astype(str),
            expected_date=cleaned.purchase_orders["expected_date"].dt.date.astype(str),
            delivered_date=cleaned.purchase_orders["delivered_date"].dt.strftime("%Y-%m-%d"),
        )[
            [
                "po_id",
                "sku",
                "supplier_id",
                "order_date",
                "expected_date",
                "delivered_date",
                "quantity",
                "unit_price",
                "status",
            ]
        ],
    }


# ------------------------------------------------------------ build
def build_db(cleaned) -> Path | str:
    """(Re)create the warehouse from cleaned frames in the configured mode."""
    frames = _prepare_frames(cleaned)
    if is_postgres():
        return _build_db_pg(frames)
    return _build_db_sqlite(frames)


def build_db_from_files() -> Path | str:
    """Populate the warehouse from the committed processed CSVs.

    Used by Docker (and operators) to initialise PostgreSQL — or SQLite —
    without running the full training pipeline.
    """
    sales = pd.read_csv(config.PROCESSED_FILES["sales"], parse_dates=["date"])
    inventory = pd.read_csv(config.PROCESSED_FILES["inventory"], parse_dates=["date"])
    pos = pd.read_csv(
        config.PROCESSED_FILES["purchase_orders"],
        parse_dates=["order_date", "expected_date", "delivered_date"],
    )
    products = pd.read_csv(config.RAW_FILES["products"])
    suppliers = pd.read_csv(config.RAW_FILES["suppliers"])
    terms = pd.read_csv(config.RAW_FILES["supply_terms"])

    from supplychainxai.data.ingest import RawData

    # statuses are derived by the cleaner; recompute here for stored CSVs
    if "status" not in pos.columns:
        pos["status"] = pos["delivered_date"].isna().map({True: "OPEN", False: "DELIVERED"})
    return build_db(
        RawData(
            products=products,
            suppliers=suppliers,
            supply_terms=terms,
            sales=sales,
            inventory=inventory,
            purchase_orders=pos,
        )
    )


def _build_db_sqlite(frames: dict[str, pd.DataFrame]) -> Path:
    if config.DB_PATH.exists():
        config.DB_PATH.unlink()
    conn = connect()
    with conn:
        for table, cols in TABLES.items():
            conn.execute(f"CREATE TABLE {table} ({', '.join(cols)})")
        for table, df in frames.items():
            conn.executemany(
                f"INSERT INTO {table} VALUES ({','.join('?' * len(df.columns))})",
                df[INSERT_COLUMNS[table]].itertuples(index=False, name=None),
            )
    conn.commit()
    conn.close()
    return config.DB_PATH


def _build_db_pg(frames: dict[str, pd.DataFrame]) -> str:
    url = get_settings().infra.database_url
    try:
        from sqlalchemy import create_engine
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "DATABASE_URL points at PostgreSQL but sqlalchemy "
            "is not installed (pip install sqlalchemy "
            "psycopg2-binary)"
        ) from exc
    engine = create_engine(url, pool_pre_ping=True)
    with engine.begin() as conn:
        raw = conn.connection  # DBAPI connection (psycopg2)
        cur = raw.cursor()
        for table, cols in TABLES.items():
            cur.execute(f'DROP TABLE IF EXISTS "{table}"')
            cur.execute(f'CREATE TABLE "{table}" ({", ".join(cols)})')
        for table, df in frames.items():
            placeholders = ",".join(["%s"] * len(INSERT_COLUMNS[table]))
            cur.executemany(
                f'INSERT INTO "{table}" VALUES ({placeholders})',
                df[INSERT_COLUMNS[table]].itertuples(index=False, name=None),
            )
    engine.dispose()
    return "postgresql"


# ------------------------------------------------------------ read path
_JULIANDAY_RE = re.compile(r"julianday\(([^)]+)\)")


def to_pg_sql(sql: str) -> str:
    """Translate the SQLite-isms used by consumer SQL into PostgreSQL.

    `julianday(x)` → `EXTRACT(EPOCH FROM x::timestamp) / 86400`, so day-level
    differences remain day-level. Everything else used (ROUND, CASE WHEN,
    COUNT, MAX) is portable as written.
    """
    return _JULIANDAY_RE.sub(r"EXTRACT(EPOCH FROM (\1::timestamp)) / 86400", sql)


def query(sql: str, params: tuple = (), db_path: Path | None = None) -> pd.DataFrame:
    """Run a read-only SQL query against the analytics store.

    NULLs are returned as None (not pandas NaN) so results stay JSON-safe.
    Uses PostgreSQL when DATABASE_URL is configured (and no explicit
    SQLite `db_path` is given).
    """
    if db_path is None and is_postgres():
        df = _query_pg(sql, params)
    else:
        with connect(db_path) as conn:
            df = pd.read_sql_query(sql, conn, params=params)
    if df.isna().any().any():
        df = df.astype(object).where(df.notna(), None)
    return df


def _query_pg(sql: str, params: tuple) -> pd.DataFrame:
    from sqlalchemy import create_engine

    engine = create_engine(get_settings().infra.database_url, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            raw = conn.connection
            cur = raw.cursor()
            cur.execute(to_pg_sql(sql), params or ())
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
        return pd.DataFrame(rows, columns=cols)
    finally:
        engine.dispose()
