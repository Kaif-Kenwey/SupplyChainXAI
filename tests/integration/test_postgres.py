"""Integration: the store works identically on PostgreSQL.

Skipped unless a PostgreSQL server is reachable. CI provides one as a
service container and sets TEST_DATABASE_URL; locally:

    docker run -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:16
    TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres pytest
"""

import os

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("psycopg2")

TEST_URL = os.getenv("TEST_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not TEST_URL, reason="TEST_DATABASE_URL not set — PostgreSQL integration skipped"
)


@pytest.fixture(scope="module")
def pg_url():
    import sqlalchemy

    engine = sqlalchemy.create_engine(TEST_URL)
    try:
        engine.connect().close()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PostgreSQL unreachable: {exc}")
    finally:
        engine.dispose()
    return TEST_URL


def _cleaned():
    import pandas as pd

    from supplychainxai.data.ingest import RawData

    n = 10
    dates = pd.date_range("2025-01-01", periods=n)
    sales = pd.DataFrame(
        {"date": dates, "sku": "P101", "units_sold": 5.0, "unit_price": 4.0, "promo_flag": 0.0}
    )
    inv = pd.DataFrame({"date": dates, "sku": "P101", "on_hand": 50.0, "on_order": 0.0})
    pos = pd.DataFrame(
        {
            "po_id": [f"PO-{i}" for i in range(3)],
            "sku": "P101",
            "supplier_id": "S1",
            "order_date": dates[:3],
            "expected_date": dates[:3] + pd.Timedelta(days=5),
            "delivered_date": dates[:3] + pd.Timedelta(days=6),
            "quantity": 100.0,
            "unit_price": 4.0,
            "status": "DELIVERED",
        }
    )
    products = pd.DataFrame(
        {
            "sku": ["P101"],
            "name": ["Bolt"],
            "category": ["F"],
            "unit_cost": [4.0],
            "pack_size": [10],
        }
    )
    suppliers = pd.DataFrame(
        {"supplier_id": ["S1"], "name": ["Acme"], "country": ["X"], "target_otd": [0.9]}
    )
    terms = pd.DataFrame(
        {
            "sku": ["P101"],
            "supplier_id": ["S1"],
            "unit_price": [4.0],
            "quoted_lead_days": [5],
            "lead_time_std": [1.0],
            "is_primary": [True],
        }
    )
    return RawData(
        products=products,
        suppliers=suppliers,
        supply_terms=terms,
        sales=sales,
        inventory=inv,
        purchase_orders=pos,
    )


def test_build_and_query(pg_url, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", pg_url)
    import supplychainxai.config.settings as settings_mod

    monkeypatch.setattr(settings_mod, "_settings", None)  # re-read env

    from supplychainxai.data import store

    assert store.is_postgres() is True
    out = store.build_db(_cleaned())
    assert out == "postgresql"

    df = store.query("SELECT COUNT(*) AS n FROM fact_sales")
    assert int(df["n"].iloc[0]) == 10

    # julianday idiom translates and computes day differences
    df = store.query(
        """SELECT AVG(julianday(delivered_date) - julianday(order_date)) AS avg_lead
             FROM fact_purchase_orders"""
    )
    assert df["avg_lead"].iloc[0] == pytest.approx(6.0)

    # parameterised query
    df = store.query("SELECT COUNT(*) AS n FROM fact_sales WHERE sku = ?", ("P101",))
    assert int(df["n"].iloc[0]) == 10

    # NULL-safety contract holds on PG too
    df = store.query("SELECT NULL AS x")
    assert df["x"].iloc[0] is None

    monkeypatch.setattr(settings_mod, "_settings", None)


def test_to_pg_sql_translation():
    from supplychainxai.data.store import to_pg_sql

    out = to_pg_sql("SELECT AVG(julianday(a.delivered_date) - julianday(a.order_date)) FROM t a")
    assert "julianday" not in out
    assert "EXTRACT(EPOCH FROM (a.delivered_date::timestamp)) / 86400" in out
