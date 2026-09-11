"""Optimization engine tests: quantities, supplier choice, simulator physics."""
import numpy as np
import pandas as pd
import pytest

from supplychainxai.optimization.engine import (choose_supplier, recommend_sku,
                                                run_recommender, simulate_scenario)


def build_world():
    products = pd.DataFrame({
        "sku": ["P101", "P102"], "name": ["Bolt", "Gasket"],
        "category": ["F", "S"], "unit_cost": [4.0, 8.0],
        "base_demand": [100.0, 50.0], "trend_per_year": [0.0, 0.0], "pack_size": [10, 5],
    })
    suppliers = pd.DataFrame({
        "supplier_id": ["S1", "S2", "S3"], "name": ["Apex", "CheapCo", "Relia"],
        "country": ["IN", "CN", "DE"], "target_otd": [0.95, 0.70, 0.98],
    })
    terms = pd.DataFrame({
        "sku": ["P101", "P101", "P102"],
        "supplier_id": ["S1", "S2", "S3"],
        "unit_price": [4.5, 4.0, 9.0],
        "quoted_lead_days": [7, 10, 14], "lead_time_std": [1.0, 3.0, 2.0],
        "is_primary": [True, False, True],
    })
    dates = pd.date_range("2025-11-01", periods=60, freq="D")
    sales = pd.DataFrame({"date": np.repeat(dates, 2),
                          "sku": np.tile(["P101", "P102"], 60),
                          "units_sold": [100] * 60 + [50] * 60,
                          "unit_price": [5.0] * 60 + [10.0] * 60,
                          "promo_flag": [0] * 120})
    inv = pd.DataFrame({"date": np.repeat(dates, 2),
                        "sku": np.tile(["P101", "P102"], 60),
                        "on_hand": [200] * 60 + [800] * 60,
                        "on_order": [0] * 120})
    pos = pd.DataFrame({"po_id": ["PO-1"], "sku": ["P101"], "supplier_id": ["S1"],
                        "order_date": [dates[0]], "expected_date": [dates[0] + pd.Timedelta(days=7)],
                        "delivered_date": [dates[0] + pd.Timedelta(days=7)],
                        "quantity": [500.0], "unit_price": [4.5]})
    from supplychainxai.data.ingest import RawData
    return RawData(products=products, suppliers=suppliers, supply_terms=terms,
                   sales=sales, inventory=inv, purchase_orders=pos)


def build_forecast(daily=(100.0, 50.0)):
    dates = pd.date_range("2025-12-31", periods=30, freq="D")
    frames = []
    for sku, d in zip(["P101", "P102"], daily):
        frames.append(pd.DataFrame({
            "sku": sku, "date": dates, "model": "Test",
            "prediction": [d] * 30, "lower_80": [d * 0.8] * 30, "upper_80": [d * 1.2] * 30,
        }))
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------ supplier
def test_choose_supplier_penalises_unreliable_cheap_option():
    data = build_world()
    sup_stats = pd.DataFrame([{"supplier_id": "S1", "on_time_rate": 0.95, "avg_lead": 7.0},
                              {"supplier_id": "S2", "on_time_rate": 0.55, "avg_lead": 12.0}])
    best, runner = choose_supplier("P101", data, sup_stats)
    # S2 is cheaper but drifts/late: S1 must win with the reliability floor
    assert best["supplier_id"] == "S1"
    assert runner["supplier_id"] == "S2"


# -------------------------------------------------------------- recommendation
def test_recommend_buys_when_short():
    data = build_world()
    fc = build_forecast()
    # shrink P101 stock: override inventory to nearly zero
    data.inventory.loc[data.inventory["sku"] == "P101", "on_hand"] = 50
    rec = recommend_sku("P101", data, fc,
                        pd.DataFrame(columns=["supplier_id", "on_time_rate", "avg_lead"]))
    assert rec.action in ("BUY", "EXPEDITE")
    assert rec.quantity > 0
    assert rec.quantity % 10 == 0                     # pack rounding
    assert rec.supplier_id in ("S1", "S2")
    assert rec.total_cost == pytest.approx(rec.quantity * rec.unit_price, abs=0.01)


def test_recommend_holds_when_covered():
    data = build_world()
    fc = build_forecast()
    data.inventory.loc[data.inventory["sku"] == "P101", "on_hand"] = 10_000
    rec = recommend_sku("P101", data, fc,
                        pd.DataFrame(columns=["supplier_id", "on_time_rate", "avg_lead"]))
    assert rec.action == "HOLD"
    assert rec.quantity == 0


def test_run_recommender_covers_all_skus():
    data = build_world()
    recs = run_recommender(data, build_forecast())
    assert {r.sku for r in recs} == {"P101", "P102"}
    assert all(isinstance(r.quantity, int) for r in recs)


# ------------------------------------------------------------------ simulator
def test_simulator_demand_increase_raises_procurement():
    data = build_world()
    fc = build_forecast()
    base = simulate_scenario(data, fc, demand_pct=0.0)
    up = simulate_scenario(data, fc, demand_pct=0.2)
    assert up["summary"]["procurement_units"] >= base["summary"]["procurement_units"]
    assert up["summary"]["avg_stockout_probability"] >= base["summary"]["avg_stockout_probability"] - 0.02


def test_simulator_inventory_drop_raises_procurement():
    data = build_world()
    fc = build_forecast()
    base = simulate_scenario(data, fc, inventory_pct=0.0)
    drop = simulate_scenario(data, fc, inventory_pct=-0.5)
    assert drop["summary"]["procurement_units"] >= base["summary"]["procurement_units"]


def test_simulator_lead_time_slip_increases_stockout_probability():
    data = build_world()
    fc = build_forecast()
    base = simulate_scenario(data, fc, lead_time_pct=0.0)
    slip = simulate_scenario(data, fc, lead_time_pct=0.5)
    assert slip["summary"]["avg_stockout_probability"] >= base["summary"]["avg_stockout_probability"] - 0.02
