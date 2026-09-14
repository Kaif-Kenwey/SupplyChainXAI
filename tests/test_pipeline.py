"""Data pipeline tests: cleaning rules, validation gates, feature engineering."""

import numpy as np
import pandas as pd
import pytest

from supplychainxai.data.clean import CleanReport, clean_purchase_orders, clean_sales
from supplychainxai.data.features import FEATURE_COLUMNS, build_features, model_frame
from supplychainxai.data.validate import validate_all


def make_sales(**over):
    df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=10, freq="D").tolist() * 1,
            "sku": ["P101"] * 10,
            "units_sold": [10, 12, 0, 14, 11, 10, 13, 12, 11, 10],
            "unit_price": [5.0] * 10,
            "promo_flag": [0] * 10,
        }
    )
    for k, v in over.items():
        df[k] = v
    return df


# ------------------------------------------------------------------ cleaning
def test_clean_sales_drops_duplicates_and_negatives():
    df = make_sales()
    dirty = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # 1 exact duplicate
    dirty.loc[3, "units_sold"] = -5  # impossible negative
    rep = CleanReport()
    out = clean_sales(dirty, rep)
    assert len(out) == 9  # dup + bad row dropped
    assert (out["units_sold"] >= 0).all()
    rules = {s["rule"] for s in rep.to_list()}
    assert "CL01" in rules and "CL02" in rules


def test_clean_sales_normalises_sku_casing():
    df = make_sales()
    df["sku"] = " p101 "
    rep = CleanReport()
    out = clean_sales(df, rep)
    assert (out["sku"] == "P101").all()


def test_clean_po_swaps_bad_dates():
    po = pd.DataFrame(
        {
            "po_id": ["PO-1"],
            "sku": ["P101"],
            "supplier_id": ["S1"],
            "order_date": [pd.Timestamp("2025-03-10")],
            "expected_date": [pd.Timestamp("2025-03-01")],  # before order -> input error
            "delivered_date": [pd.Timestamp("2025-03-15")],
            "quantity": [100.0],
            "unit_price": [5.0],
        }
    )
    rep = CleanReport()
    out = clean_purchase_orders(po, rep)
    assert out.loc[0, "order_date"] == pd.Timestamp("2025-03-01")
    assert out.loc[0, "expected_date"] == pd.Timestamp("2025-03-10")
    assert out.loc[0, "status"] == "DELIVERED"


# ---------------------------------------------------------------- validation
def test_validation_flags_unknown_sku_and_negatives():
    sales = make_sales()
    sales.loc[0, "sku"] = "P999"  # not in master
    inv = pd.DataFrame(
        {"date": sales["date"], "sku": "P101", "on_hand": [50] * 10, "on_order": [0] * 10}
    )
    inv.loc[2, "on_hand"] = -3
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
        {"supplier_id": ["S1"], "name": ["A"], "country": ["X"], "target_otd": [0.9]}
    )
    terms = pd.DataFrame(
        {
            "sku": ["P101"],
            "supplier_id": ["S1"],
            "unit_price": [4.5],
            "quoted_lead_days": [5],
            "lead_time_std": [1.0],
            "is_primary": [True],
        }
    )
    pos = pd.DataFrame(
        {
            "po_id": ["PO-1"],
            "sku": ["P101"],
            "supplier_id": ["S1"],
            "order_date": [pd.Timestamp("2025-01-01")],
            "expected_date": [pd.Timestamp("2025-01-06")],
            "delivered_date": [pd.Timestamp("2025-01-06")],
            "quantity": [50.0],
            "unit_price": [4.5],
        }
    )

    from supplychainxai.data.ingest import RawData

    rep = validate_all(
        RawData(
            products=products,
            suppliers=suppliers,
            supply_terms=terms,
            sales=sales,
            inventory=inv,
            purchase_orders=pos,
        )
    )
    assert not rep.passed
    failed = {r.rule for r in rep.results if not r.passed}
    assert any("V01" in r for r in failed)
    assert any("V06" in r for r in failed)


# ------------------------------------------------------------------ features
def test_features_lags_and_no_leakage():
    sales = make_sales()
    sales["units_sold"] = range(10, 20)
    inv = pd.DataFrame(
        {
            "date": sales["date"],
            "sku": "P101",
            "on_hand": [100 - i for i in range(10)],
            "on_order": [0] * 10,
        }
    )
    feats = build_features(sales, inv)
    row = feats.iloc[8]
    assert row["lag_1"] == 17  # units_sold[7]
    assert row["lag_7"] == 11  # units_sold[1]
    assert row["roll_mean_7"] == pytest.approx(np.mean([11, 12, 13, 14, 15, 16, 17]))
    assert set(FEATURE_COLUMNS).issubset(feats.columns)


def test_model_frame_drops_nan_head():
    n = 60
    sales = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=n, freq="D"),
            "sku": "P101",
            "units_sold": range(10, 10 + n),
            "unit_price": [5.0] * n,
            "promo_flag": [0] * n,
        }
    )
    inv = pd.DataFrame(
        {"date": sales["date"], "sku": "P101", "on_hand": [10] * n, "on_order": [0] * n}
    )
    feats = build_features(sales, inv)
    assert len(model_frame(feats, "P101")) == n - 28  # rows with lag_28 available
