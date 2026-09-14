#!/usr/bin/env python3
"""Generate a realistic synthetic procurement/sales dataset for SupplyChainXAI.

Simulates 3 years (2023-01-01 .. 2025-12-31) of daily operations for an
industrial-components distributor:

  * 12 SKUs across 5 categories, each sourced from 2-3 competing suppliers
  * demand with trend + weekday pattern + yearly seasonality + promotions + noise
  * a periodic-review (s, Q) inventory policy -> real stockouts, real POs
  * supplier lead times sampled from per-supplier Normal distributions
  * four engineered story-lines the analytics must later discover:
      1. P104 demand spike (+45% over the final 60 days)
      2. P104's primary supplier develops a +35% lead-time drift (final 90 days)
      3. P107 bulk over-purchase -> months of overstock
      4. one supplier-SKU pair gets a +15% unit-price jump (abnormal procurement)

The writer then injects realistic data-quality problems (duplicates, negative /
missing quantities, inconsistent SKU casing, swapped PO dates) so the cleaning &
validation stages of the pipeline have genuine work to do.

Usage:
    python scripts/generate_data.py            # writes 6 CSVs into data/raw/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from supplychainxai import config

RNG = np.random.default_rng(42)

# --------------------------------------------------------------------------
# master data
# --------------------------------------------------------------------------
PRODUCTS = [
    # sku, name, category, unit_cost, base_demand, trend/yr, pack_size
    ("P101", "Hex Bolt M8 Grade 8.8 (Box)", "Fasteners", 4.20, 140, 0.06, 50),
    ("P102", "Rubber Gasket 100mm", "Sealing", 7.80, 92, 0.03, 20),
    ("P103", "Hydraulic Seal Kit 40mm", "Sealing", 26.50, 46, 0.05, 10),
    ("P104", "Deep Groove Bearing 6204-2RS", "Bearings", 58.00, 84, 0.09, 20),
    ("P105", "Pillow Block Unit UCP205", "Bearings", 148.00, 31, 0.04, 5),
    ("P106", "V-Belt A-Section 914mm", "Power Transmission", 19.90, 67, -0.02, 10),
    ("P107", "Conveyor Roller 600mm", "Power Transmission", 96.00, 24, 0.05, 5),
    ("P108", "AC Induction Motor 2.2kW 4P", "Motors", 412.00, 13, 0.07, 2),
    ("P109", "Proximity Sensor M12 PNP", "Electronics", 31.00, 58, 0.10, 10),
    ("P110", "PLC I/O Module 16-Channel", "Electronics", 176.00, 19, 0.12, 4),
    ("P111", "Control Cable 2.5mm2 (100m)", "Electronics", 74.00, 112, 0.08, 25),
    ("P112", "Safety Relay 24VDC 3NO", "Electronics", 132.00, 27, 0.11, 5),
]

SUPPLIERS = [
    ("S1", "Apex Industrial Components", "India", 0.93),
    ("S2", "VoltEdge Electronics", "China", 0.78),
    ("S3", "Bharat Precision Fasteners", "India", 0.96),
    ("S4", "NordicMech Supplies", "Germany", 0.90),
    ("S5", "Shakti Polymers & Seals", "India", 0.72),
    ("S6", "GlobalDrive Motors AG", "Germany", 0.95),
]

# sku -> [(supplier, price_index, lead_days, lead_std, primary)]
SUPPLY_TERMS = {
    "P101": [("S3", 1.00, 5, 1.2, True), ("S1", 0.93, 8, 2.0, False)],
    "P102": [("S5", 1.00, 9, 2.5, True), ("S1", 1.12, 7, 1.8, False)],
    "P103": [("S5", 0.95, 10, 3.0, True), ("S1", 1.10, 8, 2.0, False)],
    "P104": [("S4", 1.00, 12, 2.5, True), ("S3", 1.08, 6, 1.5, False), ("S1", 1.05, 9, 2.2, False)],
    "P105": [("S4", 1.00, 14, 3.0, True), ("S3", 1.12, 9, 2.0, False)],
    "P106": [("S4", 1.00, 11, 2.5, True), ("S1", 0.96, 8, 2.2, False)],
    "P107": [("S4", 1.00, 15, 3.5, True), ("S1", 1.06, 10, 2.5, False)],
    "P108": [("S6", 1.00, 18, 4.0, True), ("S1", 1.14, 12, 3.0, False)],
    "P109": [
        ("S2", 0.90, 16, 4.5, True),
        ("S1", 1.08, 9, 2.0, False),
        ("S6", 1.20, 13, 3.0, False),
    ],
    "P110": [("S2", 0.92, 18, 5.0, True), ("S6", 1.15, 14, 3.2, False)],
    "P111": [("S2", 0.94, 15, 4.0, True), ("S1", 1.05, 8, 2.0, False)],
    "P112": [("S6", 1.00, 16, 3.5, True), ("S2", 0.88, 20, 5.5, False)],
}

PRICE_ANOMALY = ("S4", "P104")  # +15% unit price in the final 60 days
LEAD_DRIFT = ("S4", "P104", 0.50, 90)  # +50% lead-time drift, final 90 days
DEMAND_SPIKE = ("P104", 0.50, 60)  # +50% demand ramp, final 60 days


# --------------------------------------------------------------------------
# demand model
# --------------------------------------------------------------------------
def yearly_seasonality(dates: pd.DatetimeIndex, category: str) -> np.ndarray:
    """Category-specific annual cycle: manufacturing push in Q4, summer dip."""
    doy = dates.dayofyear.to_numpy()
    phase = {
        "Fasteners": 0.0,
        "Sealing": 0.8,
        "Bearings": 1.3,
        "Power Transmission": 1.6,
        "Motors": 2.0,
        "Electronics": 2.6,
    }[category]
    amp = {
        "Fasteners": 0.05,
        "Sealing": 0.10,
        "Bearings": 0.14,
        "Power Transmission": 0.12,
        "Motors": 0.16,
        "Electronics": 0.20,
    }[category]
    return 1.0 + amp * np.sin(2 * np.pi * (doy - 15) / 365.25 + phase)


def weekday_factor(dates: pd.DatetimeIndex) -> np.ndarray:
    """B2B pattern: strong weekdays, weak Saturday, near-dead Sunday."""
    dow = dates.dayofweek.to_numpy()
    table = np.array([1.06, 1.10, 1.08, 1.05, 1.00, 0.55, 0.22])
    return table[dow]


def simulate_demand(dates, sku, base, trend, category) -> pd.DataFrame:
    n = len(dates)
    t = np.arange(n)
    trend_f = (1.0 + trend) ** (t / 365.25)
    seas = yearly_seasonality(dates, category)
    dow = weekday_factor(dates)

    promo = np.zeros(n)
    promo_days = RNG.choice(np.arange(n), size=int(n * 0.035), replace=False)
    promo[promo_days] = 1.0
    promo_mult = np.where(promo == 1, RNG.uniform(1.5, 2.2, size=n), 1.0)

    noise = RNG.normal(1.0, 0.11, size=n)

    spike = np.ones(n)
    if sku == DEMAND_SPIKE[0]:
        _, lift, window = DEMAND_SPIKE
        ramp = np.linspace(0.0, 1.0, window)
        spike[-window:] = 1.0 + lift * ramp

    demand = base * trend_f * seas * dow * promo_mult * noise * spike
    demand = np.clip(np.round(demand), 0, None).astype(int)
    return pd.DataFrame(
        {"date": dates, "sku": sku, "demand_true": demand, "promo_flag": promo.astype(int)}
    )


# --------------------------------------------------------------------------
# inventory + procurement simulation  (periodic review, s-Q policy)
# --------------------------------------------------------------------------
def simulate_sku(df: pd.DataFrame, terms, start_inv: int) -> tuple[pd.DataFrame, list[dict]]:
    rows, pos = [], []
    on_hand = float(start_inv)
    open_pos: list[dict] = []  # {qty, arrive, supplier, expected, order_date, price}
    po_counter = 0
    bulk_done = False
    lead_primary = next(t for t in terms if t[4])

    for i, rec in df.iterrows():
        # ---- arrivals
        arriving = [po for po in open_pos if po["arrive"] <= rec["date"]]
        for po in arriving:
            on_hand += po["quantity"]
            open_pos.remove(po)

        # ---- sales (censored by availability)
        sales = min(int(rec["demand_true"]), int(on_hand))
        on_hand -= sales

        # ---- reorder decision (adaptive: tracks recent demand, like a real buyer)
        ip = on_hand + sum(po["quantity"] for po in open_pos)
        recent = df["demand_true"].iloc[max(0, i - 13) : i + 1].mean()
        rop = recent * lead_primary[2] * 1.25
        if ip < rop and not open_pos:
            # slow movers occasionally get bulk-ordered (P106/P107 stories):
            # the bulk multiplier applies to the FIRST order placed on/after the
            # target date (cycle timing is emergent, so pin the window not the day)
            bulk = 1.0
            if rec["sku"] == "P107" and not bulk_done and rec["date"] >= pd.Timestamp("2024-06-18"):
                bulk, bulk_done = 3.0, True
            if rec["sku"] == "P106" and not bulk_done and rec["date"] >= pd.Timestamp("2025-12-05"):
                bulk, bulk_done = 10.0, True
            qty = int(max(round(recent * lead_primary[2] * 1.15 * bulk), 10))
            qty = int(np.ceil(qty / 5.0) * 5)
            po_counter += 1

            sup, pidx, lead, lstd, _ = lead_primary
            if (sup, rec["sku"]) == PRICE_ANOMALY and rec["date"] >= pd.Timestamp("2025-11-01"):
                pidx *= 1.15
            actual_lead = RNG.normal(lead * 0.92, lstd)
            if RNG.random() < 0.05:
                actual_lead += lstd * 2.5  # occasional bad delay
            if (sup, rec["sku"]) == LEAD_DRIFT[:2] and rec["date"] >= pd.Timestamp("2025-10-02"):
                actual_lead *= 1.0 + LEAD_DRIFT[2] * min(1.0, 0.4 + 0.6 * RNG.random())
            actual_lead = max(2.0, actual_lead)

            arrive = rec["date"] + pd.Timedelta(days=float(round(actual_lead)))
            expected = rec["date"] + pd.Timedelta(days=int(lead))
            unit_price = round(rec["unit_cost"] * pidx * float(RNG.normal(1.0, 0.02)), 2)
            po = {
                "po_id": f"PO-{rec['sku']}-{po_counter:04d}",
                "sku": rec["sku"],
                "supplier_id": sup,
                "order_date": rec["date"],
                "expected_date": expected,
                "arrive": arrive,
                "quantity": qty,
                "unit_price": unit_price,
            }
            open_pos.append(po)
            pos.append(
                {
                    k: po[k]
                    for k in (
                        "po_id",
                        "sku",
                        "supplier_id",
                        "order_date",
                        "expected_date",
                        "arrive",
                        "quantity",
                        "unit_price",
                    )
                }
            )

        rows.append(
            {
                "date": rec["date"],
                "sku": rec["sku"],
                "on_hand": int(on_hand),
                "on_order": int(sum(po["quantity"] for po in open_pos)),
            }
        )
    return pd.DataFrame(rows), pos


# --------------------------------------------------------------------------
# data-quality corruption (so cleaning/validation have real work to do)
# --------------------------------------------------------------------------
def corrupt(sales: pd.DataFrame, pos_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(sales)
    idx = RNG.choice(n, size=int(n * 0.003), replace=False)

    sales.loc[idx[: len(idx) // 4], "units_sold"] = -sales.loc[idx[: len(idx) // 4], "units_sold"]
    sales.loc[idx[len(idx) // 4 : len(idx) // 2], "units_sold"] = np.nan
    lower = sales.loc[idx[len(idx) // 2 : (3 * len(idx)) // 4], "sku"].index
    sales.loc[lower, "sku"] = sales.loc[lower, "sku"].str.lower() + " "
    dup = sales.loc[idx[(3 * len(idx)) // 4 :]].copy()
    sales = pd.concat([sales, dup], ignore_index=True)

    swap = pos_df.sample(6, random_state=7).index
    tmp = pos_df.loc[swap, "expected_date"].copy()
    pos_df.loc[swap, "expected_date"] = pos_df.loc[swap, "order_date"]
    pos_df.loc[swap, "order_date"] = tmp
    return sales, pos_df


# --------------------------------------------------------------------------
def main() -> None:
    dates = pd.date_range(config.SIM_START, config.SIM_END, freq="D")
    products = pd.DataFrame(
        PRODUCTS,
        columns=[
            "sku",
            "name",
            "category",
            "unit_cost",
            "base_demand",
            "trend_per_year",
            "pack_size",
        ],
    )
    suppliers = pd.DataFrame(SUPPLIERS, columns=["supplier_id", "name", "country", "target_otd"])

    # ---- supply terms (product-supplier contracts)
    term_rows = []
    for sku, terms in SUPPLY_TERMS.items():
        cost = float(products.loc[products.sku == sku, "unit_cost"].iloc[0])
        for sup, pidx, lead, lstd, primary in terms:
            term_rows.append(
                {
                    "sku": sku,
                    "supplier_id": sup,
                    "unit_price": round(cost * pidx, 2),
                    "quoted_lead_days": lead,
                    "lead_time_std": lstd,
                    "is_primary": primary,
                }
            )
    terms_df = pd.DataFrame(term_rows)

    # ---- daily simulation
    sales_frames, inv_frames, po_rows = [], [], []
    for _, p in products.iterrows():
        d = simulate_demand(dates, p.sku, p.base_demand, p.trend_per_year, p.category)
        d["unit_cost"] = p.unit_cost
        markup = RNG.uniform(1.28, 1.45)
        inv, pos = simulate_sku(d, SUPPLY_TERMS[p.sku], start_inv=int(p.base_demand * 25))
        sales = d.rename(columns={"demand_true": "units_sold"})
        sales["unit_price"] = np.round(
            p.unit_cost * markup * (1 + 0.02 * np.sin(np.arange(len(d)) / 90)), 2
        )
        sales_frames.append(sales)
        inv_frames.append(inv)
        po_rows.extend(pos)

    sales = pd.concat(sales_frames, ignore_index=True)[
        ["date", "sku", "units_sold", "unit_price", "promo_flag"]
    ]
    inventory = pd.concat(inv_frames, ignore_index=True)[["date", "sku", "on_hand", "on_order"]]
    pos_df = pd.DataFrame(po_rows)[
        [
            "po_id",
            "sku",
            "supplier_id",
            "order_date",
            "expected_date",
            "arrive",
            "quantity",
            "unit_price",
        ]
    ]
    pos_df = (
        pos_df.sort_values("order_date")
        .reset_index(drop=True)
        .rename(columns={"arrive": "delivered_date"})
    )

    # ---- PO status: anything not yet arrived in the final 30 days stays open
    last_date = sales["date"].max()
    pos_df["delivered_date"] = pos_df["delivered_date"].where(
        pos_df["delivered_date"] <= last_date, pd.NaT
    )

    # ---- inject data-quality problems
    sales, pos_df = corrupt(sales, pos_df.copy())

    # ---- write raw layer
    products.to_csv(config.RAW_FILES["products"], index=False)
    suppliers.to_csv(config.RAW_FILES["suppliers"], index=False)
    terms_df.to_csv(config.RAW_FILES["supply_terms"], index=False)
    sales.to_csv(config.RAW_FILES["sales"], index=False)
    inventory.to_csv(config.RAW_FILES["inventory"], index=False)
    pos_df.to_csv(config.RAW_FILES["purchase_orders"], index=False)

    stockout_days = int((inventory["on_hand"] == 0).sum())
    print(f"raw data written to {config.DATA_RAW}")
    print(f"  sales rows        : {len(sales):,}")
    print(f"  inventory rows    : {len(inventory):,}")
    print(f"  purchase orders   : {len(pos_df):,}")
    print(f"  sku stockout days : {stockout_days:,}")
    print(f"  open POs          : {int(pos_df['delivered_date'].isna().sum())}")


if __name__ == "__main__":
    main()
