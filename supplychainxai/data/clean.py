"""Stage 2 — Cleaning: repair raw data and report every action taken.

Rules (each returns an audit count):
  SK01 normalise SKU casing / whitespace            (string hygiene)
  CL01 drop exact duplicate rows
  CL02 negative quantities  -> NaN (impossible value)
  CL03 non-numeric garbage  -> NaN (coercion during ingestion)
  CL04 NaN quantity         -> drop row (cannot be imputed safely for demand)
  PO01 expected_date < order_date -> swap correction (input error)
  PO02 missing delivered_date   -> status stays 'OPEN'
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from supplychainxai.data.ingest import RawData


@dataclass
class CleanReport:
    steps: list[dict] = field(default_factory=list)

    def add(self, table: str, rule: str, action: str, rows_affected: int) -> None:
        self.steps.append(
            {"table": table, "rule": rule, "action": action, "rows_affected": int(rows_affected)}
        )

    def total_fixed(self) -> int:
        return int(sum(s["rows_affected"] for s in self.steps))

    def to_list(self) -> list[dict]:
        return self.steps


def _normalise_sku(df: pd.DataFrame, report: CleanReport) -> pd.DataFrame:
    before = df["sku"].copy()
    df["sku"] = df["sku"].str.strip().str.upper()
    changed = int((before.notna() & (before != df["sku"])).sum())
    report.add("multiple", "SK01", "normalise sku casing/whitespace", changed)
    return df


def clean_sales(sales: pd.DataFrame, report: CleanReport) -> pd.DataFrame:
    df = _normalise_sku(sales.copy(), report)
    dups = int(df.duplicated().sum())
    df = df.drop_duplicates()
    report.add("sales", "CL01", "drop exact duplicates", dups)

    neg = int((df["units_sold"] < 0).sum())
    df.loc[df["units_sold"] < 0, "units_sold"] = pd.NA
    report.add("sales", "CL02", "negative units_sold -> NaN", neg)

    missing = int(df["units_sold"].isna().sum())
    df = df.dropna(subset=["units_sold"])
    report.add("sales", "CL04", "drop rows with un-imputable quantity", missing)

    df["units_sold"] = df["units_sold"].astype("int64")
    df["promo_flag"] = df["promo_flag"].fillna(0).astype("int64")
    return df.reset_index(drop=True)


def clean_inventory(inv: pd.DataFrame, report: CleanReport) -> pd.DataFrame:
    df = _normalise_sku(inv.copy(), report)
    dups = int(df.duplicated(subset=["date", "sku"]).sum())
    df = df.drop_duplicates(subset=["date", "sku"], keep="last")
    report.add("inventory", "CL01", "drop duplicate sku-date snapshots", dups)

    neg = int(((df["on_hand"] < 0) | (df["on_order"] < 0)).sum())
    for col in ("on_hand", "on_order"):
        df.loc[df[col] < 0, col] = 0  # physically floored at zero
    report.add("inventory", "CL02", "negative stock floored to 0", neg)
    return df.fillna({"on_hand": 0, "on_order": 0}).reset_index(drop=True)


def clean_purchase_orders(pos: pd.DataFrame, report: CleanReport) -> pd.DataFrame:
    df = _normalise_sku(pos.copy(), report)
    dups = int(df.duplicated(subset=["po_id"]).sum())
    df = df.drop_duplicates(subset=["po_id"], keep="first")
    report.add("purchase_orders", "CL01", "drop duplicate PO ids", dups)

    swapped = int((df["expected_date"] < df["order_date"]).sum())
    bad = df["expected_date"] < df["order_date"]
    fixed = df.loc[bad, ["order_date", "expected_date"]].copy()
    df.loc[bad, "order_date"] = fixed["expected_date"].values
    df.loc[bad, "expected_date"] = fixed["order_date"].values
    report.add("purchase_orders", "PO01", "swap order/expected date input errors", swapped)

    df["status"] = df["delivered_date"].isna().map({True: "OPEN", False: "DELIVERED"})
    return df.reset_index(drop=True)


def clean_all(raw: RawData) -> tuple[RawData, CleanReport]:
    report = CleanReport()
    cleaned = RawData(
        products=raw.products.copy(),
        suppliers=raw.suppliers.copy(),
        supply_terms=raw.supply_terms.copy(),
        sales=clean_sales(raw.sales, report),
        inventory=clean_inventory(raw.inventory, report),
        purchase_orders=clean_purchase_orders(raw.purchase_orders, report),
    )
    return cleaned, report
