"""Data-quality monitoring: a continuous superset of the validation gates.

The pipeline *validation* (data/validate.py) gates correctness — CRITICAL
failures abort. This module *monitors* quality as a first-class report:

  ERROR    → pipeline should stop (same integrity rules as validation)
  WARNING  → pipeline can continue, finding is recorded

Checks: missing values, duplicate rows, invalid dates, negative
quantities/prices/stock, missing SKU/supplier relationships, unexpected
categorical values, abnormal demand spikes, schema changes, stale data.

Output: `artifacts/data_quality_report.json` with status, checks,
warnings, errors, statistics, timestamp, dataset version.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd

from supplychainxai import config
from supplychainxai.data.ingest import SCHEMAS, RawData
from supplychainxai.data.validate import validate_all
from supplychainxai.mlops.provenance import compute_dataset_version

ABNORMAL_DEMAND_Z = 8.0  # extreme outlier (not promo noise) -> WARNING
REQUIRED_TABLES = ("products", "suppliers", "supply_terms", "sales", "inventory", "purchase_orders")


def _check(table: str, rule: str, severity: str, failures: int, detail: str) -> dict:
    return {
        "table": table,
        "rule": rule,
        "severity": severity,
        "status": "PASS" if failures == 0 else "FAIL",
        "failures": int(failures),
        "detail": detail,
    }


def _schema_checks(data: RawData) -> list[dict]:
    out = []
    tables = [
        ("products", data.products),
        ("suppliers", data.suppliers),
        ("supply_terms", data.supply_terms),
        ("sales", data.sales),
        ("inventory", data.inventory),
        ("purchase_orders", data.purchase_orders),
    ]
    for table, df in tables:
        expected = set(SCHEMAS.get(table, {}).get("dtypes", {}))
        actual = set(df.columns)
        missing = expected - actual
        extra = actual - expected
        out.append(
            _check(
                table,
                "DQ01 schema columns",
                "ERROR" if missing else "WARNING",
                len(missing),
                f"missing={sorted(missing)}"
                if missing
                else (f"extra columns (tracked): {sorted(extra)}" if extra else "schema unchanged"),
            )
        )
    return out


def _content_checks(data: RawData, spike_z: float = ABNORMAL_DEMAND_Z) -> list[dict]:
    checks = []

    # nulls / missing values (post-cleaning should be clean; report anyway)
    for name, df in (
        ("sales", data.sales),
        ("inventory", data.inventory),
        ("purchase_orders", data.purchase_orders),
    ):
        nulls = int(df.isna().sum().sum())
        checks.append(_check(name, "DQ02 missing values", "WARNING", nulls, f"{nulls} null cells"))

    # duplicates (should be gone after cleaning)
    dup_sales = int(data.sales.duplicated(subset=["date", "sku"]).sum())
    checks.append(
        _check("sales", "DQ03 duplicate sku-date", "ERROR", dup_sales, "duplicate demand rows")
    )
    dup_inv = int(data.inventory.duplicated(subset=["date", "sku"]).sum())
    checks.append(
        _check("inventory", "DQ04 duplicate sku-date", "ERROR", dup_inv, "duplicate snapshots")
    )

    # invalid dates
    for name, df, cols in (
        ("sales", data.sales, ["date"]),
        ("inventory", data.inventory, ["date"]),
        ("purchase_orders", data.purchase_orders, ["order_date", "expected_date"]),
    ):
        bad = sum(
            int(pd.to_datetime(df[c], errors="coerce").isna().sum())
            for c in cols
            if c in df.columns
        )
        checks.append(
            _check(name, "DQ05 unparseable dates", "ERROR", bad, f"{bad} unparseable date cells")
        )

    # domain: negative quantities / prices / stock
    checks.append(
        _check(
            "sales", "DQ06 negative units", "ERROR", int((data.sales["units_sold"] < 0).sum()), ""
        )
    )
    checks.append(
        _check(
            "sales",
            "DQ07 non-positive price",
            "ERROR",
            int((data.sales["unit_price"] <= 0).sum()),
            "",
        )
    )
    checks.append(
        _check(
            "inventory",
            "DQ08 negative stock",
            "ERROR",
            int(((data.inventory["on_hand"] < 0) | (data.inventory["on_order"] < 0)).sum()),
            "",
        )
    )
    checks.append(
        _check(
            "purchase_orders",
            "DQ09 non-positive PO quantity",
            "ERROR",
            int((data.purchase_orders["quantity"] <= 0).sum()),
            "",
        )
    )

    # referential relationships
    known_skus = set(data.products["sku"])
    for name, df in (
        ("sales", data.sales),
        ("inventory", data.inventory),
        ("purchase_orders", data.purchase_orders),
    ):
        orphan = int((~df["sku"].isin(known_skus)).sum())
        checks.append(
            _check(name, "DQ10 unknown SKU", "ERROR", orphan, f"{orphan} rows with unknown SKU")
        )
    known_sup = set(data.suppliers["supplier_id"])
    orphan_sup = int((~data.purchase_orders["supplier_id"].isin(known_sup)).sum())
    checks.append(
        _check(
            "purchase_orders",
            "DQ11 unknown supplier",
            "ERROR",
            orphan_sup,
            f"{orphan_sup} rows with unknown supplier",
        )
    )

    # unexpected categorical values (skip columns absent due to schema drift)
    bad_promo = 0
    if "promo_flag" in data.sales.columns:
        bad_promo = int((~data.sales["promo_flag"].isin([0, 1])).sum())
    checks.append(
        _check(
            "sales",
            "DQ12 promo_flag domain",
            "WARNING",
            bad_promo,
            f"{bad_promo} rows outside {{0,1}}",
        )
    )
    case_variants = int(
        sum(
            df["sku"].fillna("").str.contains(r"[a-z]", regex=True).sum()
            for df in (data.sales, data.products)
        )
    )
    checks.append(
        _check(
            "multiple",
            "DQ13 sku casing variants",
            "WARNING",
            case_variants,
            f"{case_variants} lowercase SKU cells",
        )
    )

    # abnormal demand spikes (extreme outliers in the demand distribution)
    spikes = 0
    for _, g in data.sales.groupby("sku"):
        y = g.sort_values("date")["units_sold"].astype(float)
        if len(y) > 30 and y.std() > 0:
            z = (y - y.mean()) / y.std()
            spikes += int((z.abs() > spike_z).sum())
    checks.append(
        _check(
            "sales",
            "DQ14 abnormal demand outliers",
            "WARNING",
            spikes,
            f"{spikes} days beyond {spike_z}σ per-SKU",
        )
    )

    return checks


def _freshness_checks(data: RawData, stale_days: int, now: datetime | None = None) -> list[dict]:
    """Freshness relative to `now` (wall clock by default; the training
    pipeline passes the dataset snapshot so the committed static dataset is
    not falsely flagged as stale)."""
    checks = []
    today = pd.Timestamp((now or datetime.now(UTC)).date())
    for name, df in (("sales", data.sales), ("inventory", data.inventory)):
        if len(df) and "date" in df.columns:
            age = (today - pd.to_datetime(df["date"]).max()).days
            stale = age > stale_days
            checks.append(
                _check(
                    name,
                    "DQ15 data freshness",
                    "WARNING" if stale else "PASS",
                    max(int(age), 0) if stale else 0,
                    f"latest row is {age} days old (threshold {stale_days})",
                )
            )
    return checks


def run_data_quality(
    data: RawData, stale_days: int | None = None, now: datetime | None = None
) -> dict:
    """Full data-quality sweep. Returns the report dict (and writes it)."""
    from supplychainxai.config.settings import get_settings

    stale_days = stale_days if stale_days is not None else get_settings().monitoring.stale_data_days

    checks: list[dict] = []
    checks += _schema_checks(data)
    checks += _content_checks(data)

    # bridge: pipeline validation gates become ERROR findings when failing
    vrep = validate_all(data)
    for r in vrep.results:
        if not r.passed:
            checks.append(
                _check(
                    r.table,
                    f"V:{r.rule}",
                    "ERROR" if r.severity == "CRITICAL" else "WARNING",
                    r.failures,
                    r.detail,
                )
            )

    checks += _freshness_checks(data, stale_days, now=now)

    errors = [c for c in checks if c["severity"] == "ERROR" and c["failures"] > 0]
    warnings = [c for c in checks if c["severity"] == "WARNING" and c["failures"] > 0]

    stats = {
        "rows": {
            "sales": len(data.sales),
            "inventory": len(data.inventory),
            "purchase_orders": len(data.purchase_orders),
            "products": len(data.products),
            "suppliers": len(data.suppliers),
            "supply_terms": len(data.supply_terms),
        },
        "date_min": str(pd.to_datetime(data.sales["date"]).min().date()),
        "date_max": str(pd.to_datetime(data.sales["date"]).max().date()),
        "skus": int(data.sales["sku"].nunique()),
        "null_cells_total": int(
            sum(c["failures"] for c in checks if c["rule"] == "DQ02 missing values")
        ),
    }
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset_version": compute_dataset_version(),
        "status": "FAIL" if errors else ("WARN" if warnings else "PASS"),
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "statistics": stats,
    }
    (config.ARTIFACTS / "data_quality_report.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    return report
