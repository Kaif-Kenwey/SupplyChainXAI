"""Stage 3 — Validation: assert business rules on the cleaned data.

Produces a machine-readable validation report (pass/fail per rule) that is
persisted next to the processed data. A CRITICAL failure aborts the pipeline —
analytics must never run on data that fails key integrity checks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from supplychainxai.data.ingest import RawData


class DataValidationError(RuntimeError):
    """Raised when a CRITICAL validation rule fails."""


@dataclass
class RuleResult:
    table: str
    rule: str
    severity: str          # CRITICAL | WARNING
    passed: bool
    failures: int
    detail: str


@dataclass
class ValidationReport:
    results: list[RuleResult] = field(default_factory=list)

    def add(self, table: str, rule: str, severity: str, failures: int, detail: str) -> None:
        self.results.append(RuleResult(table, rule, severity, failures == 0, int(failures), detail))

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results if r.severity == "CRITICAL")

    def to_dict(self) -> dict:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "PASS" if self.passed else "FAIL",
            "checks": [r.__dict__ for r in self.results],
            "summary": {
                "total": len(self.results),
                "failed": sum(1 for r in self.results if not r.passed),
            },
        }


def validate_all(data: RawData) -> ValidationReport:
    rep = ValidationReport()

    # ---- referential integrity
    known_skus = set(data.products["sku"])
    for name, df in (("sales", data.sales), ("inventory", data.inventory),
                     ("purchase_orders", data.purchase_orders)):
        unknown = int((~df["sku"].isin(known_skus)).sum())
        rep.add(name, "V01 sku in product master", "CRITICAL", unknown,
                f"{unknown} rows reference unknown SKUs")

    known_sup = set(data.suppliers["supplier_id"])
    for name, df in (("supply_terms", data.supply_terms),
                     ("purchase_orders", data.purchase_orders)):
        unknown = int((~df["supplier_id"].isin(known_sup)).sum())
        rep.add(name, "V02 supplier in master", "CRITICAL", unknown,
                f"{unknown} rows reference unknown suppliers")

    # ---- domain rules
    rep.add("sales", "V03 units_sold > 0", "CRITICAL",
            int((data.sales["units_sold"] <= 0).sum()), "non-positive demand rows")
    rep.add("sales", "V04 unit_price > 0", "CRITICAL",
            int((data.sales["unit_price"] <= 0).sum()), "non-positive selling price")
    # calendar coverage: tolerate small gaps left by dropped corrupt rows
    expected = data.sales["date"].nunique()
    coverage = data.sales.groupby("sku")["date"].nunique() / expected
    rep.add("sales", "V05 calendar coverage >= 98% per sku", "WARNING",
            int(coverage.lt(0.98).sum()), "skus missing >2% of the daily calendar")
    rep.add("inventory", "V06 stock levels >= 0", "CRITICAL",
            int(((data.inventory["on_hand"] < 0) | (data.inventory["on_order"] < 0)).sum()),
            "negative stock rows")
    rep.add("purchase_orders", "V07 quantity > 0", "CRITICAL",
            int((data.purchase_orders["quantity"] <= 0).sum()), "non-positive PO quantity")
    rep.add("purchase_orders", "V08 delivered >= ordered", "CRITICAL",
            int((data.purchase_orders["delivered_date"] < data.purchase_orders["order_date"]).sum()),
            "delivered before ordered")
    rep.add("purchase_orders", "V09 every sku has >=1 primary supplier", "CRITICAL",
            int(data.supply_terms.groupby("sku")["is_primary"].sum().ne(1).sum()),
            "skus without exactly one primary source")
    rep.add("supply_terms", "V10 quoted lead time > 0", "CRITICAL",
            int((data.supply_terms["quoted_lead_days"] <= 0).sum()), "non-positive lead times")

    # ---- duplicate keys after cleaning
    rep.add("sales", "V11 unique sku-date", "CRITICAL",
            int(data.sales.duplicated(subset=["date", "sku"]).sum()), "duplicate demand rows")
    rep.add("inventory", "V12 unique sku-date", "CRITICAL",
            int(data.inventory.duplicated(subset=["date", "sku"]).sum()), "duplicate snapshots")

    return rep
