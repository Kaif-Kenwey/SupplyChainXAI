"""Risk engine tests on hand-built scenarios."""
import numpy as np
import pandas as pd

from supplychainxai.risk.engine import (Alert, demand_spikes, effective_lead,
                                        lead_time_intelligence, sps_norm_cdf,
                                        supplier_reliability)


def test_norm_cdf_bounds():
    assert sps_norm_cdf(0) == 0.5
    assert sps_norm_cdf(10) > 0.99999
    assert sps_norm_cdf(-10) < 0.00001


def _sales_frame(values, end="2025-12-31"):
    dates = pd.date_range(end=end, periods=len(values), freq="D")
    return pd.DataFrame({"date": dates, "sku": "P101", "units_sold": values,
                         "unit_price": 5.0, "promo_flag": 0})


def test_demand_spike_fires_on_level_shift():
    base = [100] * 70
    spike = [150] * 21                       # +50% step, p will be ~0
    alerts = demand_spikes(_sales_frame(base + spike))
    assert len(alerts) == 1
    assert alerts[0].type == "demand_spike"
    assert alerts[0].severity == "CRITICAL"  # lift >= 30%
    assert alerts[0].data["lift_pct"] > 0.30


def test_no_spike_on_stable_series():
    alerts = demand_spikes(_sales_frame([100] * 91))
    assert alerts == []


def test_lead_time_intelligence_detects_drift():
    n = 30
    po = pd.DataFrame({
        "po_id": [f"PO-{i}" for i in range(n)],
        "sku": "P104", "supplier_id": "S4",
        "order_date": pd.date_range("2025-09-01", periods=n, freq="10D"),
        "expected_date": pd.date_range("2025-09-05", periods=n, freq="10D"),
        "delivered_date": pd.NaT, "quantity": [100.0] * n, "unit_price": [50.0] * n,
        "status": "DELIVERED",
    })
    # baseline ~12d, recent ~18d
    leads = [12] * 20 + [18] * 10
    po["delivered_date"] = [o + pd.Timedelta(days=l) for o, l in zip(po["order_date"], leads)]
    intel = lead_time_intelligence(po)
    row = intel.iloc[0]
    assert row["drift_pct"] > 0.20
    assert row["n_recent"] >= 2


def test_effective_lead_widens_with_drift():
    terms = pd.DataFrame({"sku": ["P104"], "supplier_id": ["S4"], "unit_price": [50.0],
                          "quoted_lead_days": [12], "lead_time_std": [2.0],
                          "is_primary": [True]})
    intel = pd.DataFrame([{"sku": "P104", "supplier_id": "S4", "baseline_lead": 12.0,
                           "recent_lead": 18.0, "lead_std": 3.0, "drift_pct": 0.5,
                           "n_recent": 4}])
    lead, std, sup = effective_lead("P104", terms, intel)
    assert sup == "S4"
    assert lead == 18                         # 12 * 1.5 drift-adjusted
    assert std >= 3.0


def test_supplier_reliability_small_sample_guard():
    po = pd.DataFrame({
        "po_id": ["PO-1", "PO-2"], "sku": ["P101", "P101"], "supplier_id": ["S2", "S2"],
        "order_date": pd.to_datetime(["2025-12-01", "2025-12-05"]),
        "expected_date": pd.to_datetime(["2025-12-05", "2025-12-09"]),
        "delivered_date": pd.to_datetime(["2025-12-20", "2025-12-25"]),  # both late
        "quantity": [10.0, 10.0], "unit_price": [5.0, 5.0],
        "status": ["DELIVERED", "DELIVERED"],
    })
    suppliers = pd.DataFrame({"supplier_id": ["S2"], "name": ["X"], "country": ["Y"],
                              "target_otd": [0.9]})
    # 2 POs only -> never judge; no alert despite 0% OTD
    assert supplier_reliability(po, suppliers) == []
