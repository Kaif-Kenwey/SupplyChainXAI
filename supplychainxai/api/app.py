"""SupplyChainXAI FastAPI service (phase 8 — productization).

uvicorn supplychainxai.api.app:app --reload --port 8000
-> dashboard at http://localhost:8000/  |  OpenAPI docs at /docs
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from supplychainxai import __version__, config
from supplychainxai.api import context as ctxmod
from supplychainxai.copilot import llm
from supplychainxai.copilot.engine import ask as copilot_ask

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ctx = ctxmod.load_context()
    yield


app = FastAPI(
    title="SupplyChainXAI API",
    version=__version__,
    description="Explainable demand forecasting, inventory risk and procurement optimization.",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ------------------------------------------------------------------ dashboard
@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": __version__,
        "llm_mode": "llm" if llm.llm_configured() else "grounded-narrator",
    }


@app.get("/ready")
def ready():
    """Readiness: required dependencies reachable (store, artifacts, models).

    Unlike /health (process liveness), /ready checks what the service needs
    to answer analytical questions. Returns structured JSON; never leaks
    connection strings or secrets.
    """
    checks: dict[str, str] = {}

    # analytics store reachable + has fact rows
    try:
        from supplychainxai.data import store

        n = store.query("SELECT COUNT(*) AS n FROM fact_sales")
        checks["analytics_store"] = "ok" if int(n["n"].iloc[0]) > 0 else "empty"
    except Exception as exc:
        checks["analytics_store"] = f"error: {type(exc).__name__}"

    # core artifacts loaded
    ctx = getattr(app.state, "ctx", None)
    if ctx is None:
        checks["artifacts"] = "error: context not loaded"
    else:
        ok = len(ctx.forecasts) > 0 and len(ctx.recommendations) > 0 and len(ctx.risk_alerts) >= 0
        checks["artifacts"] = "ok" if ok else "incomplete"
        checks["model_registry"] = (
            "ok" if ctx.registry_summary.get("production_versions") else "empty"
        )

    ready = all(v == "ok" for v in checks.values())
    return {"status": "ready" if ready else "not_ready", "checks": checks, "version": __version__}


# ------------------------------------------------------------------ analytics
@app.get("/api/kpis")
def kpis():
    ctx = app.state.ctx
    return {"snapshot_date": ctx.snapshot_date, **ctx.kpis, "model_selection": ctx.narrative}


@app.get("/api/inventory")
def inventory_health():
    ctx = app.state.ctx
    inv = ctx.data.inventory.sort_values("date").groupby("sku").tail(1)
    products = ctx.data.products.set_index("sku")
    fc = ctx.forecasts.groupby("sku")["prediction"].mean()
    rows = []
    for r in inv.itertuples():
        daily = float(fc.get(r.sku, 1.0)) or 1.0
        doh = r.on_hand / daily
        status = (
            "CRITICAL"
            if doh < config.CRITICAL_COVER_DAYS
            else "OVERSTOCK"
            if doh >= config.OVERSTOCK_DOH_DAYS
            else "OK"
            if doh >= 14
            else "LOW"
        )
        rows.append(
            {
                "sku": r.sku,
                "name": products.loc[r.sku, "name"],
                "category": products.loc[r.sku, "category"],
                "on_hand": int(r.on_hand),
                "on_order": int(r.on_order),
                "daily_forecast": round(daily, 1),
                "days_of_cover": round(doh, 1),
                "inventory_value": round(r.on_hand * float(products.loc[r.sku, "unit_cost"]), 2),
                "status": status,
            }
        )
    rows.sort(key=lambda x: x["days_of_cover"])
    return {"snapshot_date": ctx.snapshot_date, "items": rows}


@app.get("/api/forecast/{sku}")
def forecast(sku: str, history_days: int = Query(120, ge=30, le=365)):
    ctx = app.state.ctx
    fc = ctx.forecasts[ctx.forecasts["sku"] == sku].sort_values("date")
    if fc.empty:
        raise HTTPException(404, f"unknown sku {sku}")
    sales = ctx.data.sales[ctx.data.sales["sku"] == sku].sort_values("date").tail(history_days)
    comp = ctx.comparison[(ctx.comparison["sku"] == sku)].dropna(subset=["WAPE"])
    comp = comp.sort_values("WAPE")
    return {
        "sku": sku,
        "model": fc["model"].iloc[0],
        "history": [
            {"date": d.strftime("%Y-%m-%d"), "units": int(u)}
            for d, u in zip(sales["date"], sales["units_sold"])
        ],
        "forecast": [
            {
                "date": d.strftime("%Y-%m-%d"),
                "prediction": round(float(p), 1),
                "lower_80": round(float(lo), 1),
                "upper_80": round(float(hi), 1),
            }
            for d, p, lo, hi in zip(fc["date"], fc["prediction"], fc["lower_80"], fc["upper_80"])
        ],
        "model_comparison": comp[["model", "MAE", "RMSE", "MAPE", "WAPE"]]
        .round(2)
        .to_dict("records"),
    }


@app.get("/api/models")
def models():
    ctx = app.state.ctx
    valid = ctx.comparison.dropna(subset=["WAPE"])
    by_model = (
        valid.groupby("model")
        .agg(
            mean_wape=("WAPE", "mean"),
            mean_mae=("MAE", "mean"),
            mean_mape=("MAPE", "mean"),
            skus=("sku", "nunique"),
        )
        .round(2)
        .sort_values("mean_wape")
        .reset_index()
    )
    wins = valid.loc[valid.groupby("sku")["WAPE"].idxmin(), "model"].value_counts().to_dict()
    return {
        "by_model": by_model.to_dict("records"),
        "wins": wins,
        "narrative": ctx.narrative,
        "per_sku": valid.round(2).to_dict("records"),
    }


@app.get("/api/risk")
def risk():
    ctx = app.state.ctx
    return {"snapshot_date": ctx.snapshot_date, "alerts": [a.__dict__ for a in ctx.risk_alerts]}


@app.get("/api/suppliers")
def suppliers():
    ctx = app.state.ctx
    po = ctx.data.purchase_orders[ctx.data.purchase_orders["status"] == "DELIVERED"].copy()
    po["lead"] = (po["delivered_date"] - po["order_date"]).dt.days
    po["on_time"] = po["delivered_date"] <= po["expected_date"]
    spend = (
        ctx.data.purchase_orders.assign(v=lambda d: d["quantity"] * d["unit_price"])
        .groupby("supplier_id")["v"]
        .sum()
    )

    from supplychainxai.risk.engine import lead_time_intelligence

    lt = lead_time_intelligence(ctx.data.purchase_orders)

    rows = []
    for s in ctx.data.suppliers.itertuples():
        g = po[po["supplier_id"] == s.supplier_id]
        sku_drifts = lt[lt["supplier_id"] == s.supplier_id]
        rows.append(
            {
                "supplier_id": s.supplier_id,
                "name": s.name,
                "country": s.country,
                "pos": len(g),
                "on_time_pct": round(100 * float(g["on_time"].mean()), 1) if len(g) else None,
                "avg_lead_days": round(float(g["lead"].mean()), 1) if len(g) else None,
                "avg_lead_baseline": round(float(sku_drifts["baseline_lead"].mean()), 1)
                if len(sku_drifts)
                else None,
                "max_drift_pct": round(100 * float(sku_drifts["drift_pct"].max()), 1)
                if len(sku_drifts)
                else 0.0,
                "spend": round(float(spend.get(s.supplier_id, 0.0)), 2),
                "skus_served": int(
                    ctx.data.supply_terms[ctx.data.supply_terms["supplier_id"] == s.supplier_id][
                        "sku"
                    ].nunique()
                ),
            }
        )
    rows.sort(key=lambda r: -(r["spend"]))
    return {"suppliers": rows}


@app.get("/api/recommendations")
def recommendations():
    ctx = app.state.ctx
    out = []
    for rec in ctx.recommendations:
        expl = next((e for e in ctx.explanations if e["sku"] == rec.sku), {})
        out.append(
            {
                **rec.__dict__,
                "narrative": expl.get("narrative"),
                "components": expl.get("components", []),
            }
        )
    return {"snapshot_date": ctx.snapshot_date, "recommendations": out}


@app.get("/api/recommendations/{sku}/explain")
def explain_sku(sku: str):
    ctx = app.state.ctx
    expl = next((e for e in ctx.explanations if e["sku"] == sku), None)
    if expl is None:
        raise HTTPException(404, f"no explanation for {sku}")
    from supplychainxai.explain.engine import explain_forecast

    # SHAP-enriched forecast explanation generated by the training pipeline
    # (falls back to the on-the-fly permutation-based one for old artifacts)
    fc_expl = ctx.forecast_explanations.get(sku) or explain_forecast(
        sku, ctx.features, ctx.forecasts, ctx.importance
    )
    imp = []
    if ctx.importance is not None:
        imp = (
            ctx.importance[ctx.importance["sku"] == sku]
            .sort_values("importance", ascending=False)
            .head(6)
            .to_dict("records")
        )
    shap_imp = []
    if ctx.shap_importance is not None:
        shap_imp = (
            ctx.shap_importance[ctx.shap_importance["sku"] == sku]
            .sort_values("importance", ascending=False)
            .head(6)
            .to_dict("records")
        )
    return {
        "sku": sku,
        "explanation": expl,
        "forecast_explanation": fc_expl,
        "feature_importance": imp,
        "shap_importance": shap_imp,
    }


# ------------------------------------------------------------------ mlops
@app.get("/api/model-health")
def model_health():
    """Production-model health: latest observed performance vs reference,
    drift, interval coverage, data quality and the retraining recommendation."""
    ctx = app.state.ctx
    if not ctx.model_health:
        raise HTTPException(404, "model health not generated yet — run scripts/train.py")
    return {"snapshot_date": ctx.snapshot_date, **ctx.model_health}


@app.get("/api/business-kpis")
def business_kpis():
    """Business KPIs with an explicit honesty label per metric:
    observed (transactions) / estimated (model-dependent) / simulated."""
    ctx = app.state.ctx
    if not ctx.business_kpis:
        raise HTTPException(404, "business KPIs not generated yet — run scripts/train.py")
    return {"snapshot_date": ctx.snapshot_date, **ctx.business_kpis}


@app.get("/api/experiments")
def experiments():
    """Walk-forward evaluation summary: per-model portfolio performance,
    fold counts and stability flags — the model bake-off view."""
    ctx = app.state.ctx
    eval_dir = config.ARTIFACTS_EVALUATION
    portfolio_path = eval_dir / "walk_forward_portfolio.csv"
    if portfolio_path.exists():
        portfolio = pd.read_csv(portfolio_path).round(4)
    else:
        portfolio = pd.DataFrame()
    eval_cfg = _read_json_safe(eval_dir / "evaluation_config.json")
    per_sku = ctx.comparison  # walk-forward per-SKU means in legacy schema
    valid = per_sku.dropna(subset=["WAPE"]) if len(per_sku) else per_sku
    return {
        "evaluation": eval_cfg or {},
        "portfolio": portfolio.to_dict("records"),
        "wins": (valid.loc[valid.groupby("sku")["WAPE"].idxmin(), "model"].value_counts().to_dict())
        if len(valid)
        else {},
        "narrative": ctx.narrative,
        "registry": ctx.registry_summary,
    }


def _read_json_safe(path):
    from pathlib import Path as _P

    if _P(path).exists():
        try:
            return json.loads(_P(path).read_text())
        except Exception:
            return None
    return None


@app.get("/api/data-quality")
def data_quality_endpoint():
    ctx = app.state.ctx
    if not ctx.data_quality:
        raise HTTPException(404, "data-quality report not generated yet — run scripts/train.py")
    return {
        "snapshot_date": ctx.snapshot_date,
        "status": ctx.data_quality.get("status"),
        "generated_at": ctx.data_quality.get("generated_at"),
        "dataset_version": ctx.data_quality.get("dataset_version"),
        "summary": {
            "errors": len(ctx.data_quality.get("errors", [])),
            "warnings": len(ctx.data_quality.get("warnings", [])),
            "checks": len(ctx.data_quality.get("checks", [])),
        },
        "errors": ctx.data_quality.get("errors", []),
        "warnings": ctx.data_quality.get("warnings", []),
        "statistics": ctx.data_quality.get("statistics", {}),
        "dataset_manifest": ctx.dataset_manifest,
    }


# ------------------------------------------------------------------ simulator
class ScenarioParams(BaseModel):
    demand_pct: float = Field(0.0, ge=-0.6, le=1.5, description="demand change, e.g. 0.2 = +20%")
    lead_time_pct: float = Field(0.0, ge=-0.5, le=2.0, description="supplier lead-time change")
    inventory_pct: float = Field(0.0, ge=-0.9, le=1.0, description="on-hand inventory change")
    service_level: float = Field(0.95, ge=0.80, le=0.99)


@app.post("/api/simulate")
def simulate(params: ScenarioParams):
    ctx = app.state.ctx
    return ctxmod.run_what_if(
        ctx, params.demand_pct, params.lead_time_pct, params.inventory_pct, params.service_level
    )


# ------------------------------------------------------------------ copilot
class Question(BaseModel):
    question: str = Field(..., min_length=3, max_length=300)


@app.post("/api/copilot")
def copilot(q: Question):
    ctx = app.state.ctx
    return copilot_ask(q.question, ctx)


# ------------------------------------------------------------------ admin
@app.post("/api/admin/recompute")
def recompute():
    """Refresh hint: the pipeline is a CLI job, not an in-process request."""
    raise HTTPException(
        501,
        "Run `python scripts/run_pipeline.py` from the CLI, "
        "then restart the API (artifacts are read at startup).",
    )


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
