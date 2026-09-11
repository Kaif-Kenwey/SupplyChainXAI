"""Phase 7 — Grounded GenAI Procurement Copilot.

Pipeline (RAG-style, but the "vector store" is the analytics warehouse itself):
    question -> intent detection -> SQL / artifact retrieval -> context pack
             -> (optional) LLM re-wording  |  deterministic narrator fallback
             -> answer + the sources used

Every answer cites the queries/tables it came from. Unknown intents are
refused rather than hallucinated — enforced by design, not by prompt hope.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from supplychainxai.copilot import llm
from supplychainxai.data import store
from supplychainxai.risk.engine import Alert


@dataclass
class CopilotResult:
    question: str
    intent: str
    answer: str
    sources: list[str] = field(default_factory=list)
    context: dict = field(default_factory=dict)
    engine: str = "narrator"        # narrator | llm


SKU_RE = re.compile(r"\bP\s?-?(\d{3})\b", re.IGNORECASE)


def detect_sku(question: str) -> str | None:
    m = SKU_RE.search(question)
    return f"P{m.group(1)}" if m else None


# ---------------------------------------------------------------- retrieval
def retrieve(intent: str, sku: str | None, ctx) -> tuple[dict, list[str]]:
    """Pull exact numbers for the intent from SQL + in-memory artifacts."""
    context: dict = {}
    sources: list[str] = []

    if sku:
        row = store.query(
            """SELECT p.sku, p.name, p.category, p.unit_cost,
                      i.on_hand, i.on_order, i.date AS snapshot_date
                 FROM dim_product p
                 JOIN fact_inventory i ON i.sku = p.sku
                      AND i.date = (SELECT MAX(date) FROM fact_inventory)
                WHERE p.sku = ?""", (sku,))
        if len(row):
            context["product"] = row.to_dict("records")[0]
            sources.append("SQL: dim_product × fact_inventory (latest snapshot)")

    if intent in ("stockout", "risk"):
        alerts = [a for a in ctx.risk_alerts
                  if a.type in ("stockout_risk", "demand_spike")
                  and (sku is None or a.sku == sku)]
        context["stockout_risk_alerts"] = [
            {"sku": a.sku, "severity": a.severity, "message": a.message,
             **a.data} for a in alerts[:8]]
        sources.append("artifacts: risk engine (stockout_risk / demand_spike alerts)")

    if intent in ("reorder", "explain"):
        recs = [r for r in ctx.recommendations if r.action in ("BUY", "EXPEDITE")
                and (sku is None or r.sku == sku)]
        context["recommendations"] = [{
            "sku": r.sku, "action": r.action, "quantity": r.quantity,
            "supplier_id": r.supplier_id, "unit_price": r.unit_price,
            "total_cost": r.total_cost, "order_by": r.order_by,
            "expected_stockout_date": r.expected_stockout_date,
        } for r in recs[:8]]
        sources.append("artifacts: procurement recommendation engine")
        if sku:
            expl = next((e for e in ctx.explanations if e["sku"] == sku), None)
            if expl:
                context["explanation"] = {
                    "components": expl["components"], "narrative": expl["narrative"]}
                sources.append("artifacts: explainability layer")

    if intent == "forecast":
        fc = ctx.forecasts
        if sku:
            fc = fc[fc["sku"] == sku]
        if len(fc):
            agg = (fc.groupby("sku")
                     .agg(mean_daily=("prediction", "mean"),
                          day1=("date", "min"), day30=("date", "max"),
                          model=("model", "first")).reset_index())
            context["forecast"] = agg.to_dict("records")
            sources.append("artifacts: forecasts.csv (winning model, 30d)")

    if intent == "supplier":
        q = """
            SELECT s.supplier_id, s.name, s.country,
                   COUNT(po.po_id) AS pos_total,
                   CASE WHEN COUNT(po.po_id) = 0 THEN NULL
                        ELSE ROUND(100.0 * SUM(CASE WHEN po.delivered_date <= po.expected_date
                               THEN 1 ELSE 0 END) / COUNT(po.po_id), 1) END AS on_time_pct,
                   ROUND(AVG(julianday(po.delivered_date) - julianday(po.order_date)), 1) AS avg_lead_days
              FROM dim_supplier s
              LEFT JOIN fact_purchase_orders po ON po.supplier_id = s.supplier_id
                   AND po.status = 'DELIVERED'
             GROUP BY s.supplier_id ORDER BY pos_total DESC"""
        context["suppliers"] = store.query(q).to_dict("records")
        sources.append("SQL: dim_supplier × fact_purchase_orders (OTD %, avg lead)")
        drift = [a for a in ctx.risk_alerts if a.type in ("supplier_lead_drift",
                                                          "supplier_reliability")]
        if drift:
            context["supplier_alerts"] = [{"supplier_id": a.supplier_id, "sku": a.sku,
                                           "message": a.message} for a in drift[:6]]
            sources.append("artifacts: risk engine (supplier alerts)")

    if intent == "spend":
        q = """
            SELECT p.category,
                   ROUND(SUM(po.quantity * po.unit_price), 0) AS spend,
                   COUNT(po.po_id) AS pos
              FROM fact_purchase_orders po JOIN dim_product p ON p.sku = po.sku
             GROUP BY p.category ORDER BY spend DESC"""
        context["spend_by_category"] = store.query(q).to_dict("records")
        sources.append("SQL: fact_purchase_orders × dim_product (spend by category)")

    return context, sources


# ---------------------------------------------------------------- narrator
def _fmt(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 10 else f"{x:,.1f}"


def narrator_answer(question: str, intent: str, sku: str | None,
                    context: dict, ctx) -> str:
    if not context:
        return ("I don't have that in the analytics data. I can answer about stockouts, "
                "reorder recommendations, forecasts, supplier performance, and risk alerts.")

    if intent in ("stockout", "risk"):
        alerts = context.get("stockout_risk_alerts", [])
        if not alerts:
            return "No stockout or demand-spike alerts are currently active in the risk engine."
        crit = [a for a in alerts if a["severity"] == "CRITICAL"]
        lines = [f"{len(alerts)} active alert(s){' — ' + str(len(crit)) + ' critical' if crit else ''}. "
                 f"Most urgent: {alerts[0]['message']}"]
        for a in alerts[1:4]:
            lines.append(f"- {a['message']}")
        return " ".join(lines)

    if intent in ("reorder", "explain"):
        recs = context.get("recommendations", [])
        if recs:
            if sku:
                r = recs[0]
                expl = context.get("explanation", {})
                head = (f"{r['action']} {_fmt(r['quantity'])} units of {r['sku']} from "
                        f"{r['supplier_id']} at {r['unit_price']:.2f} "
                        f"(total {r['total_cost']:,.0f}). Order by {r['order_by']}; "
                        f"stockout projected {r['expected_stockout_date']}.")
                if expl.get("narrative"):
                    return f"{head} Why: {expl['narrative']}"
                return head
            total = sum(r["total_cost"] for r in recs)
            units = sum(r["quantity"] for r in recs)
            return (f"{len(recs)} SKUs need replenishment: {_fmt(units)} units, "
                    f"{total:,.0f} total cost. Top actions: "
                    + "; ".join(f"{r['sku']} → buy {_fmt(r['quantity'])} from {r['supplier_id']}"
                                for r in recs[:3]) + ".")
        if sku:  # SKU asked explicitly but it is a HOLD
            p = context.get("product", {})
            alerts = [a for a in ctx.risk_alerts if a.sku == sku]
            extra = f" Watch: {alerts[0].message}" if alerts else ""
            return (f"{sku} has no open purchase action — its inventory position "
                    f"({_fmt(p.get('on_hand', 0))} on hand + {_fmt(p.get('on_order', 0))} on order) "
                    f"covers the protection window.{extra}")
        return "No purchase actions are required right now — all SKUs are above their reorder points."

    if intent == "forecast":
        rows = context.get("forecast", [])
        if not rows:
            return "I don't have a forecast for that SKU."
        if sku:
            r = rows[0]
            d1, d30 = str(r["day1"])[:10], str(r["day30"])[:10]
            return (f"{sku} is forecast at ~{_fmt(r['mean_daily'])} units/day over the "
                    f"next 30 days ({d1} to {d30}), model: {r['model']}.")
        top = sorted(rows, key=lambda r: -r["mean_daily"])[:5]
        return ("Highest forecast demand: "
                + ", ".join(f"{r['sku']} ~{_fmt(r['mean_daily'])}/day" for r in top) + ".")

    if intent == "supplier":
        sups = context.get("suppliers", [])
        alerts = context.get("supplier_alerts", [])
        if sups:
            best = max((s for s in sups if s["on_time_pct"] is not None),
                       key=lambda s: s["on_time_pct"], default=None)
            worst = min((s for s in sups if s["on_time_pct"] is not None),
                        key=lambda s: s["on_time_pct"], default=None)
            lines = []
            if best:
                lines.append(f"Best on-time delivery: {best['name']} ({best['supplier_id']}) "
                             f"at {best['on_time_pct']}% across {best['pos_total']} POs.")
            if worst:
                lines.append(f"Lowest: {worst['name']} at {worst['on_time_pct']}%.")
            for a in alerts[:2]:
                lines.append(a["message"])
            return " ".join(lines)
        return "I don't have supplier delivery data in the analytics store."

    if intent == "spend":
        rows = context.get("spend_by_category", [])
        if not rows:
            return "I don't have PO spend data yet."
        total = sum(r["spend"] for r in rows)
        return (f"Total PO spend {_fmt(total)} across {len(rows)} categories. Top: "
                + ", ".join(f"{r['category']} {_fmt(r['spend'])}" for r in rows[:3]) + ".")

    return ("I can answer about stockouts, reorder recommendations, forecasts, "
            "supplier performance, and risk alerts.")


# ---------------------------------------------------------------- intents
def detect_intent(question: str) -> str:
    q = question.lower()
    rules = [
        ("reorder", r"(buy|order|reorder|replenish|purchase|procure|how many|restock)"),
        ("stockout", r"(stock\s?out|run out|runnable|breach|safety stock|cover|days left)"),
        ("risk", r"(risk|alert|anomaly|wrong|problem|danger)"),
        ("supplier", r"(supplier|vendor|lead time|reliab|on.time|delivery|otd)"),
        ("forecast", r"(forecast|predict|demand|outlook|next (30|month|week)|expect)"),
        ("explain", r"(why|explain|reason|how did|justify|driver)"),
        ("spend", r"(spend|cost|budget|expense|mone[yi])"),
    ]
    for intent, pattern in rules:
        if re.search(pattern, q):
            return intent
    return "unknown"


# ---------------------------------------------------------------- entrypoint
def ask(question: str, ctx) -> dict:
    """ctx: pipeline context object carrying artifacts + risk alerts."""
    sku = detect_sku(question)
    intent = detect_intent(question)
    if intent == "unknown" and sku:
        intent = "reorder"
    if intent == "unknown":
        return CopilotResult(question=question, intent="unknown",
                             answer=("I can answer about stockouts, reorder "
                                     "recommendations, forecasts, supplier performance, "
                                     "and risk alerts. Try: 'Which products will stock "
                                     "out next month?'")).__dict__

    context, sources = retrieve(intent, sku, ctx)
    answer = llm.llm_answer(question, context)
    engine = "llm"
    if answer is None:
        answer = narrator_answer(question, intent, sku, context, ctx)
        engine = "narrator"

    result = CopilotResult(question=question, intent=intent, answer=answer,
                           sources=sources, context=context, engine=engine)
    return result.__dict__
