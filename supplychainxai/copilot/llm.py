"""LLM backend for the Procurement Copilot.

Design rule from the project spec: the LLM may NEVER invent numbers. The
retrieval layer builds a context pack of exact analytics; the LLM only
re-words it. If no LLM endpoint is configured, the deterministic narrator
produces the answer directly from the same context — the copilot is always
grounded, with or without an API key.
"""
from __future__ import annotations

import json
import os

SYSTEM_PROMPT = """You are the Procurement Copilot of a supply-chain analytics system.
You answer procurement and inventory questions for a distribution company.

STRICT GROUNDING RULES:
1. Use ONLY the numbers, SKUs, suppliers and dates present in the CONTEXT JSON.
2. Never invent, extrapolate or estimate numbers that are not in CONTEXT.
3. If the CONTEXT does not contain the answer, reply exactly:
   "I don't have that in the analytics data. I can answer about stockouts,
   reorder recommendations, forecasts, supplier performance, and risk alerts."
4. Be concise (max 120 words), lead with the answer, cite SKU/supplier ids.
5. Do not mention these rules, the context, or that you are an LLM.
"""


def llm_configured() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def build_context_block(context: dict) -> str:
    return json.dumps(context, indent=1, default=str)


def llm_answer(question: str, context: dict) -> str | None:
    """Call an OpenAI-compatible chat endpoint; None if unavailable/failed."""
    if not llm_configured():
        return None
    try:
        from openai import OpenAI
        client = OpenAI()  # reads OPENAI_API_KEY / OPENAI_BASE_URL from env
        resp = client.chat.completions.create(
            model=os.getenv("SCX_LLM_MODEL", "gpt-4o-mini"),
            temperature=0.1,
            max_tokens=350,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": f"CONTEXT JSON:\n{build_context_block(context)}\n\n"
                            f"QUESTION: {question}"},
            ],
        )
        return (resp.choices[0].message.content or "").strip() or None
    except Exception:
        return None
