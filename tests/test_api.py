"""API tests — boots FastAPI with TestClient against the committed artifacts."""
import pytest

fastapi = pytest.importorskip("fastapi")
TestClient = pytest.importorskip("fastapi.testclient").TestClient

from supplychainxai.api.app import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok"
    assert r["llm_mode"] in ("llm", "grounded-narrator")


def test_dashboard_served(client):
    html = client.get("/").text
    assert "SupplyChainXAI" in html and "What-If" in html


def test_kpis_shape(client):
    k = client.get("/api/kpis").json()
    assert {"inventory_value", "skus_tracked", "model_selection"} <= set(k)
    assert k["skus_tracked"] > 0


def test_forecast_unknown_sku_404(client):
    assert client.get("/api/forecast/NOPE").status_code == 404


def test_forecast_payload(client):
    f = client.get("/api/forecast/P104").json()
    assert f["sku"] == "P104"
    assert len(f["forecast"]) == 30
    assert all(row["upper_80"] >= row["prediction"] >= row["lower_80"] for row in f["forecast"])
    assert len(f["model_comparison"]) >= 4


def test_simulate_stress(client):
    r = client.post("/api/simulate", json={
        "demand_pct": 0.2, "lead_time_pct": 0.3,
        "inventory_pct": -0.15, "service_level": 0.95}).json()
    assert r["summary"]["procurement_units"] >= 0
    assert len(r["per_sku"]) >= 1


def test_copilot_grounded_refusal(client):
    r = client.post("/api/copilot", json={"question": "what is the stock price of Apple?"}).json()
    assert r["intent"] == "unknown"
    assert "stockout" in r["answer"]        # the honest refusal, not a hallucination


def test_copilot_stockout_question(client):
    r = client.post("/api/copilot",
                    json={"question": "Which products will stock out next month?"}).json()
    assert r["intent"] == "stockout"
    assert len(r["answer"]) > 20
    assert r["sources"]
