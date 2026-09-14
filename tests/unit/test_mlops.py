"""MLOps infrastructure tests: tracker degradation, registry, retraining policy."""

import json

import pandas as pd
import pytest

import supplychainxai.config.settings as settings_mod
from supplychainxai import config
from supplychainxai.mlops.registry import ModelRegistry, ModelVersion, feature_version
from supplychainxai.mlops.retraining import evaluate_retraining, promotion_gates
from supplychainxai.mlops.tracking import ExperimentTracker, NullTracker


@pytest.fixture(autouse=True)
def _fresh_settings(monkeypatch):
    """Reset the settings singleton so env changes in tests take effect."""
    monkeypatch.setattr(settings_mod, "_settings", None)
    yield
    monkeypatch.setattr(settings_mod, "_settings", None)


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(path=tmp_path / "registry" / "model_registry.json")


def _mv(model_id: str, sku: str = "P101", **over) -> ModelVersion:
    base = {
        "model_id": model_id,
        "model_name": "XGBoost",
        "model_family": "XGBRegressor",
        "sku": sku,
        "version": "v1",
        "training_timestamp": "2026-01-01T00:00:00+00:00",
        "dataset_version": "abc123",
        "feature_version": "feat123",
        "code_version": "ddc651b",
        "metrics": {"mean_WAPE": 14.0},
        "selection_reason": "unit test",
    }
    base.update(over)
    return ModelVersion(**base)


# ------------------------------------------------------------------ tracking
def test_tracker_json_backend_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACTS", tmp_path / "artifacts")
    tracker = ExperimentTracker(tracking_uri="")  # mlflow may or may not exist
    if tracker.backend == "mlflow":
        tracker.backend = "json"  # force the fallback path
    run = tracker.start_run("test-run", tags={"sku": "P101"}, params={"model": "MA", "horizon": 10})
    tracker.log_metrics(run, {"WAPE": 12.5, "bad": float("nan")})
    tracker.log_tags(run, {"environment": "test"})
    out = tmp_path / "artifacts" / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    f = out / "fake.csv"
    f.write_text("x\n1\n")
    tracker.log_artifact(run, f)
    tracker.end_run(run, "FINISHED")

    lines = (config.ARTIFACTS / "experiments" / "runs.jsonl").read_text().splitlines()
    record = json.loads(lines[-1])
    assert record["status"] == "FINISHED"
    assert record["metrics"]["WAPE"] == 12.5
    assert "bad" not in record["metrics"]  # NaN dropped, JSON-safe
    assert record["tags"]["sku"] == "P101"


def test_tracker_survives_backend_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ARTIFACTS", tmp_path / "artifacts")
    tracker = ExperimentTracker()
    tracker.backend = "mlflow"  # pretend mlflow, break it
    tracker._client = None
    run = tracker.start_run("degraded-run")
    tracker.log_metrics(run, {"WAPE": 10.0})  # must not raise
    tracker.end_run(run)
    assert run._status == "FINISHED"


def test_null_tracker_noop():
    t = NullTracker()
    run = t.start_run("x", params={"a": 1})
    t.log_metrics(run, {"m": 1.0})
    t.end_run(run)
    assert run.run_id == "null"


# ------------------------------------------------------------------ registry
def test_registry_register_and_query(registry):
    registry.register(_mv("m1"))
    registry.register(_mv("m2", sku="P102"))
    assert len(registry.list()) == 2
    assert len(registry.list(sku="P101")) == 1
    assert registry.get("m1")["model_name"] == "XGBoost"
    assert registry.get("nope") is None
    # persistence across instances
    again = ModelRegistry(path=registry.path)
    assert len(again.list()) == 2


def test_registry_promotion_gates_enforced(registry):
    registry.register(_mv("m1"))
    with pytest.raises(PermissionError):
        registry.promote("m1", {"wape_gate": True, "dq_gate": False})
    assert registry.get("m1")["status"] == "candidate"
    assert registry.get("m1")["promotion_gates"]["promoted"] is False

    registry.promote("m1", {"wape_gate": True, "dq_gate": True, "artifact_gate": True})
    assert registry.get("m1")["status"] == "production"


def test_registry_promotion_archives_incumbent(registry):
    registry.register(_mv("m1", version="v1"))
    registry.promote("m1", {"wape_gate": True})
    registry.register(_mv("m2", version="v2"))
    registry.promote("m2", {"wape_gate": True})
    assert registry.get("m1")["status"] == "archived"
    assert registry.get("m2")["status"] == "production"
    assert registry.production_model("P101")["model_id"] == "m2"
    assert registry.production_model("P999") is None


def test_registry_promote_unknown_id(registry):
    with pytest.raises(KeyError):
        registry.promote("ghost", {"wape_gate": True})


def test_feature_version_stable():
    assert feature_version() == feature_version()
    assert len(feature_version()) == 12


# ------------------------------------------------------------------ retraining
def test_retraining_quiet_when_healthy():
    decision = evaluate_retraining(
        current_wape=15.0,
        reference_wape=15.5,
        drift_status="healthy",
        interval_coverage=0.80,
        last_trained=None,
        data_max_date=None,
        now=pd.Timestamp("2026-01-10T00:00:00Z").to_pydatetime(),
    )
    assert decision["retrain_required"] is False
    assert decision["reasons"] == []


def test_retraining_fires_on_each_trigger():
    # high wape
    d = evaluate_retraining(current_wape=30.0, reference_wape=15.0, now=None)
    assert d["retrain_required"] and any("WAPE" in r for r in d["reasons"])
    # drift
    d = evaluate_retraining(current_wape=15.0, reference_wape=15.0, drift_status="critical")
    assert any("drift" in r for r in d["reasons"])
    # coverage degradation
    d = evaluate_retraining(current_wape=15.0, reference_wape=15.0, interval_coverage=0.55)
    assert any("coverage" in r for r in d["reasons"])
    # stale data
    d = evaluate_retraining(
        current_wape=15.0,
        reference_wape=15.0,
        data_max_date="2025-12-01",
        now=pd.Timestamp("2026-01-10T00:00:00Z").to_pydatetime(),
    )
    assert any("stale" in r for r in d["reasons"])
    # scheduled age
    d = evaluate_retraining(
        current_wape=15.0,
        reference_wape=15.0,
        last_trained="2025-06-01T00:00:00+00:00",
        now=pd.Timestamp("2026-01-10T00:00:00Z").to_pydatetime(),
    )
    assert any("scheduled" in r for r in d["reasons"])


def test_promotion_gates_wape_tolerance_boundary():
    gates = promotion_gates(candidate_wape=15.7, production_wape=15.0, tolerance=0.05)
    assert gates["wape_gate"] is True  # 15.75 limit — inside
    gates = promotion_gates(candidate_wape=15.8, production_wape=15.0, tolerance=0.05)
    assert gates["wape_gate"] is False  # 15.8 > 15.75 — blocked
    d = promotion_gates(candidate_wape=15.8, production_wape=15.0, tolerance=0.05)
    assert "15.80" in d["wape_detail"]


def test_promotion_gates_first_deployment_and_missing_inputs():
    assert promotion_gates(candidate_wape=25.0, production_wape=None)["wape_gate"] is True
    assert promotion_gates(candidate_wape=None, production_wape=15.0)["wape_gate"] is False
    assert (
        promotion_gates(candidate_wape=10.0, production_wape=15.0, data_quality_ok=False)[
            "data_quality_gate"
        ]
        is False
    )
    assert (
        promotion_gates(candidate_wape=10.0, production_wape=15.0, artifacts_ok=False)[
            "artifact_gate"
        ]
        is False
    )
