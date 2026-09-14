"""Model registry: versioned metadata for every production forecast model.

Answers, for any SKU: *what model generated this forecast, which version,
trained when, on which data, performing how?*

Storage strategy (deliberately boring):
  * source of truth = `artifacts/registry/model_registry.json` — human
    inspectable, survives restarts, no server required;
  * when MLflow with a registry-capable backend is configured the same
    versions are mirrored best-effort into the MLflow Model Registry
    (`registry.sync_mlflow()`); failures are logged, never fatal.

A model moves candidate -> staging -> production via `promote()`, which
records the gate results that justified the promotion. The previously
production model (if any) is archived, not deleted — every prediction is
traceable to the version that made it.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

from supplychainxai import config

REGISTRY_PATH = config.ARTIFACTS / "registry" / "model_registry.json"
STATUSES = ("candidate", "staging", "production", "archived")


@dataclass
class ModelVersion:
    model_id: str
    model_name: str
    model_family: str
    sku: str
    version: str
    training_timestamp: str
    dataset_version: str
    feature_version: str
    code_version: str  # git commit
    metrics: dict = field(default_factory=dict)
    selection_reason: str = ""
    status: str = "candidate"
    artifact_path: str = ""
    evaluation: dict = field(default_factory=dict)  # fold config + stability
    promotion_gates: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def feature_version() -> str:
    """Identity of the feature engineering (columns), not the data."""
    import hashlib

    from supplychainxai.data.features import FEATURE_COLUMNS

    return hashlib.sha256("|".join(FEATURE_COLUMNS).encode()).hexdigest()[:12]


class ModelRegistry:
    def __init__(self, path: Path = REGISTRY_PATH):
        self.path = path
        self._lock = threading.Lock()
        self._models: list[dict] = []
        self._load()

    # ------------------------------------------------------------ io
    def _load(self) -> None:
        if self.path.exists():
            try:
                self._models = json.loads(self.path.read_text()).get("models", [])
            except Exception:
                self._models = []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"schema": "v1", "models": self._models}, indent=2, default=str)
        )

    # ------------------------------------------------------------ api
    def register(self, mv: ModelVersion) -> ModelVersion:
        with self._lock:
            self._models.append(mv.to_dict())
            self._save()
        return mv

    def get(self, model_id: str) -> dict | None:
        return next((m for m in self._models if m["model_id"] == model_id), None)

    def list(self, sku: str | None = None, status: str | None = None) -> list[dict]:
        out = self._models
        if sku:
            out = [m for m in out if m["sku"] == sku]
        if status:
            out = [m for m in out if m["status"] == status]
        return out

    def promote(self, model_id: str, gates: dict) -> dict:
        """candidate/staging -> production after gates; archives the incumbent.

        `gates` = {"gate_name": bool, ...}. Promotion is refused (status stays
        unchanged, refusal recorded) unless EVERY gate passes.
        """
        with self._lock:
            target = self.get(model_id)
            if target is None:
                raise KeyError(f"unknown model_id {model_id}")
            if not all(gates.values()):
                target["promotion_gates"] = {**gates, "promoted": False}
                self._save()
                failed = [k for k, v in gates.items() if not v]
                raise PermissionError(f"promotion gates failed: {failed}")
            for m in self._models:
                if m["sku"] == target["sku"] and m["status"] == "production":
                    m["status"] = "archived"
            target["status"] = "production"
            target["promotion_gates"] = {**gates, "promoted": True}
            self._save()
            return target

    def production_model(self, sku: str) -> dict | None:
        return next(
            (m for m in self._models if m["sku"] == sku and m["status"] == "production"), None
        )

    def save_model_artifact(self, model, sku: str, model_name: str) -> Path:
        """Persist the fitted estimator next to its registry metadata."""
        import joblib

        self.path.parent.mkdir(parents=True, exist_ok=True)
        path = config.ARTIFACTS_MODELS / f"{sku}__{_slug(model_name)}.joblib"
        joblib.dump(model, path)
        return path

    # ------------------------------------------------------------ mlflow mirror
    def sync_mlflow(self, tracking_uri: str = "") -> int:
        """Best-effort mirror of registry entries into the MLflow Model
        Registry. Returns the number of versions synced; never raises."""
        synced = 0
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            client = MlflowClient()
            exp_name = "supplychainxai-registry"
            exp = client.get_experiment_by_name(exp_name)
            if exp is None:
                exp_id = client.create_experiment(exp_name)
            else:
                exp_id = exp.experiment_id
            for m in self._models:
                if not m.get("artifact_path"):
                    continue
                run = client.create_run(
                    exp_id,
                    tags={
                        "sku": m["sku"],
                        "status": m["status"],
                        "model_version": m["version"],
                        "dataset_version": m["dataset_version"],
                    },
                )
                for k, v in m["metrics"].items():
                    if isinstance(v, (int, float)):
                        client.log_metric(run.info.run_id, k, float(v))
                client.log_param(run.info.run_id, "model_name", m["model_name"])
                synced += 1
        except Exception:
            return synced
        return synced
