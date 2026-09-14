"""Experiment tracking — MLflow when available, JSON-lines fallback otherwise.

Business code talks only to `ExperimentTracker`; it never imports mlflow.
That keeps the pipeline runnable anywhere:

  * MLflow installed + MLFLOW_TRACKING_URI set  -> real MLflow runs
  * MLflow installed, no URI                    -> local ./mlruns file store
  * MLflow not installed / import fails         -> JSON backend under
    artifacts/experiments/ (same API: params, metrics, tags, artifacts)

Either way the run is queryable and the pipeline never crashes because a
tracking server is down — a tracking failure is recorded and swallowed.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from supplychainxai import config


@dataclass
class RunHandle:
    run_id: str
    backend: str
    _params: dict = field(default_factory=dict)
    _metrics: dict = field(default_factory=dict)
    _tags: dict = field(default_factory=dict)
    _status: str = "RUNNING"
    _mlflow_run: object | None = field(default=None, repr=False)


class ExperimentTracker:
    """Thin façade over MLflow or a JSON-lines file backend."""

    def __init__(self, tracking_uri: str = "", experiment: str = "supplychainxai"):
        self.experiment = experiment
        self.backend, self._client = self._resolve_backend(tracking_uri)
        self.enabled = True

    # ------------------------------------------------------------ backend
    def _resolve_backend(self, uri: str) -> tuple[str, object]:
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            # Default: a local sqlite backend (single file, registry-capable).
            # Plain file stores are in maintenance mode in recent MLflow and
            # would require MLFLOW_ALLOW_FILE_STORE=true.
            effective = uri or f"sqlite:///{config.PROJECT_ROOT / 'mlflow.db'}"
            mlflow.set_tracking_uri(effective)
            self._mclient = MlflowClient()  # run-scoped calls (log_metric, set_tag…)
            return "mlflow", mlflow  # fluent API (set_experiment/start_run/end_run)
        except Exception:  # mlflow missing or broken
            return "json", None

    @property
    def backend_name(self) -> str:
        return self.backend

    # ------------------------------------------------------------ runs
    def start_run(
        self, run_name: str, tags: dict | None = None, params: dict | None = None
    ) -> RunHandle:
        tags = {k: str(v) for k, v in (tags or {}).items()}
        params = {k: str(v) for k, v in (params or {}).items()}
        if self.backend == "mlflow":
            try:
                mlflow = self._client
                mlflow.set_experiment(self.experiment)
                run = mlflow.start_run(run_name=run_name, tags=tags)
                if params:
                    mlflow.log_params(params)
                return RunHandle(
                    run_id=run.info.run_id,
                    backend="mlflow",
                    _params=params,
                    _tags=tags,
                    _mlflow_run=run,
                )
            except Exception:
                self.backend = "json"  # degrade gracefully, mid-run
        run_id = f"{int(time.time() * 1000):x}-{run_name.replace(' ', '_')[:40]}"
        return RunHandle(run_id=run_id, backend="json", _params=params, _tags=tags)

    def log_params(self, run: RunHandle, params: dict) -> None:
        params = {k: str(v) for k, v in params.items()}
        run._params.update(params)
        if run.backend == "mlflow" and params:
            try:
                run_id = run._mlflow_run.info.run_id
                for key, value in params.items():
                    self._mclient.log_param(run_id, key, value)
            except Exception:
                pass

    def log_metrics(self, run: RunHandle, metrics: dict, step: int | None = None) -> None:
        clean = {
            k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and v == v
        }  # drop NaN
        run._metrics.update(clean)
        if run.backend == "mlflow" and clean:
            try:
                run_id = run._mlflow_run.info.run_id
                for key, value in clean.items():
                    self._mclient.log_metric(run_id, key, value, step=step)
            except Exception:
                pass

    def log_tags(self, run: RunHandle, tags: dict) -> None:
        clean = {k: str(v) for k, v in tags.items()}
        run._tags.update(clean)
        if run.backend == "mlflow" and clean:
            try:
                run_id = run._mlflow_run.info.run_id
                for k, v in clean.items():
                    self._mclient.set_tag(run_id, k, v)
            except Exception:
                pass

    def log_artifact(self, run: RunHandle, path: str | Path) -> None:
        path = Path(path)
        if run.backend == "mlflow":
            try:
                self._mclient.log_artifact(run._mlflow_run.info.run_id, str(path))
                return
            except Exception:
                pass
        # JSON backend: copy small artifacts next to the run record
        dest = config.ARTIFACTS / "experiments" / run.run_id
        dest.mkdir(parents=True, exist_ok=True)
        try:
            (dest / path.name).write_bytes(path.read_bytes())
        except Exception:
            pass

    def end_run(self, run: RunHandle, status: str = "FINISHED") -> None:
        run._status = status
        if run.backend == "mlflow":
            try:
                self._client.end_run(status=status)
                return
            except Exception:
                pass
        out_dir = config.ARTIFACTS / "experiments"
        out_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "run_id": run.run_id,
            "experiment": self.experiment,
            "status": status,
            "params": run._params,
            "metrics": run._metrics,
            "tags": run._tags,
        }
        with open(out_dir / "runs.jsonl", "a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    # ------------------------------------------------------------ convenience
    def tracked(self, run_name: str, tags: dict | None = None, params: dict | None = None):
        """Context manager: `with tracker.tracked("name", ...) as run:`"""
        return _TrackedRun(self, run_name, tags, params)


class _TrackedRun:
    def __init__(
        self, tracker: ExperimentTracker, name: str, tags: dict | None, params: dict | None
    ):
        self._tracker, self._name, self._tags, self._params = tracker, name, tags, params
        self.handle: RunHandle | None = None

    def __enter__(self) -> RunHandle:
        self.handle = self._tracker.start_run(self._name, self._tags, self._params)
        return self.handle

    def __exit__(self, exc_type, exc, tb) -> bool:
        status = "FAILED" if exc_type else "FINISHED"
        self._tracker.end_run(self.handle, status)
        return False


class NullTracker(ExperimentTracker):
    """Drop-in no-op tracker for tests / users who want zero tracking I/O."""

    def __init__(self):
        super().__init__()
        self.backend = "null"

    def start_run(self, run_name, tags=None, params=None) -> RunHandle:
        return RunHandle(run_id="null", backend="null")

    def log_params(self, run, params):
        pass

    def log_metrics(self, run, metrics, step=None):
        pass

    def log_tags(self, run, tags):
        pass

    def log_artifact(self, run, path):
        pass

    def end_run(self, run, status="FINISHED"):
        pass
