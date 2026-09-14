"""Structured run context & logging for every pipeline execution.

Each pipeline run gets a `RunContext`: a run_id, timestamp, git commit,
dataset version, environment label and a JSON-lines logger. The context is
attached to experiment runs, model versions and monitoring reports so any
artifact can be traced back to *what data, what code, what configuration*
produced it.

Design rules:
  * never log secrets (keys, tokens, passwords);
  * log events, not dumps — a handful of lines per stage;
  * if git is unavailable (Docker image, tarball), `git_commit` degrades to
    "unknown" instead of failing the run.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def git_commit(project_root: Path | None = None) -> str:
    """Short SHA of the current checkout, or 'unknown' outside a git repo."""
    root = project_root or Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


@dataclass
class RunContext:
    run_id: str
    started_at: str
    git_commit: str
    environment: str
    dataset_version: str = ""
    pipeline_version: str = ""
    events: list[dict] = field(default_factory=list)

    def log(self, stage: str, message: str, **fields) -> dict:
        """Record one structured event (also printed as a single JSON line)."""
        event = {
            "ts": utc_now(),
            "run_id": self.run_id,
            "stage": stage,
            "message": message,
            **fields,
        }
        self.events.append(event)
        print(json.dumps(event, default=str), flush=True)
        return event

    def to_dict(self) -> dict:
        return asdict(self)


def new_run_context(
    dataset_version: str = "", environment: str = "development", pipeline_version: str = ""
) -> RunContext:
    return RunContext(
        run_id=uuid.uuid4().hex[:12],
        started_at=utc_now(),
        git_commit=git_commit(),
        environment=environment,
        dataset_version=dataset_version,
        pipeline_version=pipeline_version,
    )
