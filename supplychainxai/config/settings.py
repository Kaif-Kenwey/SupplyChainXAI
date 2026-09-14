"""Environment-driven runtime configuration.

`config.py` holds the *scenario* constants (paths, policy thresholds that are
part of the analytical design). This module holds the *operational* knobs an
ML engineer tunes between runs — evaluation strategy, monitoring thresholds,
retraining policy, infrastructure locations — all overridable via environment
variables with the `SCX_` prefix so the same code runs locally, in Docker and
in CI without edits.

Rules:
  * every setting has an explicit default (documented);
  * no secrets here — only non-sensitive configuration;
  * `get_settings()` is cached; tests can call it with `refresh=True` after
    changing the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, ""))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, ""))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class EvalConfig:
    """Walk-forward (rolling-origin) evaluation + model-selection policy.

    SCX_EVAL__HORIZON           forecast horizon per fold (days).   default 30
    SCX_EVAL__FOLDS             number of rolling-origin folds.      default 4
    SCX_EVAL__MIN_TRAIN_DAYS    shortest training window allowed.    default 365
    SCX_EVAL__STRIDE            days between fold origins; 0 =       default 0
                                evenly divide the evaluation span.
    SCX_EVAL__SELECTION_METRIC  WAPE | MAE | RMSE | composite.       default composite

    Composite weights (documented in README, deliberately not hardcoded):
    SCX_EVAL__W_WAPE 0.5  SCX_EVAL__W_MAE 0.3  SCX_EVAL__W_RMSE 0.2
    """

    horizon: int = 30
    folds: int = 4
    min_train_days: int = 365
    stride: int = 0  # 0 -> auto (evenly split eval span)
    selection_metric: str = "composite"
    w_wape: float = 0.50
    w_mae: float = 0.30
    w_rmse: float = 0.20
    random_seed: int = 42

    @classmethod
    def from_env(cls) -> EvalConfig:
        return cls(
            horizon=_env_int("SCX_EVAL__HORIZON", 30),
            folds=max(1, _env_int("SCX_EVAL__FOLDS", 4)),
            min_train_days=_env_int("SCX_EVAL__MIN_TRAIN_DAYS", 365),
            stride=_env_int("SCX_EVAL__STRIDE", 0),
            selection_metric=_env("SCX_EVAL__SELECTION_METRIC", "composite").lower(),
            w_wape=_env_float("SCX_EVAL__W_WAPE", 0.50),
            w_mae=_env_float("SCX_EVAL__W_MAE", 0.30),
            w_rmse=_env_float("SCX_EVAL__W_RMSE", 0.20),
            random_seed=_env_int("SCX_RANDOM_SEED", 42),
        )


@dataclass
class MonitoringConfig:
    """Model-monitoring thresholds (model health + drift + PI coverage).

    WAPE thresholds are in percent. PSI thresholds follow the common rule of
    thumb (<0.10 stable, 0.10–0.25 moderate shift, >0.25 major shift).
    Coverage is the observed fraction of actuals inside the 80% interval.
    """

    wape_warn: float = 20.0  # portfolio WAPE above this -> warning
    wape_crit: float = 28.0  # portfolio WAPE above this -> critical
    wape_degrade_pct: float = 25.0  # % worse than reference WAPE -> warning
    psi_warn: float = 0.10
    psi_crit: float = 0.25
    ks_alpha: float = 0.05
    coverage_target: float = 0.80  # nominal PI coverage
    coverage_min: float = 0.70  # below this -> calibration warning
    stale_data_days: int = 7  # data older than this -> stale warning

    @classmethod
    def from_env(cls) -> MonitoringConfig:
        return cls(
            wape_warn=_env_float("SCX_MON__WAPE_WARN", 20.0),
            wape_crit=_env_float("SCX_MON__WAPE_CRIT", 28.0),
            wape_degrade_pct=_env_float("SCX_MON__WAPE_DEGRADE_PCT", 25.0),
            psi_warn=_env_float("SCX_MON__PSI_WARN", 0.10),
            psi_crit=_env_float("SCX_MON__PSI_CRIT", 0.25),
            ks_alpha=_env_float("SCX_MON__KS_ALPHA", 0.05),
            coverage_target=_env_float("SCX_MON__COVERAGE_TARGET", 0.80),
            coverage_min=_env_float("SCX_MON__COVERAGE_MIN", 0.70),
            stale_data_days=_env_int("SCX_MON__STALE_DAYS", 7),
        )


@dataclass
class RetrainingConfig:
    """Automatic retraining policy + model-promotion gates.

    A candidate replaces production only if every gate passes:
      * candidate WAPE <= production WAPE * (1 + tolerance)
      * data quality has no ERROR findings (configurable)
      * artifacts generated successfully
    Retraining *triggers* (any one fires): WAPE above warn threshold, drift
    status warning+, PI coverage below minimum, stale data, scheduled
    interval reached.
    """

    wape_tolerance: float = 0.05  # new model may be at most 5% worse
    max_model_age_days: int = 90  # scheduled retrain interval
    new_data_rows_trigger: int = 0  # 0 = disabled
    require_no_dq_errors: bool = True

    @classmethod
    def from_env(cls) -> RetrainingConfig:
        return cls(
            wape_tolerance=_env_float("SCX_RETRAIN__WAPE_TOLERANCE", 0.05),
            max_model_age_days=_env_int("SCX_RETRAIN__MAX_AGE_DAYS", 90),
            new_data_rows_trigger=_env_int("SCX_RETRAIN__NEW_DATA_ROWS", 0),
            require_no_dq_errors=_env_bool("SCX_RETRAIN__REQUIRE_NO_DQ_ERRORS", True),
        )


@dataclass
class InfraConfig:
    """Infrastructure locations (database, MLflow, environment label)."""

    database_url: str = ""  # empty -> SQLite (config.DB_PATH)
    mlflow_tracking_uri: str = ""  # empty -> JSON fallback tracker
    mlflow_experiment: str = "supplychainxai"
    environment: str = "development"  # development | ci | production-like

    @classmethod
    def from_env(cls) -> InfraConfig:
        return cls(
            database_url=os.getenv("DATABASE_URL", ""),
            mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", ""),
            mlflow_experiment=os.getenv("SCX_MLFLOW_EXPERIMENT", "supplychainxai"),
            environment=os.getenv("SCX_ENVIRONMENT", "development"),
        )


@dataclass
class Settings:
    eval: EvalConfig = field(default_factory=EvalConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    retraining: RetrainingConfig = field(default_factory=RetrainingConfig)
    infra: InfraConfig = field(default_factory=InfraConfig)

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            eval=EvalConfig.from_env(),
            monitoring=MonitoringConfig.from_env(),
            retraining=RetrainingConfig.from_env(),
            infra=InfraConfig.from_env(),
        )


_settings: Settings | None = None


def get_settings(refresh: bool = False) -> Settings:
    """Process-wide settings singleton (env read once; `refresh=True` reloads)."""
    global _settings
    if _settings is None or refresh:
        _settings = Settings.from_env()
    return _settings
