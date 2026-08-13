"""MLflow configuration built on Ultralytics' native MLflow integration.

Not hand-rolled logging — see docs/adr/0003-native-mlflow-integration.md for why.

Env vars the native integration reads (set by `configure_ultralytics_mlflow` below):
    MLFLOW_TRACKING_URI: the project's SQLite store.
    MLFLOW_EXPERIMENT_NAME: `poultry_detection` / `poultry_segmentation`.
    MLFLOW_KEEP_RUN_ACTIVE: kept True so `finish_run` can add project-specific params/
        tags (not covered by Ultralytics' auto-logged `trainer.args`) before closing.

Reference: https://docs.ultralytics.com/integrations/mlflow/
"""

import os

import mlflow
from ultralytics import settings as ultralytics_settings

TRACKING_URI = "sqlite:///mlflow.db"

DETECTION_EXPERIMENT = "poultry_detection"
SEGMENTATION_EXPERIMENT = "poultry_segmentation"


def configure_ultralytics_mlflow(experiment_name: str, tracking_uri: str = TRACKING_URI) -> None:
    """Point Ultralytics' built-in MLflow callback at this project's store/experiment.

    Call once before any `model.train()`/`model.tune()` call. Leaves the run open after
    training (`MLFLOW_KEEP_RUN_ACTIVE=True`) so a single final training run (not a tuning
    trial — see module docstring) can append extra params/tags via `finish_run` before
    it's closed.

    Args:
        experiment_name: `DETECTION_EXPERIMENT`, `SEGMENTATION_EXPERIMENT`, or another
            MLflow experiment name.
        tracking_uri: MLflow tracking URI. Defaults to the project's local SQLite store.
    """
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
    os.environ["MLFLOW_KEEP_RUN_ACTIVE"] = "True"
    ultralytics_settings.update({"mlflow": True})


def make_run_name(model_family: str, variant: str) -> str:
    """Rename the active MLflow run to `{model_family}-{variant}-{run_id[:8]}`.

    Must be called inside an active run (i.e. after a `model.train()`/`model.tune()`
    call made with `configure_ultralytics_mlflow` in effect). The `run_id` suffix keeps
    the name globally unique even across repeated `model_family`/`variant` combinations,
    while still being searchable/readable by model at a glance — see `CLAUDE.md` §
    MLflow Conventions.

    Args:
        model_family: e.g. `"yolo26"`.
        variant: e.g. `"n-tuned"`, `"s-baseline"`.

    Returns:
        The new run name that was set as the `mlflow.runName` tag.

    Raises:
        RuntimeError: If called with no active MLflow run.
    """
    run = mlflow.active_run()
    if run is None:
        raise RuntimeError("make_run_name() requires an active MLflow run.")
    new_name = f"{model_family}-{variant}-{run.info.run_id[:8]}"
    mlflow.set_tag("mlflow.runName", new_name)
    return new_name


def finish_run(
    extra_params: dict | None = None,
    extra_tags: dict | None = None,
    extra_metrics: dict | None = None,
) -> None:
    """Log what Ultralytics' auto-logging doesn't cover, then close the run.

    Ultralytics' MLflow callback logs every YOLO training argument (`epochs`, `batch`,
    `imgsz`, `lr0`, `seed`, ...) and per-epoch metrics automatically, but under its own
    key names (parens stripped from e.g. `metrics/mAP50(B)`) — not this project's
    `box_map50`/`box_map50_95` convention (`CLAUDE.md` § MLflow Conventions). Use
    `extra_metrics` for the final, canonically-named numbers (typically from a
    re-validation of the best checkpoint — see `detection/yolo.py`'s `train`), and
    `extra_params`/`extra_tags` for project-level fields that aren't YOLO training
    arguments themselves, e.g. `model_family`.

    Args:
        extra_params: Additional params to log (`mlflow.log_params`).
        extra_tags: Additional tags to set (`mlflow.set_tags`).
        extra_metrics: Additional final metrics to log (`mlflow.log_metrics`).

    Raises:
        RuntimeError: If called with no active MLflow run.
    """
    if mlflow.active_run() is None:
        raise RuntimeError("finish_run() requires an active MLflow run.")
    if extra_params:
        mlflow.log_params(extra_params)
    if extra_tags:
        mlflow.set_tags(extra_tags)
    if extra_metrics:
        mlflow.log_metrics(extra_metrics)
    mlflow.end_run()
