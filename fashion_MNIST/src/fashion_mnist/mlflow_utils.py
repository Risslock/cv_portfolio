"""MLflow experiment setup and epoch-level logging helpers."""

from __future__ import annotations

import re
import sys
from typing import Any

import mlflow
import tensorflow as tf

EXPERIMENT_NAME = "fashion_mnist_cnn"
# MLflow bakes each experiment's artifact_location as an absolute, OS-specific
# path at creation time (confirmed: even a relative path gets resolved
# immediately). Windows and Linux absolute paths aren't mutually interpretable,
# so a run from one environment against an experiment created by the other
# silently sends its model/plot artifacts to a bogus location. Rather than
# document "don't mix environments" as a footgun, each environment gets its own
# store: the Linux container (where real GPU training happens) uses mlflow.db;
# native Windows (CPU sanity checks) uses a separate file, so a collision is
# structurally impossible rather than just discouraged.
TRACKING_URI = "sqlite:///mlflow.db" if sys.platform == "linux" else "sqlite:///mlflow-native.db"


def setup_experiment(
    experiment_name: str = EXPERIMENT_NAME, tracking_uri: str = TRACKING_URI
) -> None:
    """Point MLflow at the local SQLite backend and select the experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def sanitize_run_name(run_name: str) -> str:
    """Make an MLflow run name (human-supplied or auto-generated) filesystem-safe."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", run_name).strip("-") or "run"


class MlflowEpochLogger(tf.keras.callbacks.Callback):
    """Logs standardized per-epoch metrics to the active MLflow run during ``.fit()``."""

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        logs = logs or {}
        metrics = {
            "train_loss": logs.get("loss"),
            "train_accuracy": logs.get("accuracy"),
            "val_loss": logs.get("val_loss"),
            "val_accuracy": logs.get("val_accuracy"),
        }
        mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None}, step=epoch)
