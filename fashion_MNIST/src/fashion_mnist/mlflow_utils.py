"""MLflow experiment setup and epoch-level logging helpers."""

from __future__ import annotations

from typing import Any

import mlflow
import tensorflow as tf

EXPERIMENT_NAME = "fashion_mnist_cnn"
TRACKING_URI = "sqlite:///mlflow.db"


def setup_experiment(
    experiment_name: str = EXPERIMENT_NAME, tracking_uri: str = TRACKING_URI
) -> None:
    """Point MLflow at the local SQLite backend and select the experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


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
