"""
Unified MLflow configuration and logging for MNIST CNN/FCNN experiments.

Provides standardized parameter and metric logging across TensorFlow and PyTorch
to enable fair framework comparison and consistent experiment tracking.
"""

import time
from typing import Dict, Optional, Tuple
import mlflow


class MLflowConfig:
    """Standardized MLflow configuration for MNIST experiments."""

    # Required parameter names (11 total) - MUST be logged before training
    REQUIRED_PARAMS = [
        "learning_rate",
        "batch_size",
        "num_epochs",
        "optimizer",
        "loss_function",
        "num_conv_blocks",
        "conv_filters_initial",
        "dense_units",
        "dropout_rate",
        "random_seed",
        "framework",
    ]

    # Required per-epoch metrics
    PER_EPOCH_METRICS = [
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
    ]

    # Required final metrics (logged once after training)
    FINAL_METRICS = [
        "test_loss",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "training_time_seconds",
        "inference_time_ms_per_batch",
    ]

    def __init__(self, experiment_name: str, tracking_uri: str = "sqlite:///./mlflow.db"):
        """
        Initialize MLflow config and experiment.

        Args:
            experiment_name: Name for the MLflow experiment (e.g., "cnn_mnist_tensorflow")
            tracking_uri: MLflow tracking URI (default: local SQLite)
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.run_id = None
        self._start_time = None

        # Setup MLflow
        mlflow.set_tracking_uri(tracking_uri)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
        mlflow.set_experiment(experiment_name)
        self.experiment_id = experiment_id

    def start_run(self, tags: Optional[Dict[str, str]] = None) -> str:
        """
        Start a new MLflow run.

        Args:
            tags: Optional dict of tags for experiment filtering (e.g., framework, architecture)

        Returns:
            Run ID
        """
        mlflow.start_run(experiment_id=self.experiment_id)
        self._start_time = time.time()

        if tags:
            mlflow.set_tags(tags)

        self.run_id = mlflow.active_run().info.run_id
        return self.run_id

    def log_params(self, params: Dict) -> None:
        """
        Log training hyperparameters.

        Enforces parameter naming consistency. All param values are converted to strings
        and lowercased for consistency (except numeric values).

        Args:
            params: Dict of parameter names and values

        Raises:
            ValueError: If required parameters are missing
        """
        # Normalize parameter names: allow flexibility in input
        param_mapping = {
            "epochs": "num_epochs",
            "num_epochs": "num_epochs",
            "lr": "learning_rate",
            "learning_rate": "learning_rate",
            "batch_size": "batch_size",
            "optimizer": "optimizer",
            "loss": "loss_function",
            "loss_function": "loss_function",
            "num_conv_blocks": "num_conv_blocks",
            "conv_filters_initial": "conv_filters_initial",
            "conv_filters": "conv_filters_initial",
            "dense_units": "dense_units",
            "hidden_units": "dense_units",
            "dropout": "dropout_rate",
            "dropout_rate": "dropout_rate",
            "random_seed": "random_seed",
            "seed": "random_seed",
            "framework": "framework",
        }

        normalized_params = {}
        for key, value in params.items():
            normalized_key = param_mapping.get(key, key)
            # Convert to string and normalize case for consistency
            if isinstance(value, str):
                normalized_params[normalized_key] = value.lower()
            else:
                normalized_params[normalized_key] = str(value)

        # Validate all required parameters present
        missing = set(self.REQUIRED_PARAMS) - set(normalized_params.keys())
        if missing:
            raise ValueError(
                f"Missing required parameters: {missing}. "
                f"Required: {self.REQUIRED_PARAMS}"
            )

        # Log to MLflow (only required params)
        for param_name in self.REQUIRED_PARAMS:
            if param_name in normalized_params:
                mlflow.log_param(param_name, normalized_params[param_name])

    def log_epoch_metrics(
        self,
        epoch: int,
        train_loss: float,
        train_accuracy: float,
        val_loss: float,
        val_accuracy: float,
    ) -> None:
        """
        Log per-epoch training metrics.

        Args:
            epoch: Epoch number (0-indexed)
            train_loss: Training loss
            train_accuracy: Training accuracy [0, 1]
            val_loss: Validation loss
            val_accuracy: Validation accuracy [0, 1]
        """
        mlflow.log_metric("train_loss", float(train_loss), step=epoch)
        mlflow.log_metric("train_accuracy", float(train_accuracy), step=epoch)
        mlflow.log_metric("val_loss", float(val_loss), step=epoch)
        mlflow.log_metric("val_accuracy", float(val_accuracy), step=epoch)

    def log_final_metrics(
        self,
        test_loss: float,
        test_accuracy: float,
        test_precision: float,
        test_recall: float,
        test_f1: float,
        inference_time_ms_per_batch: float,
    ) -> None:
        """
        Log final test metrics (logged once after training completes).

        Args:
            test_loss: Test set loss
            test_accuracy: Test set accuracy [0, 1]
            test_precision: Test set precision (macro-averaged for multiclass)
            test_recall: Test set recall (macro-averaged for multiclass)
            test_f1: Test set F1-score (macro-averaged for multiclass)
            inference_time_ms_per_batch: Average inference time in milliseconds per batch
        """
        # Calculate training time
        if self._start_time is None:
            raise RuntimeError("Run not started. Call start_run() first.")

        training_time_seconds = time.time() - self._start_time

        # Log all final metrics
        mlflow.log_metric("test_loss", float(test_loss))
        mlflow.log_metric("test_accuracy", float(test_accuracy))
        mlflow.log_metric("test_precision", float(test_precision))
        mlflow.log_metric("test_recall", float(test_recall))
        mlflow.log_metric("test_f1", float(test_f1))
        mlflow.log_metric("training_time_seconds", float(training_time_seconds))
        mlflow.log_metric(
            "inference_time_ms_per_batch", float(inference_time_ms_per_batch)
        )

    def log_artifact(self, artifact_path: str, artifact_type: str = "results") -> None:
        """
        Log an artifact (file or directory) to MLflow.

        Args:
            artifact_path: Path to file or directory
            artifact_type: Type of artifact (e.g., 'results', 'models')
        """
        mlflow.log_artifacts(artifact_path, artifact_path=artifact_type)

    def end_run(self) -> None:
        """End the current MLflow run."""
        mlflow.end_run()

    @staticmethod
    def get_experiment_stats(experiment_name: str) -> Tuple[int, int]:
        """
        Get experiment statistics.

        Args:
            experiment_name: Name of experiment

        Returns:
            Tuple of (num_runs, num_with_adequate_params)
        """
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return 0, 0

        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        num_runs = len(runs)

        # Check how many runs have all required parameters
        adequate = 0
        for _, run in runs.iterrows():
            params = run.get("params", {})
            if all(p in params for p in MLflowConfig.REQUIRED_PARAMS):
                adequate += 1

        return num_runs, adequate
