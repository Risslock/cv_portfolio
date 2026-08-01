"""TensorFlow CNN training script with MLflow experiment tracking for MNIST.

Trains the CNN model on MNIST with comprehensive MLflow logging for metrics,
parameters, and model artifacts. Enables reproducible, comparable experiments.
"""

import argparse
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
from tensorflow.keras.callbacks import Callback

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tensorflow.cnn.data import load_mnist
from tensorflow.cnn.evaluate import compute_metrics
from tensorflow.cnn.models import MNISTCNNModel
from utils.mlflow_config import MLflowConfig


class MLflowLoggingCallback(Callback):
    """Keras callback to log per-epoch metrics to MLflow."""

    def on_epoch_end(self, epoch, logs=None):
        """Log training metrics at end of each epoch."""
        if logs:
            mlflow.log_metric("train_loss", float(logs.get("loss", 0)), step=epoch)
            mlflow.log_metric("train_accuracy", float(logs.get("accuracy", 0)), step=epoch)
            mlflow.log_metric("val_loss", float(logs.get("val_loss", 0)), step=epoch)
            mlflow.log_metric("val_accuracy", float(logs.get("val_accuracy", 0)), step=epoch)


def train_cnn(
    epochs: int = 60,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    num_conv_blocks: int = 2,
    conv_filters_initial: int = 32,
    dense_units: int = 128,
    dropout_rate: float = 0.5,
    random_seed: int = 42,
):
    """
    Train CNN model on MNIST with MLflow experiment tracking.

    Args:
        epochs: Number of training epochs. Default: 60.
        batch_size: Batch size for training. Default: 32.
        learning_rate: Optimizer learning rate. Default: 0.001.
        num_conv_blocks: Number of convolutional blocks. Default: 2.
        conv_filters_initial: Initial filter count. Default: 32.
        dense_units: Hidden units in dense layer. Default: 128.
        dropout_rate: Dropout probability. Default: 0.5.
        random_seed: Random seed for reproducibility. Default: 42.
    """
    # Initialize MLflow experiment
    mlflow_config = MLflowConfig(experiment_name="cnn_mnist_tensorflow")
    mlflow_config.start_run()

    try:
        # Log hyperparameters
        mlflow_config.log_params(
            {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "num_epochs": epochs,
                "optimizer": "adam",
                "loss_function": "categorical_crossentropy",
                "num_conv_blocks": num_conv_blocks,
                "conv_filters_initial": conv_filters_initial,
                "dense_units": dense_units,
                "dropout_rate": dropout_rate,
                "random_seed": random_seed,
                "framework": "tensorflow",
            }
        )

        # Load data
        print("Loading MNIST dataset...")
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist(
            normalize=True
        )
        print(
            f"  Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}"
        )

        # Create model
        print("Building CNN model...")
        model = MNISTCNNModel(
            input_shape=(28, 28, 1),
            num_classes=10,
            num_conv_blocks=num_conv_blocks,
            conv_filters_initial=conv_filters_initial,
            dense_units=dense_units,
            dropout_rate=dropout_rate,
            seed=random_seed,
        )
        model.summary()

        # Train model
        print(f"Training for {epochs} epochs...")
        start_time = time.time()

        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[MLflowLoggingCallback()],
        )

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        # Evaluate on test set
        print("Evaluating on test set...")
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

        # Get predictions for detailed metrics
        y_pred_probs = model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_test_labels = np.argmax(y_test, axis=1)

        # Compute detailed metrics
        _, precision, recall, f1 = compute_metrics(y_test_labels, y_pred)

        # Calculate inference time (ms per batch)
        batch_test_data = x_test[:batch_size]
        start_inference = time.time()
        for _ in range(10):  # Average over 10 batches
            _ = model.predict(batch_test_data, verbose=0)
        avg_inference_time_ms = ((time.time() - start_inference) / 10) * 1000

        # Log final metrics
        mlflow_config.log_final_metrics(
            test_loss=test_loss,
            test_accuracy=test_accuracy,
            test_precision=precision,
            test_recall=recall,
            test_f1=f1,
            inference_time_ms_per_batch=avg_inference_time_ms,
        )

        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1: {f1:.4f}")

        # Save model to MLflow
        print("Saving model to MLflow...")
        mlflow.tensorflow.log_model(model, artifact_path="tensorflow_cnn_model")

        print(f"MLflow Run ID: {mlflow_config.run_id}")

    finally:
        mlflow_config.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN model on MNIST")
    parser.add_argument("--epochs", type=int, default=60, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--num-conv-blocks", type=int, default=2, help="Number of conv blocks"
    )
    parser.add_argument(
        "--conv-filters-initial",
        type=int,
        default=32,
        help="Initial filter count",
    )
    parser.add_argument("--dense-units", type=int, default=128, help="Dense units")
    parser.add_argument("--dropout-rate", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    train_cnn(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_conv_blocks=args.num_conv_blocks,
        conv_filters_initial=args.conv_filters_initial,
        dense_units=args.dense_units,
        dropout_rate=args.dropout_rate,
        random_seed=args.random_seed,
    )
