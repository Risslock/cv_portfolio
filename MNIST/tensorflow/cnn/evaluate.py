"""Evaluation utilities for TensorFlow CNN model on MNIST.

Provides functions to compute evaluation metrics, visualize training history,
and perform inference on MNIST data.
"""

from typing import Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray = None
) -> Tuple[float, float, float, float]:
    """
    Compute evaluation metrics for MNIST classification.

    Args:
        y_true: True class labels (array of shape (n,) with values in [0, 9])
        y_pred: Predicted class labels (array of shape (n,) with values in [0, 9])
        y_probs: Predicted probabilities (optional, shape (n, 10))

    Returns:
        Tuple of (accuracy, precision, recall, f1) all macro-averaged in range [0, 1]
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return accuracy, precision, recall, f1


def plot_history(
    history: Dict, save_path: str = "training_history.png"
) -> None:
    """
    Plot and save training history (loss and accuracy over epochs).

    Args:
        history: Dictionary with keys 'loss', 'accuracy', 'val_loss', 'val_accuracy'
                 (as returned by Keras model.fit())
        save_path: Path to save the plot figure. Default: "training_history.png"
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot loss
    ax1.plot(history["loss"], label="Training Loss", linewidth=2)
    ax1.plot(history["val_loss"], label="Validation Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Model Loss Over Training")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    ax2.plot(history["accuracy"], label="Training Accuracy", linewidth=2)
    ax2.plot(history["val_accuracy"], label="Validation Accuracy", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Model Accuracy Over Training")
    ax2.set_ylim([0.9, 1.0])  # Zoom in on accuracy range
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    print(f"Training history saved to {save_path}")
    plt.close()


def infer_batch(model, batch_data: np.ndarray) -> np.ndarray:
    """
    Perform inference on a batch of images.

    Args:
        model: Trained Keras model
        batch_data: Batch of images (shape (batch_size, 28, 28, 1))

    Returns:
        Array of predicted class labels (shape (batch_size,))
    """
    predictions = model.predict(batch_data, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    return predicted_labels
