"""Test-set evaluation: scikit-learn metrics, confusion matrix, and classification report."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from fashion_mnist.data import CLASS_NAMES, configure_gpu, load_fashion_mnist_data


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Macro-averaged precision/recall/F1 (classes are balanced, ~equal support)."""
    return {
        "test_precision": precision_score(y_true, y_pred, average="macro"),
        "test_recall": recall_score(y_true, y_pred, average="macro"),
        "test_f1": f1_score(y_true, y_pred, average="macro"),
    }


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> Path:
    labels = sorted(CLASS_NAMES)
    display_labels = [CLASS_NAMES[label] for label in labels]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(matrix, display_labels=display_labels).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def save_classification_report(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> Path:
    labels = sorted(CLASS_NAMES)
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=[CLASS_NAMES[label] for label in labels]
    )
    output_path.write_text(report)
    return output_path


def evaluate_model(
    model: tf.keras.Model,
    test_images: tf.Tensor,
    test_labels: tf.Tensor,
    output_dir: Path,
) -> dict[str, float | Path]:
    """Run the full evaluation pipeline and write artifacts to ``output_dir``.

    Returns a dict of scalar metrics (test_loss, test_accuracy, test_precision,
    test_recall, test_f1) plus the paths of the saved confusion matrix and
    classification report artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    y_pred = np.argmax(model.predict(test_images, verbose=0), axis=1)
    y_true = np.asarray(test_labels)

    metrics = compute_metrics(y_true, y_pred)
    confusion_matrix_path = save_confusion_matrix(
        y_true, y_pred, output_dir / "confusion_matrix.png"
    )
    classification_report_path = save_classification_report(
        y_true, y_pred, output_dir / "classification_report.txt"
    )

    return {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        **metrics,
        "confusion_matrix_path": confusion_matrix_path,
        "classification_report_path": classification_report_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Fashion MNIST model.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    configure_gpu()

    model = tf.keras.models.load_model(args.model_path)
    _, _, (test_images, test_labels) = load_fashion_mnist_data()

    results = evaluate_model(model, test_images, test_labels, args.output_dir)
    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
