"""
Standardized metrics calculation for MNIST experiments.

Provides utilities to calculate F1-score and other metrics consistently
across TensorFlow and PyTorch implementations.
"""

from typing import Tuple
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def calculate_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate macro-averaged F1-score for multiclass classification.

    Args:
        y_true: True labels (array or list of class indices)
        y_pred: Predicted labels (array or list of class indices)

    Returns:
        F1-score (macro-averaged) in range [0, 1]
    """
    return float(f1_score(y_true, y_pred, average="macro"))


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1-score (all macro-averaged).

    Args:
        y_true: True labels (array or list of class indices)
        y_pred: Predicted labels (array or list of class indices)

    Returns:
        Tuple of (precision, recall, f1_score) all macro-averaged in range [0, 1]
    """
    precision = float(precision_score(y_true, y_pred, average="macro"))
    recall = float(recall_score(y_true, y_pred, average="macro"))
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    return precision, recall, f1
