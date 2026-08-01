"""MNIST dataset loading and preprocessing for TensorFlow CNN training.

Provides utilities to load MNIST data, normalize it, and create train/validation/test splits
for CNN model training and evaluation.
"""

from typing import Tuple
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


def load_mnist(
    normalize: bool = True,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """
    Load MNIST dataset and split into train, validation, and test sets.

    Args:
        normalize: If True, normalize images to [0, 1] range. Default: True.

    Returns:
        Tuple of ((train_data, train_labels), (val_data, val_labels), (test_data, test_labels))
        where:
        - train_data: shape (50000, 28, 28, 1), float32, normalized if requested
        - train_labels: shape (50000, 10), one-hot encoded
        - val_data: shape (10000, 28, 28, 1), float32, normalized if requested
        - val_labels: shape (10000, 10), one-hot encoded
        - test_data: shape (10000, 28, 28, 1), float32, normalized if requested
        - test_labels: shape (10000, 10), one-hot encoded
    """
    # Load MNIST dataset (automatically downloads if needed)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Split training data into train (50k) and validation (10k)
    x_train, x_val = x_train[:50000], x_train[50000:]
    y_train, y_val = y_train[:50000], y_train[50000:]

    # Reshape to include channel dimension: (N, 28, 28) → (N, 28, 28, 1)
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32")
    x_val = x_val.reshape(-1, 28, 28, 1).astype("float32")
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32")

    # Normalize to [0, 1] range
    if normalize:
        x_train = x_train / 255.0
        x_val = x_val / 255.0
        x_test = x_test / 255.0

    # One-hot encode labels (10 classes: 0-9)
    y_train = to_categorical(y_train, 10)
    y_val = to_categorical(y_val, 10)
    y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
