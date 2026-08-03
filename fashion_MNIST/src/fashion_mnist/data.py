"""Fashion MNIST dataset loading, splitting, and GPU runtime setup."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

CLASS_NAMES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}

Split = tuple[tf.Tensor, tf.Tensor]


def configure_gpu() -> None:
    """Enable memory growth on any visible GPU so TensorFlow doesn't pre-allocate all VRAM."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def load_fashion_mnist_data(val_split: float = 0.1) -> tuple[Split, Split, Split]:
    """Load Fashion MNIST, normalize to [0, 1], and carve a validation split out of train.

    Returns ((train_images, train_labels), (val_images, val_labels), (test_images, test_labels)).
    """
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    train_images = tf.expand_dims(train_images, -1)
    test_images = tf.expand_dims(test_images, -1)

    val_size = int(len(train_images) * val_split)
    train_images, val_images = train_images[:-val_size], train_images[-val_size:]
    train_labels, val_labels = train_labels[:-val_size], train_labels[-val_size:]

    return (
        (train_images, train_labels),
        (val_images, val_labels),
        (test_images, test_labels),
    )
