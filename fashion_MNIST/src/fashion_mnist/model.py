"""Parametrized CNN builder for Fashion MNIST."""

from __future__ import annotations

from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAvgPool2D,
    GlobalMaxPool2D,
    Input,
)
from tensorflow.keras.models import Model

from fashion_mnist.augmentation import create_image_augmentation

INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 10
POOLING_TYPES = ("max", "avg", "flatten")


def core_fashion_model(
    n_conv: int = 2,
    n_dense: int = 2,
    kernel_size: int = 3,
    global_pooling_type: str = "max",
    conv_dropout_rate: float = 0.5,
    dense_dropout_rate: float = 0.2,
) -> Model:
    """Build the core CNN (no augmentation layers): this is the deployable model.

    Each conv block doubles the previous block's filters (32, 64, 128, ...) and is
    followed by BatchNorm + ReLU. Each dense block is a fixed 64 units + BatchNorm +
    ReLU + dropout.
    """
    if n_conv < 1 or n_conv > 6:
        raise ValueError("n_conv must be between 1 and 6")
    if n_dense < 1:
        raise ValueError("n_dense must be greater than 0")
    if global_pooling_type not in POOLING_TYPES:
        raise ValueError(f"global_pooling_type must be one of {POOLING_TYPES}")

    input_layer = Input(shape=INPUT_SHAPE)
    x = input_layer
    for i in range(n_conv):
        filters = 32 * (2**i)
        x = Conv2D(filters, kernel_size, strides=1)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

    if global_pooling_type == "max":
        x = GlobalMaxPool2D()(x)
    elif global_pooling_type == "avg":
        x = GlobalAvgPool2D()(x)
    else:
        x = Flatten()(x)
    x = Dropout(conv_dropout_rate)(x)

    for _ in range(n_dense):
        x = Dense(64, kernel_regularizer="l2")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(dense_dropout_rate)(x)

    output = Dense(NUM_CLASSES, activation="softmax")(x)
    return Model(inputs=input_layer, outputs=output, name="fashion_mnist_cnn")


def build_training_model(
    core_model: Model,
    horizontal_flip: bool = False,
    rotation: float = 0.0,
    brightness: float = 0.0,
    zoom: float = 0.0,
    translation: float = 0.0,
) -> Model:
    """Wrap ``core_model`` with a GPU-side augmentation head, for use in ``.fit()`` only.

    ``core_model`` and the returned training model share the same weight tensors
    (the augmentation head has none), so training the returned model also trains
    ``core_model`` in place. Evaluate/save ``core_model`` directly, not this wrapper.
    """
    aug_input = Input(shape=INPUT_SHAPE)
    augmentation = create_image_augmentation(
        horizontal_flip=horizontal_flip,
        rotation=rotation,
        brightness=brightness,
        zoom=zoom,
        translation=translation,
    )
    return Model(
        inputs=aug_input,
        outputs=core_model(augmentation(aug_input)),
        name="fashion_mnist_training",
    )
