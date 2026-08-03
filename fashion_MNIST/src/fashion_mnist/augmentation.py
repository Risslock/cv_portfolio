"""GPU-side image augmentation, wired directly into the model graph.

These layers are meant to be applied to the model's input tensor (not via
``tf.data.Dataset.map``) so augmentation runs on the GPU as part of the forward
pass during ``model.fit()``. Keras preprocessing layers are automatically inert
during ``.evaluate()``/``.predict()``, so no manual train/eval toggling is needed.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import (
    RandomBrightness,
    RandomFlip,
    RandomRotation,
    RandomTranslation,
    RandomZoom,
)


def create_image_augmentation(
    horizontal_flip: bool = False,
    rotation: float = 0.0,
    brightness: float = 0.0,
    zoom: float = 0.0,
    translation: float = 0.0,
) -> tf.keras.Sequential:
    """Build a stack of Keras preprocessing layers for training-time augmentation."""
    augmentation = tf.keras.Sequential(name="augmentation")
    if horizontal_flip:
        augmentation.add(RandomFlip("horizontal"))
    augmentation.add(RandomRotation(rotation, fill_mode="constant", fill_value=0))
    augmentation.add(RandomBrightness(brightness))
    augmentation.add(RandomZoom(zoom, fill_mode="constant", fill_value=0))
    augmentation.add(
        RandomTranslation(translation, translation, fill_mode="constant", fill_value=0)
    )
    return augmentation
