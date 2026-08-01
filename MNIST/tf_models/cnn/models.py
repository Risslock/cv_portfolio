"""CNN model implementation for MNIST digit classification using TensorFlow.

Provides a Convolutional Neural Network model optimized for MNIST (28×28 grayscale images)
with reproducibility via fixed random seed.
"""

import tensorflow as tf
from tensorflow import keras


class MNISTCNNModel(keras.Sequential):
    """
    Convolutional Neural Network for MNIST digit classification.

    Architecture:
    - Conv2D(32, 3×3, ReLU) + MaxPooling(2×2)
    - Conv2D(64, 3×3, ReLU) + MaxPooling(2×2)
    - Flatten
    - Dense(128, ReLU) + Dropout(0.5)
    - Dense(10, Softmax)

    This architecture is specifically designed for 28×28 grayscale MNIST images.
    Two convolutional blocks extract spatial features, followed by dense layers
    for classification. Dropout prevents overfitting on the relatively small
    training set.

    Example:
        >>> model = MNISTCNNModel(seed=42)
        >>> model.summary()  # View architecture
        >>> # Training: model.fit(train_data, validation_data=val_data, epochs=60)
    """

    def __init__(
        self,
        input_shape: tuple = (28, 28, 1),
        num_classes: int = 10,
        num_conv_blocks: int = 2,
        conv_filters_initial: int = 32,
        dense_units: int = 128,
        dropout_rate: float = 0.5,
        seed: int = 42,
    ):
        """
        Initialize CNN model for MNIST classification.

        Args:
            input_shape: Input image shape (height, width, channels). Default: (28, 28, 1)
                for MNIST grayscale images.
            num_classes: Number of output classes (digit categories). Default: 10 (0-9).
            num_conv_blocks: Number of Conv→MaxPool blocks. Default: 2.
                Each block halves spatial dimensions (28→14→7).
            conv_filters_initial: Initial number of convolutional filters. Default: 32.
                Doubles with each block (32 → 64).
            dense_units: Number of hidden units in dense layer. Default: 128.
            dropout_rate: Dropout probability after dense layer. Default: 0.5.
            seed: Random seed for reproducibility. Default: 42.
                CRITICAL: Must be 42 for reproducible experiments across runs.

        Returns:
            Compiled Keras Sequential model ready for training with:
            - Optimizer: Adam (learning rate 0.001)
            - Loss: Categorical crossentropy
            - Metrics: Accuracy
        """
        # Set random seed for reproducibility
        tf.random.set_seed(seed)

        # Build layer stack
        layers = []

        # Convolutional blocks
        for block_idx in range(num_conv_blocks):
            filters = conv_filters_initial * (2 ** block_idx)

            # First block includes input_shape
            if block_idx == 0:
                layers.append(
                    keras.layers.Conv2D(
                        filters,
                        kernel_size=(3, 3),
                        activation="relu",
                        input_shape=input_shape,
                    )
                )
            else:
                layers.append(
                    keras.layers.Conv2D(
                        filters, kernel_size=(3, 3), activation="relu"
                    )
                )

            layers.append(keras.layers.MaxPooling2D(pool_size=(2, 2)))

        # Dense layers
        layers.append(keras.layers.Flatten())
        layers.append(keras.layers.Dense(dense_units, activation="relu"))
        layers.append(keras.layers.Dropout(dropout_rate))
        layers.append(keras.layers.Dense(num_classes, activation="softmax"))

        # Initialize Sequential model
        super().__init__(layers)

        # Compile model
        self.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
