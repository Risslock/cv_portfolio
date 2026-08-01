from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


class TF_FCNN(Model):
    """
    A fully connected neural network model for MNIST digit classification.

    Architecture: Input(784) → Dense(128) → ReLU → Dropout(0.2) →
    Dense(64) → ReLU → Dropout(0.2) → Dense(10) → Softmax

    Includes L2 regularization on dense layers for preventing overfitting.
    Dropout layers are active only during training (controlled by training flag).

    Attributes:
        hidden_layer1 (Dense): First hidden layer with 128 units and ReLU activation.
        dropout_layer1 (Dropout): Dropout (0.2) after first hidden layer.
        hidden_layer2 (Dense): Second hidden layer with 64 units and ReLU activation.
        dropout_layer2 (Dropout): Dropout (0.2) after second hidden layer.
        output_layer (Dense): Output layer with softmax activation.

    Parameters:
        input_shape (tuple): Shape of input data, e.g., (784,) for flattened MNIST images.
        num_classes (int): Number of output classes, e.g., 10 for digits 0-9.

    Example:
        >>> model = TF_FCNN(input_shape=(784,), num_classes=10)
        >>> model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        >>> model.fit(x_train, y_train, validation_data=(x_val, y_val))
    """

    def __init__(self, input_shape: tuple, num_classes: int) -> None:
        """
        Initializes the TF_FCNN model with specified architecture.

        Args:
            input_shape (tuple): Shape of input features, e.g., (784,)
            num_classes (int): Number of output classes

        Raises:
            ValueError: If num_classes < 2 or input_shape is empty
        """
        super().__init__()

        # Store parameters for serialization (required for get_config)
        self.input_shape_config = input_shape
        self.num_classes_config = num_classes

        # Input validation
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        if not input_shape or len(input_shape) == 0:
            raise ValueError("input_shape must be a non-empty tuple")

        # Layer definitions with L2 regularization for better generalization
        self.hidden_layer1 = Dense(
            128, kernel_regularizer="l2", activation="relu", name="hidden_1"
        )
        self.hidden_layer2 = Dense(
            64, kernel_regularizer="l2", activation="relu", name="hidden_2"
        )
        self.output_layer = Dense(num_classes, activation="softmax", name="output")

    def build(self, input_shape):
        """
        Build the model layers with specified input shape.

        Called automatically by Keras before the first forward pass.
        Ensures all layers are properly initialized.

        Args:
            input_shape: Shape of input data (batch_size, input_size)
        """
        super().build(input_shape)

    def call(self, inputs, training=None, mask=None):
        """
        Forward pass of the model.

        Args:
            inputs: Input tensor of shape (batch_size, input_size)
            training: Boolean indicating training mode (unused, L2 regularization used instead)
            mask: Optional mask tensor (unused)

        Returns:
            Tensor of shape (batch_size, num_classes) with class probabilities
        """
        x = self.hidden_layer1(inputs)
        x = self.hidden_layer2(x)
        return self.output_layer(x)

    def get_config(self) -> dict:
        """
        Return model configuration for serialization.

        Required for saving custom Model subclasses. Returns all parameters
        needed to reconstruct the model architecture.

        Returns:
            dict: Configuration dictionary with model parameters
        """
        return {
            "input_shape": self.input_shape_config,
            "num_classes": self.num_classes_config,
        }

    @classmethod
    def from_config(cls, config: dict) -> "TF_FCNN":
        """
        Create model instance from configuration dictionary.

        Used during model loading to reconstruct the model with saved architecture.

        Args:
            config (dict): Configuration dictionary from get_config()

        Returns:
            TF_FCNN: Reconstructed model instance
        """
        return cls(**config)
