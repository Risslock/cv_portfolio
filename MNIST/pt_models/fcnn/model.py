"""
A simple fully connected neural network for MNIST classification on PyTorch.
"""

import torch
import torch.nn as nn


class PT_FCNN(nn.Module):
    """
    A fully connected neural network model for MNIST digit classification.

    Architecture: Flatten -> Linear(784, 128) -> ReLU -> Dropout(0.2) ->
    Linear(128, 64) -> ReLU -> Dropout(0.2) -> Linear(64, num_classes)
    """

    def __init__(self, input_shape: tuple, num_classes: int) -> None:
        """
        Initializes the PT_FCNN model with specified architecture.

        Args:
            input_shape (tuple): Shape of input image, e.g., (1, 28, 28) or (28, 28).
                                 Will be flattened to 1D internally.
            num_classes (int): Number of output classes (10 for MNIST digits)

        Raises:
            ValueError: If num_classes < 2 or input_shape is empty
        """
        super().__init__()

        self.input_shape_config = input_shape
        self.num_classes_config = num_classes

        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        if not input_shape or len(input_shape) == 0:
            raise ValueError("input_shape must be a non-empty tuple")

        flattened_size = 1
        for dim in input_shape:
            flattened_size *= dim

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
                             or (batch_size, height, width)

        Returns:
            torch.Tensor: Raw logits of shape (batch_size, num_classes).
                         Use with CrossEntropyLoss for training.
        """
        return self.network(x)

    def predict(self, x):
        """
        Predicts the class labels for the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
                             or (batch_size, height, width)

        Returns:
            torch.Tensor: Predicted class labels of shape (batch_size,).
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)
