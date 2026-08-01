"""CNN model implementation for MNIST digit classification using PyTorch.

Provides a Convolutional Neural Network model optimized for MNIST (28×28 grayscale images)
with reproducibility via fixed random seed. Architecturally equivalent to TensorFlow version.
"""

import torch
import torch.nn as nn


class MNISTCNNModel(nn.Module):
    """
    Convolutional Neural Network for MNIST digit classification.

    Architecture:
    - Conv2d(1→32, 3×3, ReLU) + MaxPool(2×2)
    - Conv2d(32→64, 3×3, ReLU) + MaxPool(2×2)
    - Flatten
    - Linear(64×7×7 → 128, ReLU) + Dropout(0.5)
    - Linear(128 → 10)

    This architecture is specifically designed for 28×28 grayscale MNIST images
    and is architecturally equivalent to the TensorFlow version for fair comparison.
    Two convolutional blocks extract spatial features via learned filters and
    pooling operations, followed by dense layers for classification.

    Input format: (batch_size, 1, 28, 28) — PyTorch NCHW convention
    Output format: (batch_size, 10) — logits (not softmax; applied in loss function)

    Example:
        >>> model = MNISTCNNModel(seed=42)
        >>> print(model)  # View architecture
        >>> # Training: custom loop with optimizer, loss computation, backward pass
        >>> # Inference: model.eval(); logits = model(batch); predictions = logits.argmax(dim=1)
    """

    def __init__(
        self,
        input_channels: int = 1,
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
            input_channels: Number of input channels. Default: 1 (grayscale).
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
            Uncompiled PyTorch nn.Module ready for training with custom loop.
        """
        super().__init__()

        # Set random seed for reproducibility
        torch.manual_seed(seed)

        # Convolutional layers (2 blocks)
        in_channels = input_channels
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        for block_idx in range(num_conv_blocks):
            out_channels = conv_filters_initial * (2 ** block_idx)
            self.conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
            self.pool_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels

        # Calculate flattened size after conv blocks
        # Input: (batch, 1, 28, 28)
        # After block 1: (batch, 32, 14, 14)
        # After block 2: (batch, 64, 7, 7)
        self.flatten_size = (conv_filters_initial * (2 ** (num_conv_blocks - 1))) * 7 * 7

        # Dense layers
        self.fc1 = nn.Linear(self.flatten_size, dense_units)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(dense_units, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN.

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28).
               Values should be in [0, 1] range (normalized).

        Returns:
            Output logits of shape (batch_size, 10) — raw class scores without softmax.
            Note: Softmax is typically applied in the loss function (CrossEntropyLoss).
        """
        # Convolutional blocks with ReLU activation and max pooling
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            x = conv(x)
            x = torch.nn.functional.relu(x)
            x = pool(x)

        # Flatten for dense layers
        x = x.flatten(start_dim=1)

        # Dense layers with dropout
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
