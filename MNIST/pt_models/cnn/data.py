"""MNIST dataset loading and preprocessing for PyTorch CNN training.

Provides utilities to load MNIST data, normalize it, and create PyTorch DataLoaders
for CNN model training and evaluation.
"""

from typing import Tuple
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


def load_mnist(
    batch_size: int = 32,
    normalize: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load MNIST dataset and create PyTorch DataLoaders for train, validation, and test sets.

    Args:
        batch_size: Batch size for DataLoader. Default: 32.
        normalize: If True, normalize images to [0, 1] range. Default: True.

    Returns:
        Tuple of (train_loader, val_loader, test_loader) where each is a PyTorch DataLoader
        with:
        - train_loader: 50000 samples / batch_size batches
        - val_loader: 10000 samples / batch_size batches
        - test_loader: 10000 samples / batch_size batches
        Each batch contains (images, labels) where:
        - images: shape (batch_size, 1, 28, 28), float32, normalized if requested
        - labels: shape (batch_size,), long tensor with class indices [0, 9]
    """
    # Define transform: normalize to [0, 1] if requested
    if normalize:
        transform = transforms.Compose(
            [transforms.ToTensor()]  # Automatically converts to [0, 1]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 255),  # Keep in [0, 255]
            ]
        )

    # Download and load MNIST training dataset
    train_full = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    # Split into train (50k) and validation (10k)
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_full, [50000, 10000], generator=torch.Generator().manual_seed(42)
    )

    # Load test dataset
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
