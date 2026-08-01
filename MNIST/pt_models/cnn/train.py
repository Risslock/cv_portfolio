"""PyTorch CNN training script with MLflow experiment tracking for MNIST.

Trains the CNN model on MNIST with comprehensive MLflow logging for metrics,
parameters, and model artifacts. Enables reproducible, comparable experiments.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn

# Add project root to path for utils imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import local modules from current package (now that pt_models doesn't shadow pytorch)
from .data import load_mnist
from .models import MNISTCNNModel
from .evaluate import compute_metrics
from utils.mlflow_config import MLflowConfig


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch and return average loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, val_loader, criterion, device):
    """Evaluate model on validation/test set and return loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)

            outputs = model(data)
            loss = criterion(outputs, target)

            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_cnn(
    epochs: int = 60,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    num_conv_blocks: int = 2,
    conv_filters_initial: int = 32,
    dense_units: int = 128,
    dropout_rate: float = 0.5,
    random_seed: int = 42,
    checkpoint_dir: str = "./results/pytorch_cnn",
    patience: int = 5,
):
    """
    Train CNN model on MNIST with MLflow experiment tracking.

    Args:
        epochs: Number of training epochs. Default: 60.
        batch_size: Batch size for training. Default: 32.
        learning_rate: Optimizer learning rate. Default: 0.001.
        num_conv_blocks: Number of convolutional blocks. Default: 2.
        conv_filters_initial: Initial filter count. Default: 32.
        dense_units: Hidden units in dense layer. Default: 128.
        dropout_rate: Dropout probability. Default: 0.5.
        random_seed: Random seed for reproducibility. Default: 42.
        checkpoint_dir: Directory to save model checkpoints. Default: "./results/pytorch_cnn"
        patience: Early stopping patience (epochs without improvement). Default: 5.
    """
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Set device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seeds for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Initialize MLflow experiment
    mlflow_config = MLflowConfig(experiment_name="cnn_mnist_pytorch")
    mlflow_config.start_run()

    try:
        # Log hyperparameters
        mlflow_config.log_params(
            {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "num_epochs": epochs,
                "optimizer": "adam",
                "loss_function": "crossentropyloss",
                "num_conv_blocks": num_conv_blocks,
                "conv_filters_initial": conv_filters_initial,
                "dense_units": dense_units,
                "dropout_rate": dropout_rate,
                "random_seed": random_seed,
                "framework": "pytorch",
            }
        )

        # Load data
        print("Loading MNIST dataset...")
        train_loader, val_loader, test_loader = load_mnist(batch_size=batch_size)
        print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        # Create model
        print("Building CNN model...")
        model = MNISTCNNModel(
            input_channels=1,
            num_classes=10,
            num_conv_blocks=num_conv_blocks,
            conv_filters_initial=conv_filters_initial,
            dense_units=dense_units,
            dropout_rate=dropout_rate,
            seed=random_seed,
        )
        model.to(device)
        print(model)

        # Setup optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Train model with early stopping
        print(f"Training for {epochs} epochs (with early stopping, patience={patience})...")
        start_time = time.time()

        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            # Log epoch metrics to MLflow
            mlflow_config.log_epoch_metrics(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_acc,
                val_loss=val_loss,
                val_accuracy=val_acc,
            )

            print(
                f"Epoch {epoch + 1}/{epochs}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )

            # Early stopping and checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                # Save best model
                torch.save(model.state_dict(), f"{checkpoint_dir}/best_model.pth")
                print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(
                        f"Early stopping triggered after {epoch + 1} epochs "
                        f"({patience} epochs without improvement)"
                    )
                    break

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        # Evaluate on test set
        print("Evaluating on test set...")
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

        # Get predictions for detailed metrics
        model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(target.numpy())

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        # Compute detailed metrics
        _, precision, recall, f1 = compute_metrics(all_labels, all_predictions)

        # Calculate inference time (ms per batch)
        batch_test_data = next(iter(test_loader))[0][:batch_size].to(device)
        start_inference = time.time()
        with torch.no_grad():
            for _ in range(10):  # Average over 10 batches
                _ = model(batch_test_data)
        avg_inference_time_ms = ((time.time() - start_inference) / 10) * 1000

        # Log final metrics
        mlflow_config.log_final_metrics(
            test_loss=test_loss,
            test_accuracy=test_accuracy,
            test_precision=precision,
            test_recall=recall,
            test_f1=f1,
            inference_time_ms_per_batch=avg_inference_time_ms,
        )

        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1: {f1:.4f}")

        # Save model to MLflow
        print("Saving model to MLflow...")
        mlflow.pytorch.log_model(
            model,
            artifact_path="pytorch_cnn_model",
            serialization_format="pickle",
        )

        print(f"MLflow Run ID: {mlflow_config.run_id}")

    finally:
        mlflow_config.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN model on MNIST")
    parser.add_argument("--epochs", type=int, default=60, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--num_conv_blocks", type=int, default=2, help="Number of conv blocks"
    )
    parser.add_argument(
        "--conv_filters_initial",
        type=int,
        default=32,
        help="Initial filter count",
    )
    parser.add_argument("--dense_units", type=int, default=128, help="Dense units")
    parser.add_argument("--dropout_rate", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    train_cnn(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_conv_blocks=args.num_conv_blocks,
        conv_filters_initial=args.conv_filters_initial,
        dense_units=args.dense_units,
        dropout_rate=args.dropout_rate,
        random_seed=args.random_seed,
        patience=5,
    )
