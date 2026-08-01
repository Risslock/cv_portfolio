import argparse
import datetime
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from utils.mlflow_config import MLflowConfig
from utils.metrics import calculate_metrics

from model import PT_FCNN

FRAMEWORK = "pytorch"
ARCHITECTURE = "fcnn"
MLFLOW_TRACKING_URI = "sqlite:///./mlflow.db"
MLFLOW_EXPERIMENT_NAME = f"mnist_{FRAMEWORK}_{ARCHITECTURE}"
RANDOM_SEED = 42
NUM_CLASSES = 10
NUM_CONV_BLOCKS = 0
CONV_FILTERS_INITIAL = 0
DENSE_UNITS = 128
DROPOUT_RATE = 0.2
LOSS_FUNCTION = "crossentropyloss"


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a FCNN model on MNIST dataset with MLflow tracking"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="Optimizer type",
    )
    return parser.parse_args()


def load_mnist_dataset(data_dir: str = "./data"):
    """Load MNIST dataset with standard normalization."""
    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transforms.ToTensor()
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transforms.ToTensor()
    )
    return train_dataset, test_dataset


def measure_inference_time(model, input_size, num_iterations=100):
    """Measure average inference time in milliseconds per batch."""
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, *input_size).to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
        start_time = time.time()
        for _ in range(num_iterations):
            _ = model(dummy_input)
        end_time = time.time()
    avg_time_ms = (end_time - start_time) * 1000 / num_iterations
    return avg_time_ms


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate_epoch(model, val_loader, criterion, device):
    """Validate on one epoch."""
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_predictions = []

    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        predicted = outputs.argmax(dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

    avg_loss = running_loss / len(val_loader.dataset)
    return avg_loss, np.array(all_labels), np.array(all_predictions)


def train_with_early_stopping(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs,
    device,
    checkpoint_dir: str = "./checkpoints",
    patience: int = 5,
):
    """Train with early stopping and checkpointing."""
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    training_history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(epochs):
        train_loss, train_accuracy = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_labels, val_preds = validate_epoch(
            model, val_loader, criterion, device
        )
        val_accuracy = np.mean(val_labels == val_preds)

        training_history["train_loss"].append(train_loss)
        training_history["train_accuracy"].append(train_accuracy)
        training_history["val_loss"].append(val_loss)
        training_history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch [{epoch + 1}/{epochs}] - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} - "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), f"{checkpoint_dir}/best_model.pth")
            print("Best model saved.")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                break

    return model, training_history


def main():
    """Train PyTorch FCNN with standardized MLflow tracking."""
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    args = parse_arguments()

    # Create result directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = f"results/{FRAMEWORK}_{ARCHITECTURE}/{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    print(f"\n{'=' * 60}")
    print(f"Results will be saved to: {result_dir}")
    print(f"{'=' * 60}\n")

    # Initialize MLflow
    mlflow_cfg = MLflowConfig(MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI)
    print(f"MLflow Experiment: {MLFLOW_EXPERIMENT_NAME}\n")

    # Load data
    data_dir = "MNIST/data"
    train_dataset, test_dataset = load_mnist_dataset(data_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Data Loaded:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Classes: {NUM_CLASSES}\n")

    # Setup model and training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PT_FCNN(input_shape=(1, 28, 28), num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()

    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    print("Model initialized.")
    print()

    # Start MLflow run
    mlflow_cfg.start_run(
        tags={
            "framework": FRAMEWORK,
            "architecture": ARCHITECTURE,
            "project": "mnist_digits",
        }
    )

    try:
        # Log standardized parameters
        params = {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "num_epochs": args.epochs,
            "optimizer": args.optimizer,
            "loss_function": LOSS_FUNCTION,
            "num_conv_blocks": NUM_CONV_BLOCKS,
            "conv_filters_initial": CONV_FILTERS_INITIAL,
            "dense_units": DENSE_UNITS,
            "dropout_rate": DROPOUT_RATE,
            "random_seed": RANDOM_SEED,
            "framework": FRAMEWORK,
        }
        mlflow_cfg.log_params(params)

        # Train
        print("Training model...\n")
        os.makedirs(result_dir, exist_ok=True)
        trained_model, training_history = train_with_early_stopping(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            epochs=args.epochs,
            device=device,
            checkpoint_dir=result_dir,
            patience=5,
        )

        # Log per-epoch metrics
        print("Logging training history to MLflow...")
        for epoch in range(len(training_history["train_loss"])):
            mlflow_cfg.log_epoch_metrics(
                epoch=epoch,
                train_loss=training_history["train_loss"][epoch],
                train_accuracy=training_history["train_accuracy"][epoch],
                val_loss=training_history["val_loss"][epoch],
                val_accuracy=training_history["val_accuracy"][epoch],
            )

        # Evaluate
        print("Evaluating on test set...")
        best_model_path = f"{result_dir}/best_model.pth"
        if os.path.exists(best_model_path):
            trained_model.load_state_dict(torch.load(best_model_path))

        all_labels = []
        all_predictions = []
        test_loss_total = 0.0

        trained_model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = trained_model(images)
                loss = criterion(outputs, labels)
                test_loss_total += loss.item() * images.size(0)
                predicted = outputs.argmax(dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        test_loss = test_loss_total / len(test_dataset)
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        test_accuracy = np.mean(all_labels == all_predictions)

        test_precision, test_recall, test_f1 = calculate_metrics(
            all_labels, all_predictions
        )

        print("\nTest Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_accuracy:.4f}")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall: {test_recall:.4f}")
        print(f"  F1-Score: {test_f1:.4f}\n")

        # Measure inference time
        print("Measuring inference time...")
        input_size = (1, 28, 28)
        inference_time_ms = measure_inference_time(
            trained_model, input_size, num_iterations=100
        )
        print(
            f"Average inference time: {inference_time_ms:.4f} ms "
            f"(over 100 predictions)\n"
        )

        # Log final metrics with training time
        mlflow_cfg.log_final_metrics(
            test_loss=test_loss,
            test_accuracy=test_accuracy,
            test_precision=test_precision,
            test_recall=test_recall,
            test_f1=test_f1,
            inference_time_ms_per_batch=inference_time_ms,
        )

        # Save training curves
        plt.figure(figsize=(14, 5))

        plt.subplot(1, 2, 1)
        plt.plot(training_history["train_loss"], label="Train Loss")
        plt.plot(training_history["val_loss"], label="Validation Loss")
        plt.title("Model Loss Over Epochs", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(training_history["train_accuracy"], label="Training Accuracy")
        plt.plot(training_history["val_accuracy"], label="Validation Accuracy")
        plt.title("Model Accuracy Over Epochs", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        curves_path = f"{result_dir}/training_curves.png"
        plt.savefig(curves_path, dpi=100)
        plt.close()
        mlflow_cfg.log_artifact(curves_path)

        # Save confusion matrix
        print("Generating confusion matrix...")
        cm = confusion_matrix(all_labels, all_predictions)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=list(range(NUM_CLASSES))
        )
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix", fontsize=12, fontweight="bold")
        cm_path = f"{result_dir}/confusion_matrix.png"
        plt.savefig(cm_path, dpi=100)
        plt.close()
        mlflow_cfg.log_artifact(cm_path)

        # Log artifacts
        mlflow_cfg.log_artifact(result_dir)

        print("Training complete!")
        print(f"Results saved to: {result_dir}\n")

    finally:
        mlflow_cfg.end_run()


if __name__ == "__main__":
    main()
