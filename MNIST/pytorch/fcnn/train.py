import argparse
import datetime
import time
import os

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import PT_FCNN
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

FRAMEWORK = "PyTorch"
ARCHITECTURE = "FCNN"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = f"mnist_{FRAMEWORK}_{ARCHITECTURE}"
RANDOM_SEED = 42
NUM_CLASSES = 10


def load_mnist_dataset(data_dir: str = "./data") -> tuple:
    """
    Load MNIST dataset with standard normalization.

    Args:
        data_dir: Directory to store/load MNIST data

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor(),  ## Just load it raw without normalization
    )
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transforms.ToTensor(),  ## Just load it raw without normalization
    )
    return train_dataset, test_dataset


def parse_arguments():
    """
    Parse command-line arguments for training configuration.

    Returns arguments for:
        - epochs: Number of training epochs
        - batch_size: Batch size for training
        - learning_rate: Learning rate for optimizer
        - optimizer: Optimizer type (e.g., 'adam', 'sgd')
        - experiment_name: MLflow experiment name
        - log_dir: Directory for checkpoints
    """
    parser = argparse.ArgumentParser(
        description="Train a fully connected neural network on MNIST."
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training"
    )
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
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=MLFLOW_EXPERIMENT_NAME,
        help="MLflow experiment name",
    )
    return parser.parse_args()


def measure_inference_time(model, input_size, num_iterations=100):
    """
    Measure average inference time of the model on a dummy input.

    Args:
        model: PyTorch model to evaluate
        input_size: Tuple representing the input size (C, H, W)
        num_iterations: Number of iterations to average over

    Returns:
        Average inference time in milliseconds
    """
    device = next(model.parameters()).device
    print(f"Measuring inference time on device: {device}")
    dummy_input = torch.randn(1, *input_size).to(device)
    model.eval()
    with torch.no_grad():
        # Warm-up
        for _ in range(10):
            _ = model(dummy_input)

        # Measure time
        start_time = time.time()
        for _ in range(num_iterations):
            _ = model(dummy_input)
        end_time = time.time()

    avg_time_ms = (end_time - start_time) * 1000 / num_iterations
    return avg_time_ms


def calculate_metrics(y_true, y_pred):
    """
    Calculate accuracy, recall, and precision metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dict with keys: 'accuracy', 'recall', 'precision'
    """
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average="macro")
    return {"accuracy": accuracy, "recall": recall, "precision": precision}


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cpu/cuda)

    Returns:
        Tuple of (avg_loss, accuracy)
    """
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
    """
    Evaluate model on validation set.

    Args:
        model: PyTorch model to evaluate
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to evaluate on (cpu/cuda)

    Returns:
        Tuple of (avg_loss, metrics_dict)
        where metrics_dict contains accuracy, recall, precision
    """
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
    metrics_dict = calculate_metrics(all_labels, all_predictions)
    return avg_loss, metrics_dict


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
    """
    Train model with early stopping and model checkpointing.

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        epochs: Number of epochs to train
        device: Device to train on (cpu/cuda)
        checkpoint_dir: Directory to save best model checkpoint
        patience: Patience for early stopping (epochs without improvement)

    Returns:
        Tuple of (trained_model, training_history)
        where training_history is a dict with lists of train/val losses and metrics
    """
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    training_history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_recall": [],
        "val_precision": [],
    }

    for epoch in range(epochs):
        train_loss, train_accuracy = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)

        training_history["train_loss"].append(train_loss)
        training_history["train_accuracy"].append(train_accuracy)
        training_history["val_loss"].append(val_loss)
        training_history["val_accuracy"].append(val_metrics["accuracy"])
        training_history["val_recall"].append(val_metrics["recall"])
        training_history["val_precision"].append(val_metrics["precision"])

        print(
            f"Epoch [{epoch + 1}/{epochs}] - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} - "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f}"
        )

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save the best model checkpoint
            torch.save(model.state_dict(), f"{checkpoint_dir}/best_model.pth")
            print("Best model saved.")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                break

    return model, training_history


def main():
    """
    Main training pipeline:
    1. Parse arguments
    2. Load and prepare data
    3. Setup MLflow
    4. Initialize model and training components
    5. Train with early stopping and checkpointing
    6. Log results to MLflow
    """

    # Set random seed for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Parse command-line arguments
    args = parse_arguments()

    # set data directory for MNIST dataset
    data_dir = "MNIST/data"

    # Load and prepare data
    train_dataset, test_dataset = load_mnist_dataset(data_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # Create or get the experiment
    experiment = mlflow.get_experiment_by_name(args.experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(args.experiment_name)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(args.experiment_name)

    # Create result directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = f"results/{FRAMEWORK.lower()}_{ARCHITECTURE.lower()}/{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    print(f"\n{'=' * 60}")
    print(f"Results will be saved to: {result_dir}")
    print(f"{'=' * 60}\n")

    # Initialize model, criterion, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PT_FCNN(input_shape=(1, 28, 28), num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    # Setup optimizer based on user choice
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError(
            f"Unsupported optimizer: {args.optimizer}, selected from ['adam', 'sgd']"
        )

    with mlflow.start_run(experiment_id=experiment_id):
        # Log tags for experiment filtering
        mlflow.set_tags(
            {
                "framework": FRAMEWORK,
                "architecture": ARCHITECTURE,
                "project": "mnist_digits",
            }
        )

        # Log hyperparameters
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("optimizer", args.optimizer)

        # Train the model with early stopping and checkpointing
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
        # Log training history metrics to MLflow
        for epoch in range(len(training_history["train_loss"])):
            mlflow.log_metric(
                "train_loss", training_history["train_loss"][epoch], step=epoch
            )
            mlflow.log_metric(
                "train_accuracy", training_history["train_accuracy"][epoch], step=epoch
            )
            mlflow.log_metric(
                "val_loss", training_history["val_loss"][epoch], step=epoch
            )
            mlflow.log_metric(
                "val_accuracy", training_history["val_accuracy"][epoch], step=epoch
            )
            mlflow.log_metric(
                "val_recall", training_history["val_recall"][epoch], step=epoch
            )
            mlflow.log_metric(
                "val_precision", training_history["val_precision"][epoch], step=epoch
            )

        # evaluate the best model on the test set
        best_model_path = f"{result_dir}/best_model.pth"
        if not os.path.exists(best_model_path):
            print(
                f"Warning: Best model not found at {best_model_path}. Using final model."
            )
        else:
            trained_model.load_state_dict(torch.load(best_model_path))
        test_loss, test_metrics = validate_epoch(
            trained_model, test_loader, criterion, device
        )
        # Log test metrics to MLflow
        mlflow.log_metric("test_loss", test_loss, step=0)
        mlflow.log_metric("test_accuracy", test_metrics["accuracy"], step=0)
        mlflow.log_metric("test_recall", test_metrics["recall"], step=0)
        mlflow.log_metric("test_precision", test_metrics["precision"], step=0)

        # print results
        print(f"\n{'=' * 60}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test Recall: {test_metrics['recall']:.4f}")
        print(f"Test Precision: {test_metrics['precision']:.4f}")
        print(f"{'=' * 60}\n")

        # Save the trained model to MLflow
        example_input = torch.randn(1, 1, 28, 28).to(device).cpu().numpy()
        mlflow.pytorch.log_model(trained_model, "model", input_example=example_input)

        # Generate and save training curves plot
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        plt.plot(training_history["train_loss"], label="Train Loss")
        plt.plot(training_history["val_loss"], label="Validation Loss")
        plt.title("Model Loss Over Epochs", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Accuracy plot
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
        mlflow.log_artifact(curves_path)

        # Generate and save confusion matrix for test set
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            trained_model.eval()
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                predicted = trained_model.predict(images)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        cm = confusion_matrix(all_labels, all_predictions)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=list(range(NUM_CLASSES))
        )
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix", fontsize=12, fontweight="bold")
        cm_path = f"{result_dir}/confusion_matrix.png"
        plt.savefig(cm_path, dpi=100)
        plt.close()
        mlflow.log_artifact(cm_path)

        # measure average inference time for 100 iterations with dummy input
        input_size = (1, 28, 28)  # MNIST image size
        inference_time_ms = measure_inference_time(
            trained_model, input_size, num_iterations=100
        )
        mlflow.log_metric("inference_time_ms", inference_time_ms)
        print(f"Average inference time over 100 iterations: {inference_time_ms:.4f} ms")

        # Log all local artifacts to MLflow
        mlflow.log_artifacts(result_dir, artifact_path="results")

        print(f"\nTraining complete!")
        print(f"Results saved to: {result_dir}")
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}\n")


if __name__ == "__main__":
    main()
