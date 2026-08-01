import argparse
import datetime
import os
import time

import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
import numpy as np
from model import TF_FCNN
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.random import set_seed

FRAMEWORK = "TensorFlow"
ARCHITECTURE = "FCNN"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = f"mnist_{FRAMEWORK}_{ARCHITECTURE}"
RANDOM_SEED = 42
NUM_CLASSES = 10  # MNIST has 10 classes (0-9)


# Create arguments for the training function from command line
parser = argparse.ArgumentParser(
    description="Train a FCNN model on MNIST dataset with MLflow tracking"
)
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument(
    "--batch_size", type=int, default=32, help="Batch size for training"
)
parser.add_argument(
    "--learning_rate", type=float, default=0.001, help="Learning rate for optimizer"
)
parser.add_argument(
    "--optimizer",
    type=str,
    default="adam",
    choices=["adam", "sgd"],
    help="Optimizer to use: adam or sgd",
)
args = parser.parse_args()


def measure_inference_time(model, input_size: int, num_iterations: int = 100) -> float:
    """
    Measure average inference time on a single example.

    Runs inference multiple times on dummy data to get average time.
    Useful for benchmarking model performance on target hardware.

    Args:
        model: Trained Keras model
        input_size (int): Size of input features (e.g., 784 for MNIST)
        num_iterations (int): Number of inference runs (default: 100)

    Returns:
        float: Average inference time in milliseconds
    """
    # Create dummy input with batch size 1
    dummy_input = np.zeros((1, input_size))

    # Warm-up pass to exclude compilation overhead
    _ = model.predict(dummy_input, verbose=0)

    # Measure inference time over multiple iterations
    start_time = time.time()
    for _ in range(num_iterations):
        _ = model.predict(dummy_input, verbose=0)
    end_time = time.time()

    # Calculate average time in milliseconds
    total_time = (end_time - start_time) * 1000  # Convert to milliseconds
    avg_time_ms = total_time / num_iterations

    return avg_time_ms


def load_mnist_data():
    """
    Load the MNIST dataset and preprocess it for training.

    Returns:
        tuple: A tuple containing the preprocessed training and testing data.
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Preprocess the data: reshape to 1D and normalize
    x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0

    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


def main():
    """
    Train a FCNN model on MNIST dataset with MLflow experiment tracking.

    This function:
    1. Loads and preprocesses MNIST data
    2. Creates and compiles a FCNN model
    3. Trains with early stopping and model checkpointing
    4. Evaluates on test set
    5. Logs all metrics, parameters, and artifacts to MLflow
    6. Saves training history and visualization plots
    """

    # Set random seeds for reproducibility
    np.random.seed(RANDOM_SEED)
    set_seed(RANDOM_SEED)

    # Define timestamp for the experiment
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Create result directory
    result_dir = f"results/{FRAMEWORK.lower()}_{ARCHITECTURE.lower()}/{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    print(f"\n{'=' * 60}")
    print(f"Results will be saved to: {result_dir}")
    print(f"{'=' * 60}\n")

    # Set MLflow tracking URI to sqlite
    mlflow.set_tracking_uri("sqlite:///./mlflow.db")

    # Create or set MLflow experiment
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print(f"MLflow Experiment: {MLFLOW_EXPERIMENT_NAME} (ID: {experiment_id})\n")

    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    print("Data Loaded:")
    print(f"  Training: shape={x_train.shape}, labels={y_train.shape}")
    print(f"  Testing:  shape={x_test.shape}, labels={y_test.shape}")

    # Derive input size from data (don't hardcode it)
    input_size = x_train.shape[1]
    print(f"  Input size: {input_size}, Classes: {NUM_CLASSES}\n")

    # Create and compile the model
    input_shape = (input_size,)
    model = TF_FCNN(input_shape=input_shape, num_classes=NUM_CLASSES)

    # Select optimizer based on command line argument
    if args.optimizer.lower() == "adam":
        optimizer = Adam(learning_rate=args.learning_rate)
        print(f"Using Adam optimizer with learning rate: {args.learning_rate}")
    elif args.optimizer.lower() == "sgd":
        optimizer = SGD(learning_rate=args.learning_rate)
        print(f"Using SGD optimizer with learning rate: {args.learning_rate}")
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}. Choose 'adam' or 'sgd'.")

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", "recall", "precision"],
    )
    print("Model compiled successfully.\n")

    # Initialize layers with a dummy forward pass (required for subclassed Models)
    print("Initializing model layers...")
    dummy_input = np.zeros((1, input_size))
    _ = model(dummy_input, training=False)
    print("Layers initialized.\n")

    print("Model Architecture:")
    model.summary()
    print()

    with mlflow.start_run():
        # Log tags for experiment filtering
        mlflow.set_tags(
            {
                "framework": FRAMEWORK,
                "architecture": ARCHITECTURE,
                "project": "mnist_digits",
            }
        )

        # Log hyperparameters
        mlflow.log_params(vars(args))
        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_param("input_size", input_size)
        mlflow.log_param("num_classes", NUM_CLASSES)

        # Define callbacks for early stopping and model checkpointing
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        model_checkpoint = ModelCheckpoint(
            f"{result_dir}/best_model.keras", monitor="val_loss", save_best_only=True
        )

        # Train the model
        print("Training model...\n")
        hist = model.fit(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1,
        )

        # Load best model weights
        model.load_weights(f"{result_dir}/best_model.keras")

        # Log training history to MLflow (per epoch metrics)
        print("Logging training history to MLflow...")
        for metric_name, metric_values in hist.history.items():
            for epoch, value in enumerate(metric_values):
                mlflow.log_metric(metric_name, float(value), step=epoch)

        # Evaluate the model on test set
        print("Evaluating model on test set...")
        metrics = model.evaluate(x_test, y_test, return_dict=True, verbose=0)
        loss = float(metrics["loss"])
        accuracy = float(metrics["accuracy"])
        recall = float(metrics["recall"])
        precision = float(metrics["precision"])

        # Log final test metrics to MLflow
        mlflow.log_metrics(
            {
                "test_loss": loss,
                "test_accuracy": accuracy,
                "test_recall": recall,
                "test_precision": precision,
            }
        )

        print("\nTest Results:")
        print(f"  Loss: {loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  Precision: {precision:.4f}\n")

        # Generate and save training curves plot
        plt.figure(figsize=(14, 5))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(hist.history["loss"], label="Training Loss", linewidth=2)
        plt.plot(hist.history["val_loss"], label="Validation Loss", linewidth=2)
        plt.title("Model Loss Over Epochs", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(hist.history["accuracy"], label="Training Accuracy", linewidth=2)
        plt.plot(hist.history["val_accuracy"], label="Validation Accuracy", linewidth=2)
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

        # Generate and save confusion matrix
        print("Generating confusion matrix...")
        y_pred = model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)  # Convert one-hot back to class indices

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=[i for i in range(NUM_CLASSES)]
        )
        disp.plot(cmap="Blues", values_format="d")
        plt.title("Confusion Matrix - MNIST Test Set", fontweight="bold")
        cm_path = f"{result_dir}/confusion_matrix.png"
        plt.savefig(cm_path, dpi=100, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(cm_path)

        # Measure inference time on single example
        print("Measuring inference time...")
        inference_time_ms = measure_inference_time(model, input_size, num_iterations=100)
        mlflow.log_metric("inference_time_ms", inference_time_ms)
        print(f"Average inference time: {inference_time_ms:.4f} ms (over 100 predictions)\n")

        # Log all local artifacts to MLflow
        mlflow.log_artifacts(result_dir, artifact_path="results")

        # Log the model to MLflow
        mlflow.keras.log_model(model, "best_model")

        print(f"\nTraining complete!")
        print(f"Results saved to: {result_dir}")
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}\n")


if __name__ == "__main__":
    main()
