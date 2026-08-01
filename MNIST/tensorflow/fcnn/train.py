import argparse
import datetime
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Import TensorFlow EARLY (before local model import)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.random import set_seed

# Setup path for utils imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from utils.mlflow_config import MLflowConfig
from utils.metrics import calculate_metrics

# Now import local model (TensorFlow already loaded)
from model import TF_FCNN

FRAMEWORK = "tensorflow"
ARCHITECTURE = "fcnn"
MLFLOW_TRACKING_URI = "sqlite:///./mlflow.db"
MLFLOW_EXPERIMENT_NAME = f"mnist_{FRAMEWORK}_{ARCHITECTURE}"
RANDOM_SEED = 42
NUM_CLASSES = 10
NUM_CONV_BLOCKS = 0
CONV_FILTERS_INITIAL = 0
DENSE_UNITS = 128
DROPOUT_RATE = 0.2
LOSS_FUNCTION = "categorical_crossentropy"


parser = argparse.ArgumentParser(
    description="Train a FCNN model on MNIST dataset with MLflow tracking"
)
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
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


def measure_inference_time(
    model, input_size: int, num_iterations: int = 100
) -> float:
    """Measure average inference time in milliseconds per batch."""
    dummy_input = np.zeros((1, input_size), dtype=np.float32)
    _ = model(dummy_input, training=False)

    start_time = time.time()
    for _ in range(num_iterations):
        _ = model(dummy_input, training=False)
    end_time = time.time()

    total_time = (end_time - start_time) * 1000
    avg_time_ms = total_time / num_iterations
    return avg_time_ms


def load_mnist_data():
    """Load and preprocess MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)


def main():
    """Train FCNN with standardized MLflow tracking."""
    np.random.seed(RANDOM_SEED)
    set_seed(RANDOM_SEED)

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
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    print("Data Loaded:")
    print(f"  Training: shape={x_train.shape}, labels={y_train.shape}")
    print(f"  Testing:  shape={x_test.shape}, labels={y_test.shape}")

    input_size = x_train.shape[1]
    print(f"  Input size: {input_size}, Classes: {NUM_CLASSES}\n")

    # Create and compile model
    input_shape = (input_size,)
    model = TF_FCNN(input_shape=input_shape, num_classes=NUM_CLASSES)

    if args.optimizer.lower() == "adam":
        optimizer = Adam(learning_rate=args.learning_rate)
    elif args.optimizer.lower() == "sgd":
        optimizer = SGD(learning_rate=args.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", "recall", "precision"],
    )
    print("Model compiled successfully.\n")

    dummy_input = np.zeros((1, input_size))
    _ = model(dummy_input, training=False)
    print("Model initialized.\n")
    print("Model Architecture:")
    model.summary()
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

        # Setup callbacks
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        model_checkpoint = ModelCheckpoint(
            f"{result_dir}/best_model.keras",
            monitor="val_loss",
            save_best_only=True,
        )

        # Train
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

        model.load_weights(f"{result_dir}/best_model.keras")

        # Log per-epoch metrics
        print("Logging training history to MLflow...")
        for epoch in range(len(hist.history["loss"])):
            mlflow_cfg.log_epoch_metrics(
                epoch=epoch,
                train_loss=hist.history["loss"][epoch],
                train_accuracy=hist.history["accuracy"][epoch],
                val_loss=hist.history["val_loss"][epoch],
                val_accuracy=hist.history["val_accuracy"][epoch],
            )

        # Evaluate
        print("Evaluating model on test set...")
        y_pred = model.predict(x_test, verbose=0)
        y_true = np.argmax(y_test, axis=1)
        y_pred_class = np.argmax(y_pred, axis=1)

        eval_results = model.evaluate(x_test, y_test, return_dict=True, verbose=0)
        test_loss = float(eval_results["loss"])
        test_accuracy = float(eval_results["accuracy"])

        test_precision, test_recall, test_f1 = calculate_metrics(
            y_true, y_pred_class
        )

        print("\nTest Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_accuracy:.4f}")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall: {test_recall:.4f}")
        print(f"  F1-Score: {test_f1:.4f}\n")

        # Measure inference time
        print("Measuring inference time...")
        inference_time_ms = measure_inference_time(model, input_size, num_iterations=100)
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
        plt.plot(hist.history["loss"], label="Training Loss", linewidth=2)
        plt.plot(hist.history["val_loss"], label="Validation Loss", linewidth=2)
        plt.title("Model Loss Over Epochs", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(hist.history["accuracy"], label="Training Accuracy", linewidth=2)
        plt.plot(
            hist.history["val_accuracy"], label="Validation Accuracy", linewidth=2
        )
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
        cm = confusion_matrix(y_true, y_pred_class)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=[i for i in range(NUM_CLASSES)]
        )
        disp.plot(cmap="Blues", values_format="d")
        plt.title("Confusion Matrix - MNIST Test Set", fontweight="bold")
        cm_path = f"{result_dir}/confusion_matrix.png"
        plt.savefig(cm_path, dpi=100, bbox_inches="tight")
        plt.close()
        mlflow_cfg.log_artifact(cm_path)

        # Log artifacts
        mlflow_cfg.log_artifact(result_dir)

        print(f"\nTraining complete!")
        print(f"Results saved to: {result_dir}\n")

    finally:
        mlflow_cfg.end_run()


if __name__ == "__main__":
    main()
