# Fashion MNIST: CNN Image Classification

A TensorFlow/Keras convolutional neural network project for classifying clothing articles from the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset. Built to demonstrate practical deep learning workflow: from an exploratory notebook to a production-style training pipeline with GPU-accelerated augmentation, configurable architectures, and MLflow experiment tracking.

## Objective

Implement and train a CNN to classify 10 categories of clothing (28x28 grayscale images), showcasing:
- Idiomatic TensorFlow/Keras model building (Functional API)
- GPU-accelerated image augmentation as part of the model graph
- Configurable, reproducible training runs via CLI hyperparameters
- Experiment tracking and artifact logging with MLflow
- Rigorous evaluation using scikit-learn (precision, recall, F1, confusion matrix)

## Labels

| Label | Description   |
|-------|---------------|
| 0     | T-shirt/top   |
| 1     | Trouser       |
| 2     | Pullover      |
| 3     | Dress         |
| 4     | Coat          |
| 5     | Sandal        |
| 6     | Shirt         |
| 7     | Sneaker       |
| 8     | Bag           |
| 9     | Ankle boot    |

## Tech Stack

- **TensorFlow / tf.keras** — model definition, training, GPU-side image augmentation
- **scikit-learn** — precision, recall, F1, confusion matrix
- **MLflow** — experiment tracking, parameter/metric logging, model artifacts (local SQLite backend)
- **matplotlib** — training curves and confusion matrix visualization
- **uv** — dependency and environment management

## Project Structure

```
fashion_MNIST/
├── src/
│   └── fashion_mnist/
│       ├── __init__.py
│       ├── data.py           # load, split, and batch the Fashion MNIST dataset
│       ├── augmentation.py   # GPU-side Keras augmentation layers
│       ├── model.py          # parametrized CNN builder (core_fashion_model)
│       ├── train.py          # CLI entry point: training + MLflow logging
│       ├── evaluate.py       # sklearn metrics, confusion matrix, plots
│       └── mlflow_utils.py   # experiment setup, param/metric logging helpers
├── notebooks/
│   └── fashion_mnist.ipynb   # exploratory notebook — step-by-step walkthrough
├── results/                  # saved models, confusion matrix plots, run outputs
├── data/                     # cached dataset (gitignored)
├── mlruns/ + mlflow.db       # MLflow tracking store (gitignored)
├── pyproject.toml
├── uv.lock
└── README.md
```

## Approach

This project was built in two phases:

1. **Notebook exploration** (`notebooks/fashion_mnist.ipynb`) — an unhurried, step-by-step build: data loading, augmentation pipeline, a configurable CNN builder, and a first training run. This notebook is the reference design; it's kept as-is rather than rewritten.
2. **Production pipeline** (`src/fashion_mnist/`) — the notebook's ideas promoted into reusable, CLI-driven scripts with proper experiment tracking and evaluation, so different architectures/hyperparameters can be run and compared without editing code.

### GPU-accelerated augmentation

Image augmentation (`RandomFlip`, `RandomRotation`, `RandomBrightness`, `RandomZoom`, `RandomTranslation`) is implemented as Keras preprocessing layers wired directly into the model graph (ahead of the core CNN), rather than as a `tf.data` map step. This means augmentation runs on the GPU as part of the forward pass during `model.fit()`, and is automatically skipped at inference/evaluation time (Keras preprocessing layers are no-ops outside of training).

### Configurable architecture

The CNN builder supports tuning the number of conv blocks, number of dense blocks, kernel size, and the global pooling strategy (`max`, `avg`, or `flatten`), so different capacity/regularization trade-offs can be explored from the CLI without touching code.

## Getting Started

### Installation

```bash
cd fashion_MNIST
uv sync
```

### Training

```bash
uv run python -m fashion_mnist.train \
  --epochs 100 \
  --batch-size 512 \
  --learning-rate 0.001 \
  --n-conv 4 \
  --n-dense 2 \
  --global-pooling-type flatten \
  --horizontal-flip \
  --rotation 0.1 \
  --brightness 0.001 \
  --zoom 0.1 \
  --translation 0.1
```

Each run:
- Trains with `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint` (best validation loss)
- Logs hyperparameters, per-epoch metrics, and final test metrics to MLflow
- Evaluates on the held-out test set with scikit-learn (precision/recall/F1, confusion matrix) and logs the confusion matrix plot as an MLflow artifact
- Saves the best model to `results/`

### Viewing Experiments

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open http://localhost:5000 to compare runs, training curves, and confusion matrices.

### Standalone Evaluation

```bash
uv run python -m fashion_mnist.evaluate --model-path results/best_model.keras
```

### Exploring the Notebook

```bash
uv run jupyter notebook notebooks/fashion_mnist.ipynb
```

## GPU Notes

TensorFlow ≥2.11 has no native GPU support on Windows — it silently falls back to CPU even with CUDA/cuDNN installed. To actually exercise the GPU-accelerated augmentation path on Windows hardware, run this project inside **WSL2** (with CUDA installed there) or a Linux/Colab environment. `configure_gpu()` and the augmentation-in-model design work unchanged either way; only the runtime environment differs.

## Data Splits

- 54,000 train / 6,000 validation / 10,000 test (standard Fashion MNIST test set, 10% of train held out for validation)
- Pixel values normalized to [0, 1]
- Fixed random seed for reproducibility

## Status

🚧 **In Progress** — notebook prototype complete (~88.6% test accuracy); production pipeline (`src/fashion_mnist/`) in development.
