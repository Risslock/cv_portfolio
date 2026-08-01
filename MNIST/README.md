# MNIST Digit Classification: ML & Software Engineering Showcase

A comprehensive machine learning project demonstrating expertise in neural network architectures, multiple deep learning frameworks, professional software development practices, and MLflow experiment tracking.

## Project Overview

This project implements digit classification using the MNIST dataset (28×28 grayscale images, 10 classes) with multiple framework and architecture combinations:

- **TensorFlow**: Fully Connected Neural Networks (FCNN) & Convolutional Neural Networks (CNN)
- **PyTorch**: Fully Connected Neural Networks (FCNN) & Convolutional Neural Networks (CNN)

Each implementation features:
✅ Production-grade OOP design  
✅ MLflow experiment tracking with standardized parameter/metric naming  
✅ Early stopping and model checkpointing  
✅ Comprehensive documentation and Jupyter notebooks  
✅ Professional code quality (PEP 8, linting, type hints)  

## Quick Start

### Prerequisites

- Python 3.10+
- UV (Python package manager) - [Install here](https://docs.astral.sh/uv/getting-started/installation/)

### Installation

1. **Navigate to the MNIST project**:
   ```bash
   cd MNIST
   ```

2. **Create and activate virtual environment**:
   ```bash
   uv sync
   ```

3. **Verify setup**:
   ```bash
   uv run python -c "import tensorflow; import torch; print('Setup successful!')"
   ```

### Running Models

Each framework/model combination has a training script:

```bash
# TensorFlow FCNN (baseline architecture)
uv run python tf_models/fcnn/train.py --epochs 10

# TensorFlow CNN (modern best practices)
uv run python tf_models/cnn/train.py --epochs 50 --batch_size 32

# PyTorch FCNN (custom training loop)
uv run python pt_models/fcnn/train.py --epochs 10

# PyTorch CNN (with early stopping)
uv run python pt_models/cnn/train.py --epochs 50 --batch_size 32
```

### Training Script Options

All training scripts support customizable hyperparameters:

```bash
uv run python tf_models/cnn/train.py \
  --epochs 60 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --num_conv_blocks 2 \
  --conv_filters_initial 32 \
  --dense_units 128 \
  --dropout_rate 0.5
```

### Viewing Experiment Results

After training, view all experiment metrics and artifacts:

```bash
mlflow ui --port 5000
```

Then navigate to http://localhost:5000 to compare:
- Experiment runs (TensorFlow vs PyTorch)
- Training metrics (loss, accuracy per epoch)
- Test metrics (precision, recall, F1)
- Model artifacts
- Training times and inference performance

### Exploring Notebooks

Interactive Jupyter notebooks demonstrate complete pipelines:

```bash
uv run jupyter notebook notebooks/
```

Available notebooks:
- **tensorflow_cnn_guide.ipynb** - Complete CNN walkthrough with data loading, model building, training, evaluation
- **pytorch_cnn_guide.ipynb** - Parallel PyTorch implementation showcasing framework differences

Each notebook:
- ✅ Executes end-to-end without errors (<2 minutes on CPU)
- ✅ Includes 8+ explanatory markdown cells
- ✅ Shows data loading, preprocessing, model definition, training, evaluation
- ✅ Integrates MLflow experiment tracking
- ✅ Demonstrates model persistence and loading

## Project Structure

```
MNIST/
├── tf_models/                   # TensorFlow implementations
│   ├── fcnn/                   # Fully Connected Neural Network
│   │   ├── __init__.py
│   │   ├── model.py            # Model class definition
│   │   ├── train.py            # Training with MLflow tracking
│   │   ├── data.py             # Data loading and preprocessing
│   │   ├── evaluate.py         # Evaluation metrics and visualization
│   │   └── README.md           # Framework-specific guide
│   │
│   └── cnn/                    # Convolutional Neural Network
│       ├── __init__.py
│       ├── models.py           # MNISTCNNModel class
│       ├── data.py             # MNIST loading (50k/10k/10k split)
│       ├── train.py            # Training with early stopping & MLflow
│       ├── evaluate.py         # Metrics & visualization
│       └── README.md           # Complete guide with MLflow info
│
├── pt_models/                   # PyTorch implementations
│   ├── fcnn/                   # Fully Connected Neural Network
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── README.md
│   │
│   └── cnn/                    # Convolutional Neural Network
│       ├── __init__.py
│       ├── models.py           # MNISTCNNModel class
│       ├── data.py             # DataLoader creation
│       ├── train.py            # Custom loop with early stopping & MLflow
│       ├── evaluate.py         # Metrics & visualization
│       └── README.md           # Complete guide with MLflow info
│
├── notebooks/                   # Jupyter learning guides
│   ├── tensorflow_cnn_guide.ipynb   # 8 sections, fully executable
│   ├── pytorch_cnn_guide.ipynb      # Parallel PyTorch guide
│   └── README.md                    # Notebook navigation guide
│
├── utils/                       # Shared utilities
│   ├── __init__.py
│   ├── metrics.py              # F1, precision, recall computation
│   └── mlflow_config.py        # MLflow setup & parameter logging
│
├── specs/                       # Feature specifications & design docs
│   └── 001-cnn-mnist-frameworks/
│       ├── spec.md             # Feature specification
│       ├── plan.md             # Implementation plan
│       ├── tasks.md            # Task breakdown
│       ├── data-model.md       # Data model & entities
│       ├── contracts/          # Interface contracts
│       └── research.md         # Technical research & decisions
│
├── data/                        # MNIST dataset (auto-downloaded)
│
├── results/                     # Training outputs & checkpoints
│   ├── tf_models_cnn/          # Best models & metrics
│   └── pt_models_cnn/
│
├── pyproject.toml              # Project dependencies (UV)
├── uv.lock                     # Locked dependency versions
├── .gitignore                  # Git ignore patterns
└── README.md                   # This file
```

## Key Features

### 1. MLflow Experiment Tracking (Principle VIII)
All training scripts automatically log to MLflow:

**Logged Parameters (11 total):**
- Hyperparameters: `learning_rate`, `batch_size`, `num_epochs`
- Model config: `num_conv_blocks`, `conv_filters_initial`, `dense_units`, `dropout_rate`
- Training setup: `optimizer`, `loss_function`, `random_seed`, `framework`

**Logged Metrics:**
- Per-epoch: `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy`
- Final test: `test_loss`, `test_accuracy`, `test_precision`, `test_recall`, `test_f1`
- Performance: `training_time_seconds`, `inference_time_ms_per_batch`

**Model Artifacts:**
- Saved as `tensorflow_cnn_model` or `pytorch_cnn_model`
- Fully reproducible from logged parameters
- Loadable via MLflow for inference

### 2. Early Stopping & Model Checkpointing

**TensorFlow:**
- Uses Keras `EarlyStopping` callback
- Monitors validation loss with 5-epoch patience
- Auto-saves best model to `results/tf_models_cnn/best_model.keras`
- Restores best weights automatically

**PyTorch:**
- Custom early stopping implementation
- Tracks best validation loss across epochs
- Saves best model state_dict to `results/pt_models_cnn/best_model.pth`
- Stops after 5 epochs without improvement

### 3. Object-Oriented Design (Principle I)
All code follows OOP principles:
- **CNN Models**: `MNISTCNNModel` class inheriting keras.Sequential (TF) or torch.nn.Module (PT)
- **Data Loading**: Modular functions returning normalized tensors/DataLoaders
- **Evaluation**: Reusable evaluation utilities for metrics and visualization
- **MLflow Config**: `MLflowConfig` class managing experiment tracking

### 4. Professional Code Quality (Principle V)
- **PEP 8 Compliance**: All code passes flake8 checks
- **Type Hints**: Fully annotated for clarity
- **Docstrings**: NumPy-style docstrings for all public functions
- **Snake_case Parameters**: Standardized naming convention (learning_rate, batch_size, etc.)

### 5. Comprehensive Documentation (Principle III & VI)

**Framework-Specific READMEs:**
- `tf_models/cnn/README.md` - TensorFlow CNN guide with MLflow integration details
- `pt_models/cnn/README.md` - PyTorch CNN guide with custom training loop explanation

**Jupyter Notebooks:**
- Interactive step-by-step walkthroughs
- Executable cells demonstrating full pipeline
- Clear explanations of CNN concepts and framework differences

**Docstrings:**
- Every function and class fully documented
- Parameter descriptions and return value specifications
- Usage examples for complex functions

### 6. Reproducible Experiments (Principle VIII & IV)

**Reproducibility Features:**
- Fixed random seed (seed=42) in all models
- Standardized data splits (50k train / 10k val / 10k test)
- All hyperparameters logged to MLflow
- Model artifacts saved for exact reproduction
- Identical architecture across frameworks

**To Reproduce:**
```python
import mlflow

# Load parameters and model from MLflow run
run_id = "abc123..."
model = mlflow.tensorflow.load_model(f"runs:/{run_id}/tensorflow_cnn_model")
# Predictions are deterministic given same input
```

## Model Architectures

### Fully Connected Neural Network (FCNN)
**Purpose**: Baseline demonstrating fundamental neural network concepts

- Input: 784 neurons (28×28 flattened)
- Hidden1: 128 neurons + ReLU
- Hidden2: 64 neurons + ReLU
- Output: 10 neurons + Softmax
- Expected Accuracy: ~97% on MNIST

### Convolutional Neural Network (CNN)
**Purpose**: Demonstrates spatial feature learning and modern best practices

**Architecture:**
- Conv Block 1: Conv2D(32, 3×3, ReLU) → MaxPool(2×2) [28×28→14×14]
- Conv Block 2: Conv2D(64, 3×3, ReLU) → MaxPool(2×2) [14×14→7×7]
- Flatten: 64×7×7 = 3,136 features
- Dense1: 128 neurons + ReLU + Dropout(0.5)
- Output: 10 neurons + Softmax

**Expected Performance:**
- Test Accuracy: ≥98%
- Training Time: ~1 min/epoch on CPU
- Inference: <50ms per batch

### Framework Comparison

| Aspect | TensorFlow | PyTorch |
|--------|-----------|---------|
| **Model Definition** | keras.Sequential | torch.nn.Module |
| **Training Loop** | model.fit() (high-level) | Custom loop (explicit) |
| **Data Format** | NHWC (channels-last) | NCHW (channels-first) |
| **Output** | Softmax probabilities | Logits + loss softmax |
| **Use Case** | Production pipelines | Research & experimentation |
| **Learning Curve** | Beginner-friendly | More control required |

## Training Pipeline

### Step-by-Step Process

1. **Data Loading** (auto-download if needed)
   - MNIST: 60k training, 10k test
   - Split: 50k train, 10k validation, 10k test
   - Normalization: [0, 1] range

2. **Model Initialization**
   - Load model class
   - Set seed=42 for reproducibility
   - Compile (TF) or prepare optimizer (PT)

3. **MLflow Setup**
   - Create experiment: `cnn_mnist_tensorflow` or `cnn_mnist_pytorch`
   - Log 11 standardized parameters

4. **Training with Early Stopping**
   - Epoch-level training loop
   - Log metrics per epoch (train_loss, train_accuracy, val_loss, val_accuracy)
   - Monitor validation loss
   - Save best model checkpoint
   - Stop if no improvement for 5 epochs

5. **Evaluation**
   - Compute test metrics (accuracy, precision, recall, F1)
   - Measure inference time
   - Log final metrics to MLflow

6. **Artifact Logging**
   - Save trained model to MLflow
   - Store all parameters and metrics
   - Enable full reproducibility

## Expected Results

### Performance Metrics
- **FCNN Accuracy**: ~97%
- **CNN Accuracy**: ≥98%
- **Training Time**: 2-3 minutes (50 epochs) on CPU
- **Inference Time**: <50ms per batch

### Comparison
Running both frameworks shows:
- CNN consistently outperforms FCNN (~1-2% improvement)
- PyTorch often trains slightly faster than TensorFlow on CPU
- Both achieve highly reproducible results with seed=42

## Dependencies

Managed via UV in `pyproject.toml`:

**Core ML Frameworks:**
- tensorflow 2.13+ - TensorFlow framework
- torch 2.0+ - PyTorch framework
- torchvision - PyTorch computer vision utilities

**Data & Math:**
- numpy - Numerical computations
- scikit-learn - Metrics, preprocessing

**Experiment Tracking:**
- mlflow 2.0+ - Experiment tracking and model registry

**Notebooks & Visualization:**
- jupyter - Notebook environment
- matplotlib - Plotting library

**Development:**
- flake8 - Code linting
- pylint - Advanced linting

## Usage Examples

### Training a Model

```bash
# Quick 10-epoch test run
uv run python tf_models/cnn/train.py --epochs 10

# Full 60-epoch training with custom batch size
uv run python pt_models/cnn/train.py --epochs 60 --batch_size 64
```

### Comparing Experiments

```bash
# View all experiments in MLflow UI
mlflow ui --port 5000

# Open browser to http://localhost:5000
# Compare TensorFlow vs PyTorch performance
# View training curves and final metrics
```

### Loading a Trained Model

```python
import mlflow

# Load from MLflow
model = mlflow.tensorflow.load_model("runs:/{run_id}/tensorflow_cnn_model")

# Make predictions
predictions = model.predict(test_data)
```

## Constitutional Principles

This project follows the MNIST Showcase Constitution (`.specify/memory/constitution.md`):

✅ **Principle I**: Object-Oriented Design - All code uses classes with clear interfaces  
✅ **Principle II**: Folder Structure Clarity - Framework-first organization with clear hierarchy  
✅ **Principle III**: Usage Clarity - Complete docstrings and usage examples  
✅ **Principle IV**: Jupyter Notebooks - Learning guides with 8+ explanatory cells each  
✅ **Principle V**: Code Quality - PEP 8 compliant, fully documented, type-hinted  
✅ **Principle VI**: Documentation - Comprehensive READMEs at all levels  
✅ **Principle VII**: UV Management - All dependencies in pyproject.toml  
✅ **Principle VIII**: MLflow Tracking - Standardized experiment tracking with naming conventions  

## For Each Framework

- **tf_models/cnn/README.md** - TensorFlow CNN complete guide
- **pt_models/cnn/README.md** - PyTorch CNN complete guide
- **notebooks/README.md** - Notebook navigation and learning objectives

## Status

✅ **Complete & Ready for Use**

- All architectures implemented (FCNN, CNN for both frameworks)
- MLflow tracking integrated and tested
- Early stopping and model checkpointing working
- Comprehensive documentation and notebooks
- Code passes linting and style checks
- Experiments fully reproducible

## Getting Started

1. Install dependencies: `uv sync`
2. Run a quick experiment: `uv run python tf_models/cnn/train.py --epochs 10`
3. View results: `mlflow ui --port 5000`
4. Explore notebooks: `uv run jupyter notebook notebooks/`

---

**Created**: 2026-07-31  
**Last Updated**: 2026-08-01  
**Status**: Production Ready  
**Python Version**: 3.10+  
**Lead Framework**: TensorFlow & PyTorch  
**Model Architectures**: FCNN, CNN  
