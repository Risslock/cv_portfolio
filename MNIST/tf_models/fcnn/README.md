# TensorFlow Fully Connected Neural Network (FCNN)

A production-grade implementation of a fully connected neural network for MNIST digit classification using TensorFlow/Keras with comprehensive MLflow experiment tracking.

## Architecture

**Model Structure**:
- Input Layer: 784 neurons (28×28 flattened images)
- Hidden Layer 1: 128 neurons + ReLU activation
- Hidden Layer 2: 64 neurons + ReLU activation
- Output Layer: 10 neurons + Softmax activation

**Total Parameters**: 109,386 trainable parameters (~427 KB)

**Regularization**:
- L2 regularization (0.01) on all dense layers for weight penalty
- Early stopping on validation loss (patience: 5 epochs) to prevent overfitting

**Optimization**:
- Optimizer: Adam (default) or SGD (configurable)
- Loss Function: Categorical Crossentropy (one-hot encoded labels)
- Metrics: Accuracy, Precision, Recall

## Key Components

### `model.py`
Defines the `TF_FCNN` subclassed Model that encapsulates the fully connected architecture:
- Implements `__init__()`, `call()`, `build()`, `get_config()`, and `from_config()`
- Full docstring documentation with examples
- Serializable for MLflow model logging
- Input validation for robust error handling

### `train.py`
Complete training pipeline including:
- MNIST dataset loading with one-hot label encoding
- Dynamic input size detection from data (not hardcoded)
- Model instantiation with configurable optimizer
- Reproducible training with fixed random seeds (RANDOM_SEED=42)
- MLflow experiment tracking with per-epoch metric logging
- Dual model saving: best model (by validation accuracy) and final model
- Test set evaluation with multiple metrics
- Inference time benchmarking (100 predictions, average in ms)
- Visualization generation (training curves, confusion matrix)
- Timestamped results directory for organization

## Training

### Basic Usage

To train with default settings (Adam optimizer, 20 epochs):

```bash
python train.py
```

### Configurable Options

```bash
# Using SGD optimizer
python train.py --optimizer sgd --learning_rate 0.01 --epochs 20

# Quick test (5 epochs)
python train.py --epochs 5 --batch_size 32

# Full configuration
python train.py --epochs 30 --batch_size 64 --learning_rate 0.0005 --optimizer adam
```

**Command Line Arguments**:
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size for training (default: 32)
- `--learning_rate`: Learning rate for optimizer (default: 0.001)
- `--optimizer`: Choice of `adam` or `sgd` (default: adam)

### Training Flow

Training will automatically:
1. Load and preprocess MNIST data (60K train, 10K test)
2. Create timestamped results directory
3. Initialize model layers with dummy forward pass
4. Display model architecture summary
5. Create MLflow experiment run and log tags
6. Train with callbacks (early stopping, model checkpointing)
7. Load best model and evaluate on test set
8. Log training history per epoch to MLflow
9. Measure inference time (100 predictions)
10. Generate visualizations and confusion matrix
11. Log all artifacts to MLflow and local directory

## MLflow Experiment Tracking

All training runs are tracked in MLflow under experiment `mnist_TensorFlow_FCNN`.

### Tagged Experiments

Each run is tagged with:
- `framework`: TensorFlow
- `architecture`: FCNN
- `project`: mnist_digits

### Metrics Logged Per Epoch

Training automatically logs:
- `loss` - Training loss per epoch
- `accuracy` - Training accuracy per epoch
- `precision` - Training precision per epoch
- `recall` - Training recall per epoch
- `val_loss` - Validation loss per epoch
- `val_accuracy` - Validation accuracy per epoch
- `val_precision` - Validation precision per epoch
- `val_recall` - Validation recall per epoch
- `test_loss` - Final test set loss
- `test_accuracy` - Final test set accuracy
- `test_precision` - Final test set precision
- `test_recall` - Final test set recall
- `inference_time_ms` - Average inference time in milliseconds

### Hyperparameters Logged

All command-line arguments are logged:
- `epochs`, `batch_size`, `learning_rate`, `optimizer`
- `random_seed` (for reproducibility)
- `input_size` (derived from data)
- `num_classes` (10 for MNIST)

### Artifacts Logged

Each training run saves:
- **best_model.keras** - Best model (Keras native format)
- **confusion_matrix.png** - Test set confusion matrix heatmap
- **training_curves.png** - Loss and accuracy plots over epochs
- All artifacts logged to MLflow with prefix `results/`

### Accessing MLflow Results

View all experiments and compare runs:

```bash
python -m mlflow ui --host 127.0.0.1 --port 5000
```

Navigate to `http://127.0.0.1:5000` to:
- View all training runs
- Compare metrics across runs
- Filter by tags (framework, architecture)
- Download logged models
- Analyze hyperparameter impact

## Actual Performance

Performance on MNIST test set:
- **Test Accuracy**: ~94.4%
- **Test Precision**: ~95.7%
- **Test Recall**: ~93.1%
- **Test Loss**: ~0.36
- **Inference Time**: ~1-2 ms per prediction
- **Training Time**: ~3-5 seconds per epoch (CPU)
- **Model Size**: ~427 KB (weights only)

## Output Structure

After each training, a timestamped results directory is created:

```
results/tensorflow_fcnn/YYYY-MM-DD_HH-MM-SS/
├── best_model.keras          # Best model weights
├── confusion_matrix.png       # Test confusion matrix
└── training_curves.png        # Loss & accuracy plots
```

Results are automatically organized by date/time for easy tracking of multiple runs.

## Hyperparameter Experimentation

To compare different configurations:

```bash
# Run 1: Adam optimizer
python train.py --optimizer adam --learning_rate 0.001 --epochs 20

# Run 2: SGD optimizer  
python train.py --optimizer sgd --learning_rate 0.01 --epochs 20

# Run 3: Higher learning rate
python train.py --optimizer adam --learning_rate 0.005 --epochs 20
```

View all runs in MLflow UI and compare side-by-side to identify best configuration.

## Code Quality & Standards

This implementation adheres to the MNIST Constitution:

**Object-Oriented Design**:
- Model implemented as reusable `TF_FCNN` class
- Clear separation of concerns (model.py, train.py)
- Configurable and extensible architecture

**Documentation**:
- Complete docstrings for all public methods (NumPy style)
- Type hints on function signatures
- Inline comments explaining non-obvious logic
- README with setup, usage, and results

**Code Standards**:
- PEP 8 compliant formatting
- Passes flake8 linting checks
- Clear variable and function naming
- No procedural scripts (pure OOP)

**Reproducibility**:
- Fixed random seeds (RANDOM_SEED=42)
- All hyperparameters logged to MLflow
- Deterministic data preprocessing
- Version-controlled model architecture

## Jupyter Notebook

See `../notebooks/tensorflow_fcnn_walkthrough.ipynb` for an interactive walkthrough of:
1. Data loading and preprocessing
2. Model architecture building  
3. Training process with real-time metric tracking
4. Model evaluation and metrics
5. Visualization generation
6. MLflow experiment browsing

## Troubleshooting

**Shape mismatch errors during training**: 
- Ensure data is properly one-hot encoded (used `to_categorical`)
- Use `categorical_crossentropy` loss with one-hot labels
- Dummy forward pass initializes layers before training

**Slow training on Windows**:
- TensorFlow GPU is not supported natively on Windows
- Consider WSL2 or TensorFlow-DirectML for GPU acceleration
- CPU training is still reasonable for MNIST (~3s per epoch)

**MLflow dependency warnings**:
- Safe to ignore - environmental version mismatches only
- Model trains and logs successfully despite warnings
- Warnings are MLflow being cautious about reproducibility

## References

- TensorFlow Documentation: https://www.tensorflow.org/api_docs
- Keras Sequential Models: https://keras.io/guides/functional_api/
- Subclassing Model Guide: https://keras.io/guides/making_new_layers_and_models_via_subclassing/
- MLflow Documentation: https://mlflow.org/docs/
- MNIST Dataset: http://yann.lecun.com/exdb/mnist/
