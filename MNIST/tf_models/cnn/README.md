# TensorFlow Convolutional Neural Network for MNIST

A production-grade implementation of a Convolutional Neural Network for MNIST digit classification using TensorFlow/Keras with MLflow experiment tracking for reproducible, comparable results.

## Architecture

**Model Structure** (2 convolutional blocks):
- Conv Block 1: Conv2D(32 filters, 3×3 kernel, ReLU) → MaxPool(2×2)
  - Input: (batch, 28, 28, 1) → Output: (batch, 14, 14, 32)
- Conv Block 2: Conv2D(64 filters, 3×3 kernel, ReLU) → MaxPool(2×2)
  - Input: (batch, 14, 14, 32) → Output: (batch, 7, 7, 64)
- Dense Layers:
  - Flatten → Dense(128, ReLU) + Dropout(0.5) → Dense(10, Softmax)
  - Output: (batch, 10) probability distribution over digits

**Why This Architecture?**
- Two conv blocks extract features at multiple scales (low-level edges, high-level shapes)
- Pooling reduces spatial dimensions while retaining important information
- Dropout prevents overfitting on relatively small dataset (60k training samples)
- Softmax output enables probability-based inference and multi-class loss

## Components

### `models.py`
Defines the `MNISTCNNModel` class (inherits keras.Sequential):
```python
model = MNISTCNNModel(
    input_shape=(28, 28, 1),
    num_classes=10,
    num_conv_blocks=2,
    conv_filters_initial=32,
    dense_units=128,
    dropout_rate=0.5,
    seed=42  # For reproducibility
)
```

### `data.py`
Data loading and preprocessing:
```python
(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist()
```
- Automatically downloads MNIST if needed
- Normalizes to [0, 1] range
- Splits into 50k train, 10k validation, 10k test
- One-hot encodes labels

### `train.py`
Complete training pipeline with MLflow tracking:
```bash
uv run python tensorflow/cnn/train.py --epochs 60 --batch-size 32 --learning-rate 0.001
```

Options:
- `--epochs`: Number of training epochs (default: 60)
- `--batch-size`: Batch size for training (default: 32)
- `--learning-rate`: Adam learning rate (default: 0.001)
- `--num-conv-blocks`: Convolutional blocks (default: 2)
- `--conv-filters-initial`: Initial filter count (default: 32)
- `--dense-units`: Dense layer units (default: 128)
- `--dropout-rate`: Dropout probability (default: 0.5)
- `--random-seed`: Random seed (default: 42, MUST BE 42 for reproducibility)

### `evaluate.py`
Evaluation utilities:
- `compute_metrics(y_true, y_pred)` → (accuracy, precision, recall, f1)
- `plot_history(history)` → Visualizes loss/accuracy curves
- `infer_batch(model, batch_data)` → Performs inference on images

## Training

### Quick Start (Default Parameters)
```bash
cd ../../  # Go to project root
uv run python tensorflow/cnn/train.py
```

### Full Training (60 Epochs)
```bash
uv run python tensorflow/cnn/train.py --epochs 60
```

### Expected Results
- **Test Accuracy**: ≥98% (validates CNN architecture effectiveness)
- **Training Time**: ~60 seconds per epoch on CPU
- **Total Training**: ~60 minutes for 60 epochs
- **Final Test Metrics**: Logged to MLflow with frame details

## MLflow Experiment Tracking

All training automatically logs to MLflow. View results:

```bash
mlflow ui --port 5000
```

Then navigate to http://localhost:5000 and select experiment `cnn_mnist_tensorflow`

**Logged Parameters** (11 total):
- learning_rate, batch_size, num_epochs
- optimizer, loss_function
- num_conv_blocks, conv_filters_initial, dense_units, dropout_rate
- random_seed (always 42), framework ("tensorflow")

**Logged Metrics (Per Epoch)**:
- train_loss, train_accuracy, val_loss, val_accuracy

**Final Metrics**:
- test_loss, test_accuracy, test_precision, test_recall, test_f1
- training_time_seconds, inference_time_ms_per_batch

**Model Artifact**:
- Saved as `tensorflow_cnn_model` for later retrieval

## Reproducibility

Training is fully reproducible via:
1. Fixed random seed (seed=42) in model initialization
2. All hyperparameters logged to MLflow
3. Deterministic data splits (50k/10k/10k)
4. Model artifact saved for exact reproduction

**To reproduce a run:**
```python
import mlflow

# Load model from MLflow
model = mlflow.tensorflow.load_model("runs:/{run_id}/tensorflow_cnn_model")

# Get parameters from MLflow UI
# Re-run with identical hyperparameters
```

## Comparison with PyTorch

See `../pytorch/cnn/README.md` for PyTorch implementation with identical architecture.

**Framework Comparison via MLflow:**
1. Run both `tensorflow/cnn/train.py` and `pytorch/cnn/train.py`
2. Open MLflow UI: `mlflow ui --port 5000`
3. Compare experiments: `cnn_mnist_tensorflow` vs `cnn_mnist_pytorch`
4. Analyze: accuracy, training speed, inference time

## Learning Guide

Interactive Jupyter notebook: `notebooks/tensorflow_cnn_guide.ipynb`

Covers:
- Data loading and visualization
- Model architecture explanation
- Training with MLflow logging
- Evaluation and metrics computation
- Model persistence

Run:
```bash
cd ../..  # Go to project root
uv run jupyter notebook notebooks/tensorflow_cnn_guide.ipynb
```

## Key Differences from FCNN

| Aspect | FCNN | CNN |
|--------|------|-----|
| **Spatial Info** | Lost (flattened) | Preserved via convolution |
| **Feature Learning** | Manual or FC-based | Automatic via filters |
| **Parameters** | Higher (~1M for MNIST) | Lower (~1-2M) |
| **Test Accuracy** | ~97% | ≥98% |
| **Training Speed** | Slower | Faster |

## References

- [TensorFlow Keras Sequential API](https://www.tensorflow.org/api/keras/Sequential)
- [Conv2D Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D)
- [MaxPooling2D Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPooling2D)
- [CNN Fundamentals (CS231n)](http://cs231n.github.io/convolutional-networks/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)

---

**Created:** 2026-08-01  
**Constitutional Principle:** VIII (Experiment Tracking & Reproducibility)  
**Framework:** TensorFlow 2.13+
