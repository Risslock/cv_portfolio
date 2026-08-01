# PyTorch Convolutional Neural Network for MNIST

A production-grade implementation of a Convolutional Neural Network for MNIST digit classification using PyTorch with MLflow experiment tracking for reproducible, comparable results.

## Architecture

**Model Structure** (2 convolutional blocks):
- Conv Block 1: Conv2d(1→32 filters, 3×3 kernel, ReLU) → MaxPool(2×2)
  - Input: (batch, 1, 28, 28) → Output: (batch, 32, 14, 14)
- Conv Block 2: Conv2d(32→64 filters, 3×3 kernel, ReLU) → MaxPool(2×2)
  - Input: (batch, 32, 14, 14) → Output: (batch, 64, 7, 7)
- Dense Layers:
  - Flatten (3136 features) → Linear(128, ReLU) + Dropout(0.5) → Linear(10)
  - Output: (batch, 10) logits (no softmax; applied in loss function)

**Key Differences from TensorFlow:**
- Inputs use NCHW format (channels-first) vs TensorFlow's NHWC (channels-last)
- Outputs logits instead of softmax probabilities
- Loss function (CrossEntropyLoss) applies softmax internally

**Why This Architecture?**
- Two conv blocks extract features at multiple scales
- Pooling reduces spatial dimensions while retaining features
- Dropout prevents overfitting
- Logit output enables efficient loss computation (CrossEntropyLoss)

## Components

### `models.py`
Defines the `MNISTCNNModel` class (inherits torch.nn.Module):
```python
model = MNISTCNNModel(
    input_channels=1,
    num_classes=10,
    num_conv_blocks=2,
    conv_filters_initial=32,
    dense_units=128,
    dropout_rate=0.5,
    seed=42  # For reproducibility
)
```

### `data.py`
Data loading with PyTorch DataLoaders:
```python
train_loader, val_loader, test_loader = load_mnist(batch_size=32)
```
- Automatically downloads MNIST if needed
- Normalizes to [0, 1] range (torchvision.transforms.ToTensor)
- Creates DataLoaders for efficient batching
- Splits into 50k train, 10k validation, 10k test

### `train.py`
Complete training pipeline with MLflow tracking:
```bash
uv run python pytorch/cnn/train.py --epochs 60 --batch-size 32 --learning-rate 0.001
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
- `infer_batch(model, batch_data, device)` → Performs inference on images

## Training

### Quick Start (Default Parameters)
```bash
cd ../../  # Go to project root
uv run python pytorch/cnn/train.py
```

### Full Training (60 Epochs)
```bash
uv run python pytorch/cnn/train.py --epochs 60
```

### GPU Training (if available)
The script auto-detects GPU. Speedup: ~5-10x vs CPU

### Expected Results
- **Test Accuracy**: ≥98% (validates CNN architecture effectiveness)
- **Training Time**: ~60 seconds per epoch on CPU (~10 sec on GPU)
- **Total Training**: ~60 minutes for 60 epochs on CPU
- **Final Test Metrics**: Logged to MLflow with detailed breakdown

## MLflow Experiment Tracking

All training automatically logs to MLflow. View results:

```bash
mlflow ui --port 5000
```

Then navigate to http://localhost:5000 and select experiment `cnn_mnist_pytorch`

**Logged Parameters** (11 total):
- learning_rate, batch_size, num_epochs
- optimizer, loss_function
- num_conv_blocks, conv_filters_initial, dense_units, dropout_rate
- random_seed (always 42), framework ("pytorch")

**Logged Metrics (Per Epoch)**:
- train_loss, train_accuracy, val_loss, val_accuracy

**Final Metrics**:
- test_loss, test_accuracy, test_precision, test_recall, test_f1
- training_time_seconds, inference_time_ms_per_batch

**Model Artifact**:
- Saved as `pytorch_cnn_model` for later retrieval

## Reproducibility

Training is fully reproducible via:
1. Fixed random seed (seed=42) in model initialization
2. All hyperparameters logged to MLflow
3. Deterministic data splits (50k/10k/10k using torch.Generator)
4. Model artifact saved for exact reproduction

**To reproduce a run:**
```python
import torch
import mlflow

# Load model from MLflow
model = mlflow.pytorch.load_model("runs:/{run_id}/pytorch_cnn_model")

# Get parameters from MLflow UI
# Re-run with identical hyperparameters
```

## Comparison with TensorFlow

See `../tensorflow/cnn/README.md` for TensorFlow implementation with identical architecture.

**Framework Comparison via MLflow:**
1. Run both `tensorflow/cnn/train.py` and `pytorch/cnn/train.py`
2. Open MLflow UI: `mlflow ui --port 5000`
3. Compare experiments: `cnn_mnist_tensorflow` vs `cnn_mnist_pytorch`
4. Analyze: accuracy, training speed, inference time, model sizes

## Learning Guide

Interactive Jupyter notebook: `notebooks/pytorch_cnn_guide.ipynb`

Covers:
- MNIST data loading with torchvision
- Custom model definition with torch.nn.Module
- Custom training loop (forward, backward, step)
- Evaluation with torch.no_grad() context
- MLflow integration
- Inference patterns

Run:
```bash
cd ../..  # Go to project root
uv run jupyter notebook notebooks/pytorch_cnn_guide.ipynb
```

## PyTorch-Specific Patterns

### Model Definition
```python
class MNISTCNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # ... more layers
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        # ... forward pass
        return x
```

### Training Loop
```python
model.train()
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

### Inference
```python
model.eval()
with torch.no_grad():
    logits = model(test_images)
    predictions = torch.argmax(logits, dim=1)
```

## Key Differences from FCNN

| Aspect | FCNN | CNN |
|--------|------|-----|
| **Spatial Info** | Lost (flattened) | Preserved via convolution |
| **Feature Learning** | Manual | Automatic |
| **Parameters** | Higher | Lower |
| **Test Accuracy** | ~97% | ≥98% |
| **Training Speed** | Slower | Faster |

## References

- [PyTorch Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
- [PyTorch nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [CNN Fundamentals (CS231n)](http://cs231n.github.io/convolutional-networks/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)

---

**Created:** 2026-08-01  
**Constitutional Principle:** VIII (Experiment Tracking & Reproducibility)  
**Framework:** PyTorch 2.0+
