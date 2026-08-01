# Contract: CNN Model Interface

**Phase**: 1 (Design & Contracts)
**Date**: 2026-08-01
**Feature**: CNN MNIST implementation with MLflow tracking

## Purpose

This contract defines the interface that both TensorFlow and PyTorch CNN models MUST implement. Ensures consistent behavior across frameworks, enabling fair comparison and interchangeable usage patterns.

## Model Interface Specification

### TensorFlow CNN Model

**Class**: Inherits from `tensorflow.keras.Sequential` or `tensorflow.keras.Model`

**Constructor**:
```python
class TensorFlowCNNModel(keras.Sequential):
    def __init__(self, input_shape=(28, 28, 1), num_classes=10, 
                 num_conv_blocks=2, conv_filters_initial=32, dense_units=128, 
                 dropout_rate=0.5, seed=42):
        """
        Initialize CNN model for MNIST classification.
        
        Args:
            input_shape (tuple): Input image shape (height, width, channels). 
                Default: (28, 28, 1) for MNIST grayscale.
            num_classes (int): Number of output classes. Default: 10 (digits 0-9).
            num_conv_blocks (int): Number of Conv→MaxPool blocks. Default: 2.
            conv_filters_initial (int): Initial filter count, doubles per block. Default: 32.
            dense_units (int): Hidden layer units after flattening. Default: 128.
            dropout_rate (float): Dropout probability. Default: 0.5.
            seed (int): Random seed for reproducibility. Default: 42.
        
        Returns:
            Compiled Keras Sequential model ready for training.
        """
        super().__init__([
            # Conv block 1
            keras.layers.Conv2D(conv_filters_initial, (3, 3), activation='relu', 
                               input_shape=input_shape),
            keras.layers.MaxPooling2D((2, 2)),
            
            # Conv block 2
            keras.layers.Conv2D(conv_filters_initial * 2, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            
            # Dense layers
            keras.layers.Flatten(),
            keras.layers.Dense(dense_units, activation='relu'),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile with standard settings
        self.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
```

**Interface Contract**:
- Input shape: (batch_size, 28, 28, 1) — required
- Output shape: (batch_size, 10) — probabilities over digit classes
- Output values: Sum to 1.0 per sample (softmax)
- Training method: `model.fit(train_data, validation_data=val_data, epochs=...)`
- Inference method: `predictions = model.predict(test_data)`
- Serialization: `mlflow.tensorflow.log_model(model, artifact_path="tensorflow_cnn_model")`

**Required Methods**:
- `fit(X_train, y_train, validation_data=...)` — Training with epoch-level callback hooks for MLflow logging
- `predict(X)` — Inference; returns logits or probabilities
- `evaluate(X_test, y_test)` — Evaluation; returns loss and metrics
- `get_config()` — Architecture summary (inherited from Sequential)

**Expected Behavior**:
- Deterministic when seed=42 (enforced at initialization)
- Achieves ≥98% test accuracy on MNIST
- Training completes in <2 minutes per epoch on CPU
- Handles batch prediction (vectorized inference)

---

### PyTorch CNN Model

**Class**: Inherits from `torch.nn.Module`

**Constructor & Forward**:
```python
class MNISTCNNModel(torch.nn.Module):
    def __init__(self, input_channels=1, num_classes=10, 
                 num_conv_blocks=2, conv_filters_initial=32, dense_units=128, 
                 dropout_rate=0.5, seed=42):
        """
        Initialize CNN model for MNIST classification.
        
        Args:
            input_channels (int): Input image channels. Default: 1 (grayscale).
            num_classes (int): Number of output classes. Default: 10 (digits 0-9).
            num_conv_blocks (int): Number of Conv→MaxPool blocks. Default: 2.
            conv_filters_initial (int): Initial filter count, doubles per block. Default: 32.
            dense_units (int): Hidden layer units after flattening. Default: 128.
            dropout_rate (float): Dropout probability. Default: 0.5.
            seed (int): Random seed for reproducibility. Default: 42.
        
        Returns:
            PyTorch model ready for training.
        """
        super().__init__()
        torch.manual_seed(seed)
        
        # Conv layers
        self.conv1 = torch.nn.Conv2d(input_channels, conv_filters_initial, 
                                      kernel_size=3, padding=1)
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(conv_filters_initial, conv_filters_initial * 2, 
                                      kernel_size=3, padding=1)
        self.pool2 = torch.nn.MaxPool2d(2, 2)
        
        # Flattened size: (28→14→7) = 7×7, filters = conv_filters_initial * 2
        self.flatten_size = conv_filters_initial * 2 * 7 * 7
        
        # Dense layers
        self.fc1 = torch.nn.Linear(self.flatten_size, dense_units)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.fc2 = torch.nn.Linear(dense_units, num_classes)
    
    def forward(self, x):
        """
        Forward pass through CNN.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, 10).
        """
        # Conv block 1
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.pool1(x)
        
        # Conv block 2
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Dense layers
        x = x.flatten(1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
```

**Interface Contract**:
- Input shape: (batch_size, 1, 28, 28) — required (PyTorch: NCHW format)
- Output shape: (batch_size, 10) — logits (NOT softmax; applied in loss)
- Training method: Custom loop with `model.train()`, `optimizer.zero_grad()`, backprop, `optimizer.step()`
- Inference method: `model.eval()` + `torch.no_grad()` + `model(X)`
- Serialization: `mlflow.pytorch.log_model(model, artifact_path="pytorch_cnn_model")`

**Required Methods**:
- `forward(X)` — Forward pass; returns logits
- `train()` — Set to training mode (dropout active)
- `eval()` — Set to evaluation mode (dropout inactive)
- `parameters()` — Access to trainable weights for optimizer

**Expected Behavior**:
- Deterministic when seed=42 (enforced at initialization)
- Achieves ≥98% test accuracy on MNIST
- Training completes in <2 minutes per epoch on CPU
- Handles batch prediction (vectorized inference)
- Output logits before softmax (PyTorch convention)

---

## Equivalence Contract

Both models MUST be architecturally equivalent:

| Component | TensorFlow | PyTorch | Equivalence |
|-----------|------------|---------|-------------|
| **Input** | (batch, 28, 28, 1) NHWC | (batch, 1, 28, 28) NCHW | Same logical input (grayscale 28×28) |
| **Conv Block 1** | Conv2D(32, 3×3, ReLU) + MaxPool(2×2) | Conv2d(1→32, 3×3, ReLU) + MaxPool(2×2) | Identical layer structure |
| **Conv Block 2** | Conv2D(64, 3×3, ReLU) + MaxPool(2×2) | Conv2d(32→64, 3×3, ReLU) + MaxPool(2×2) | Identical layer structure |
| **Flatten** | Flatten() | x.flatten(1) | Both reshape (batch, ...) → (batch, -1) |
| **Dense 1** | Dense(128, ReLU) + Dropout(0.5) | Linear(64×7×7, 128) + ReLU + Dropout(0.5) | Identical functionality |
| **Dense 2** | Dense(10, Softmax) | Linear(128, 10) | TF applies softmax; PT applies in loss |
| **Output** | (batch, 10) probabilities | (batch, 10) logits | Differs by activation; both valid |

**Prediction Behavior**:
- TensorFlow: `predictions = model.predict(X)` returns softmax probabilities [0,1]
- PyTorch: `logits = model(X)` returns raw logits; softmax must be applied externally or via loss function
- Both produce identical class predictions: `argmax(predictions)` ≈ `argmax(logits)`

---

## Usage Patterns (Framework-Agnostic Intent)

### Training Pattern

```python
# Both frameworks should enable:
model = ModelClass(seed=42)
model.train()
for epoch in range(num_epochs):
    # Forward pass on batch
    outputs = model(batch_data)
    # Compute loss
    loss = criterion(outputs, batch_labels)
    # Backward pass (framework-specific)
    # Log metrics
    mlflow.log_metric("train_loss", loss, step=epoch)

# Both frameworks should enable:
model.eval()
test_loss, test_acc = evaluate(model, test_loader)
mlflow.log_metric("test_accuracy", test_acc)
```

### Inference Pattern

```python
# Both frameworks should enable:
model.eval()
with torch.no_grad():  # PyTorch
    # or equivalent TensorFlow
    predictions = model(test_data)
class_predictions = predictions.argmax(axis=1)
```

### Serialization Pattern

```python
# Both frameworks:
mlflow.{framework}.log_model(model, artifact_path=f"{framework}_cnn_model")
# Later retrieval:
loaded_model = mlflow.{framework}.load_model(model_uri)
predictions = loaded_model.predict(new_data)
```

---

## Validation Checklist

- [ ] TensorFlow model inherits from `keras.Sequential` or `keras.Model`
- [ ] PyTorch model inherits from `torch.nn.Module`
- [ ] Both accept `seed=42` at initialization
- [ ] Both have identical high-level architecture (2 conv blocks + 2 dense layers)
- [ ] TensorFlow input: (batch, 28, 28, 1); PyTorch input: (batch, 1, 28, 28)
- [ ] TensorFlow output: softmax probabilities; PyTorch output: logits
- [ ] Both compile/initialize without errors
- [ ] Both achieve ≥98% test accuracy on MNIST
- [ ] Both training completes in <2 minutes/epoch on CPU
- [ ] Both models serializable to MLflow artifacts
- [ ] Both models loadable from MLflow and produce predictions

---

## Notes

- Input format difference (NHWC vs NCHW) is framework convention; both are equivalent
- Output activation difference (softmax vs logits) is framework convention; predictions equivalent via argmax
- Seed=42 enforced for reproducibility (per constitutional requirement)
- No custom layers or experimental features; use standard conv/pool/dense/dropout
- Both models should be deterministic and reproducible
