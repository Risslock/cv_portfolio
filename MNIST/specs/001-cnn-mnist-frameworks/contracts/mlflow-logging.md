# Contract: MLflow Logging

**Phase**: 1 (Design & Contracts)
**Date**: 2026-08-01
**Feature**: CNN MNIST implementation with MLflow tracking

## Purpose

This contract defines the MLflow experiment tracking interface for CNN model training in both TensorFlow and PyTorch. Establishes consistent parameter naming, metric logging patterns, and artifact conventions to enable reproducible, comparable experiments.

## MLflow Experiment Setup Contract

### Experiment Initialization

**Interface**:
```python
import mlflow

experiment_name = f"cnn_mnist_{framework}"  # e.g., "cnn_mnist_tensorflow"
mlflow.set_experiment(experiment_name)
```

**Mandatory Before Training**:
- Experiment created with framework-specific name
- Run started: `mlflow.start_run()`
- Run ID retrievable for verification

### Parameter Logging Contract

**Logged Pre-Training** (all values required):

```python
mlflow.log_param("learning_rate", learning_rate)  # float: e.g., 0.001
mlflow.log_param("batch_size", batch_size)        # int: e.g., 32
mlflow.log_param("num_epochs", num_epochs)        # int: e.g., 60
mlflow.log_param("optimizer", optimizer)          # string: "Adam" or "SGD"
mlflow.log_param("loss_function", loss_fn)        # string: "categorical_crossentropy"
mlflow.log_param("num_conv_blocks", 2)            # int: 2
mlflow.log_param("conv_filters_initial", 32)      # int: 32
mlflow.log_param("dense_units", 128)              # int: 128
mlflow.log_param("dropout_rate", 0.5)             # float: 0.5
mlflow.log_param("random_seed", 42)               # int: **MUST be 42**
mlflow.log_param("framework", framework)          # string: "tensorflow" or "pytorch"
```

**Validation**:
- All 11 parameters logged
- Parameter names use snake_case (no camelCase, no spaces)
- `random_seed = 42` (non-negotiable, enforced)
- String values lowercase (e.g., "adam" NOT "ADAM")

### Metrics Logging Contract (Per Epoch)

**Logged at Each Epoch** (during training):

```python
mlflow.log_metric("train_loss", train_loss, step=epoch)
mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
mlflow.log_metric("val_loss", val_loss, step=epoch)
mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
```

**Validation**:
- Metric names use snake_case
- Scoped prefixes: `train_`, `val_`
- Step parameter: `step=epoch` (enables epoch-level tracking)
- Values numeric (float or int)
- Logged at consistent epochs (every epoch, no gaps)

### Final Metrics Logging Contract

**Logged After Training** (once, no step parameter):

```python
mlflow.log_metric("test_loss", test_loss)
mlflow.log_metric("test_accuracy", test_accuracy)
mlflow.log_metric("test_precision", test_precision)
mlflow.log_metric("test_recall", test_recall)
mlflow.log_metric("test_f1", test_f1)
mlflow.log_metric("training_time_seconds", training_time)
mlflow.log_metric("inference_time_ms_per_batch", inference_time)
```

**Validation**:
- Metric names use snake_case
- Scope: `test_` for test set metrics
- No step parameter (logged once as final values)
- Values numeric, ≥ 0
- `test_accuracy ≥ 0.98` (success criterion)

### Artifact Logging Contract

**Saved Post-Training**:

```python
# Model artifact (TensorFlow SavedModel or PyTorch state_dict)
mlflow.pytorch.log_model(model, artifact_path="pytorch_cnn_model")
# OR
mlflow.tensorflow.save_model(model, path=artifact_dir)

# Metrics visualization (optional PNG)
plt.savefig(f"{artifact_dir}/training_history.png")
mlflow.log_artifact(f"{artifact_dir}/training_history.png")

# Metadata (optional JSON)
metadata = {"architecture": "CNN_2blocks", "framework": "pytorch", "input_shape": (28, 28, 1)}
import json
with open(f"{artifact_dir}/metadata.json", "w") as f:
    json.dump(metadata, f)
mlflow.log_artifact(f"{artifact_dir}/metadata.json")
```

**Validation**:
- Model serialized and loadable (test by loading from MLflow URI)
- Artifact paths descriptive: `pytorch_cnn_model`, `tensorflow_cnn_model`
- Optional visualizations aid portfolio presentation
- All artifacts retrievable via MLflow API

## Usage Example (TensorFlow)

```python
import mlflow
import tensorflow as tf
from tensorflow import keras

# Setup
mlflow.set_experiment("cnn_mnist_tensorflow")

with mlflow.start_run() as run:
    # Log parameters
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("num_epochs", 60)
    mlflow.log_param("optimizer", "adam")
    mlflow.log_param("loss_function", "categorical_crossentropy")
    mlflow.log_param("num_conv_blocks", 2)
    mlflow.log_param("conv_filters_initial", 32)
    mlflow.log_param("dense_units", 128)
    mlflow.log_param("dropout_rate", 0.5)
    mlflow.log_param("random_seed", 42)
    mlflow.log_param("framework", "tensorflow")
    
    # Build model, compile, train
    model = keras.Sequential([...])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Training loop with epoch-level logging
    for epoch in range(60):
        history = model.fit(train_data, validation_data=val_data, epochs=1, verbose=0)
        mlflow.log_metric("train_loss", float(history.history['loss'][0]), step=epoch)
        mlflow.log_metric("train_accuracy", float(history.history['accuracy'][0]), step=epoch)
        mlflow.log_metric("val_loss", float(history.history['val_loss'][0]), step=epoch)
        mlflow.log_metric("val_accuracy", float(history.history['val_accuracy'][0]), step=epoch)
    
    # Test and log final metrics
    test_loss, test_accuracy = model.evaluate(test_data)
    mlflow.log_metric("test_loss", float(test_loss))
    mlflow.log_metric("test_accuracy", float(test_accuracy))
    mlflow.log_metric("test_precision", precision_value)
    mlflow.log_metric("test_recall", recall_value)
    mlflow.log_metric("test_f1", f1_value)
    mlflow.log_metric("training_time_seconds", elapsed_time)
    mlflow.log_metric("inference_time_ms_per_batch", inference_ms)
    
    # Save model
    mlflow.tensorflow.log_model(model, artifact_path="tensorflow_cnn_model")
    
    print(f"Run ID: {run.info.run_id}")
```

## Usage Example (PyTorch)

```python
import mlflow
import torch
import torch.nn as nn

# Setup
mlflow.set_experiment("cnn_mnist_pytorch")

with mlflow.start_run() as run:
    # Log parameters
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("num_epochs", 60)
    mlflow.log_param("optimizer", "adam")
    mlflow.log_param("loss_function", "crossentropyloss")
    mlflow.log_param("num_conv_blocks", 2)
    mlflow.log_param("conv_filters_initial", 32)
    mlflow.log_param("dense_units", 128)
    mlflow.log_param("dropout_rate", 0.5)
    mlflow.log_param("random_seed", 42)
    mlflow.log_param("framework", "pytorch")
    
    # Build model
    model = MNISTCNNModel(...)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop with epoch-level logging
    for epoch in range(60):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_accuracy", train_acc, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)
    
    # Test and log final metrics
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate_test(model, test_loader)
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_precision", test_prec)
    mlflow.log_metric("test_recall", test_rec)
    mlflow.log_metric("test_f1", test_f1)
    mlflow.log_metric("training_time_seconds", elapsed_time)
    mlflow.log_metric("inference_time_ms_per_batch", inference_ms)
    
    # Save model
    mlflow.pytorch.log_model(model, artifact_path="pytorch_cnn_model")
    
    print(f"Run ID: {run.info.run_id}")
```

## Validation Checklist

- [ ] Experiment name includes framework: `cnn_mnist_tensorflow` or `cnn_mnist_pytorch`
- [ ] All 11 parameters logged (including `random_seed = 42`)
- [ ] Parameter names: snake_case, no camelCase
- [ ] Metrics logged per epoch with `step=epoch`
- [ ] Metric names: snake_case with `train_`, `val_`, `test_` prefixes
- [ ] Final metrics logged once (no step parameter)
- [ ] Model artifact saved with descriptive path
- [ ] `test_accuracy ≥ 0.98`
- [ ] Training time < 120 seconds
- [ ] All runs reproducible from logged parameters alone

## Notes

- MLflow tracking URI defaults to `./mlruns` (local file tracking)
- Experiments accessible via MLflow UI: `mlflow ui --port 5000`
- No remote MLflow server required for v1
- Parameter/metric consistency enables framework comparison
- Fixed seed (42) enables deterministic replication
