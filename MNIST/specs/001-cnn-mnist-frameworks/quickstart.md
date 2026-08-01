# Quickstart: CNN MNIST Validation Guide

**Phase**: 1 (Design & Contracts)
**Date**: 2026-08-01
**Feature**: CNN MNIST implementation with MLflow tracking

## Purpose

This guide provides step-by-step validation that CNN implementations meet all specification requirements. Run this after completing implementation to verify:
- Models achieve ≥98% test accuracy
- MLflow experiment tracking works with consistent naming
- Notebooks execute end-to-end without errors
- Code passes linting checks
- Framework comparison is valid

## Prerequisites

```bash
# Environment setup
uv sync  # Install all dependencies from pyproject.toml

# Verify MLflow and frameworks are installed
uv run python -c "import mlflow, tensorflow, torch; print('✓ All dependencies ready')"

# Start MLflow tracking server (optional, for UI)
# uv run mlflow ui --port 5000
# Then visit http://localhost:5000 to browse experiments
```

## Validation Checklist

### 1. Code Quality Checks

**Goal**: Verify zero linting warnings

```bash
# Check TensorFlow CNN code
uv run flake8 tensorflow/cnn/ --max-line-length=100
uv run pylint tensorflow/cnn/ --disable=C0111 --disable=R0913

# Check PyTorch CNN code
uv run flake8 pytorch/cnn/ --max-line-length=100
uv run pylint pytorch/cnn/ --disable=C0111 --disable=R0913

# Expected result: No warnings
```

**Pass Criteria**: ✅ Zero warnings from both flake8 and pylint

### 2. Model Interface Validation

**Goal**: Verify both models match [Model Interface Contract](contracts/model-interface.md)

```python
# Test TensorFlow model
python -c "
from tensorflow_cnn_models import MNISTCNNModel as TFModel
import numpy as np

model = TFModel(seed=42)
x = np.random.randn(1, 28, 28, 1).astype('float32')
y = model.predict(x)

print(f'TensorFlow input shape: {x.shape}')
print(f'TensorFlow output shape: {y.shape}')
print(f'Output sum per sample: {y.sum(axis=1)}')  # Should be ~1.0
assert y.shape == (1, 10), 'Output shape mismatch'
assert np.allclose(y.sum(axis=1), 1.0), 'Softmax constraint violated'
print('✓ TensorFlow model interface OK')
"

# Test PyTorch model
python -c "
from pytorch_cnn_models import MNISTCNNModel
import torch

model = MNISTCNNModel(seed=42)
model.eval()
x = torch.randn(1, 1, 28, 28)
with torch.no_grad():
    y = model(x)

print(f'PyTorch input shape: {x.shape}')
print(f'PyTorch output shape: {y.shape}')
assert y.shape == (1, 10), 'Output shape mismatch'
print('✓ PyTorch model interface OK')
"
```

**Pass Criteria**: ✅ Both models instantiate and forward pass works with correct shapes

### 3. Unit Tests Execution

**Goal**: Run all unit tests; verify >70% coverage

```bash
# Run tests
uv run pytest tests/ -v --cov=tensorflow/cnn --cov=pytorch/cnn --cov-report=term-summary

# Expected output:
# - test_tensorflow_cnn_models.py: PASSED
# - test_pytorch_cnn_models.py: PASSED
# - test_data_loading.py: PASSED
# - Coverage: ≥70%
```

**Pass Criteria**: ✅ All tests pass, coverage ≥70%

### 4. TensorFlow CNN Training & MLflow Logging

**Goal**: Train TensorFlow model; verify MLflow tracks parameters and metrics correctly

```bash
# Run training script
uv run python tensorflow/cnn/train.py --epochs 2 --batch_size 32 --seed 42

# Expected output:
# Epoch 1/2
# - train_loss, train_accuracy logged to MLflow per epoch
# - val_loss, val_accuracy logged to MLflow per epoch
# Epoch 2/2
# [same as above]
# Training complete. Final metrics logged to MLflow.
# Run ID: <uuid>
```

**Verify MLflow Logs** (inspect directly):

```bash
# View MLflow run details
uv run python -c "
import mlflow
mlflow.set_experiment('cnn_mnist_tensorflow')
runs = mlflow.search_runs()
latest_run = runs.iloc[0]

print('Latest TensorFlow run:')
print(f'  Run ID: {latest_run.run_id}')
print(f'  Parameters:')
for k, v in latest_run.params.items():
    print(f'    {k}: {v}')
print(f'  Metrics:')
for k, v in latest_run.metrics.items():
    print(f'    {k}: {v}')
"

# Verify specific parameters logged
python -c "
import mlflow
mlflow.set_experiment('cnn_mnist_tensorflow')
run = mlflow.search_runs().iloc[0]
params = run.params

assert params.get('learning_rate') == '0.001', 'learning_rate not logged'
assert params.get('batch_size') == '32', 'batch_size not logged'
assert params.get('random_seed') == '42', '⚠️ random_seed MUST be 42'
assert params.get('framework') == 'tensorflow', 'framework not logged'
print('✓ All parameters logged correctly')
"
```

**Pass Criteria**: 
- ✅ Training completes without errors
- ✅ All 11 parameters logged (including random_seed=42)
- ✅ Metrics logged per epoch (train_loss, train_accuracy, val_loss, val_accuracy)
- ✅ Final metrics logged (test_loss, test_accuracy, test_precision, test_recall, test_f1)

### 5. PyTorch CNN Training & MLflow Logging

**Goal**: Train PyTorch model; verify MLflow tracks parameters and metrics correctly

```bash
# Run training script
uv run python pytorch/cnn/train.py --epochs 2 --batch_size 32 --seed 42

# Expected output:
# Epoch 1/2
# - train_loss, train_accuracy logged to MLflow per epoch
# - val_loss, val_accuracy logged to MLflow per epoch
# Epoch 2/2
# [same as above]
# Training complete. Final metrics logged to MLflow.
# Run ID: <uuid>
```

**Verify MLflow Logs**:

```bash
uv run python -c "
import mlflow
mlflow.set_experiment('cnn_mnist_pytorch')
runs = mlflow.search_runs()
latest_run = runs.iloc[0]

print('Latest PyTorch run:')
print(f'  Run ID: {latest_run.run_id}')
print(f'  Framework: {latest_run.params.get(\"framework\")}')
print(f'  Test Accuracy: {latest_run.metrics.get(\"test_accuracy\")}')
assert float(latest_run.metrics.get('test_accuracy', 0)) >= 0.98, 'Accuracy below 98%'
print('✓ PyTorch model meets accuracy requirement')
"
```

**Pass Criteria**:
- ✅ Training completes without errors
- ✅ All 11 parameters logged (including random_seed=42)
- ✅ Metrics logged per epoch
- ✅ Final metrics logged (test_accuracy ≥ 0.98)

### 6. Framework Comparison via MLflow

**Goal**: View both TensorFlow and PyTorch experiments in MLflow; verify comparison is valid

```bash
# Start MLflow UI (if not already running)
uv run mlflow ui --port 5000

# In browser, navigate to http://localhost:5000
# Expected view:
# - Left sidebar: Experiments
#   - cnn_mnist_tensorflow
#   - cnn_mnist_pytorch
# - Select both experiments to compare
# - Verify:
#   - Both have identical parameter names (snake_case)
#   - Both have identical metric names (train_loss, val_loss, test_accuracy, etc.)
#   - Both have similar test accuracy (≥0.98)
#   - Training times similar (<2 min each)
```

**Programmatic Verification**:

```bash
uv run python -c "
import mlflow
import pandas as pd

# Get both experiments
tf_runs = mlflow.search_runs(experiment_names=['cnn_mnist_tensorflow'])
pt_runs = mlflow.search_runs(experiment_names=['cnn_mnist_pytorch'])

print('Comparison:')
print('TensorFlow CNN:')
print(f'  Test Accuracy: {tf_runs.iloc[0].metrics.get(\"test_accuracy\", \"N/A\")}')
print(f'  Training Time: {tf_runs.iloc[0].metrics.get(\"training_time_seconds\", \"N/A\")} sec')
print()
print('PyTorch CNN:')
print(f'  Test Accuracy: {pt_runs.iloc[0].metrics.get(\"test_accuracy\", \"N/A\")}')
print(f'  Training Time: {pt_runs.iloc[0].metrics.get(\"training_time_seconds\", \"N/A\")} sec')
print()
print('✓ Framework comparison data available in MLflow')
"
```

**Pass Criteria**:
- ✅ Both experiments visible in MLflow UI
- ✅ Parameter naming consistent across frameworks
- ✅ Metric naming consistent across frameworks
- ✅ Both models achieve ≥98% accuracy
- ✅ Both training times <2 minutes

### 7. Jupyter Notebooks Validation

**Goal**: Execute notebooks end-to-end; verify no errors and markdown content present

```bash
# TensorFlow notebook
uv run jupyter nbconvert --to notebook --execute notebooks/tensorflow_cnn_guide.ipynb

# PyTorch notebook
uv run jupyter nbconvert --to notebook --execute notebooks/pytorch_cnn_guide.ipynb

# Expected result: Both notebooks execute without errors
# Check output for:
# - Data loading visualization
# - Model architecture summary
# - Training history plots
# - Evaluation metrics
# - Model artifact save confirmation
```

**Manual Inspection**:

```bash
# Open notebooks and verify:
# 1. ✅ At least 8 markdown cells with explanatory content
# 2. ✅ All code cells execute in sequence without errors
# 3. ✅ Visualizations (plots) render correctly
# 4. ✅ Final accuracy displayed (≥98%)
# 5. ✅ Model saved to MLflow artifact (message displayed)
# 6. ✅ Execution time <2 minutes total per notebook
```

**Pass Criteria**:
- ✅ Both notebooks execute end-to-end without errors
- ✅ 8+ markdown cells per notebook
- ✅ Visualizations present
- ✅ Execution time <2 min per notebook
- ✅ MLflow artifacts saved

### 8. Reproducibility Validation

**Goal**: Verify seed=42 ensures deterministic training

```bash
# Train model twice with seed=42
uv run python tensorflow/cnn/train.py --epochs 1 --seed 42
echo "First training done. Note test_accuracy."

uv run python tensorflow/cnn/train.py --epochs 1 --seed 42
echo "Second training done. Verify test_accuracy is identical."

# Verify via MLflow
python -c "
import mlflow
runs = mlflow.search_runs(experiment_names=['cnn_mnist_tensorflow'], order_by=['start_time DESC'], max_results=2)
acc1 = runs.iloc[0].metrics.get('test_accuracy')
acc2 = runs.iloc[1].metrics.get('test_accuracy')
print(f'Run 1 accuracy: {acc1}')
print(f'Run 2 accuracy: {acc2}')
if acc1 == acc2:
    print('✓ Reproducibility confirmed: Identical results with seed=42')
else:
    print('⚠️ Accuracies differ; check random seed initialization')
"
```

**Pass Criteria**:
- ✅ Multiple runs with seed=42 produce identical results
- ✅ Model weights and predictions reproducible
- ✅ MLflow logs confirm seed parameter

### 9. Model Serialization & Reproducibility

**Goal**: Save model to MLflow, load it, and verify predictions match

```bash
python -c "
import mlflow
import numpy as np
from tensorflow_cnn_models import MNISTCNNModel

# Train and save model
mlflow.set_experiment('cnn_mnist_tensorflow')
with mlflow.start_run():
    model = MNISTCNNModel(seed=42)
    # [training code...]
    mlflow.tensorflow.log_model(model, artifact_path='tensorflow_cnn_model')
    run_id = mlflow.active_run().info.run_id

# Load model from MLflow and test
model_uri = f'runs:/{run_id}/tensorflow_cnn_model'
loaded_model = mlflow.tensorflow.load_model(model_uri)

# Verify predictions match
x_test = np.random.randn(5, 28, 28, 1).astype('float32')
original_preds = model.predict(x_test)
loaded_preds = loaded_model.predict(x_test)

if np.allclose(original_preds, loaded_preds):
    print('✓ Model serialization and loading verified')
else:
    print('⚠️ Predictions differ after load; check serialization')
"
```

**Pass Criteria**:
- ✅ Model artifact saved to MLflow
- ✅ Model loads from artifact path
- ✅ Loaded model produces identical predictions
- ✅ Inference time documented

### 10. Documentation & Clarity

**Goal**: Verify README files and docstrings are clear

```bash
# Check README files exist
ls -la tensorflow/cnn/README.md
ls -la pytorch/cnn/README.md
ls -la notebooks/README.md

# Verify docstrings
uv run pydoc3 tensorflow.cnn.models | head -20
uv run pydoc3 pytorch.cnn.models | head -20

# Expected: Clear docstrings for MNISTCNNModel class and public functions
```

**Pass Criteria**:
- ✅ README in tensorflow/cnn/ describes TensorFlow implementation
- ✅ README in pytorch/cnn/ describes PyTorch implementation
- ✅ All public functions/classes have docstrings
- ✅ Docstrings explain parameters and return values

## Quick Validation Script

Run this to validate all requirements at once:

```bash
#!/bin/bash

echo "=== CNN MNIST Validation ==="

echo "1. Linting..."
uv run flake8 tensorflow/cnn pytorch/cnn && echo "✓ Linting passed" || echo "✗ Linting failed"

echo "2. Unit tests..."
uv run pytest tests/ -q && echo "✓ Tests passed" || echo "✗ Tests failed"

echo "3. TensorFlow model..."
uv run python tensorflow/cnn/train.py --epochs 2 && echo "✓ TensorFlow training passed" || echo "✗ Failed"

echo "4. PyTorch model..."
uv run python pytorch/cnn/train.py --epochs 2 && echo "✓ PyTorch training passed" || echo "✗ Failed"

echo "5. Notebooks..."
uv run jupyter nbconvert --to notebook --execute notebooks/tensorflow_cnn_guide.ipynb && echo "✓ TF notebook passed" || echo "✗ Failed"
uv run jupyter nbconvert --to notebook --execute notebooks/pytorch_cnn_guide.ipynb && echo "✓ PT notebook passed" || echo "✗ Failed"

echo ""
echo "=== Validation Complete ==="
echo "Check MLflow UI: mlflow ui --port 5000"
```

## Success Criteria Summary

| Criterion | Passing Value | Command to Verify |
|-----------|---------------|-------------------|
| Test accuracy | ≥98% | `mlflow runs.metrics['test_accuracy']` |
| Training time | <2 min | `mlflow runs.metrics['training_time_seconds']` |
| Linting | 0 warnings | `flake8 && pylint` |
| Tests | All pass | `pytest` |
| Notebooks | Execute without errors | `jupyter nbconvert --execute` |
| Parameters | All 11 logged | `mlflow runs.params` (includes seed=42) |
| Reproducibility | Identical results | Run twice with seed=42 |
| Model save/load | Predictions match | Load from MLflow URI |

## Notes

- All commands use `uv run` to execute in project environment
- MLflow tracking URI defaults to `./mlruns` (local)
- Seed=42 is non-negotiable per constitutional requirement
- Framework comparison validity depends on identical hyperparameters and metrics
