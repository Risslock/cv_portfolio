# Research: CNN MNIST Frameworks

**Phase**: 0 (Research & Clarification)
**Date**: 2026-08-01
**Feature**: CNN MNIST implementation with MLflow tracking

## Research Summary

This document consolidates research on technical decisions, best practices, and confirmed approach for implementing CNN models in TensorFlow and PyTorch with MLflow experiment tracking.

### 1. CNN Architecture for MNIST

**Decision**: Implement a small, practical CNN with 2-3 convolutional blocks followed by dense layers.

**Rationale**: 
- MNIST images are small (28×28 grayscale), so deep networks are unnecessary
- 2-3 conv blocks with pooling demonstrate CNN concepts without excessive complexity
- Architecture suitable for portfolio: shows understanding of conv/pool/dense layers without over-engineering
- Achieves 98%+ accuracy, meeting performance goals

**Recommended Architecture**:
```
Conv2D(32, 3×3, ReLU) → MaxPool → Conv2D(64, 3×3, ReLU) → MaxPool → Flatten → Dense(128, ReLU) → Dense(10, Softmax)
```

**Alternatives Considered**:
- Deeper networks (ResNet, VGG): Unnecessary for MNIST; adds complexity without portfolio benefit
- Simpler networks (1 conv layer): Insufficient to demonstrate CNN knowledge
- Automated architecture search: Overkill for this project scope

### 2. Framework Consistency

**Decision**: Implement identical high-level architecture in both TensorFlow and PyTorch; demonstrate framework-specific patterns while maintaining same layer structure.

**Rationale**:
- Side-by-side comparison is portfolio goal; identical architecture makes frameworks directly comparable
- Framework idioms should be respected (TensorFlow Sequential vs PyTorch nn.Module)
- Consistent results validate both implementations are correct

**Approach**:
- TensorFlow: Use Sequential API for simplicity (educational value)
- PyTorch: Use nn.Module base class with __init__ and forward methods
- Both: Same layer configuration (filters, kernel sizes, activation functions)

**Alternatives Considered**:
- Different architectures per framework: Undermines comparison goal
- Functional API for both: Loses educational clarity of Sequential for TensorFlow
- Custom training loops: Feasible but adds unnecessary complexity; TensorFlow's fit() and PyTorch's standard loop are industry standard

### 3. MLflow Integration

**Decision**: Integrate MLflow tracking into training scripts; log parameters pre-training, metrics per epoch, artifacts post-training.

**Rationale**:
- Constitutional requirement (Principle VIII)
- Enables portfolio-quality experiment comparison
- MLflow local tracking (./mlruns) requires no external infrastructure
- Standard naming conventions enable repeatability

**Naming Conventions** (Constitutional Requirement):
- **Parameters** (pre-training): `learning_rate`, `batch_size`, `num_epochs`, `optimizer`, `num_conv_blocks`, `conv_filters`, `dense_units`
- **Metrics per epoch**: `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy`
- **Final metrics**: `test_loss`, `test_accuracy`, `test_precision`, `test_recall`, `test_f1`
- **Artifacts**: `tensorflow_cnn_model` (SavedModel), `pytorch_cnn_model` (state_dict)

**Alternatives Considered**:
- Weights & Biases: Requires external account; local MLflow sufficient
- Manual logging to CSV: Not portfolio-grade; MLflow provides UI comparison
- Tensorboard: Good for visualization; MLflow provides experiment tracking and artifact management

### 4. Data Loading & Preprocessing

**Decision**: Use framework-native dataset loaders (Keras for TensorFlow, torchvision for PyTorch); normalize to [0,1]; optional data augmentation.

**Rationale**:
- Framework loaders handle automatic MNIST download and standard test/train split
- Normalization to [0,1] is standard preprocessing; ensures both frameworks use identical scale
- Data augmentation (rotation, shift) optional for v1; can improve accuracy if needed

**Implementation**:
- TensorFlow: `tensorflow.keras.datasets.mnist.load_data()`
- PyTorch: `torchvision.datasets.MNIST` with `torchvision.transforms`
- Normalization: (pixel - mean) / std; equivalently [0, 255] → [0, 1]
- Preprocessing encapsulated in `data.py` for reusability

**Alternatives Considered**:
- Custom download logic: Unnecessarily complex; framework loaders are reliable
- No preprocessing: Suboptimal training; normalization is standard practice
- Aggressive augmentation: Overkill for MNIST; accuracy already saturates

### 5. Jupyter Notebooks as Learning Guides

**Decision**: Create two parallel notebooks (tensorflow_cnn_guide.ipynb, pytorch_cnn_guide.ipynb) with identical conceptual structure and 8+ markdown cells explaining each step.

**Rationale**:
- Constitutional requirement (Principle IV)
- Notebooks are primary portfolio artifacts: reviewers run them to verify understanding
- Side-by-side structure highlights framework differences while maintaining conceptual flow
- Markdown-rich narrative supports learning value

**Structure**:
1. **Setup**: Imports, paths, MLflow experiment setup
2. **Data Loading**: Load MNIST, inspect shapes, visualize samples
3. **Preprocessing**: Normalization, train/val/test split, batch creation
4. **Model Definition**: Explain architecture; show layer-by-layer construction
5. **Training Setup**: Optimizer, loss function, MLflow parameter logging
6. **Training Loop**: Execute training; log metrics per epoch
7. **Evaluation**: Compute test metrics (accuracy, precision, recall, F1)
8. **Visualization**: Plot training history, show sample predictions, compare with FCNN baseline
9. **Model Artifact**: Save model to MLflow

**Alternatives Considered**:
- Single combined notebook: Harder to compare framework differences
- Minimal markdown: Less educational; harder for reviewer to follow
- Separate notebooks for models vs training: Fragments learning flow

### 6. Random Seed & Reproducibility

**Decision**: Fix random seeds for NumPy, TensorFlow, and PyTorch at start of training script and notebooks.

**Rationale**:
- Constitutional requirement (Principle VIII)
- Enables exact reproduction of results
- Logged in MLflow as parameter `random_seed` for downstream reproducibility

**Implementation**:
```python
import random
import numpy as np
import tensorflow as tf
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
torch.manual_seed(SEED)
```

**Limitations**:
- GPU non-determinism: Even with seed, GPU operations may vary slightly; document in notebooks
- Thread-based variations: Acceptable for this scope

### 7. Model Checkpointing

**Decision**: Save only final trained model to MLflow (v1); no intermediate checkpoints.

**Rationale**:
- Scope simplification: Final model is sufficient for comparison
- MLflow artifact storage handles versioning
- Checkpointing can be added in future iteration if needed for long-running training

**Alternatives Considered**:
- Save best validation model: Adds complexity; MNIST training is fast, overfitting unlikely
- Save all checkpoints: Consumes unnecessary storage; not needed for portfolio

### 8. Performance & Resource Constraints

**Decision**: Target 2-minute training per notebook on standard CPU; batch size 32-128; single-threaded for reproducibility.

**Rationale**:
- CPU-only requirement: Portable, no GPU dependency
- 2-minute target: Fast feedback loop for portfolio reviewers
- Small batch size: MNIST is small dataset; no optimization needed
- Single-threaded: Ensures determinism

**Validation**:
- Time budget per epoch: ~1-2 seconds on CPU (60 epochs → ~60-120 seconds)
- Memory: ~200-500MB; standard laptop capable
- Inference: <10ms per batch (suitable for real-time scenarios)

### 9. Code Quality & Testing

**Decision**: Implement unit tests for model classes and data loaders; integration tests for training pipeline; full linting compliance (flake8, pylint).

**Rationale**:
- Constitutional requirement (Principle V)
- Tests validate reproducibility
- Linting ensures professional code quality

**Test Coverage**:
- Model instantiation and forward pass
- Data loading and shape validation
- Metrics computation (accuracy, precision, recall)
- MLflow parameter and metric logging
- Model save/load cycle

**Linting**:
- flake8: Style guide compliance (E501 line length, etc.)
- pylint: Code quality (naming, unused variables, etc.)
- Target: Zero warnings

## Confirmed Approach

### Technical Decisions Summary

| Decision | Approach | Confidence |
|----------|----------|------------|
| CNN Architecture | 2-3 conv blocks + pooling → flatten → 2 dense layers | High |
| Framework Consistency | Identical high-level structure; framework idioms respected | High |
| MLflow Integration | Epoch-level metrics + final artifacts; local tracking | High |
| Naming Conventions | Snake_case parameters, scoped metrics (train_/val_/test_) | High |
| Data Loading | Framework natives (Keras/torchvision); [0,1] normalization | High |
| Notebooks | Parallel structure; 8+ markdown cells per notebook | High |
| Reproducibility | Fixed seeds (42); deterministic training | High |
| Checkpointing | Final model only; no intermediate saves | Medium (nice-to-have for future) |
| Performance Budget | 2 min training; CPU-only; single-threaded | High |
| Code Quality | Unit + integration tests; flake8 + pylint zero warnings | High |

## Dependencies & Prerequisites

- Python 3.10+
- TensorFlow 2.13+ (pip install via pyproject.toml)
- PyTorch 2.0+ (pip install via pyproject.toml)
- MLflow 2.0+ (pip install via pyproject.toml)
- Jupyter (pip install via pyproject.toml)
- pytest (dev dependency)

## Next Steps

1. Phase 1: Generate data model, contracts, quickstart guide
2. Phase 2: Generate tasks.md with dependency ordering
3. Implementation: Execute tasks in order, following design decisions above
4. Validation: Run quickstart.md guide; verify all success criteria met
