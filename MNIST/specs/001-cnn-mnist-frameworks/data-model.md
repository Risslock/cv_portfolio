# Data Model: CNN MNIST Frameworks

**Phase**: 1 (Design & Contracts)
**Date**: 2026-08-01
**Feature**: CNN MNIST implementation with MLflow tracking

## Entities

### 1. CNN Model

**Concept**: A trained convolutional neural network for MNIST digit classification.

**Variants**:
- **TensorFlow CNN**: `tensorflow.keras.Sequential` model
- **PyTorch CNN**: `torch.nn.Module` subclass

**Architecture** (both frameworks):
```
Input (28×28×1) 
  → Conv2D(32, 3×3, ReLU) 
  → MaxPool(2×2) 
  → Conv2D(64, 3×3, ReLU) 
  → MaxPool(2×2) 
  → Flatten 
  → Dense(128, ReLU) 
  → Dropout(0.5) 
  → Dense(10, Softmax)
```

**Attributes**:
- Input shape: (batch_size, 28, 28, 1) — MNIST grayscale images
- Output shape: (batch_size, 10) — 10 digit classes (0-9)
- Parameters: ~100K-200K trainable weights
- Output values: Probabilities [0, 1] per class

**State Transitions**:
1. **Instantiated**: Model created, random initialization
2. **Compiled** (TensorFlow): Optimizer and loss function attached
3. **Trained**: Weights updated via backpropagation
4. **Evaluated**: Test metrics computed
5. **Saved**: Serialized to MLflow artifact storage

**Validation Rules**:
- Input must be 4D tensor: (batch, 28, 28, 1)
- Output must sum to 1.0 per sample (softmax constraint)
- Accuracy on test set ≥ 98%

### 2. Training Hyperparameters

**Concept**: Configuration for model training; logged to MLflow as parameters for reproducibility.

**Fields**:
| Field | Type | Default Value | Constraint | Notes |
|-------|------|----------------|-----------|-------|
| `learning_rate` | float | 0.001 | 0.00001 - 0.1 | Controls optimization step size |
| `batch_size` | int | 32 | 8 - 256 | Samples per training step |
| `num_epochs` | int | 60 | 1 - 200 | Training iterations over dataset |
| `optimizer` | string | "Adam" | "Adam", "SGD" | Optimization algorithm |
| `loss_function` | string | "categorical_crossentropy" | standard losses | Classification loss |
| `num_conv_blocks` | int | 2 | 1 - 4 | Convolutional layers |
| `conv_filters_initial` | int | 32 | 8 - 64 | Starting filter count |
| `dense_units` | int | 128 | 32 - 512 | Hidden layer size |
| `dropout_rate` | float | 0.5 | 0.0 - 0.8 | Regularization |
| `random_seed` | int | **42** | **must be 42** | **For reproducible, deterministic training** |

**Reproducibility Requirement** ⚠️ **CRITICAL**:
- `random_seed = 42` MUST be used for all training runs
- Fixed seed ensures identical weight initialization and stochastic operations
- Logged to MLflow for reproducibility audit trail
- Enables exact replication of results from MLflow logs alone

**Validation Rules**:
- learning_rate > 0
- batch_size > 0
- num_epochs > 0
- dense_units > 0
- dropout_rate ∈ [0, 1]
- random_seed = 42 (enforced)

### 3. Training Metrics

**Concept**: Computed during training; logged to MLflow per epoch.

**Metrics Per Epoch**:
| Metric | Type | Range | Meaning |
|--------|------|-------|---------|
| `train_loss` | float | [0, ∞) | Cross-entropy loss on training batch |
| `train_accuracy` | float | [0, 1] | Fraction of training samples correctly classified |
| `val_loss` | float | [0, ∞) | Cross-entropy loss on validation set |
| `val_accuracy` | float | [0, 1] | Fraction of validation samples correctly classified |

**Final Metrics** (logged after training):
| Metric | Type | Range | Meaning |
|--------|------|-------|---------|
| `test_loss` | float | [0, ∞) | Cross-entropy loss on test set |
| `test_accuracy` | float | [0, 1] | Fraction of test samples correctly classified |
| `test_precision` | float | [0, 1] | Weighted precision across 10 classes |
| `test_recall` | float | [0, 1] | Weighted recall across 10 classes |
| `test_f1` | float | [0, 1] | Weighted F1-score across 10 classes |
| `training_time_seconds` | float | [0, ∞) | Wall-clock time for training |
| `inference_time_ms_per_batch` | float | [0, ∞) | Average inference time per batch |

### 4. MNIST Dataset

**Concept**: Training, validation, and test splits of MNIST digit images.

**Fields**:
| Field | Type | Values | Meaning |
|-------|------|--------|---------|
| `image` | array | (28, 28, 1), values ∈ [0, 1] | Grayscale digit image (normalized) |
| `label` | int | ∈ {0, 1, ..., 9} | Digit class |

**Splits**:
| Split | Samples | Purpose |
|-------|---------|---------|
| Training | 50,000 | Model weight optimization |
| Validation | 10,000 | Hyperparameter tuning, early stopping |
| Test | 10,000 | Final performance evaluation |

**Validation Rules**:
- Image shape: (28, 28, 1)
- Image values: [0, 1]
- Label: integer in {0, 1, ..., 9}
- No missing values

### 5. MLflow Experiment Run

**Concept**: A single training execution tracked in MLflow with full reproducibility information.

**Fields**:
| Field | Type | Logged By | Logged When |
|-------|------|-----------|-------------|
| `params` | dict | Training script | Pre-training (all hyperparameters including seed=42) |
| `metrics` | dict | Training script | Per epoch (train/val) + post-training (test) |
| `artifacts` | files | Training script | Post-training (model file, plots) |
| `run_id` | string | MLflow | Auto-generated on run creation |
| `timestamp` | datetime | MLflow | Auto-recorded on run creation |

**Artifact Types**:
| Artifact | Format | Meaning |
|----------|--------|---------|
| Model | SavedModel (TF) / state_dict (PT) | Trained weights, serializable |
| Metrics plot | PNG | Training/validation curves over epochs |
| Metadata | JSON | Architecture summary, layer info |

**Validation Rules**:
- All parameters present including `random_seed = 42`
- Metrics logged consistently per epoch (no gaps)
- Model artifact serializable and loadable
- Run reproducible from logged parameters (including seed)
- Identical inputs → identical outputs (deterministic)

### 6. Comparison Experiment

**Concept**: Set of runs representing different framework/hyperparameter combinations for portfolio comparison.

**Fields**:
| Field | Meaning | Example |
|-------|---------|---------|
| Framework | TensorFlow or PyTorch | "tensorflow", "pytorch" |
| Architecture | CNN variant | "cnn_2blocks" |
| Learning Rate | Hyperparameter | 0.001, 0.0001 |
| Random Seed | Reproducibility anchor | 42 |
| Best Test Accuracy | Final outcome | 0.987 |
| Training Time | Performance metric | 65 seconds |

**Comparison Criteria** (from spec):
1. Test accuracy ≥ 98%
2. Training time < 2 minutes
3. Consistent hyperparameter naming across frameworks (including `random_seed = 42`)
4. Reproducible from MLflow logs alone (all parameters logged)

## Entity Relationships

```
Experiment Run (MLflow)
├── Training Hyperparameters (parameters, including random_seed = 42)
├── Training Metrics (metrics per epoch)
├── Test Metrics (metrics after training)
├── CNN Model (artifact)
│   ├── TensorFlow: Sequential model
│   └── PyTorch: nn.Module subclass
└── MNIST Dataset (external reference)
    ├── Training split (50k images)
    ├── Validation split (10k images)
    └── Test split (10k images)

Comparison Experiment
├── Multiple Experiment Runs (TensorFlow + PyTorch)
├── Each run: Same seed (42), possibly different hyperparameters
├── All runs: Deterministic, reproducible
└── Output: Portfolio-quality comparison via MLflow UI
```

## Validation & Quality Gates

**Data Integrity**:
- MNIST images: shape (28, 28, 1), values ∈ [0, 1]
- Labels: integer ∈ {0-9}
- No missing data

**Model Quality**:
- Test accuracy ≥ 98%
- Model file serializable and loadable
- Inference works on unseen data

**Experiment Tracking & Reproducibility**:
- All hyperparameters logged (snake_case naming)
- `random_seed = 42` explicitly logged (non-negotiable)
- Metrics logged per epoch (consistent naming)
- Final metrics + training time logged
- Model artifact saved
- **Run reproducible from logs alone** (seed guarantees determinism)

**Code Quality**:
- Model class instantiates without errors
- Training loop completes without exceptions
- All dimensions match expected shapes
- Reproducible across runs (seed = 42 enforced)
- Deterministic: Same seed + same parameters = identical results
