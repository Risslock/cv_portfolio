# Implementation Plan: CNN MNIST Frameworks

**Branch**: `001-cnn-mnist-frameworks` | **Date**: 2026-08-01 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/001-cnn-mnist-frameworks/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command; its definition describes the execution workflow.

## Summary

Implement Convolutional Neural Network (CNN) models for MNIST digit classification in both TensorFlow and PyTorch frameworks, with MLflow experiment tracking for reproducible, comparable results. Deliverables include production-grade OOP implementations, comprehensive Jupyter notebooks as learning guides, and logged experiments enabling framework comparison.

## Technical Context

**Language/Version**: Python 3.10+

**Primary Dependencies**: TensorFlow 2.13+, PyTorch 2.0+, MLflow 2.0+, NumPy, Jupyter, scikit-learn (for metrics)

**Storage**: File-based (local MNIST dataset cache, MLflow tracking directory ./mlruns)

**Testing**: pytest (unit/integration tests for data loaders and utilities)

**Target Platform**: CPU-capable systems (GPU optional, will auto-detect and use if available)

**Project Type**: Machine Learning model showcase library with educational notebooks

**Performance Goals**: Model training <2 minutes per notebook, 98%+ accuracy on MNIST test set

**Constraints**: Reproducible across runs (fixed random seeds), portable (no external service dependencies), offline-capable (dataset auto-downloads on first run)

**Scale/Scope**: Single dataset (MNIST 60k training + 10k test), two frameworks, one architecture (CNN; FCNN already implemented)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Principle Compliance Requirements

| Principle | Requirement | Implementation Approach |
|-----------|-------------|------------------------|
| **I. Object-Oriented Design** | All production code MUST follow OOP with clear separation of concerns | CNN models as reusable classes (TensorFlow Sequential/Functional, PyTorch nn.Module); data loading as separate utility classes; training pipeline as orchestrator class |
| **II. Folder Structure Clarity** | Project structure MUST be self-documenting with clear hierarchy by framework then model type | `tensorflow/cnn/`, `pytorch/cnn/` with separate modules for models, data, training, evaluation |
| **III. Usage Clarity** | Every artifact MUST have clear usage documentation and docstrings | Public functions/classes with complete NumPy-style docstrings; notebooks with step-by-step explanations |
| **IV. Jupyter Notebooks** | Notebooks MUST serve as easy-to-follow learning guides with executable cells in sequence | `notebooks/tensorflow_cnn_guide.ipynb`, `notebooks/pytorch_cnn_guide.ipynb` with 8+ markdown cells and complete pipeline walkthrough |
| **V. Code Quality Standards** | Code MUST pass linting (flake8, pylint), have complete docstrings, include type hints where helpful | Pre-commit linting verification; docstring validation; type hints on public APIs |
| **VI. README & Documentation** | Documentation MUST explain purpose, dataset, models, and how to run everything | README in tensorflow/cnn/ and pytorch/cnn/ directories; section in main project README |
| **VII. Virtual Environment Management** | All Python dependencies MUST be managed via UV with pyproject.toml | Dependencies added to pyproject.toml; confirmed via `uv lock` |
| **VIII. Experiment Tracking & Reproducibility** | **MANDATORY** All experiments MUST use MLflow with consistent parameter/metric naming (snake_case parameters, scoped metrics) | MLflow client integrated in training scripts; naming: `learning_rate`, `batch_size`, `train_loss`, `val_accuracy`, `test_precision`; model artifacts saved as `tensorflow_cnn_model`, `pytorch_cnn_model` |

### Gate Evaluation

✅ **PASS**: All constitutional requirements are addressable within this feature scope. No conflicts or unjustifiable violations detected. MLflow tracking (Principle VIII) is primary requirement driver; OOP structure (Principle I) and notebooks (Principle IV) are core deliverables.

## Project Structure

### Documentation (this feature)

```text
specs/001-cnn-mnist-frameworks/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md        # Phase 1 output (/speckit-plan command)
├── quickstart.md        # Phase 1 output (/speckit-plan command)
├── contracts/           # Phase 1 output (/speckit-plan command)
├── checklists/
│   └── requirements.md  # Spec quality checklist
└── tasks.md             # Phase 2 output (/speckit-tasks command - NOT created by /speckit-plan)
```

### Source Code (repository root)

```text
tensorflow/
├── cnn/
│   ├── __init__.py
│   ├── models.py        # CNN model class (Sequential or Functional)
│   ├── data.py          # MNIST data loading, preprocessing
│   ├── train.py         # Training loop with MLflow logging
│   ├── evaluate.py      # Evaluation metrics and visualization
│   └── README.md        # Framework-specific documentation

pytorch/
├── cnn/
│   ├── __init__.py
│   ├── models.py        # CNN model class (nn.Module)
│   ├── data.py          # MNIST data loading, preprocessing (PyTorch DataLoader)
│   ├── train.py         # Training loop with MLflow logging
│   ├── evaluate.py      # Evaluation metrics and visualization
│   └── README.md        # Framework-specific documentation

notebooks/
├── tensorflow_cnn_guide.ipynb    # Learning guide: data → model → training → evaluation
├── pytorch_cnn_guide.ipynb       # Learning guide: parallel structure to TensorFlow
└── README.md                      # Notebook navigation guide

tests/
├── test_tensorflow_cnn_models.py # Unit tests for TensorFlow model class
├── test_pytorch_cnn_models.py    # Unit tests for PyTorch model class
├── test_data_loading.py          # Integration tests for data utilities
└── conftest.py                    # pytest fixtures (fixtures for temporary files, MLflow tracking)

utils/
├── metrics.py           # Shared metrics computation (accuracy, precision, recall, loss)
├── mlflow_config.py     # MLflow experiment setup and logging utilities
└── __init__.py
```

**Structure Decision**: Framework-first organization (tensorflow/, pytorch/) ensures clear separation per Principle II. Within each framework, CNN-specific modules (models.py, data.py, train.py, evaluate.py) follow convention established by prior FCNN implementations. Notebooks in dedicated `notebooks/` directory enable portfolio presentation. Shared utilities in `utils/` reduce duplication. Tests organized by module type (models, data, integration). This structure is self-documenting: reviewers immediately understand framework comparison approach and can find TensorFlow or PyTorch code independently.

## Phase 0: Research & Clarification

**Status**: Ready to proceed. No NEEDS CLARIFICATION markers remain in spec. Technical context complete.

**Research Tasks** (if any unknowns existed, would be dispatched here):
- ✓ MLflow integration patterns for ML training pipelines
- ✓ CNN architecture best practices for MNIST (28×28 images)
- ✓ Cross-framework model architecture standardization
- ✓ Jupyter notebook best practices for portfolio presentation

**Artifacts Generated**: research.md (Phase 0 output)

## Phase 1: Design & Contracts

**Deliverables**:
1. `data-model.md` — Define entities (CNN model, MLflow experiment, metrics)
2. `contracts/mlflow-logging.md` — MLflow parameter/metric naming contract
3. `contracts/model-interface.md` — Model class interface for both frameworks
4. `quickstart.md` — Validation guide for running and comparing experiments

## Phase 2: Tasks Generation

**Next Command**: `/speckit-tasks` will generate dependency-ordered tasks in `tasks.md`

**Expected Output**: 
- Setup tasks (environment, dependencies)
- Implementation tasks (models, data loading, training, evaluation)
- Testing tasks (unit tests, integration tests)
- Documentation tasks (notebooks, READMEs)
- Validation tasks (linting, experiment comparison)
