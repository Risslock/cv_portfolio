# CLAUDE.md — fashion_MNIST

Guidance for Claude Code when working in this directory.

## Project Intent

A portfolio piece showing a full deep learning workflow on Fashion MNIST: exploratory notebook -> production-style, CLI-driven training pipeline with MLflow tracking. It is intentionally scoped narrower than the sibling `../MNIST/` project (TensorFlow only, single CNN architecture family, no PyTorch comparison, no spec-kit docs) — do not import patterns from `../MNIST/` wholesale without checking they fit here.

## Decisions Already Made (don't re-litigate these)

- **Framework**: TensorFlow / tf.keras only. No PyTorch.
- **Environment**: `uv` with `pyproject.toml` + `uv.lock`, scoped to this directory (not the repo root).
- **Code style**: Function-based modules (mirrors the notebook), not OOP/class-based like `../MNIST/`. Keep `src/fashion_mnist/*.py` as plain functions with type hints, not model/data classes.
- **CLI**: `argparse`, one flag per hyperparameter, run via `python -m fashion_mnist.train`.
- **No formal spec-kit workflow** (no `.specify/`, `spec.md`, `plan.md`, `constitution.md`). Use the README and this file as the source of truth.
- **No test suite for now.** Don't add pytest scaffolding unless explicitly asked.
- **MLflow backend**: local SQLite (`sqlite:///mlflow.db`), not the plain file store — matches `../MNIST/` for consistency across the repo.

## Source of Truth

`notebooks/fashion_mnist.ipynb` is the reference implementation for the modeling approach. It is kept as-is (not rewritten) and should be treated as the design spec for `src/fashion_mnist/`:
- `load_fashion_mnist_data()` -> becomes `data.py`
- `create_image_augmentation()` -> becomes `augmentation.py`
- `core_fashion_model()` -> becomes `model.py`
- The training cell (`model.fit(...)` + callbacks) -> becomes `train.py`, parametrized via CLI

When productionizing a piece of the notebook, preserve its behavior exactly unless there's a good reason to change it — this is a translation/refactor, not a redesign.

## Key Architectural Pattern: GPU Augmentation

Augmentation layers (`RandomFlip`, `RandomRotation`, `RandomBrightness`, `RandomZoom`, `RandomTranslation`) must be wired into the model graph *before* the core CNN (i.e. `training_model = Model(inputs=aug_input, outputs=core_model(augmentation_layers(aug_input)))`), not applied via `tf.data.Dataset.map()`. This is what makes augmentation run on the GPU as part of `.fit()`. Keras preprocessing layers are automatically inert during `.evaluate()`/`.predict()`, so no manual toggling is needed — but this means **the saved/loaded model for evaluation should be the augmentation-free core model**, not the full training model with the augmentation head, unless you specifically want to evaluate through the (inert) augmentation layers too.

## GPU Caveat (verified during scaffolding)

TensorFlow >=2.11 has no native Windows GPU support (confirmed via `tf.config.list_physical_devices("GPU")` returning empty and TF's own warning on this machine, running TF 2.21 on native Windows). CUDA/cuDNN being installed doesn't change this. If GPU-accelerated training/augmentation needs to be demonstrated live, it must run under WSL2 or Linux — don't assume `configure_gpu()` finding a GPU on a native Windows shell; that's expected to report none here.

## MLflow Conventions

- Experiment name: `fashion_mnist_cnn`
- Tracking URI: `sqlite:///mlflow.db` (relative to `fashion_MNIST/`)
- **Params to log**: `learning_rate`, `batch_size`, `num_epochs`, `n_conv`, `n_dense`, `kernel_size`, `global_pooling_type`, `horizontal_flip`, `rotation`, `brightness`, `zoom`, `translation`, `random_seed`
- **Per-epoch metrics**: `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy`
- **Final test metrics**: `test_loss`, `test_accuracy`, `test_precision`, `test_recall`, `test_f1` (macro-averaged — classes are balanced, ~6k samples each)
- **Artifacts**: best model (`.keras`), confusion matrix plot (`.png`), classification report (`.txt`)
- Use snake_case for all logged param/metric names.

## Commands

```bash
uv sync                                                    # install deps
uv run python -m fashion_mnist.train --epochs 50 ...       # train
uv run python -m fashion_mnist.evaluate --model-path ...   # standalone eval
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # view experiments
uv run jupyter notebook notebooks/fashion_mnist.ipynb      # explore notebook
```

## Gitignore Reminders

Make sure these stay gitignored: `data/` (cached dataset), `mlruns/`, `mlflow.db`, `results/*.keras`, `.venv/`.
