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
- **GPU on Windows**: solved via a Dev Container (`.devcontainer/`), not WSL2-by-hand or a `tensorflow-directml` variant. Deps + CUDA lib registration are baked into the image at build time — see `.devcontainer/Dockerfile` and the GPU Caveat section below before touching this.

## Source of Truth

`notebooks/fashion_mnist.ipynb` is the reference implementation for the modeling approach. It is kept as-is (not rewritten) and should be treated as the design spec for `src/fashion_mnist/`:
- `load_fashion_mnist_data()` -> becomes `data.py`
- `create_image_augmentation()` -> becomes `augmentation.py`
- `core_fashion_model()` -> becomes `model.py`
- The training cell (`model.fit(...)` + callbacks) -> becomes `train.py`, parametrized via CLI

When productionizing a piece of the notebook, preserve its behavior exactly unless there's a good reason to change it — this is a translation/refactor, not a redesign.

## Key Architectural Pattern: GPU Augmentation

Augmentation layers (`RandomFlip`, `RandomRotation`, `RandomBrightness`, `RandomZoom`, `RandomTranslation`) must be wired into the model graph *before* the core CNN (i.e. `training_model = Model(inputs=aug_input, outputs=core_model(augmentation_layers(aug_input)))`), not applied via `tf.data.Dataset.map()`. This is what makes augmentation run on the GPU as part of `.fit()`. Keras preprocessing layers are automatically inert during `.evaluate()`/`.predict()`, so no manual toggling is needed — but this means **the saved/loaded model for evaluation should be the augmentation-free core model**, not the full training model with the augmentation head, unless you specifically want to evaluate through the (inert) augmentation layers too.

## GPU Caveat (verified during scaffolding) + Docker Fix (verified working)

TensorFlow >=2.11 has no native Windows GPU support (confirmed via `tf.config.list_physical_devices("GPU")` returning empty and TF's own warning on this machine, running TF 2.21 on native Windows). CUDA/cuDNN being installed doesn't change this. Don't assume `configure_gpu()` finding a GPU on a native Windows shell; that's expected to report none there.

The fix in this project is a Dev Container (`.devcontainer/`), verified end-to-end on this machine's RTX 2060 SUPER via Docker Desktop's WSL2 GPU passthrough (`docker run --gpus=all ...`). Two things had to be true simultaneously for this to work — both are already handled, but if GPU detection ever silently reverts to CPU inside the container, check these first:

1. **`tensorflow[and-cuda]` only on Linux.** `pyproject.toml` conditions the dependency on `sys_platform`: plain `tensorflow` on native Windows (`sys_platform != 'linux'`), `tensorflow[and-cuda]` inside the Linux container (`sys_platform == 'linux'`). This is what makes `uv sync` install CUDA/cuDNN as pip wheels (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, etc.) automatically inside the container without needing a CUDA toolkit baked into the base image.
2. **rpath doesn't work in a `uv`-managed venv — `ldconfig` does.** `tensorflow[and-cuda]`'s normal trick for finding its pip-installed CUDA libs is an rpath relative to its own install location; this does **not** resolve correctly when TensorFlow lives in a `uv`-created venv (confirmed by testing: GPU detection failed with "Cannot dlopen some GPU libraries" until fixed). The fix — `find .../site-packages/nvidia -mindepth 2 -maxdepth 2 -type d -name lib > /etc/ld.so.conf.d/nvidia-pip.conf && ldconfig` — is baked into `.devcontainer/Dockerfile` as a `RUN` step at build time, so it's permanent (part of the image layer) rather than something that has to be redone per container start. Doing this at container-start time instead of build time works too, but only for the lifetime of that specific container instance — a fresh `docker run` (as opposed to a rebuilt image) won't have it, which is why it belongs in the Dockerfile, not `postCreateCommand`.

This is not actually Docker-specific — the same rpath issue would hit anyone running `uv sync` with `tensorflow[and-cuda]` on native Linux too. Worth remembering if this project's GPU story ever moves off Windows.

The venv lives at `/opt/venv` (set via `UV_PROJECT_ENVIRONMENT` in the Dockerfile), **not** inside the bind-mounted `/workspace`, so the Linux-built venv never collides with a native-Windows `.venv` sitting in the same project folder.

## MLflow Conventions

- Experiment name: `fashion_mnist_cnn`
- Tracking URI is **environment-dependent, on purpose** (`mlflow_utils.TRACKING_URI`): `sqlite:///mlflow.db` inside the Linux devcontainer, `sqlite:///mlflow-native.db` on native Windows. See "Cross-environment fix" below before changing this.
- **Params to log**: `learning_rate`, `batch_size`, `num_epochs`, `n_conv`, `n_dense`, `kernel_size`, `global_pooling_type`, `horizontal_flip`, `rotation`, `brightness`, `zoom`, `translation`, `random_seed`
- **Per-epoch metrics**: `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy`
- **Final test metrics**: `test_loss`, `test_accuracy`, `test_precision`, `test_recall`, `test_f1` (macro-averaged — classes are balanced, ~6k samples each)
- **Artifacts**: best model (`.keras`, logged both as a raw file artifact and via `mlflow.tensorflow.log_model` with a `signature`/`input_example` for inference), confusion matrix plot (`.png`), classification report (`.txt`)
- Use snake_case for all logged param/metric names.
- **Run naming**: `train.py` accepts `--run-name` (optional; MLflow auto-generates a human-readable one like `capable-shrike-728` if omitted). Local artifacts are written to `results/<sanitized-run-name>-<short-run-id>/` (see `train.py`, `mlflow_utils.sanitize_run_name`) — every run gets its own directory, named for both human readability and guaranteed uniqueness, so results are recoverable and traceable back to their MLflow run instead of overwriting the previous run's files.

### Cross-environment fix: separate tracking stores, not a shared one

MLflow's local artifact store bakes an experiment's `artifact_location` as an **absolute path**, fixed at whichever moment/environment first created the experiment — confirmed by testing: an experiment created inside the Docker container got `artifact_location` pointing at `/workspace/mlruns/...` (a Linux path); a later native-Windows run against the *same* `mlflow.db` then tried to write artifacts there too, which Windows silently resolved to `C:\workspace\mlruns\...` — a bogus path at the drive root, completely outside the project. Also confirmed: passing a *relative* `artifact_location` doesn't help either — `mlflow.create_experiment()` normalizes it to an absolute, OS-specific path immediately at creation time, so there's no lazy-resolution trick available within MLflow's local file store. Params/metrics still log fine regardless (they go through the SQLite backend store, not the filesystem artifact store) — it's only the model/confusion-matrix/report *artifacts* that vanish to the wrong location.

Rather than document "don't mix environments" as a footgun to remember, `mlflow_utils.TRACKING_URI` makes a collision structurally impossible: each environment gets its own sqlite file (`mlflow.db` for the Linux container, `mlflow-native.db` for native Windows), decided automatically via `sys.platform`. Tradeoff knowingly accepted: this means two separate MLflow histories/UIs (`mlflow ui --backend-store-uri sqlite:///mlflow.db` vs `sqlite:///mlflow-native.db`) rather than one unified one — the alternative (a real fix) would be running an `mlflow server` process as the single source of truth that both environments talk to over HTTP instead of touching sqlite/local-filesystem directly, which was considered and explicitly deferred as unnecessary infrastructure for this project (native Windows is only ever used for quick CPU sanity checks, not real experiment history — the Docker container, with the GPU, is the canonical place real training results should accumulate).

## Commands

```bash
uv sync                                                    # install deps
uv run python -m fashion_mnist.train --epochs 50 ...       # train
uv run python -m fashion_mnist.evaluate --model-path ...   # standalone eval
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db          # view container/GPU runs
uv run mlflow ui --backend-store-uri sqlite:///mlflow-native.db   # view native-Windows runs
uv run jupyter notebook notebooks/fashion_mnist.ipynb      # explore notebook

# GPU training on Windows via Docker (see README "GPU Training via Docker" for full detail)
docker build -t fashion-mnist-gpu -f .devcontainer/Dockerfile .
docker run --rm --gpus=all -v "${PWD}:/workspace" -w /workspace fashion-mnist-gpu \
  bash -c "uv sync --extra dev && uv run python -m fashion_mnist.train --epochs 50 ..."
```

## Gitignore Reminders

Make sure these stay gitignored: `data/` (cached dataset), `mlruns/`, `mlflow.db`, `mlflow-native.db`, `results/*.keras`, `.venv/`.
