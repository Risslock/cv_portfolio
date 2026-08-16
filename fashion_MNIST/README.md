# Fashion MNIST: CNN Image Classification

A TensorFlow/Keras convolutional neural network project for classifying clothing articles from the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset. Built to demonstrate practical deep learning workflow: from an exploratory notebook to a production-style training pipeline with GPU-accelerated augmentation, configurable architectures, and MLflow experiment tracking.

## Objective

Implement and train a CNN to classify 10 categories of clothing (28x28 grayscale images), showcasing:
- Idiomatic TensorFlow/Keras model building (Functional API)
- GPU-accelerated image augmentation as part of the model graph
- Configurable, reproducible training runs via CLI hyperparameters
- Experiment tracking and artifact logging with MLflow
- Rigorous evaluation using scikit-learn (precision, recall, F1, confusion matrix)

## Labels

| Label | Description   |
|-------|---------------|
| 0     | T-shirt/top   |
| 1     | Trouser       |
| 2     | Pullover      |
| 3     | Dress         |
| 4     | Coat          |
| 5     | Sandal        |
| 6     | Shirt         |
| 7     | Sneaker       |
| 8     | Bag           |
| 9     | Ankle boot    |

## Tech Stack

- **TensorFlow / tf.keras** — model definition, training, GPU-side image augmentation
- **scikit-learn** — precision, recall, F1, confusion matrix
- **MLflow** — experiment tracking, parameter/metric logging, model artifacts (local SQLite backend)
- **matplotlib** — training curves and confusion matrix visualization
- **uv** — dependency and environment management
- **Docker / Dev Containers** — GPU-enabled training environment for Windows (see [GPU Training via Docker](#gpu-training-via-docker-windows))

## Project Structure

```
fashion_MNIST/
├── .devcontainer/
│   ├── Dockerfile             # GPU-enabled image: deps + CUDA libs baked in at build time
│   └── devcontainer.json      # VS Code Dev Container config (--gpus=all, port 5000 forwarded)
├── src/
│   └── fashion_mnist/
│       ├── __init__.py
│       ├── data.py           # load, split, and batch the Fashion MNIST dataset
│       ├── augmentation.py   # GPU-side Keras augmentation layers
│       ├── model.py          # parametrized CNN builder (core_fashion_model)
│       ├── train.py          # CLI entry point: training + MLflow logging
│       ├── evaluate.py       # sklearn metrics, confusion matrix, plots
│       └── mlflow_utils.py   # experiment setup, param/metric logging helpers
├── notebooks/
│   └── fashion_mnist.ipynb   # exploratory notebook — step-by-step walkthrough
├── results/                  # saved models, confusion matrix plots, run outputs
├── data/                     # cached dataset (gitignored)
├── mlruns/ + mlflow.db       # MLflow tracking store (gitignored)
├── pyproject.toml
├── uv.lock
└── README.md
```

## Approach

This project was built in two phases:

1. **Notebook exploration** (`notebooks/fashion_mnist.ipynb`) — an unhurried, step-by-step build: data loading, augmentation pipeline, a configurable CNN builder, and a first training run. This notebook is the reference design; it's kept as-is rather than rewritten.
2. **Production pipeline** (`src/fashion_mnist/`) — the notebook's ideas promoted into reusable, CLI-driven scripts with proper experiment tracking and evaluation, so different architectures/hyperparameters can be run and compared without editing code.

### GPU-accelerated augmentation

Image augmentation (`RandomFlip`, `RandomRotation`, `RandomBrightness`, `RandomZoom`, `RandomTranslation`) is implemented as Keras preprocessing layers wired directly into the model graph (ahead of the core CNN), rather than as a `tf.data` map step. This means augmentation runs on the GPU as part of the forward pass during `model.fit()`, and is automatically skipped at inference/evaluation time (Keras preprocessing layers are no-ops outside of training).

### Configurable architecture

The CNN builder supports tuning the number of conv blocks, number of dense blocks, kernel size, the two dropout rates (conv-side and dense-side, set independently), and the global pooling strategy (`max`, `avg`, or `flatten`), so different capacity/regularization trade-offs can be explored from the CLI without touching code. The pooling strategy turned out to matter most — see [Results](#results).

## Getting Started

### Installation

```bash
cd fashion_MNIST
uv sync
```

### Training

```bash
# Best known configuration — reproduces the 93.6% run in Results below
uv run python -m fashion_mnist.train \
  --epochs 200 \
  --batch-size 512 \
  --learning-rate 0.001 \
  --n-conv 5 \
  --n-dense 3 \
  --global-pooling-type flatten \
  --horizontal-flip \
  --zoom 0.05 \
  --translation 0.05 \
  --run-name baseline
```

Everything else falls back to its default: `--kernel-size 3`, `--conv-dropout-rate 0.5`, `--dense-dropout-rate 0.2`, `--rotation 0.0`, `--brightness 0.0`, `--seed 42`, `--val-split 0.1`, plus the callback settings (`--early-stopping-patience 5`, `--lr-patience 3`, `--lr-factor 0.1`).

`--run-name` is optional — MLflow auto-generates a human-readable name (e.g. `capable-shrike-728`) if omitted.

Each run:
- Trains with `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint` (best validation loss)
- Logs hyperparameters, per-epoch metrics, and final test metrics to MLflow
- Evaluates on the held-out test set with scikit-learn (precision/recall/F1, confusion matrix)
- Saves everything — the model (`.keras`), confusion matrix plot, and classification report — to its own `results/<run-name>-<run-id>/` directory, and logs the same files as MLflow artifacts, so every run's outputs are recoverable and traceable back to its MLflow entry instead of overwriting the previous run
- Logs the model with an inferred **signature** and a small **input example**, so it's ready for `mlflow.pyfunc.load_model()`-based inference without guessing input shape/dtype

### Viewing Experiments

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open http://localhost:5000 to compare runs, training curves, and confusion matrices.

> Native Windows and the Docker container log to separate MLflow stores (`mlflow.db` vs `mlflow-native.db`) — see [GPU Training via Docker](#gpu-training-via-docker-windows) for why.

### Standalone Evaluation

```bash
uv run python -m fashion_mnist.evaluate --model-path results/<run-name>-<run-id>/best_model.keras
```

Artifacts land next to the model by default (same run directory) unless `--output-dir` is given explicitly.

### Exploring the Notebook

```bash
uv run jupyter notebook notebooks/fashion_mnist.ipynb
```

## GPU Training via Docker (Windows)

TensorFlow ≥2.11 has no native GPU support on Windows — it silently falls back to CPU even with CUDA/cuDNN installed (see [GPU Notes](#gpu-notes) below). The fix used here: run training inside a **Dev Container** with the NVIDIA GPU passed through via Docker Desktop's WSL2 backend. Verified working end-to-end on this project's own hardware (RTX 2060 SUPER) — TensorFlow loads cuDNN, XLA compiles for the CUDA platform, and training is markedly faster than the CPU path.

### Prerequisites

1. **Docker Desktop**, using the WSL2 backend (default on modern installs)
2. An **NVIDIA GPU driver** on the Windows host that supports GPU passthrough to WSL2 (the regular Game Ready / Studio driver is enough — do *not* install a separate Linux driver inside WSL2)
3. Verify passthrough works before going further:
   ```bash
   docker run --rm --gpus=all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
   ```
   If this prints your GPU, you're set. If not, fix Docker Desktop/driver setup first — nothing below will work until this does.
4. (For the interactive workflow) VS Code with the **Dev Containers** extension

### How it's wired up

- `.devcontainer/Dockerfile` builds a `python:3.12-slim` image, installs `uv`, and runs `uv sync` **at build time** — so TensorFlow, MLflow, scikit-learn, etc. are baked into the image, not reinstalled on every container start.
- `pyproject.toml` conditions the TensorFlow dependency on platform: native Windows gets plain `tensorflow` (CPU only, since GPU isn't available there anyway); the Linux container gets `tensorflow[and-cuda]`, which pulls CUDA/cuDNN as pip wheels — no CUDA toolkit needs to be installed in the image itself.
- **Gotcha found while building this**: `tensorflow[and-cuda]`'s bundled CUDA/cuDNN libraries aren't discovered automatically when TensorFlow lives in a `uv`-managed venv (its rpath-based lookup assumes a plain pip/venv layout). The fix — registering the pip-installed library directories with `ldconfig` — is baked into the Dockerfile as a build step, so it's permanent and doesn't need to be redone per container.
- The venv lives at `/opt/venv` **outside** the bind-mounted project folder, so it never collides with a native-Windows `.venv` in the same directory.
- The project folder is bind-mounted at `/workspace`, so `results/`, `data/`, and `mlflow.db` written during a container training run land directly on the host filesystem — `mlflow ui` and `uv run python -m fashion_mnist.evaluate` on native Windows see them immediately, no copying needed.

### Usage: VS Code Dev Container (interactive)

1. Open the `fashion_MNIST/` folder in VS Code
2. Command Palette → **Dev Containers: Reopen in Container**
3. Once it's up, use the integrated terminal exactly like the native workflow:
   ```bash
   uv run python -m fashion_mnist.train --epochs 100 --n-conv 4 --n-dense 2 --global-pooling-type flatten
   ```
4. Port 5000 is forwarded automatically, so `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db` is reachable from the host browser.
5. Rebuild the container (**Dev Containers: Rebuild Container**) whenever `pyproject.toml` or `uv.lock` changes.

> **Why native Windows and the container use separate MLflow stores.** MLflow bakes each experiment's artifact storage location as an absolute path at creation time. If the experiment were shared, a run from one environment could silently send its model/plot artifacts to a path that only makes sense in the *other* environment (confirmed while building this: a native-Windows run against a container-created experiment tried to write to `C:\workspace\...`, outside the project entirely — and a relative path doesn't dodge this either, since MLflow resolves it to an absolute one immediately). Rather than rely on remembering not to mix environments, the container uses `mlflow.db` and native Windows automatically uses a separate `mlflow-native.db` (see `mlflow_utils.py`) — a collision is structurally impossible. Native Windows is meant for quick CPU sanity checks anyway; the container (with the GPU) is where real experiment history should accumulate. View each with `mlflow ui --backend-store-uri sqlite:///mlflow.db` or `sqlite:///mlflow-native.db` respectively.

### Usage: plain Docker CLI (scripting/automation, no VS Code needed)

```bash
docker build -t fashion-mnist-gpu -f .devcontainer/Dockerfile .

docker run --rm --gpus=all \
  -v "${PWD}:/workspace" -w /workspace \
  fashion-mnist-gpu \
  bash -c "uv sync --extra dev && uv run python -m fashion_mnist.train --epochs 100 --n-conv 4 --n-dense 2 --global-pooling-type flatten"
```

The `uv sync` inside the container is fast (dependencies are already in the image; this just links the local `fashion_mnist` package now that the source is mounted in).

## GPU Notes

TensorFlow ≥2.11 has no native GPU support on Windows — it silently falls back to CPU even with CUDA/cuDNN installed. See [GPU Training via Docker](#gpu-training-via-docker-windows) above for the fix used in this project. `configure_gpu()` and the augmentation-in-model design work unchanged in the container; only the runtime environment differs.

## Data Splits

- 54,000 train / 6,000 validation / 10,000 test (standard Fashion MNIST test set, 10% of train held out for validation)
- Pixel values normalized to [0, 1]
- Fixed random seed for reproducibility

## Results

Best run to date: **93.62% test accuracy** (macro F1 0.936) on the standard 10,000-image test set, from the production pipeline with GPU training in the Dev Container. For reference, the exploratory notebook prototype reached ~88.6%.

All runs share `batch_size=512`, `learning_rate=0.001`, `seed=42`, Adam, and early stopping on validation loss; metrics are read from `mlflow.db`. **One run per configuration, no repeated seeds** — so no noise floor was measured here, and sub-1-point deltas below should be treated as unresolved rather than real. The larger effects are called out on that basis.

### Architecture sweep

Seven configurations, no augmentation, 100-epoch cap — so architecture is the only variable across the whole block:

| Conv blocks | Dense blocks | Pooling | Test accuracy |
|---|---|---|---|
| 5 | 3 | `flatten` | **93.56%** |
| 5 | 3 | `max` | 92.55% |
| 6 | 2 | `max` | 92.44% |
| 4 | 2 | `max` | 92.28% |
| 4 | 3 | `max` | 91.68% |
| 4 | 2 | `avg` | 90.92% |
| 3 | 3 | `max` | 90.43% |

- **`flatten` beats global pooling**, and it's the largest architectural effect here: +1.01 points over `max` at identical 5-conv/3-dense depth, and `max` in turn is +1.36 over `avg` at 4-conv/2-dense. On 28×28 inputs the final feature map is small enough that flattening preserves spatial layout that global pooling averages away — the usual case for global pooling assumes larger maps and a parameter count worth cutting, neither of which applies at this scale.
- **Depth helps, then plateaus** — at `max` pooling, 3 → 4 → 5 conv blocks climbs 90.43% → 92.28% → 92.55%, but 6 blocks (92.44%) does not continue the trend.

### Augmentation: the clearest result, and it's negative

Three runs at 200 epochs with a horizontal flip and mild (0.05) zoom/translation isolate rotation and pooling one at a time:

| Conv/Dense | Pooling | Rotation | Test accuracy | Δ vs. row above |
|---|---|---|---|---|
| 5 / 3 | `flatten` | 0.0 | **93.62%** | — |
| 5 / 3 | `flatten` | 0.1 | 90.52% | **−3.10** |
| 5 / 3 | `max` | 0.1 | 88.02% | **−2.50** |

**Rotation is actively harmful** — 0.1 (±10%) costs 3.10 points against an otherwise byte-identical run, the largest single effect measured in this project and comfortably beyond anything a seed difference would plausibly explain. Fashion MNIST items are centered, upright, and consistently scaled by construction, so rotation manufactures poses that never appear at test time and spends model capacity on them. The second row-pair independently reproduces the pooling finding from the sweep above, at +2.50 for `flatten`.

The uncomfortable corollary: **augmentation bought essentially nothing overall.** The best augmented run (93.62%, flip + mild zoom/translation, 200 epochs) beats the best un-augmented one (93.56%, no augmentation at all, 100 epochs) by 0.06 points — noise, by any reasonable standard, for twice the training budget. This dataset is large, balanced, and low-variance enough that the augmentations available here have little left to add, and the one aggressive setting tried made things distinctly worse.

## Status

✅ **Production pipeline complete.** `src/fashion_mnist/` (data, augmentation, model, train, evaluate, MLflow utils) is built and in use, with 11 completed runs tracked (one a 1-epoch smoke test) and the GPU Dev Container verified end-to-end.

Not planned, but the honest next steps: **repeated seeds per configuration**, to establish a noise floor and settle whether the ~1-point pooling deltas are real — the rotation and `flatten`-vs-`max` effects are large enough to survive that scrutiny, the depth ordering probably isn't. After that, a proper hyperparameter search rather than a hand-driven sweep.
