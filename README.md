# Computer Vision Portfolio

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c) ![TensorFlow](https://img.shields.io/badge/TensorFlow-ff6f00) ![Env](https://img.shields.io/badge/env-uv-purple) ![Tracking](https://img.shields.io/badge/tracking-MLflow-0194e2)

A monorepo of self-contained computer vision projects — image classification, object detection, instance segmentation, and a planned generative/geospatial track. Each project is treated as its own deliverable: a real problem, a reproducible environment, tracked experiments, and results reported honestly, including the negative ones.

**Start here:** [`poultry_monitoring/`](poultry_monitoring/README.md) — the most developed project, and the one that best represents how I work.

<p align="center">
  <img src="poultry_monitoring/docs/images/test_prediction_1.jpg" width="800" alt="YOLO26 detecting 60 chickens in a dense overhead poultry-farm image from the held-out test split">
</p>
<p align="center"><em>YOLO26 on a held-out test frame: 60 birds detected in a dense, high-occlusion overhead scene.</em></p>

## Projects at a Glance

| Project | Task | Status | Core stack | Headline result |
|---|---|---|---|---|
| [**Poultry Monitoring**](poultry_monitoring/README.md) | Object detection + instance segmentation | 🟡 Active | Ultralytics YOLO26, Albumentations, Optuna, MLflow | Detection val mAP50-95 **0.893**; all four segmentation arms beat the published mask baseline |
| [**MNIST**](MNIST/README.md) | Multi-class classification | ✅ Complete | TensorFlow + PyTorch, MLflow, spec-kit | CNN test accuracy **99.18%** — matched across both frameworks |
| [**Fashion MNIST**](fashion_MNIST/README.md) | Multi-class classification | ✅ Pipeline complete | TensorFlow/Keras, GPU Dev Container, MLflow | **93.6%** test accuracy; augmentation measured as a net negative |
| [Medical Segmentation](medical_segmentation/) | Semantic segmentation | 🔲 Planned | — | — |
| [Satellite Water & Irrigation](satellite_water/) | Multispectral segmentation | 🔲 Planned | — | — |

Legend: ✅ done · 🟡 in progress · 🔲 not started

---

## 🐔 Poultry Monitoring — Detection & Instance Segmentation

**Detecting and segmenting individual chickens in dense, high-occlusion overhead farm imagery**, plus a synthetic copy-paste pipeline that makes flock density a controllable training variable rather than whatever the camera happened to capture.

The dataset is [ChickenVerse](https://github.com/amirivojdan/ChickenVerse) (ChickenDet subset): 6,539 overhead images, 153,764 annotated instances, ~23.5 birds per image and 50+ in the densest frames — a genuine occlusion and small-object problem, not a single-object toy set.

**What's done:**

- **Detection** — `yolo26n` tuned with Optuna, augmented with a domain-informed directional brightness/contrast transform, and progressively unfrozen to val **mAP50-95 = 0.893**, marginally ahead of ChickenVerse's published baseline.
- **Instance segmentation** — baselines and copy-paste arms trained at two model sizes, all four ahead of the published mask mAP50-95, then scored once on a genuinely held-out test split.
- **A real finding, not just a number** — synthetic copy-paste *flips sign with model capacity*: box mAP50-95 **+1.19** on `yolo26n-seg`, **−0.72** on `yolo26s-seg`. The effect reproduces on the held-out split and scales monotonically with scene density, up to **+2.01** and **−2.30** in the most crowded third of frames.

<p align="center">
  <img src="poultry_monitoring/docs/images/copy_paste_density_effect.png" width="720" alt="Line chart showing copy-paste's change in box mAP50-95 across sparse, medium and dense test images, rising for yolo26n-seg and falling for yolo26s-seg">
</p>

**Why it's the flagship piece:** it carries the practices the rest of the repo aims at — a metric-noise floor established before any delta is called real, a test split held back until every decision was fixed, 18 [ADRs](poultry_monitoring/docs/adr/README.md) recording rejected alternatives, and a deployment section on why a model trained at five facilities doesn't simply transfer to a sixth.

**Next:** DETR as a secondary transformer track, NVIDIA DALI data loading, and ONNX/LiteRT export with latency benchmarks.

→ [Full README](poultry_monitoring/README.md) · [Roadmap](poultry_monitoring/plan.md) · [Engineering notes](poultry_monitoring/docs/engineering-notes.md)

---

## 🔤 MNIST — Framework & Architecture Comparison

The foundational classification pipeline, built as a **four-way comparison**: FCNN and CNN, each implemented in both TensorFlow and PyTorch, with identical architectures, splits, and seeds so the frameworks are the only variable.

| Architecture | TensorFlow | PyTorch |
|---|---|---|
| CNN | **99.18%** | **99.18%** |
| FCNN | 97.04% | 98.04% |

*Test accuracy, best tracked run per experiment.*

Both CNNs converge to the same number, which is the point — the comparison is about idioms (`model.fit()` vs. an explicit training loop, NHWC vs. NCHW, callbacks vs. hand-rolled early stopping), not a winner. All runs are MLflow-tracked with standardized snake_case parameter and metric names, early stopping, and checkpointing.

This is also the repo's **spec-kit** project: `.specify/`, per-feature `specs/` with plan and task breakdowns, and a versioned constitution.

→ [Full README](MNIST/README.md)

---

## 👗 Fashion MNIST — Notebook to Production Pipeline

Classifying 10 clothing categories with a configurable Keras CNN. The project's real subject is the **path from an exploratory notebook to a CLI-driven, tracked training pipeline** — and solving GPU training on Windows.

- **93.6% test accuracy**, best of 11 MLflow-tracked runs from the production pipeline.
- **A clean negative result on augmentation** — rotation at ±10% costs **3.1 points** against an otherwise identical run, and augmentation overall bought 0.06 points over no augmentation at all. Fashion MNIST items are centered, upright, and consistently scaled, so geometric jitter fabricates poses the test set never contains.
- **GPU augmentation inside the model graph** — Keras preprocessing layers wired ahead of the core CNN, so augmentation runs on the GPU during `fit()` and goes inert at inference automatically.
- **GPU on Windows via a Dev Container** — TensorFlow ≥2.11 has no native Windows GPU support. Solved with a CUDA-enabled Dev Container over Docker's WSL2 passthrough, including a non-obvious fix: `tensorflow[and-cuda]`'s rpath lookup fails inside a `uv`-managed venv, so the CUDA library paths are registered with `ldconfig` at image build time.
- **Structurally separated tracking stores** — MLflow bakes artifact paths as absolute at experiment creation, so the container and native Windows write to different SQLite backends rather than relying on remembering not to mix them.

→ [Full README](fashion_MNIST/README.md)

---

## 🗺️ Roadmap

Placeholder directories exist for the first two; the rest are scoped but unstarted.

| Planned project | Focus |
|---|---|
| **Medical Segmentation** | Pixel-level segmentation of CT/MRI/X-ray regions of interest; U-Net variants, class imbalance, IoU/Dice |
| **Satellite Water & Irrigation** | Multispectral Sentinel-2/Landsat-8 segmentation; NDWI/SWI indices, Google Earth Engine, large-scale geospatial tiling |
| **Agricultural Segmentation** | Crop-type and crop-health segmentation from drone/satellite imagery |
| **CNN Explainability** | Grad-CAM, activation analysis, and failure debugging on models already trained in this repo |
| **Style Transfer** | Perceptual loss, Gram matrices, content/style trade-offs |
| **Image Generation** | VAE → GAN → diffusion, with a focus on training stability |
| **Multi-Object Tracking** | Temporal association and re-identification, built on the poultry detector |

Rather than standalone "technique" folders, cross-cutting concerns — transfer learning, augmentation, quantization, export — are demonstrated inside the project where they actually matter.

---

## Repository Conventions

Projects are deliberately **not** forced into one template, but they do share a spine:

| Concern | Convention |
|---|---|
| **Environments** | `uv` + `pyproject.toml`/`uv.lock`, scoped **per project** — no shared root environment. Python versions differ by project. |
| **Experiment tracking** | MLflow with a local SQLite backend, snake_case param/metric names, and per-run artifact directories |
| **Exploration** | Notebook-first, then productionized into a `src/` package — the notebook records what was learned, not what ships |
| **Code quality** | `ruff` (lint + format) and `pytest` smoke tests on deterministic, non-ML code; no gate on model convergence |
| **Design decisions** | Non-obvious calls get an ADR with the rejected alternative and why it failed, instead of long inline comments |
| **Results** | Deltas are reported against a measured noise floor; held-out splits stay untouched until decisions are final |

## AI-Assisted Working Styles

This repo is also a deliberate comparison of **how much process to bring to an AI-assisted project**, scaled to each project's actual size and shape:

| Project | Formality | Setup |
|---|---|---|
| [`MNIST/`](MNIST/) | Heaviest | Full spec-kit — `.specify/`, `specs/` with per-feature plan and task breakdowns, versioned `constitution.md`. Suits work built incrementally, feature by feature. |
| [`fashion_MNIST/`](fashion_MNIST/) | Lightest | A single `CLAUDE.md` alongside the README as the sole source of truth. Suits a small, single-framework, single-architecture project. |
| [`poultry_monitoring/`](poultry_monitoring/) | Hybrid | `constitution.md` (principles) + `plan.md` (phased roadmap) + `CLAUDE.md` (conventions) + `docs/adr/` — no spec-kit tooling. Suits a project planned end-to-end upfront, too large for one file but not incremental enough for full spec-kit. |

None of these is "the right way." They're presented side by side because picking the wrong formality — either direction — is a real cost, and the comparison is more useful than an opinion.

## Getting Started

Each project is independent; there is no root-level environment to install.

```bash
git clone <repository-url>
cd cv_portfolio

# Pick a project and set it up there
cd poultry_monitoring
uv sync --extra dev

# Run something
uv run python -m poultry_monitoring.detection.yolo predict --weights <path/to/best.pt> --source <images/>

# Inspect the experiment history
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Exact commands vary — datasets, GPU requirements, and entry points differ per project, so **the project's own README is authoritative**. `fashion_MNIST/` additionally needs Docker Desktop with WSL2 GPU passthrough for GPU training.

## Licensing

Licensing is **per project**, not repo-wide. [`poultry_monitoring/`](poultry_monitoring/LICENSE) is AGPL-3.0 (required by its Ultralytics dependency), and its dataset, donor bank, and trained weights inherit ChickenVerse's CC BY-NC-SA 4.0 terms — non-commercial, attribution, share-alike. Check each project's own `LICENSE`/`NOTICE` before reusing anything.
