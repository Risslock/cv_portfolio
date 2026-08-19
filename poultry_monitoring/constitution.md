# Poultry Monitoring Constitution

A living document defining the principles governing this project. This is a **hybrid, lighter-weight sibling** to `../MNIST/`'s full spec-kit constitution — same *kind* of document (non-negotiable-ish project law), smaller footprint (no `.specify/` tooling, no per-feature `specs/`). See `CLAUDE.md` for why this project uses this particular workflow, and `plan.md` for the phased roadmap this constitution governs.

## Core Principles

### I. Functional, Pipeline-First Code

All first-party code (data loading, augmentation, export, inference, benchmarking, MLflow helpers) MUST be plain functions with type hints — no classes for the sake of structure. Where a framework's idiomatic API is itself class-based (`ultralytics.YOLO`, Hugging Face `Trainer`), use it directly and unwrapped rather than forcing a functional facade over it. The line is: **our glue code is functional; the frameworks' own APIs are used as designed.**

**Rationale**: Consistent with the repo's `fashion_MNIST` convention. Avoids busywork abstraction over two already-opinionated training frameworks.

### II. Notebook-First Exploration, Then Productionize

Each major new capability (a model family's first fine-tune, a new augmentation idea, an export/quantization path) MUST start as a Jupyter notebook — run on Google Colab's free T4, or on the local `uv`-managed GPU environment (Principle IX), whichever's more convenient at the time — before being productionized into `src/poultry_monitoring/`. The notebook establishes *that it works* and is kept as the record of that exploration; the productionized function is not required to mirror it line-for-line — improving structure, naming, or approach while translating is expected and encouraged, not just tolerated. Deliberate deviations worth remembering (a redesign, not just a refactor) SHOULD be noted in the module docstring or a short comment, so it's clear the notebook and the shipped code are meant to differ there.

**Rationale**: Separates the learning/exploration loop (fast iteration, visual feedback, free GPU, no packaging overhead) from the production loop (reusable, testable, best-practice functions gated per Principle VIII). Unlike `fashion_MNIST` — a small, already-correct reference notebook — this project's notebooks are exploratory by nature (two unfamiliar architectures, new pipeline components), so faithfully preserving their behavior isn't the goal; capturing *what was learned* is.

### III. Task-Separated, Shared-Utility Structure

The package is organized by **task** (`detection/` vs. `segmentation/`), not by model family — each task directory holds both models' wrappers (`yolo.py`, `detr.py`) side by side. Genuinely shared logic (COCO/data loading, MLflow helpers, export, the model-load/inference path, the benchmarking harness) lives once in shared top-level modules — never duplicated per task or per model. Metric computation is the documented exception: box+mask extraction lives in `segmentation/yolo.py`'s `extract_metrics` as the single source of truth, imported by the shared harnesses, rather than being lifted into a `metrics.py` that would duplicate it.

**Rationale**: Keeping both models' code for a given task together keeps the codebase legible task-by-task regardless of which model is currently getting attention, and leaves a fair comparison possible in the source itself if/when Principle IV's comparison work happens — without making that comparison the reason the structure exists (see Principle IV and `CLAUDE.md` § Project Intent: YOLO26 is the primary deliverable, DETR a secondary practice track).

### IV. Fair Head-to-Head Comparison, If and When Made (SHOULD)

YOLO26 is this project's primary deliverable; DETR is a secondary track pursued as the user's own transformer-architecture practice, not a requirement for calling YOLO26's detection/segmentation work done. Comparison is not the project's goal for its own sake — but *if* a comparison claim is ever made in the README or results write-up, YOLO26 and DETR SHOULD be evaluated on identical train/val/test splits with identical metrics (box mAP@50 / mAP@50-95 for detection, mask mAP for segmentation). If a deviation is necessary (e.g. DETR needing a different image size, or a segmentation-capable variant swapped in), it SHOULD be stated explicitly next to the claim, not left implicit.

**Rationale**: A soft rule, not a hard gate — real constraints (compute, framework quirks) may force asymmetries. What matters is that asymmetries are disclosed, not eliminated at all costs.

### V. Benchmarking Rigor (SHOULD)

Latency/throughput numbers (export target comparisons) SHOULD report hardware, batch size, precision (FP32/FP16/INT8) and the device the number was measured on, and SHOULD be averaged over multiple runs with warmup iterations excluded. CPU measurements SHOULD additionally pin and disclose the thread count — left unpinned, they are not reproducible run-to-run.

**Rationale**: Unqualified benchmark numbers are not credible in a portfolio context. Soft rule for the same reason as Principle IV — favors disclosure over blocking on perfect methodology.

### VI. Dataset Licensing & Attribution (MUST)

ChickenVerse is **CC BY-NC-SA 4.0**. Non-commercial use only, attribution to the original authors required, any derivative or adaptation shared under the same license. This project MUST NOT be repurposed for a commercial use case without revisiting the dataset's license first.

**Rationale**: Only hard legal/ethical constraint in this project; unlike Principles IV/V, there's no tradeoff to weigh here.

### VII. Reproducibility

Random seeds MUST be fixed and logged. Dependencies MUST be pinned via `uv.lock`, committed to the repo. Every training/benchmark run MUST be tracked in MLflow with its parameters, metrics, and artifacts (see `CLAUDE.md` § MLflow Conventions for the concrete schema).

**Rationale**: Baseline expectation for any experiment claimed as a comparison — without it, "YOLO26 beat DETR" is not a reproducible claim.

### VIII. Basic Quality Gates

`ruff` lint and format checks MUST pass. `pytest` smoke tests MUST pass for deterministic, non-ML code — COCO annotation parsing, augmentation output shapes/ranges, export I/O round-trips. Model training/convergence itself is explicitly **not** gated (that's an experiment outcome, not a correctness bug). These gates are run **locally, on demand** via the command line — no CI pipeline. See `CLAUDE.md` § Gates for exact commands.

**Rationale**: Gates should catch pipeline bugs (a malformed COCO parse, a broken export) cheaply and fast, without pretending training scripts are unit-testable in the same way.

### IX. Environment

Two GPU-capable environments, not mutually exclusive: **Google Colab** (free T4 GPU), and a local **`uv`-managed virtual environment on native Windows** (no Docker/WSL2 — unlike `fashion_MNIST`, PyTorch has native Windows CUDA support, so no container workaround is needed). The local env is the required target for productionized training/benchmarking runs that need to be reproducible on this machine (Principle VII); for Principle II's exploratory notebooks it's now a verified alternative to Colab (`torch.cuda.is_available()` confirmed `True` against the local RTX 2060 SUPER via `uv sync --extra dev`) — use whichever is more convenient at the time, Colab for a disposable/shareable run, local when the dataset's already on disk and Colab's session limits aren't worth the trip. `torch`/`torchvision` in the local env MUST be pinned against a CUDA wheel index in `pyproject.toml`, not left to default to the CPU-only PyPI wheel. NVIDIA DALI's packaging story for this env was never resolved and no longer needs to be — the DALI phase is closed on measured grounds ([ADR 0019](docs/adr/0019-discard-dali.md)), so the question is moot rather than answered. The same caution still applies to anything new: **do not assume a backend has native Windows/current-Python wheels without checking first.** Phase 6's export backends are the live instance of this — each one's wheel availability is verified before it enters the benchmark matrix, and a backend that can't be installed is dropped and documented rather than worked around by changing the pinned interpreter.

**Rationale**: Colab's free T4 is still the right tool for cheap, disposable exploration when nothing's on this machine yet; the local RTX 2060 env is what real experiment history (Principle VII) accumulates against, matching this machine's verified hardware (CUDA 13.1-capable driver) and avoiding the TensorFlow-on-Windows problem `fashion_MNIST` hit — which doesn't apply here since this project is PyTorch-based. Now that both are confirmed GPU-capable, Principle II notebooks aren't locked to Colab specifically — the point of Principle II is the notebook-first workflow, not the hosting.

## Technology Stack

- **Training framework**: PyTorch (via `ultralytics` for YOLO26, via `transformers` for DETR)
- **Models**: YOLO26 (Ultralytics) and DETR (Hugging Face `transformers`), both for detection and instance segmentation
- **Dataset**: ChickenVerse / ChickenDet (COCO-format boxes + masks)
- **Augmentation**: [Albumentations](https://albumentations.ai/) — chosen over DALI for augmentation specifically ([rationale](https://albumentations.ai/docs/albumentations-vs-dali/)): richer target-aware transform set (bbox/mask/keypoint-aware in one `A.Compose`), reproducible/serializable configs, no GPU-memory contention with training itself. Runs CPU-side during data loading.
- **Data loading**: standard PyTorch `DataLoader`. NVIDIA DALI was held as a possible decode/loading throughput optimization, gated on the data path proving to be the bottleneck — **that gate was evaluated and closed**: the training loop is GPU-bound, so DALI is not pursued ([ADR 0019](docs/adr/0019-discard-dali.md))
- **Experiment tracking**: MLflow, local SQLite backend
- **Export**: ONNX Runtime (CPU + CUDA EP), OpenVINO (CPU, INT8 via NNCF), TensorRT (GPU, FP16/INT8). TensorFlow Lite / LiteRT was in the original stack but is demoted to a documented "attempted / why not" — its `tensorflow` + `onnx2tf` toolchain is the least likely to install on this env (Principle IX). Each backend's wheel availability is verified before it enters the benchmark matrix
- **Dependency management**: `uv` with `pyproject.toml` + `uv.lock`
- **Quality gates**: `ruff` (lint + format), `pytest` (smoke tests only)
- **Language**: Python 3.11+

## Governance

This constitution governs `poultry_monitoring/` only — it does not apply repo-wide (see root [README.md](../README.md) § Project Philosophy for why different projects in this portfolio intentionally use different workflow formalities). Amendments happen by editing this file directly; no sync-report tooling like `../MNIST/.specify/memory/constitution.md` — that's the point of the "hybrid" tier being lighter than full spec-kit.

**Version**: 1.1.0 | **Ratified**: 2026-08-06 | **Last Amended**: 2026-08-19

Amendments in 1.1.0: DALI removed from Principles I/II/III/V/IX and the Technology Stack, following [ADR 0019](docs/adr/0019-discard-dali.md) (phase closed on measured grounds, not deleted from the record). Principle V tightened to require the measurement device and a pinned CPU thread count. Principle III now names the `extract_metrics` exception explicitly. Export stack updated to ONNX Runtime / OpenVINO / TensorRT with LiteRT demoted.
