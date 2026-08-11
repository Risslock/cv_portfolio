# CLAUDE.md — poultry_monitoring

Guidance for Claude Code when working in this directory.

## Project Intent

A portfolio piece training and comparing two architecture families — Ultralytics **YOLO26** (CNN, anchor-free/NMS-free) and **DETR** (transformer, set-prediction) — head-to-head on the same dense, high-occlusion dataset (ChickenVerse/ChickenDet), for both object detection and instance segmentation. Beyond model accuracy, it also covers domain-specific augmentation, a GPU-accelerated data loading pipeline (NVIDIA DALI vs. standard `DataLoader`), and export/optimization (ONNX Runtime, TFLite/LiteRT) with latency/throughput benchmarking. See the [README](README.md) for the full problem statement and dataset details.

## Workflow: Why This Project Uses Hybrid (constitution.md + plan.md + CLAUDE.md)

This repo intentionally uses **different workflow formalities per project**, as a deliberate portfolio point about working styles in the AI-assisted-development era (see root [README.md](../README.md) § Project Philosophy):

- `../MNIST/` — full spec-kit (`.specify/`, `specs/`, `plan`/`tasks` per feature): heaviest formality, suited to a project built feature-by-feature over time.
- `../fashion_MNIST/CLAUDE.md` — lightest: just a CLAUDE.md + README, no separate governance docs. Suited to a small, single-framework, single-architecture project.
- **`poultry_monitoring/` (here) — hybrid**: `constitution.md` (non-negotiable-ish principles) + `plan.md` (phased roadmap, updated as work progresses) + this file (concrete working conventions, "decisions already made," commands). No `.specify/` tooling, no per-feature `specs/` — those are overkill for a project planned end-to-end upfront. This tier fits a project too large/multi-part for a single CLAUDE.md, but not built incrementally enough to need full spec-kit.

**Read `constitution.md` and `plan.md` before making structural decisions here.** This file assumes both and focuses on the "how to actually work in this repo, today" layer.

## Decisions Already Made (don't re-litigate these)

- **Framework**: PyTorch, via `ultralytics` (YOLO26) and `transformers` (DETR). No TensorFlow.
- **Code style**: Plain functions with type hints for all first-party code; framework classes (`ultralytics.YOLO`, HF `Trainer`) used natively, unwrapped. See constitution Principle I.
- **Workflow**: Notebook-first exploration — on Colab or the local `.venv` (both GPU-capable, see § Environment below) — then productionize into `src/poultry_monitoring/`; translation is allowed to improve on the notebook, not required to mirror it. See constitution Principle II.
- **Package layout**: task-separated (`detection/` vs. `segmentation/`), shared utilities live once at the top level. See § Package Layout below and constitution Principle III.
- **Environment**: `uv` + `pyproject.toml`/`uv.lock`, native Windows (no Docker/WSL2 — PyTorch has native Windows CUDA support, unlike the TensorFlow situation in `../fashion_MNIST/`). `torch`/`torchvision` pinned to a CUDA wheel index — verify `torch.cuda.is_available()` is `True` after `uv sync`, don't assume it (confirmed `True` locally against an RTX 2060 SUPER). Run `uv sync --extra dev` (not plain `uv sync`) to also get `jupyter`/`ipykernel`/`torchinfo` for running notebooks locally; a kernel named `poultry_monitoring` is registered for the VS Code/Jupyter kernel picker.
- **Gates**: `ruff` (lint + format) + `pytest` smoke tests on deterministic non-ML code only. No gate on model training/convergence. See § Gates below.
- **Dataset license**: ChickenVerse is CC BY-NC-SA 4.0 — non-commercial, attribution, share-alike. Don't let export/deployment discussions drift into commercial framing without flagging this.

## Source of Truth

Per constitution Principle II, exploratory notebooks (`notebooks/`, run on Colab) come first and record what was learned; `src/poultry_monitoring/` is the productionized, possibly-redesigned version. When asked to productionize a notebook, check the notebook for *what it proved works*, not for exact code to port — improving structure/naming/approach is expected. If a deliberate redesign happens, leave a short note (module docstring or comment) saying so.

## Package Layout

```
src/poultry_monitoring/
  data/
    coco.py            # ChickenDet COCO parsing (shared: boxes + masks are in the same annotation file)
    dali_pipeline.py   # GPU-accelerated DALI loader (Phase 5)
  augmentation/
    shared.py           # task-agnostic: lighting/color jitter
    detection.py         # bbox-aware: mosaic, occlusion-aware crops
    segmentation.py       # mask-aware: copy-paste
  detection/
    yolo.py              # YOLO26 detection train/predict wrappers
    detr.py              # DETR detection train/predict wrappers
  segmentation/
    yolo.py              # YOLO26-seg wrappers
    detr.py              # DETR (panoptic head) or Mask2Former wrappers; SAM stretch goal (see plan.md Future Work)
  export.py               # ONNX/LiteRT export, shared across task + model
  benchmark.py             # latency/throughput harness, shared
  metrics.py               # box/mask mAP computation, shared
  mlflow_utils.py          # MLflow config/logging helpers, shared

tests/                # pytest smoke tests, mirrors package structure
notebooks/            # Colab exploration notebooks (Phase 1 onward)
data/                 # gitignored: cached ChickenVerse download
results/              # gitignored: per-run artifacts, one dir per MLflow run
```

Rule of thumb for "does this go in a task dir or a shared module": if segmentation and detection would do the *exact same thing* here, it's shared; if the task changes the logic (not just the input), it belongs in `detection/` or `segmentation/`.

## MLflow Conventions

- **Experiments — one per task, not one unified experiment**: `poultry_detection` and `poultry_segmentation`. Model family (`yolo26`/`detr`/future `sam`) is a run tag/param within each, not a separate experiment — this keeps mAP/mask-mAP metric columns directly comparable within an experiment's runs table, since detection and segmentation don't share the same metrics anyway.
- **Tracking URI**: `sqlite:///mlflow.db` (native Windows env only — no cross-environment split needed here, unlike `../fashion_MNIST/`, since this project doesn't use a container).
- **Run naming — hint at the model, keep MLflow's human-readable name for search**: don't fully hand-roll run names. Start the run unnamed so MLflow assigns its own readable name (e.g. `capable-shrike-728`), then rename it via `mlflow.set_tag("mlflow.runName", f"{model}_{variant}-{auto_name}")` — e.g. `yolo26n-baseline-capable-shrike-728`, `detr-augmented-jovial-otter-42`. This is what `mlflow_utils.make_run_name()` should do once written: readable and searchable by model at a glance, still globally unique and easy to reference verbally (mirrors `../fashion_MNIST/CLAUDE.md`'s `sanitize_run_name` pattern, applied to the run name itself instead of the local results directory).
- **Params to log**: `model_family` (`yolo26`/`detr`/`sam`), `variant` (scale/backbone), `learning_rate`, `batch_size`, `num_epochs`, `image_size`, `augmentation_enabled`, `dali_enabled`, `random_seed`.
- **Metrics**: `box_map50`, `box_map50_95` (in `poultry_detection`); `mask_map50`, `mask_map50_95` (in `poultry_segmentation`); `train_loss`, `val_loss` per epoch; `throughput_img_per_sec` (DALI benchmark runs); `latency_ms` (export benchmark runs, tag with `precision` and `export_target`).
- **Artifacts**: best model weights, sample predictions (boxes/masks overlaid), PR curves, and — for benchmark runs — the hardware/batch-size/precision table required by constitution Principle V.
- Use snake_case for all logged param/metric names, matching repo-wide convention (`../fashion_MNIST/CLAUDE.md`, `../MNIST/`).

## Skills

- **`dataviz`** — load before building any comparison chart or benchmark plot (Phase 7 write-up, and any earlier ad hoc results plot). This project's whole point is a legible YOLO26-vs-DETR comparison; don't default to raw matplotlib without checking the skill first.
- **`run`** — use to actually launch training/benchmark scripts and confirm a pipeline change works end-to-end in the real environment, not just that it type-checks.

## Gates

Per constitution Principle VIII — lint/format + smoke tests only, no training-correctness gate:

```bash
uv run ruff check .          # lint
uv run ruff format --check .  # format check
uv run pytest tests/          # smoke tests: COCO parsing, augmentation shapes, export I/O round-trips
```

No CI — these are run locally, on demand, before considering a change done. If this ever changes, wire a workflow to run exactly these three commands rather than inventing CI-specific steps.

## Commands

```bash
uv sync --extra dev                                              # install deps incl. jupyter/torchinfo
uv run jupyter notebook notebooks/                                # Phase 1 exploration — Colab or local GPU
uv run python -m poultry_monitoring.detection.yolo --help          # once scaffolded (Phase 2+)
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db            # view experiment runs
```

(Training/export/benchmark CLI entry points don't exist yet — Phase 0/2+ in `plan.md`. Update this section as they're built.)

## Gitignore Reminders

Make sure these stay gitignored: `data/` (cached ChickenVerse download), `mlruns/`, `mlflow.db`, `results/*` (per-run artifacts, keep `.gitkeep` only), `runs/` (Ultralytics' default output dir for ad hoc `model.val()`/`model.predict()` calls that skip `project=`/`name=` — real runs already land under the gitignored `data/`), `.venv/`, `*.onnx`, `*.tflite`, `*.pt` (trained weights — too large for git; document how to regenerate instead).
