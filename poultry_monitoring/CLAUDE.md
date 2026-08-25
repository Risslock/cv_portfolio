# CLAUDE.md — poultry_monitoring

Guidance for Claude Code when working in this directory.

## Project Intent

A portfolio piece whose primary goal is a genuinely good object detection + instance segmentation model for poultry monitoring on ChickenVerse/ChickenDet (dense, high-occlusion overhead imagery) — **not a comparison exercise for its own sake**. Ultralytics **YOLO26** (CNN, anchor-free/NMS-free) is the model actually getting productionized: real fine-tuning, hyperparameter tuning, a size sweep, MLflow-tracked runs. **DETR** (transformer, set-prediction) is a secondary track — the user's own hands-on practice with transformer architectures, trained and evaluated the same way when there's time for it, but it does not gate or dilute focus from getting YOLO26 right. If/when both are far enough along to compare, do so fairly (constitution Principle IV), but don't default to "need both before productionizing" reasoning. Beyond model accuracy, the project also covers domain-specific augmentation and export/optimization (ONNX Runtime, OpenVINO, TensorRT) with quantization, accuracy-degradation measurement, and latency/throughput benchmarking. A DALI-vs-`DataLoader` track was scoped and then closed on measured grounds — the training loop is GPU-bound ([ADR 0019](docs/adr/0019-discard-dali.md)). See the [README](README.md) for the full problem statement and dataset details.

## Workflow: Why This Project Uses Hybrid (constitution.md + plan.md + CLAUDE.md)

This repo intentionally uses **different workflow formalities per project**, as a deliberate portfolio point about working styles in the AI-assisted-development era (see root [README.md](../README.md) § Project Philosophy):

- `../MNIST/` — full spec-kit (`.specify/`, `specs/`, `plan`/`tasks` per feature): heaviest formality, suited to a project built feature-by-feature over time.
- `../fashion_MNIST/CLAUDE.md` — lightest: just a CLAUDE.md + README, no separate governance docs. Suited to a small, single-framework, single-architecture project.
- **`poultry_monitoring/` (here) — hybrid**: `constitution.md` (non-negotiable-ish principles) + `plan.md` (phased roadmap, updated as work progresses) + this file (concrete working conventions, "decisions already made," commands). No `.specify/` tooling, no per-feature `specs/` — those are overkill for a project planned end-to-end upfront. This tier fits a project too large/multi-part for a single CLAUDE.md, but not built incrementally enough to need full spec-kit.

**Read `constitution.md` and `plan.md` before making structural decisions here.** This file assumes both and focuses on the "how to actually work in this repo, today" layer.

## Decisions Already Made (don't re-litigate these)

- **Framework**: PyTorch, via `ultralytics` (YOLO26) and `transformers` (DETR). No TensorFlow.
- **Code style**: Plain functions with type hints for all first-party code; framework classes (`ultralytics.YOLO`, HF `Trainer`) used natively, unwrapped. See constitution Principle I.
- **Docstrings**: [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) (`Args:`/`Returns:`/`Raises:` sections) for every first-party function in `src/poultry_monitoring/`. Notebook helper functions should follow it too where it's not more noise than signal for a short exploratory function.
- **Workflow**: Notebook-first exploration — on Colab or the local `.venv` (both GPU-capable, see § Environment below) — then productionize into `src/poultry_monitoring/`; translation is allowed to improve on the notebook, not required to mirror it. See constitution Principle II. **YOLO26 is the priority** — it's the model actually getting productionized (hyperparameter tuning, size sweep, MLflow-tracked runs); the DETR notebook is secondary/practice and isn't a prerequisite for YOLO26 productionization work. See § Project Intent.
- **Package layout**: task-separated (`detection/` vs. `segmentation/`), shared utilities live once at the top level. See § Package Layout below and constitution Principle III.
- **Environment**: `uv` + `pyproject.toml`/`uv.lock`, native Windows (no Docker/WSL2 — PyTorch has native Windows CUDA support, unlike the TensorFlow situation in `../fashion_MNIST/`). `torch`/`torchvision` pinned to a CUDA wheel index — verify `torch.cuda.is_available()` is `True` after `uv sync`, don't assume it (confirmed `True` locally against an RTX 2060 SUPER). Run `uv sync --extra dev` (not plain `uv sync`) to also get `jupyter`/`ipykernel`/`torchinfo` for running notebooks locally; a kernel named `poultry_monitoring` is registered for the VS Code/Jupyter kernel picker.
- **Gates**: `ruff` (lint + format) + `pytest` smoke tests on deterministic non-ML code only. No gate on model training/convergence. See § Gates below.
- **Dataset license**: ChickenVerse is CC BY-NC-SA 4.0 — non-commercial, attribution, share-alike. Don't let export/deployment discussions drift into commercial framing without flagging this.
- **Working style**: nano/micro updates, not full-feature pushes in one turn — smallest coherent step, surfaced, before the next one. Track multi-step work with a todo list. Rationale-heavy explanations (why an approach was tried and rejected, not just what was decided) go in `docs/adr/`, not inlined as long comments/docstrings — see § Architecture Decision Records. When something breaks or a design assumption turns out wrong, surface it and ask before running an autonomous multi-step fix loop.

## Architecture Decision Records

`docs/adr/` holds non-obvious design decisions, especially ones arrived at by testing an
assumption and finding it wrong (the reasoning, not just the conclusion, is the point —
see `docs/adr/0001-*.md` for what "worth an ADR" looks like: a decision with a real
rejected alternative and a concrete reason it didn't work). Template at
`docs/adr/template.md`; index at `docs/adr/README.md`. Code should point here with a
short comment (`# see docs/adr/0001-...md`) instead of inlining the full rationale.

## Source of Truth

Per constitution Principle II, exploratory notebooks (`notebooks/`, run on Colab) come first and record what was learned; `src/poultry_monitoring/` is the productionized, possibly-redesigned version. When asked to productionize a notebook, check the notebook for *what it proved works*, not for exact code to port — improving structure/naming/approach is expected. If a deliberate redesign happens, leave a short note (module docstring or comment) saying so.

## Package Layout

```
src/poultry_monitoring/
  data/
    coco.py            # ChickenDet COCO parsing + prepare_data (shared: boxes + masks in one annotation file)
  augmentation/
    shared.py           # task-agnostic: lighting/color jitter (build_domain_transforms)
    visualize.py         # before/after grids for shared.py's transforms — no torch import
    detection.py          # bbox-aware: occlusion simulation (CoarseDropout, boxes untouched)
    segmentation.py        # mask-aware: copy-paste primitives + donor bank (framework-free)
  detection/
    yolo.py              # YOLO26 core: train/predict, TrainOutcome; also owns the unified CLI (imports tuning.py/preprocessing_eval.py locally inside main() — see docs/adr/0010)
    tuning.py             # multi-run strategies: tune/augtune/unfreeze/sweep, built on yolo.py's train()
    preprocessing_eval.py  # test-time-only preprocessing comparison harness (ttp CLI) — see docs/adr/0004
    detr.py              # DETR detection train/predict wrappers (secondary/practice track)
  segmentation/
    yolo.py              # YOLO26-seg wrappers; owns the seg CLI (train/val/predict/ttp)
    evaluation.py         # held-out + density-stratified scoring (val CLI) — see docs/adr/0010
    copy_paste_training.py  # on-the-fly copy-paste: Ultralytics transform/dataset/trainer (ADR 0017)
    synthetic_data.py       # offline synthetic-split materializer — documented fallback, not the default
    detr.py              # DETR (panoptic head) or Mask2Former wrappers; SAM stretch goal (see plan.md Future Work)
  export.py               # ONNX/OpenVINO/TensorRT export + quantization + the artifact manifest, shared
  inference.py             # THE single model-load path (owns task=, device, thread pinning, warmup) + runner CLI
  benchmark.py              # latency/throughput harness, shared
  mlflow_utils.py            # MLflow config/logging helpers, shared

tests/                # pytest smoke tests, mirrors package structure
notebooks/            # Colab exploration notebooks (Phase 1 onward)
data/                 # gitignored: cached ChickenVerse download
results/              # gitignored: per-run artifacts, one dir per MLflow run
```

Rule of thumb for "does this go in a task dir or a shared module": if segmentation and detection would do the *exact same thing* here, it's shared; if the task changes the logic (not just the input), it belongs in `detection/` or `segmentation/`.

**There is deliberately no `metrics.py`.** An earlier version of this layout reserved one for "box/mask mAP computation, shared", but `segmentation/yolo.py`'s `extract_metrics` already *is* that single source of truth — it owns the Ultralytics-result-to-metric-name mapping and is imported by `segmentation/evaluation.py`, `segmentation/preprocessing_eval.py` and the benchmark harness. Adding `metrics.py` would duplicate it, not share it. Don't re-create it; extend `extract_metrics` instead.

**Loading a model is `inference.load_model`'s job, everywhere.** Export, accuracy scoring, benchmarking and prediction all open artifacts through that one function, so *what you benchmark is what you serve* holds structurally. It also concentrates the non-obvious bits in one place — chiefly the mandatory `task="segment"`, since Ultralytics' `guess_model_task` matches only `"-seg" in path.stem` and an exported `…/weights/best.onnx` silently resolves to `"detect"` (see `docs/adr/0020`). Don't call `YOLO(...)` directly in new code.

## MLflow Conventions

- **Experiments — one per task, not one unified experiment**: `poultry_detection` and `poultry_segmentation`. Model family (`yolo26`/`detr`/future `sam`) is a run tag/param within each, not a separate experiment — this keeps mAP/mask-mAP metric columns directly comparable within an experiment's runs table, since detection and segmentation don't share the same metrics anyway.
- **Tracking URI**: `sqlite:///mlflow.db` (native Windows env only — no cross-environment split needed here, unlike `../fashion_MNIST/`, since this project doesn't use a container).
- **Run naming — hint at the model, `run_id` suffix for uniqueness**: `mlflow_utils.make_run_name(model_family, variant)` renames the active run to `f"{model_family}-{variant}-{run_id[:8]}"` — e.g. `yolo26-n-tuned-7e954a89`. Not MLflow's own auto-generated adjective-animal name (that was the original plan; not reliably reachable through `model.train()`'s public kwargs — see `docs/adr/0003-native-mlflow-integration.md`).
- **Params to log**: `model_family` (`yolo26`/`detr`/`sam`), `variant` (scale/backbone) — plus everything Ultralytics' native MLflow integration already auto-logs from `trainer.args` (`lr0`, `batch`, `epochs`, `imgsz`, `seed`, ...), so don't re-log those by hand.
- **Metrics**: `box_map50`, `box_map50_95`, `box_precision`, `box_recall` (in `poultry_detection`, via `mlflow_utils.finish_run`'s `extra_metrics` — Ultralytics' own auto-logged per-epoch metrics use different key names, e.g. `metrics/mAP50(B)`); the nine box+mask keys from `segmentation/yolo.py`'s `extract_metrics` (in `poultry_segmentation`); `latency_ms_mean`/`latency_ms_median`/`latency_ms_p95` and `throughput_img_per_sec` (export benchmark runs — tag with `export_target`, `precision`, `device` and `batch`, per constitution Principle V).
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

**Cadence**: run before shipping/committing a big-enough update, not after every nano-step — matches § Working Style above and avoids overprocessing. Exception: **never run `pytest` while a real GPU training job is active in the background** — it imports torch/touches CUDA and can starve/crash the live job (this has happened — cost ~58 epochs of a training run). `ruff check`/`ruff format --check` are pure static analysis and stay safe to run anytime, including mid-training.

## Commands

```bash
uv sync --extra dev                                                        # install deps incl. jupyter/torchinfo
uv run jupyter notebook notebooks/                                          # Phase 1 exploration — Colab or local GPU
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db                     # view experiment runs

# detection/yolo.py CLI — see README.md for a walked-through example of each
uv run python -m poultry_monitoring.detection.yolo tune --data-dir <dir>       # hyperparameter search (yolo26n)
uv run python -m poultry_monitoring.detection.yolo augtune --data-dir <dir>     # custom augmentation search (yolo26n)
uv run python -m poultry_monitoring.detection.yolo train --data-dir <dir>        # fine-tune one size
uv run python -m poultry_monitoring.detection.yolo sweep --data-dir <dir>         # tune, then train every size in --sizes
uv run python -m poultry_monitoring.detection.yolo predict --weights <pt> --source <img>  # inference on custom images

# synthetic copy-paste (Phase 3 Stage B) — build the donor bank once, then train with it
uv run python -m poultry_monitoring.augmentation.segmentation build-bank --annotations <json> --img-dir <dir> --bank-dir <dir>
uv run python -m poultry_monitoring.segmentation.yolo train --data-dir <dir> --copy-paste-bank <dir>  # on-the-fly, per sample

# held-out evaluation — --split Test is deliberately opt-in; --by-density reproduces the README's density figure
uv run python -m poultry_monitoring.segmentation.yolo val --data-dir <dir> --weights <pt> --split Test --by-density

# Phase 6 export/optimization — export writes an artifact manifest the later stages enrich
uv run python -m poultry_monitoring.export --data-dir <dir> --weights <pt> [<pt> ...]  # full matrix
uv run python -m poultry_monitoring.export --data-dir <dir> --variants onnx-fp32 --weights <pt>  # one cell
uv run python -m poultry_monitoring.export --data-dir <dir> --list                     # show the manifest

# inference.py — run ANY exported artifact (images only); --variant resolves via the manifest
uv run python -m poultry_monitoring.inference --list
uv run python -m poultry_monitoring.inference --data-dir <dir> --variant <name> --source <img> --device cpu

# benchmark.py — latency/throughput per constitution Principle V (warmup excluded, averaged)
uv run python -m poultry_monitoring.benchmark --data-dir <dir> --batch 1 4 8

# join accuracy (manifest) + latency (benchmark_results.json) into the COMMITTED report.
# The JSON results live under the gitignored /data/, so docs/export_results.md is the
# durable record -- regenerate it after any re-export/re-score/re-benchmark, never hand-edit.
uv run python -m poultry_monitoring.benchmark --data-dir <dir> --summary docs/export_results.md

# augmentation/visualize.py CLI — pure Albumentations/numpy, no torch import, safe to
# run alongside a live GPU training job (unlike anything above, which all touch torch)
uv run python -m poultry_monitoring.augmentation.visualize --image <img> --label <txt>  # before/after grid + boxes
```

**Export/inference/benchmark gotchas that are already handled — don't undo them** (see [ADR 0020](docs/adr/0020-export-quantization-matrix.md)): `YOLO_AUTOINSTALL=false` is set at the top of `export.py`/`inference.py`/`benchmark.py` **before** the ultralytics import, because otherwise loading an ONNX model on CPU pip-installs `onnxruntime` next to the pinned `onnxruntime-gpu` and breaks both. `inference.load_model` always passes `task="segment"` (an exported `best.onnx` otherwise silently scores as `detect`, with no mask metrics) and registers torch's `lib/` on the Windows DLL path (ONNX Runtime's CUDA provider can't find CUDA/cuDNN otherwise, and falls back to CPU *without raising*). Always check `describe_runtime()`'s `providers` before believing a GPU number.

## Gitignore Reminders

Make sure these stay gitignored: `/data/` (cached ChickenVerse download), `mlruns/`, `mlflow.db`, `results/*` (per-run artifacts, keep `.gitkeep` only), `runs/` (Ultralytics' default output dir for ad hoc `model.val()`/`model.predict()` calls that skip `project=`/`name=` — real runs already land under the gitignored `/data/`), `.venv/`, `*.onnx`, `*.tflite`, `*.pt` (trained weights — too large for git; document how to regenerate instead).

**Footgun already hit once**: both this project's `.gitignore` and the monorepo root's had an *unanchored* `data/` rule — which also matches `src/poultry_monitoring/data/` (the COCO-parsing package module) and silently hides it from git, not just the dataset cache. Fixed via `/data/` (anchored) here plus an explicit `!/src/poultry_monitoring/data/` negation for the root rule's unanchored copy. If a new top-level directory is ever named `data` again anywhere under `src/`, check `git status`/`git check-ignore -v <path>` before assuming a missing file is a bug elsewhere.
