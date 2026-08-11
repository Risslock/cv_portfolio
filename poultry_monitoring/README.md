# Poultry Monitoring: Object Detection & Instance Segmentation

Detecting and segmenting individual chickens in dense, overhead poultry farm imagery — a portfolio project applying modern object detection and segmentation architectures to an industrial computer vision problem. **YOLO26 is the primary deliverable**; a DETR track exists as separate practice with transformer architectures, not a gating comparison (see [`CLAUDE.md`](CLAUDE.md) § Project Intent).

**Status:** 🟡 In progress — `yolo26n` tuned, trained, and progressively unfrozen (val mAP50 = 0.987, mAP50-95 = 0.892), now **ahead of** ChickenVerse's own published baseline; `yolo26s` trained (close on mAP50, gap remains on mAP50-95 — no unfreezing applied yet). See [Status](#status) for the full picture and [`plan.md`](plan.md) for the live phase-by-phase roadmap.

## Table of Contents

- [Dataset](#dataset)
- [Industrial Applications & Real-World Challenges](#industrial-applications--real-world-challenges)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Usage](#usage)
- [Results](#results)
- [Key Findings](#key-findings)
- [Portfolio Scope & Objectives](#portfolio-scope--objectives)
- [Project Docs](#project-docs)
- [Status](#status)

## Dataset

[**ChickenVerse**](https://github.com/amirivojdan/ChickenVerse) (ChickenDet subset), via [Zenodo](https://zenodo.org/records/20672799):

- 6,539 overhead-view images from 5 poultry facilities
- 153,764 annotated chicken instances (~23.5 per image on average, up to 50+ in the test split)
- COCO-format annotations with both bounding boxes and pixel-level segmentation masks
- Pre-split into train / validation / test sets

### License

ChickenVerse is released under **CC BY-NC-SA 4.0** (Attribution-NonCommercial-ShareAlike). It is used here strictly for **non-commercial, educational/portfolio purposes**, with attribution to the original authors. Any derivative work or adaptation must be shared under the same license.

## Industrial Applications & Real-World Challenges

Automated visual monitoring of poultry flocks has direct relevance to precision livestock farming:

- **Flock counting** — automated headcounts without manual inspection
- **Density & welfare monitoring** — detecting overcrowding or abnormal distribution, an early welfare indicator
- **Behavioral analysis** — time spent near feeders and water outlets, and time spent walking, laying, or standing, as activity/welfare indicators
- **Automated inspection** — surfacing anomalies (e.g. isolated or motionless birds) from continuous camera feeds
- **Scalable monitoring** — replacing manual, spot-check inspection with continuous, camera-based coverage across facilities

A production system for this problem has to hold up against conditions this dataset won't necessarily cover on its own: varying litter color/composition, inconsistent lighting between facilities and times of day, differing camera distance/angle across installations, and birds changing in size and appearance as a flock ages. Worth keeping in mind when interpreting results and generalizing beyond this dataset.

**Future work:** the detection/segmentation output is the foundation for a small analytics dashboard — extracting per-video metrics (counts, density over time, feeder/water-outlet dwell time, behavior breakdowns) from farm house footage. See [`plan.md`](plan.md) § Future Work for the full backlog.

## Tech Stack

- **[Ultralytics YOLO26](https://docs.ultralytics.com/)** — real-time, anchor-free/NMS-free CNN detector and segmenter; the model this project actually productionizes
- **[Albumentations](https://albumentations.ai/)** — custom domain-specific augmentation (color invariance, lighting/contrast, occlusion simulation), chosen over DALI for augmentation specifically — richer transform set, no GPU-memory contention with training
- **[NVIDIA DALI](https://developer.nvidia.com/dali)** — GPU-accelerated data *loading*, evaluated later only if profiling shows it's the actual bottleneck
- **[MLflow](https://mlflow.org/)** — experiment tracking (local SQLite store), via Ultralytics' native integration
- **[DETR](https://huggingface.co/docs/transformers/model_doc/detr)** (Hugging Face `transformers`) — transformer-based, set-prediction detector/segmenter; a secondary practice track, not gating the above
- **[ONNX Runtime](https://onnxruntime.ai/)** and **[TensorFlow Lite / LiteRT](https://ai.google.dev/edge/litert)** — model export and optimization for deployment
- **PyTorch** as the underlying training framework

## Architecture

```mermaid
flowchart LR
    A["ChickenVerse<br/>COCO annotations<br/>(boxes + masks)"] --> B["data/coco.py<br/>COCO to YOLO labels + data.yaml"]
    B --> C["augmentation/shared.py<br/>color invariance, lighting"]
    B --> D["augmentation/detection.py<br/>occlusion (CoarseDropout)"]
    C --> E["detection/yolo.py"]
    D --> E
    E -->|tune| F["model.tune()<br/>built-in hyperparameter search"]
    E -->|augtune| G["tune_augmentation_parameters()<br/>custom augmentation random search"]
    F --> H["train() / unfreeze()<br/>YOLO26 fine-tuning +<br/>progressive unfreezing"]
    G --> H
    H --> I[("MLflow<br/>poultry_detection")]
    H --> J["best.pt checkpoint"]
    J --> K["predict() / export"]
```

Custom augmentation search runs separately from `model.tune()` because Ultralytics' tuner executes each trial as its own subprocess, which can't carry live Python objects like a custom Albumentations transform list — see [ADR 0001](docs/adr/0001-custom-augmentation-search-separate-from-tuner.md). Both search results feed into `train()`, which fine-tunes a checkpoint, validates the best epoch, and logs everything to MLflow via [ADR 0003](docs/adr/0003-native-mlflow-integration.md)'s native integration.

Full package layout (module-by-module, including the not-yet-built segmentation/DALI/export pieces) lives in [`CLAUDE.md`](CLAUDE.md) § Package Layout — this diagram covers the detection path that's actually implemented today.

## Usage

```bash
uv sync --extra dev   # installs deps incl. jupyter/torchinfo; verify torch.cuda.is_available()
```

Everything below runs through `detection/yolo.py`'s CLI (`--data-dir` points at a ChickenDet
root containing `images/` and `annotations/`):

```bash
# Hyperparameter search on yolo26n (Ultralytics' native genetic-algorithm tuner)
uv run python -m poultry_monitoring.detection.yolo tune --data-dir data/ChickenDet \
    --iterations 20 --epochs 15 --fraction 0.3

# Custom Albumentations parameter search (color invariance, lighting/contrast, occlusion)
uv run python -m poultry_monitoring.detection.yolo augtune --data-dir data/ChickenDet \
    --trials 8 --epochs 15 --fraction 0.3

# Fine-tune one size with fixed hyperparameters (e.g. a winning tune/augtune result)
uv run python -m poultry_monitoring.detection.yolo train --data-dir data/ChickenDet \
    --model-name yolo26n --variant tuned --epochs 300 --patience 15 \
    --hyperparameters-json '{"lr0": 0.01, "fliplr": 0.5, "p_lighting": 0.12}'

# Progressive-unfreezing refinement on top of an already-trained checkpoint: three stages of
# decreasing freeze (10 -> 5 -> 0) and decreasing lr0 (5e-4 -> 1e-4 -> 2e-5), each continuing
# from the previous stage's best weights — see DEFAULT_UNFREEZE_STAGES in detection/yolo.py
uv run python -m poultry_monitoring.detection.yolo unfreeze --data-dir data/ChickenDet \
    --model-name yolo26n --initial-weights data/ChickenDet/YOLO/yolo26n-tuned/weights/best.pt \
    --hyperparameters-json '{"lr0": 0.01, "fliplr": 0.5, "p_lighting": 0.12}'

# Full pipeline: tune, then train every size in --sizes with the winning config
uv run python -m poultry_monitoring.detection.yolo sweep --data-dir data/ChickenDet \
    --sizes n s --train-epochs 300 --train-patience 15

# Inference on your own images with a trained checkpoint
uv run python -m poultry_monitoring.detection.yolo predict \
    --weights data/ChickenDet/YOLO/yolo26n-tuned/weights/best.pt \
    --source path/to/image_or_directory --conf 0.36 --save-dir predictions/
```

**Visualize what the custom augmentations actually do** to a sample image (and its boxes,
if a label file is given) before committing to a search over their parameters — pure
Albumentations/numpy, no GPU needed, so it's safe to run alongside a live GPU training job
(unlike everything above, which all touch torch):

```bash
uv run python -m poultry_monitoring.augmentation.visualize \
    --image data/ChickenDet/images/Train/<file>.jpeg \
    --label data/ChickenDet/labels/Train/<file>.txt \
    --shared p_color_invariance=1.0 p_lighting=1.0 \
    --detection p_occlusion=1.0 \
    --n-samples 6 --save augmentation_preview.png
```

`--shared`/`--detection` are `key=value` overrides for `augmentation/shared.py` (color
invariance, lighting/contrast) and `augmentation/detection.py` (occlusion simulation via
`CoarseDropout`) respectively — omit `--save` to open an interactive window instead.

**Experiment tracking** — every `train`/`unfreeze`/`sweep` run logs params, per-epoch
metrics, and final `box_map50`/`box_map50_95`/`box_precision`/`box_recall` to MLflow under
the `poultry_detection` experiment:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Export/benchmark CLI entry points don't exist yet — Phase 6 in [`plan.md`](plan.md).

## Results

| Model | Split | Precision | Recall | mAP50 | mAP50-95 | Notes |
|---|---|---|---|---|---|---|
| `yolo26n` (this project) | val | 0.965 | 0.953 | **0.987** | **0.892** | Tuned hyperparameters + custom augmentation + progressive unfreezing (3 stages, `freeze` 10→5→0), full dataset. |
| `yolo26n` (ChickenVerse published) | val | — | — | 0.987 | 0.890 | Reference baseline, same architecture/scale — **we're now ahead on both columns.** |
| `yolo26s` (this project) | val | 0.967 | 0.951 | 0.989 | 0.904 | Tuned config, straight `train()` — **no** progressive unfreezing applied yet. |
| `yolo26s` (ChickenVerse published) | val | — | — | 0.990 | 0.919 | Reference baseline — close on mAP50, a real gap remains on mAP50-95 (same shape `yolo26n` had *before* unfreezing). |

All numbers are validation-split metrics from an explicit `model.val()` pass after training (not training-loop numbers). ChickenVerse's own benchmark table reports two separate mAP columns — only the `val_mAP50`/`val_mAP50-95` ones are comparable to the numbers above; see [`plan.md`](plan.md) Phase 2 for the full note. **The test split is intentionally not used for any comparison or tuning decision**, to avoid implicitly fitting it.

This table reflects the state as of the last update — [`plan.md`](plan.md) Phase 2 is the live source of truth for anything still in flight.

### Sample Predictions

Two images from the **test split** — never used for training, tuning, or any validation-based decision — run through `yolo26n`'s final checkpoint (`conf=0.36`, CPU inference). Ground truth alongside each prediction, same images:

| Ground truth | Prediction |
|---|---|
| ![Ground truth example 1: dense overhead flock, annotated boxes](docs/images/test_ground_truth_1.jpg) | ![Prediction example 1: dense overhead flock with detection boxes](docs/images/test_prediction_1.jpg) |
| ![Ground truth example 2: dense overhead flock, annotated boxes](docs/images/test_ground_truth_2.jpg) | ![Prediction example 2: dense overhead flock with detection boxes](docs/images/test_prediction_2.jpg) |

56 ground-truth birds vs. 60 predicted in the first image, 42 vs. 44 in the second — close counts, and the extra predictions are mostly genuine partial birds at the frame edge that the model still caught. Confidence is mostly 0.8+ even in tightly overlapping clusters and against low-contrast litter. The overlapping labels in the densest clusters are the scene, not a rendering artifact — this is genuinely what "~23 birds/image, up to 50+" looks like at full resolution.

## Key Findings

Learnings worth keeping visible here, not just buried in `plan.md`'s working history:

- **Test-time preprocessing doesn't help** ([ADR 0004](docs/adr/0004-no-test-time-preprocessing.md)) — autocontrast/CLAHE/histogram-equalization applied only at inference (no retraining) on a trained checkpoint were flat-to-negative: autocontrast was a wash (≤0.001 on every metric — the model already saw similarly mild autocontrast during training), while CLAHE and histogram-equalization measurably hurt (−0.013 to −0.014 mAP50-95) by creating a train/inference distribution mismatch instead of correcting one.
- **`close_mosaic` must scale with stage length, not get inherited from a different run's tune result** — reusing `close_mosaic=10` (sized for a 300-epoch run) unchanged on 30-epoch progressive-unfreezing stages disabled mosaic for the last third of each stage, visible below as a sharp train-loss drop plus a transient val-loss/mAP50-95 dip right at epoch 20 (mosaic switches off at `epoch == epochs - close_mosaic`). Fixed with two rules of thumb: `epochs >> close_mosaic`, and `patience < close_mosaic` (since Ultralytics' `best.pt` always tracks the best-ever-observed fitness regardless of when training stops, a shorter patience bounds how much of a bad post-transition regime gets trained through before reverting).

  ![Training curves for the final progressive-unfreezing stage, showing the mAP50-95 dip and recovery right at the close_mosaic transition (epoch 20 of 30)](docs/images/training_curves_stage2.png)
- **`AutoContrast`'s cutoff needed to vary per-application, not get tuned to one fixed value** — Albumentations' `RandomBrightnessContrast` auto-symmetrizes a tuned scalar into a fresh `±range` every call, but `AutoContrast.cutoff` doesn't have that built in; the augmentation search had converged it to a single always-identical value. Fixed with a small subclass that resamples `cutoff` from a fixed range each application instead.
- **`copy_paste_mode` (`"flip"` vs `"mixup"`) is a wash at proxy scale** — every metric within 0.006 between the two; not a meaningful lever for this dataset as tested, despite a real theoretical scale-mismatch risk for `"mixup"` (unmatched cutout/background scale) that didn't clearly show up in the numbers either.

## Portfolio Scope & Objectives

### Detection track — done

1. **Fine-tuned and hyperparameter-optimized a CNN-based detector (YOLO26)** to a genuinely strong result on a dense, occluded dataset: `yolo26n` beats ChickenVerse's own published baseline on both mAP50 and mAP50-95 after a built-in hyperparameter search, a custom-augmentation random search, and progressive unfreezing — see [Results](#results).
2. **Built a domain-specific augmentation strategy** for dense, small, occluded objects: Albumentations-based color-invariance, lighting, and occlusion-simulation transforms, searched alongside Ultralytics' own hyperparameters, with a bounding-box-aware visualization CLI to eyeball the effect before committing to a search.
3. **Ran and documented negative results, not just wins**: test-time preprocessing and `copy_paste_mode` A/B both came back flat-or-negative — kept and written up (ADR + README [Key Findings](#key-findings)) instead of quietly dropped, because knowing what *doesn't* help is part of the actual engineering record.

### Production practices demonstrated along the way

- **Task-separated, shared-utility package structure**: `src/poultry_monitoring/` — `data/`, `augmentation/`, `detection/` — each a small set of typed, docstringed functions, not one long script; a single CLI (`python -m poultry_monitoring.detection.yolo <command>`) drives tuning, training, progressive unfreezing, inference, and comparison runs.
- **Experiment tracking**: every tuning/training run logged to MLflow (params, per-epoch metrics, final val metrics, artifacts) — see [`CLAUDE.md`](CLAUDE.md) § MLflow Conventions for the schema.
- **Decision records**: 4 [ADRs](docs/adr/README.md) capturing non-obvious calls, several reached by testing an assumption and finding it wrong (e.g. why the augmentation search can't share Ultralytics' own tuner) rather than just asserting a conclusion.
- **Exploratory-then-productionized workflow**: notebooks establish that an approach works; `src/poultry_monitoring/` is the redesigned, tested, CLI-driven version — not a line-for-line port (constitution Principle II).
- **Test coverage scoped deliberately**: `pytest` smoke tests cover deterministic pipeline code (COCO parsing, augmentation shapes/behavior, hyperparameter routing) — not training convergence, which isn't a unit-testable property (constitution Principle VIII).
- **Governance sized to the project**: `constitution.md` + `plan.md` + `CLAUDE.md` — heavier than a single README, lighter than full spec-kit — see [Project Docs](#project-docs).

### Beyond detection — planned, not started

4. Instance segmentation on the same dataset (YOLO26-seg), same tune/train/MLflow treatment as detection — Phase 3 in [`plan.md`](plan.md).
5. Hands-on practice with a transformer-based detector (DETR) as a secondary track — compared fairly against YOLO26 if/when both are far enough along, not a gating requirement.
6. A GPU-accelerated data loading pipeline (DALI vs. a standard loader), if profiling shows it's warranted.
7. Exporting and optimizing trained models (ONNX, TFLite/LiteRT) and comparing inference latency/throughput across targets.

## Project Docs

This project uses a hybrid workflow — heavier than a single `CLAUDE.md`, lighter than full spec-kit — see [`CLAUDE.md`](CLAUDE.md) § Workflow for why.

| Doc | What's in it |
|---|---|
| [`constitution.md`](constitution.md) | Non-negotiable-ish principles governing this project (code style, notebook-first workflow, benchmarking rigor, dataset licensing) |
| [`plan.md`](plan.md) | Phased roadmap and **live status** of every task — the first place to check for "where are we right now" |
| [`CLAUDE.md`](CLAUDE.md) | Concrete working conventions for Claude Code in this repo: package layout, MLflow schema, gate commands, CLI reference |
| [`docs/adr/`](docs/adr/README.md) | Architecture Decision Records — non-obvious design calls, especially ones reached by testing an assumption and finding it wrong |
| [Root `README.md`](../README.md) | How this project fits into the broader portfolio, and why different projects here use different workflow formalities |

## Status

🟡 **In progress.** Environment set up and GPU-verified (local + Colab); data exploration and YOLO26 baseline notebooks done; `src/poultry_monitoring/` productionized with a working train/tune/augtune/unfreeze/sweep/predict/ttp CLI. `yolo26n` is fully trained through progressive unfreezing and now beats ChickenVerse's published baseline; `yolo26s` is trained but hasn't had the same unfreezing treatment yet (see [Results](#results)). `copy_paste_mode` A/B and test-time-preprocessing comparisons are done (see [Key Findings](#key-findings)); a `multi_scale` hyperparameter re-tune is running now. See [`plan.md`](plan.md) for phase-by-phase status and the [root repository README](../README.md) for how this project fits into the broader portfolio.
