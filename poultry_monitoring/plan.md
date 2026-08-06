# Plan — Poultry Monitoring

Phased roadmap for this project, governed by [constitution.md](constitution.md). Update phase status as work progresses — this file is the single place tracking "where are we," replacing per-feature `specs/` docs that a full spec-kit workflow would use.

Legend: 🔲 not started · 🟡 in progress · ✅ done · ⏭️ stretch/deferred

## Phase 0 — Setup

- 🔲 Download ChickenVerse (ChickenDet split) from Zenodo; verify COCO annotation loading (boxes + masks) for train/val/test splits
- 🔲 Scaffold `src/poultry_monitoring/` package (see `CLAUDE.md` § Package Layout)
- 🔲 `uv init` + `pyproject.toml` with `torch`/`torchvision` pinned to a CUDA wheel index, `ultralytics`, `transformers`, `mlflow`, dev deps (`ruff`, `pytest`)
- 🔲 Verify `torch.cuda.is_available()` is `True` in the new local venv (see constitution Principle IX — don't repeat the `fashion_MNIST`/CPU-wheel mistake)
- 🔲 `.gitignore`, MLflow SQLite tracking store wired up

## Phase 1 — Exploration & Learning (Colab notebooks)

Per constitution Principle II: this phase happens on **Google Colab** (free T4 GPU), not the local env — it's where the actual learning about these two architectures happens before anything gets productionized.

- 🔲 Notebook: load + visualize ChickenDet (boxes + masks, density distribution across splits)
- 🔲 Notebook: first YOLO26 fine-tune (detection) on a subset — sanity-check the `ultralytics` API, default hyperparameters, output format
- 🔲 Notebook: first DETR fine-tune (detection) on a subset — sanity-check the `transformers` training loop, Hungarian matching loss behavior, output format
- 🔲 Notebook: prototype segmentation heads for both (YOLO26-seg; DETR panoptic head or Mask2Former fallback)
- 🔲 Notebook: prototype augmentation ideas (mosaic/copy-paste, occlusion-aware crops) and eyeball results visually
- 🔲 Notebook: sanity-check DALI availability/behavior on Colab as an early signal, even though the real DALI work happens locally (Phase 5)
- 🔲 Capture what was learned from each notebook (API quirks, what worked/didn't) — this is the design input for Phases 2-6, per constitution Principle II

## Phase 2 — Detection Baseline (productionized)

- 🔲 YOLO26n/s detection fine-tune on ChickenDet, standard `DataLoader`, no custom augmentation
- 🔲 DETR detection baseline (Hugging Face `transformers`) on the same splits
- 🔲 Log both to MLflow under identical metric names (box mAP@50, mAP@50-95); note any split/config asymmetry per constitution Principle IV

## Phase 3 — Segmentation Baseline (productionized)

- 🔲 YOLO26-seg fine-tune on ChickenDet masks
- 🔲 DETR segmentation-capable variant (native panoptic/mask head, or swap to Mask2Former if it underperforms, per Phase 1 findings)
- 🔲 Log mask mAP@50 / mAP@50-95 for both, same disclosure rule as Phase 2

## Phase 4 — Domain-Specific Augmentation

- 🔲 Occlusion-aware crops, mosaic/copy-paste, lighting/color jitter (Ultralytics built-ins for YOLO26; custom `albumentations` pipeline for DETR parity)
- 🔲 Re-run detection + segmentation baselines with augmentation on; measure delta over Phase 2/3 numbers

## Phase 5 — DALI Accelerated Loading

- 🔲 Resolve DALI's Linux/CUDA packaging story on the local Windows env (unconfirmed — see constitution Principle IX)
- 🔲 Build GPU-accelerated decode+augment DALI pipeline
- 🔲 Benchmark throughput (img/sec) against the standard `DataLoader` baseline, per constitution Principle V (hardware/batch size/precision disclosed)

## Phase 6 — Export & Optimization

- 🔲 Export both models to ONNX Runtime; verify inference parity vs. native PyTorch
- 🔲 Export YOLO26 to LiteRT (native support); attempt DETR → ONNX → `onnx2tf` → LiteRT as a stretch goal (⏭️ if it doesn't convert cleanly)
- 🔲 INT8 quantization via `onnxruntime.quantization` for both models
- 🔲 Benchmark latency/throughput: CPU vs. GPU vs. (if available) an edge target, FP32 vs. INT8

## Phase 7 — Write-Up

- 🔲 Results tables + comparison charts in `README.md` (via `dataviz` skill — see `CLAUDE.md` § Skills)
- 🔲 Final narrative: YOLO26 (CNN, real-time, NMS-free) vs. DETR (transformer, set-prediction) tradeoffs, backed by the actual numbers from Phases 2-6

## Future Work (out of scope for now)

- ⏭️ Analytics dashboard: per-video metrics (counts, density over time, feeder/water-outlet dwell time, behavior breakdowns) from farm house footage, built on top of the Phase 2-6 detection/segmentation models
- ⏭️ Bring in ChickenAct (15,250 single-bird behavior clips, 15 classes) for the behavior-classification piece of the dashboard
- ⏭️ Evaluate Meta's **SAM** (Segment Anything Model) as a third segmentation comparison point — promptable/zero-shot segmentation vs. the trained YOLO26-seg and DETR-seg models. Would log to the same `poultry_segmentation` MLflow experiment (see `CLAUDE.md` § MLflow Conventions) for direct comparison once pursued.
- ⏭️ **Synthetic scene generation via cutout compositing** — a data-generation idea, bigger in scope than the Phase 4 augmentation transforms, so tracked separately:
  1. **Extract per-bird RGBA cutouts** from the existing annotated data. Two possible sources: (a) directly from ChickenDet's segmentation masks (crop the bbox, apply the mask as an alpha channel), or (b) for any future data that only has boxes, crop the bbox and run a segmentation model on the crop (SAM is the natural fit here, per the bullet above) to derive the mask.
  2. **Build a cutout library** — tag each cutout with source metadata (original image/split, approximate bird size/scale) so later sampling can control for it.
  3. **Composite onto backgrounds** — either the original training images (with existing birds already present) or new/unseen background plates (empty house shots, or backgrounds with birds removed via inpainting) — placing cutouts according to chosen density/knowledge rules, e.g. no-overlap-with-existing-birds for a clean baseline, or controlled overlap to deliberately manufacture occlusion training examples.
  4. **Per-cutout transforms before placement** — rotation, scale, flip, color/lighting jitter — applied to each cutout independently for diversity, on top of whatever scene-level density is chosen.
  5. **Generates new (image, box, mask) annotation triples automatically** — since placement is programmatic, ground truth for the synthetic instances is free, unlike manual annotation.
  - **Open questions to resolve before building this**: how to get clean birds-removed backgrounds (naive cropping vs. an inpainting pass); how much scale/lighting mismatch between cutout and background is tolerable before it reads as obviously synthetic; whether synthetic images should be a separate, clearly-tagged data source in MLflow (a `data_source: synthetic` param) so results can be reported with vs. without synthetic data — in the spirit of constitution Principle V (benchmarking/comparison disclosure) even though this is a training-data question, not a benchmark one.
  - **Why it's worth doing**: ChickenDet's density is fixed by what was actually filmed; this would let density (and occlusion pattern) become a controllable variable instead, which is exactly the axis Principle IV's "dense, high-occlusion" framing cares about.

## Status

🚧 Planning complete (this file + `constitution.md` + `CLAUDE.md`). Phase 0 not yet started.
