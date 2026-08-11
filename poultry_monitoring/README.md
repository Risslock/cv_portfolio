# Poultry Monitoring: Object Detection & Instance Segmentation

Detecting and segmenting individual chickens in dense, overhead poultry farm imagery — a portfolio project applying modern object detection and segmentation architectures to an industrial computer vision problem. **YOLO26 is the primary deliverable**; a DETR track exists as separate practice with transformer architectures, not a gating comparison (see `CLAUDE.md` § Project Intent).

## Dataset

[**ChickenVerse**](https://github.com/amirivojdan/ChickenVerse) (ChickenDet subset), via [Zenodo](https://zenodo.org/records/20672799):

- 6,539 overhead-view images from 5 poultry facilities
- 153,764 annotated chicken instances (~23.5 per image on average, up to 50+ in the test split)
- COCO-format annotations with both bounding boxes and pixel-level segmentation masks
- Pre-split into train / validation / test sets

## Industrial Applications & Real-World Challenges

Automated visual monitoring of poultry flocks has direct relevance to precision livestock farming:

- **Flock counting** — automated headcounts without manual inspection
- **Density & welfare monitoring** — detecting overcrowding or abnormal distribution, an early welfare indicator
- **Behavioral analysis** — time spent near feeders and water outlets, and time spent walking, laying, or standing, as activity/welfare indicators
- **Automated inspection** — surfacing anomalies (e.g. isolated or motionless birds) from continuous camera feeds
- **Scalable monitoring** — replacing manual, spot-check inspection with continuous, camera-based coverage across facilities

A production system for this problem has to hold up against conditions this dataset won't necessarily cover on its own: varying litter color/composition, inconsistent lighting between facilities and times of day, differing camera distance/angle across installations, and birds changing in size and appearance as a flock ages. Worth keeping in mind when interpreting results and generalizing beyond this dataset.

**Future work:** the detection/segmentation output is the foundation for a small analytics dashboard — extracting per-video metrics (counts, density over time, feeder/water-outlet dwell time, behavior breakdowns) from farm house footage.

### License

ChickenVerse is released under **CC BY-NC-SA 4.0** (Attribution-NonCommercial-ShareAlike). It is used here strictly for **non-commercial, educational/portfolio purposes**, with attribution to the original authors. Any derivative work or adaptation must be shared under the same license.

## Tech Stack

- **Ultralytics YOLO26** — real-time, anchor-free/NMS-free CNN detector and segmenter; the model this project actually productionizes
- **Albumentations** — custom domain-specific augmentation (color invariance, lighting/contrast), chosen over DALI for augmentation specifically — richer transform set, no GPU-memory contention with training
- **NVIDIA DALI** — GPU-accelerated data *loading*, evaluated later only if profiling shows it's the actual bottleneck
- **MLflow** — experiment tracking (local SQLite store), via Ultralytics' native integration
- **DETR** (Hugging Face `transformers`) — transformer-based, set-prediction detector/segmenter; a secondary practice track, not gating the above
- **ONNX Runtime** and **TensorFlow Lite / LiteRT** — model export and optimization for deployment
- **PyTorch** as the underlying training framework

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

# Custom Albumentations parameter search (color invariance, lighting/contrast)
uv run python -m poultry_monitoring.detection.yolo augtune --data-dir data/ChickenDet \
    --trials 8 --epochs 15 --fraction 0.3

# Fine-tune one size with fixed hyperparameters
uv run python -m poultry_monitoring.detection.yolo train --data-dir data/ChickenDet \
    --model-name yolo26n --variant baseline --epochs 100 --patience 15

# Full pipeline: tune, then train every size in --sizes with the winning config
uv run python -m poultry_monitoring.detection.yolo sweep --data-dir data/ChickenDet \
    --sizes n s --train-epochs 300 --train-patience 15

# Inference on your own images with a trained checkpoint
uv run python -m poultry_monitoring.detection.yolo predict \
    --weights data/ChickenDet/YOLO/yolo26n-baseline/weights/best.pt \
    --source path/to/image_or_directory --conf 0.36 --save-dir predictions/
```

`train`/`sweep` log every run to MLflow (`uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`)
under the `poultry_detection` experiment — params, per-epoch metrics, and the final
`box_map50`/`box_map50_95`/`box_precision`/`box_recall` numbers. A real run on `yolo26n`
(full dataset, tuned hyperparameters) reached **mAP50 ≈ 0.98, mAP50-95 ≈ 0.86** by epoch 57 —
see `docs/adr/` for the design decisions behind the tuning pipeline.

## Portfolio Scope & Objectives

This project is scoped to demonstrate:

1. Fine-tuning and hyperparameter-optimizing a CNN-based detector (YOLO26) to a genuinely strong result on a dense, occluded dataset — the primary deliverable
2. A domain-specific augmentation strategy suited to dense, small, occluded objects (Albumentations, searched alongside hyperparameters)
3. Hands-on practice with a transformer-based detector (DETR) as a secondary track — compared fairly against YOLO26 if/when both are far enough along, not a gating requirement
4. Building and benchmarking a GPU-accelerated data loading pipeline (DALI vs. a standard loader), if profiling shows it's warranted
5. Exporting and optimizing trained models (ONNX, TFLite/LiteRT) and comparing inference latency/throughput across targets

## Status

🟡 **In progress** — environment set up and GPU-verified (local + Colab); data exploration and YOLO26 baseline notebooks done; `src/poultry_monitoring/` productionized with a working train/tune/sweep/predict CLI; first full hyperparameter + augmentation search and size sweep underway. See [`plan.md`](plan.md) for phase-by-phase status and the root [repository README](../README.md) for how this project fits into the broader portfolio.
