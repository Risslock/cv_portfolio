# Poultry Monitoring: Object Detection & Instance Segmentation

Detecting and segmenting individual chickens in dense, overhead poultry farm imagery — a portfolio project applying modern object detection and segmentation architectures to an industrial computer vision problem.

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

- **Ultralytics YOLO26** — real-time, anchor-free/NMS-free CNN detector and segmenter
- **DETR** (Hugging Face `transformers`) — transformer-based, set-prediction detector/segmenter, as an architectural point of comparison
- **NVIDIA DALI** — GPU-accelerated data loading and augmentation pipeline
- **ONNX Runtime** and **TensorFlow Lite / LiteRT** — model export and optimization for deployment
- **PyTorch** as the underlying training framework

## Portfolio Scope & Objectives

This project is scoped to demonstrate:

1. Training and fine-tuning both a CNN-based (YOLO26) and a transformer-based (DETR) model on the same dataset, for both detection and instance segmentation
2. Designing an augmentation strategy suited to dense, small, occluded objects
3. Building and benchmarking a GPU-accelerated data loading pipeline (DALI vs. a standard loader)
4. Exporting and optimizing trained models (ONNX, TFLite/LiteRT) and comparing inference latency/throughput across targets

## Status

🚧 **Planned** — not yet started. See the root [repository README](../README.md) for how this project fits into the broader portfolio.
