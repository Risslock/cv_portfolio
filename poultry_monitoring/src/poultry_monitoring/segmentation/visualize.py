"""Visualize `-seg` training labels and ground-truth-vs-prediction, for eyeballing and figures.

Two distinct uses: (1) `plot_training_samples` renders a grid of real training images
with their converted YOLO-seg mask labels overlaid — meant to be run *before* a real
training job, to catch a bad label conversion (e.g. a mostly-bbox-filled mask from
`docs/adr/0013-...md`'s failure mode) by eye rather than discovering it in a trained
model's metrics. (2) `plot_ground_truth_vs_prediction` compares one image's ground
truth against a trained checkpoint's actual prediction — for spot-checking a run and
for generating the same kind of before/after figure this project's README already uses
for the synthetic copy-paste pipeline (see README.md § Synthetic Copy-Paste Data
Augmentation).

Unlike `augmentation/visualize.py`, this module imports `ultralytics.YOLO` (needed for
`plot_ground_truth_vs_prediction`'s inference) — it is **not** safe to run alongside a
live GPU training job.

Mask-rasterization and overlay helpers here are ported from
`notebooks/03_explore_segmentation.ipynb`'s `rasterize_yolo_polygon`/`overlay_masks`.
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLO

from poultry_monitoring.segmentation.yolo import DEFAULT_CONF_THRESHOLD, DEFAULT_IOU_THRESHOLD

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def rasterize_yolo_seg_label(label_path: Path, img_w: int, img_h: int) -> list[np.ndarray]:
    """Turn a YOLO-seg label file (`class x1 y1 x2 y2 ...` per line, normalized) into binary masks.

    Args:
        label_path: Path to a YOLO-format `.txt` label file. Missing file returns no masks.
        img_w: Image width in pixels, for de-normalizing coordinates.
        img_h: Image height in pixels, for de-normalizing coordinates.

    Returns:
        One binary mask (shape `(img_h, img_w)`) per label line.
    """
    if not label_path.exists():
        return []

    masks = []
    for line in label_path.read_text().splitlines():
        parts = list(map(float, line.split()))
        if len(parts) < 7:  # class + at least 3 (x, y) points
            continue
        coords = np.array(parts[1:]).reshape(-1, 2)
        coords[:, 0] *= img_w
        coords[:, 1] *= img_h
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(mask, [coords.astype(np.int32)], 1)
        masks.append(mask)
    return masks


def masks_from_prediction(result) -> list[np.ndarray]:
    """Rasterize an Ultralytics `-seg` prediction's masks to the original image's resolution.

    Uses `result.masks.xy` (already in original-image pixel coordinates) rather than
    `result.masks.data` (at the model's internal, letterboxed resolution) so the output
    lines up pixel-for-pixel with `result.orig_img`/the label-derived masks above.

    Args:
        result: One Ultralytics `Results` object from a `-seg` model's `.predict()`.

    Returns:
        One binary mask per predicted instance, shape `result.orig_shape`. Empty list
        if the model predicted no instances.
    """
    if result.masks is None:
        return []
    h, w = result.orig_shape
    masks = []
    for polygon in result.masks.xy:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
        masks.append(mask)
    return masks


def overlay_masks(image: np.ndarray, masks: list[np.ndarray], alpha: float = 0.45) -> np.ndarray:
    """Draw each instance mask as a distinct semi-transparent color, for display only.

    Args:
        image: RGB image array.
        masks: One binary mask per instance, e.g. from `rasterize_yolo_seg_label` or
            `masks_from_prediction`.
        alpha: Blend strength of the mask color over the underlying pixels.

    Returns:
        A copy of `image` with each mask tinted a distinct random color.
    """
    overlay = image.copy()
    rng = np.random.default_rng(0)  # reproducible per-instance colors across calls
    for mask in masks:
        if mask.sum() == 0:
            continue
        color = rng.integers(50, 255, size=3)
        region = mask.astype(bool)
        overlay[region] = (overlay[region] * (1 - alpha) + color * alpha).astype(np.uint8)
    return overlay


def _sample_split_images(
    data_dir: Path, split: str, n_samples: int, seed: int | None
) -> list[Path]:
    """Pick `n_samples` random image paths from `data_dir/images/<split>`."""
    image_dir = data_dir / "images" / split
    candidates = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    rng = np.random.default_rng(seed)
    n_samples = min(n_samples, len(candidates))
    indices = rng.choice(len(candidates), size=n_samples, replace=False)
    return [candidates[i] for i in sorted(indices)]


def plot_training_samples(
    data_dir: Path,
    split: str = "Train",
    n_samples: int = 6,
    seed: int | None = None,
    save_path: Path | None = None,
) -> None:
    """Show a grid of real training images with their converted YOLO-seg labels overlaid.

    Meant to be run once before a training job, or after any relabeling
    (`--force-relabel`), to catch a bad conversion by eye — e.g. masks that look
    suspiciously box-shaped (the exact failure mode `data.coco.cache_polygon_annotations`
    exists to prevent, see docs/adr/0013-...md) or an empty/missing label file for an
    image that clearly has birds in it.

    Args:
        data_dir: ChickenDet root — must already have `labels/<split>/` from
            `data.coco.prepare_data`.
        split: Which split's images/labels to sample from.
        n_samples: How many images to show (capped at however many the split has).
        seed: Optional seed for reproducible sampling.
        save_path: If given, save the figure here instead of opening an interactive window.
    """
    data_dir = Path(data_dir)
    image_paths = _sample_split_images(data_dir, split, n_samples, seed)

    ncols = min(3, len(image_paths))
    nrows = -(-len(image_paths) // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows), squeeze=False)

    for ax, image_path in zip(axes.flat, image_paths):
        image = np.array(Image.open(image_path).convert("RGB"))
        label_path = data_dir / "labels" / split / f"{image_path.stem}.txt"
        masks = rasterize_yolo_seg_label(label_path, image.shape[1], image.shape[0])
        ax.imshow(overlay_masks(image, masks))
        ax.set_title(f"{image_path.name} ({len(masks)} instances)", fontsize=9)
        ax.axis("off")
    for ax in axes.flat[len(image_paths) :]:
        ax.axis("off")

    fig.suptitle(f"{split} samples — labels as fed to training")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def plot_ground_truth_vs_prediction(
    weights_path: Path,
    image_path: Path,
    label_path: Path | None = None,
    conf: float = DEFAULT_CONF_THRESHOLD,
    iou: float = DEFAULT_IOU_THRESHOLD,
    save_path: Path | None = None,
) -> None:
    """Show one image's ground-truth masks side by side with a trained checkpoint's predictions.

    Args:
        weights_path: Path to a trained `-seg` checkpoint.
        image_path: Image to run inference on.
        label_path: YOLO-seg label file for `image_path`, e.g.
            `<data_dir>/labels/Validation/<stem>.txt`. If omitted, the left panel shows
            the plain image with no overlay (labeled accordingly).
        conf: Minimum detection confidence to keep a prediction — see
            `segmentation.yolo.DEFAULT_CONF_THRESHOLD`.
        iou: NMS IoU threshold for de-duplicating overlapping predictions.
        save_path: If given, save the figure here instead of opening an interactive window.
    """
    image = np.array(Image.open(image_path).convert("RGB"))
    if label_path is not None:
        gt_masks = rasterize_yolo_seg_label(Path(label_path), image.shape[1], image.shape[0])
        gt_title = f"Ground truth ({len(gt_masks)} instances)"
    else:
        gt_masks, gt_title = [], "Ground truth (no label given)"

    model = YOLO(str(Path(weights_path).resolve()))
    result = model.predict(source=str(image_path), conf=conf, iou=iou, verbose=False)[0]
    pred_masks = masks_from_prediction(result)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(overlay_masks(image, gt_masks) if gt_masks else image)
    axes[0].set_title(gt_title)
    axes[0].axis("off")
    axes[1].imshow(overlay_masks(image, pred_masks))
    axes[1].set_title(f"Prediction ({len(pred_masks)} instances)")
    axes[1].axis("off")

    fig.suptitle(Path(image_path).name)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the `python -m poultry_monitoring.segmentation.visualize` CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    labels_parser = subparsers.add_parser(
        "labels", help="Grid of training images with their YOLO-seg labels overlaid."
    )
    labels_parser.add_argument(
        "--data-dir", type=Path, required=True, help="ChickenDet dataset root."
    )
    labels_parser.add_argument("--split", default="Train")
    labels_parser.add_argument("--n-samples", type=int, default=6)
    labels_parser.add_argument("--seed", type=int, default=None)
    labels_parser.add_argument(
        "--save", type=Path, default=None, help="Save here instead of opening a window."
    )

    compare_parser = subparsers.add_parser(
        "compare", help="Ground truth vs. a trained checkpoint's prediction, for one image."
    )
    compare_parser.add_argument(
        "--weights", type=Path, required=True, help="Trained -seg .pt file."
    )
    compare_parser.add_argument("--image", type=Path, required=True)
    compare_parser.add_argument(
        "--label", type=Path, default=None, help="YOLO-seg label file — omit to skip ground truth."
    )
    compare_parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD)
    compare_parser.add_argument("--iou", type=float, default=DEFAULT_IOU_THRESHOLD)
    compare_parser.add_argument(
        "--save", type=Path, default=None, help="Save here instead of opening a window."
    )

    return parser


def main() -> None:
    """CLI entry point — see `_build_arg_parser` for `--help` on each subcommand."""
    args = _build_arg_parser().parse_args()

    if args.command == "labels":
        plot_training_samples(
            args.data_dir,
            split=args.split,
            n_samples=args.n_samples,
            seed=args.seed,
            save_path=args.save,
        )
    elif args.command == "compare":
        plot_ground_truth_vs_prediction(
            args.weights,
            args.image,
            label_path=args.label,
            conf=args.conf,
            iou=args.iou,
            save_path=args.save,
        )


if __name__ == "__main__":
    main()
