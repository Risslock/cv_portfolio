"""Visualize augmentation effects on sample images — for experimentation, not training.

Standalone helper for eyeballing what a transform list actually does before committing
to a search over its parameters (see `detection/yolo.py`'s `tune_augmentation_parameters`).
Built-in Ultralytics augmentations (mosaic, hsv, flips, copy_paste) already have some
visibility via the `train_batch*.jpg` artifacts `model.train(plots=True)` writes — this
module covers what those don't: this project's own custom Albumentations transforms
(`shared.build_domain_transforms`) in isolation. Pure Albumentations/numpy — no torch
import, safe to run alongside a live GPU training job.
"""

import argparse
from pathlib import Path

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from poultry_monitoring.augmentation.detection import build_detection_transforms
from poultry_monitoring.augmentation.shared import build_domain_transforms

BoxXcYcWH = tuple[float, float, float, float]


def load_yolo_label_boxes(label_path: Path) -> list[BoxXcYcWH]:
    """Read a YOLO-format label file as normalized `(x_center, y_center, w, h)` boxes.

    Handles both plain box labels (`class xc yc w h`) and the segment/polygon labels
    `data/coco.py`'s `convert_coco_to_yolo_labels` writes when `use_segments=True`
    (`class x1 y1 x2 y2 ...`) — polygons are reduced to their bounding box via min/max
    extent, since this is for drawing a box, not training on a mask.

    Args:
        label_path: Path to a YOLO-format `.txt` label file.

    Returns:
        One normalized `(x_center, y_center, w, h)` tuple per label line.
    """
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        _, *coords = line.split()
        coords = list(map(float, coords))
        if len(coords) == 4:
            boxes.append(tuple(coords))
        else:
            xs, ys = coords[0::2], coords[1::2]
            x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
            boxes.append(((x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min))
    return boxes


def generate_augmented_samples(
    image: np.ndarray,
    transforms: list,
    bboxes: list[BoxXcYcWH] | None = None,
    n_samples: int = 6,
    seed: int | None = None,
) -> list[dict]:
    """Run `transforms` on `image` (and `bboxes`, if given) `n_samples` times.

    Args:
        image: RGB image array (e.g. `np.array(Image.open(path).convert("RGB"))`).
        transforms: Albumentations transform list, e.g. `build_domain_transforms(...)`.
        bboxes: Normalized YOLO-format boxes to carry through the pipeline alongside
            the image — see `load_yolo_label_boxes`. None of this project's current
            custom transforms are spatial (image-only color/contrast jitter), so boxes
            won't actually move yet, but this keeps the function correct if a spatial
            transform is ever added here.
        n_samples: How many independent draws to generate.
        seed: Optional seed for reproducible draws.

    Returns:
        `n_samples` dicts, each with an `"image"` key and, if `bboxes` was given, a
        `"bboxes"` key (same normalized format, post-transform).
    """
    bbox_params = A.BboxParams(format="yolo", label_fields=["class_labels"]) if bboxes else None
    pipeline = A.Compose(transforms, bbox_params=bbox_params)
    if seed is not None:
        pipeline.set_random_seed(seed)

    samples = []
    for _ in range(n_samples):
        if bboxes:
            sample = pipeline(image=image, bboxes=bboxes, class_labels=[0] * len(bboxes))
        else:
            sample = pipeline(image=image)
        samples.append(sample)
    return samples


def _draw_image_with_boxes(ax, image: np.ndarray, bboxes, title: str | None = None) -> None:
    """Plot `image` on `ax` with normalized YOLO-format `bboxes` overlaid, if given."""
    ax.imshow(image)
    if bboxes:
        h, w = image.shape[:2]
        for x_c, y_c, bw, bh in bboxes:
            x1, y1 = (x_c - bw / 2) * w, (y_c - bh / 2) * h
            box = Rectangle((x1, y1), bw * w, bh * h, fill=False, color="#1baf7a", linewidth=1.5)
            ax.add_patch(box)
    if title:
        ax.set_title(title)
    ax.axis("off")


def plot_augmented_grid(
    image_path: Path,
    transforms: list,
    label_path: Path | None = None,
    n_samples: int = 6,
    seed: int | None = None,
    save_path: Path | None = None,
) -> None:
    """Show the original image alongside `n_samples` random draws from `transforms`.

    Probabilities in `transforms` are often low by design (see `build_domain_transforms`),
    so most draws may look unchanged — pass an explicitly high-probability transform list
    (e.g. `build_domain_transforms(p_color_invariance=1.0, p_lighting=1.0)`) to always see
    the effect.

    Args:
        image_path: Path to a sample image.
        transforms: Albumentations transform list, e.g. `build_domain_transforms(...)`.
        label_path: Optional path to the image's YOLO-format label file — if given,
            draws bounding boxes on the original and every augmented draw.
        n_samples: How many augmented draws to show.
        seed: Optional seed for reproducible draws.
        save_path: If given, save the figure here instead of opening an interactive
            window (e.g. for a headless run via the CLI below).
    """
    image = np.array(Image.open(image_path).convert("RGB"))
    bboxes = load_yolo_label_boxes(label_path) if label_path else None
    samples = generate_augmented_samples(
        image, transforms, bboxes=bboxes, n_samples=n_samples, seed=seed
    )

    fig, axes = plt.subplots(1, n_samples + 1, figsize=(3 * (n_samples + 1), 3))
    _draw_image_with_boxes(axes[0], image, bboxes, title="original")
    for ax, sample in zip(axes[1:], samples):
        _draw_image_with_boxes(ax, sample["image"], sample.get("bboxes"))

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def _parse_kv_floats(pairs: list[str]) -> dict[str, float]:
    """Parse `key=value` CLI args (e.g. `["p_color_invariance=1.0"]`) into a float dict."""
    result = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        result[key] = float(value)
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the `python -m poultry_monitoring.augmentation.visualize` CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Sample image to augment.")
    parser.add_argument(
        "--label", type=Path, default=None, help="YOLO-format label file — draws boxes if given."
    )
    parser.add_argument("--n-samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--save", type=Path, default=None, help="Save here instead of opening a window."
    )
    parser.add_argument(
        "--shared",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="build_domain_transforms overrides, e.g. p_color_invariance=1.0 p_lighting=1.0",
    )
    parser.add_argument(
        "--detection",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="build_detection_transforms overrides, e.g. p_occlusion=1.0",
    )
    return parser


def main() -> None:
    """CLI entry point — see `_build_arg_parser` for `--help`."""
    args = _build_arg_parser().parse_args()
    transforms = build_domain_transforms(
        **_parse_kv_floats(args.shared)
    ) + build_detection_transforms(**_parse_kv_floats(args.detection))

    plot_augmented_grid(
        args.image,
        transforms,
        label_path=args.label,
        n_samples=args.n_samples,
        seed=args.seed,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
