"""Test-time image preprocessing: compare deterministic enhancements against a checkpoint.

See docs/adr/0004-no-test-time-preprocessing.md — three candidates already tested and
rejected against this project's trained distribution. This harness stays in the codebase
because a genuinely different transform is still worth re-testing, not because any of
them are expected to win by default.
"""

import shutil
from collections.abc import Callable
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils import YAML

from poultry_monitoring.detection.yolo import CLASS_NAMES


def _autocontrast_image(image: np.ndarray, cutoff: float = 1.0) -> np.ndarray:
    """Per-channel percentile contrast stretch.

    Same idea as `augmentation/shared.py`'s `AutoContrast`, applied unconditionally here
    instead of as a probabilistic transform.
    """
    result = image.astype(np.float32)
    for c in range(image.shape[2]):
        channel = result[:, :, c]
        low, high = np.percentile(channel, [cutoff, 100 - cutoff])
        if high > low:
            result[:, :, c] = np.clip((channel - low) * 255.0 / (high - low), 0, 255)
    return result.astype(np.uint8)


def _clahe_image(image: np.ndarray) -> np.ndarray:
    """CLAHE on the L channel of LAB.

    Avoids amplifying color-channel noise the way per-channel CLAHE on RGB directly would.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _histogram_equalize_image(image: np.ndarray) -> np.ndarray:
    """Global histogram equalization on the Y channel of YCrCb."""
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def _brightness_contrast_image(
    image: np.ndarray, brightness: float = -15.0, contrast: float = 1.2
) -> np.ndarray:
    """Darken and boost contrast on the L channel of LAB.

    Domain-informed alternative to `_autocontrast_image`'s symmetric stretch: ChickenVerse
    birds are white against dark brown/grey litter, so darkening (crushing the low end)
    plus a mild contrast gain should widen that separation, rather than stretching the
    already-bright bird pixels further.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lightness, a, b = cv2.split(lab)
    lightness = cv2.convertScaleAbs(lightness, alpha=contrast, beta=brightness)
    return cv2.cvtColor(cv2.merge([lightness, a, b]), cv2.COLOR_LAB2RGB)


# Test-time-only preprocessing candidates — no retraining involved, see
# `evaluate_test_time_preprocessing`. The brightness_contrast_*/autocontrast_cutoff*
# entries came out of a visual sweep (docs/images/preprocessing_sweep_v2_*.png,
# clahe_low_range_sweep.png) — brightness_contrast at the aggressive end was the clear
# winner there (clean bird/background separation, no artifacts); autocontrast is only
# viable in this narrow low-cutoff band before it introduces color-cast/noise; clahe
# didn't show a usable range at all, so it's kept only at its original mild default.
TEST_TIME_PREPROCESSORS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "autocontrast": _autocontrast_image,
    "clahe": _clahe_image,
    "hist_eq": _histogram_equalize_image,
    "brightness_contrast": _brightness_contrast_image,
    "brightness_contrast_b30_c18": partial(
        _brightness_contrast_image, brightness=-30.0, contrast=1.8
    ),
    "brightness_contrast_b30_c20": partial(
        _brightness_contrast_image, brightness=-30.0, contrast=2.0
    ),
    "autocontrast_cutoff4": partial(_autocontrast_image, cutoff=4.0),
    "autocontrast_cutoff5": partial(_autocontrast_image, cutoff=5.0),
    "autocontrast_cutoff6": partial(_autocontrast_image, cutoff=6.0),
}


def _write_preprocessed_validation_split(
    data_dir: Path, dest_dir: Path, preprocess: Callable[[np.ndarray], np.ndarray]
) -> None:
    """Write a `preprocess`-transformed copy of `images/Validation` + its labels under `dest_dir`.

    Labels are copied unchanged — none of `TEST_TIME_PREPROCESSORS` are spatial, so boxes
    don't move. A full copy (not a symlink) because Ultralytics resolves a split's label
    directory by swapping `images` for `labels` in its *own* path, which wouldn't find the
    original `labels/Validation` if `dest_dir` used a different split folder name.
    """
    src_images, src_labels = data_dir / "images" / "Validation", data_dir / "labels" / "Validation"
    dst_images, dst_labels = dest_dir / "images" / "Validation", dest_dir / "labels" / "Validation"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    for label_path in src_labels.glob("*.txt"):
        shutil.copy2(label_path, dst_labels / label_path.name)
    for image_path in src_images.iterdir():
        if not image_path.is_file():
            continue
        processed = preprocess(np.array(Image.open(image_path).convert("RGB")))
        Image.fromarray(processed).save(dst_images / image_path.name)


def evaluate_test_time_preprocessing(
    data_dir: Path,
    weights_path: Path,
    project: Path,
    preprocessors: dict[str, Callable[[np.ndarray], np.ndarray]] | None = None,
) -> dict[str, dict[str, float]]:
    """Compare validation-set metrics of each preprocessed variant against the baseline.

    Each preprocessor runs over every validation image once, unconditionally — no
    retraining involved, inference/eval only.

    Args:
        data_dir: ChickenDet root (must already have `chickendet.yaml` from
            `data.coco.prepare_data`).
        weights_path: Trained checkpoint to evaluate (unchanged across every variant).
        project: Local save dir for `model.val()` artifacts and each variant's image copy.
        preprocessors: Name -> image-array-in, image-array-out function. Defaults to
            `TEST_TIME_PREPROCESSORS`.

    Returns:
        `{"baseline": {...}, <preprocessor_name>: {...}, ...}`, each value the same
        `box_map50`/`box_map50_95`/`box_precision`/`box_recall` shape as `TrainOutcome`.
    """
    data_dir = Path(data_dir).resolve()
    project = Path(project).resolve()
    preprocessors = preprocessors if preprocessors is not None else TEST_TIME_PREPROCESSORS
    model = YOLO(str(Path(weights_path).resolve()))

    def box_metrics(val_metrics) -> dict[str, float]:
        return {
            "box_map50": float(val_metrics.box.map50),
            "box_map50_95": float(val_metrics.box.map),
            "box_precision": float(val_metrics.box.mp),
            "box_recall": float(val_metrics.box.mr),
        }

    results = {
        "baseline": box_metrics(
            model.val(
                data=str(data_dir / "chickendet.yaml"),
                project=str(project),
                name="ttp-baseline",
                exist_ok=True,
            )
        )
    }
    for name, preprocess in preprocessors.items():
        variant_dir = project / "ttp" / name
        _write_preprocessed_validation_split(data_dir, variant_dir, preprocess)
        yaml_path = variant_dir / "data.yaml"
        # `train` key is required by Ultralytics' data-YAML schema but never read for a
        # val()-only call (only the active split's path is existence-checked) — points at
        # the same processed folder rather than a real, unused training split.
        YAML.save(
            yaml_path,
            {
                "path": str(variant_dir),
                "train": "images/Validation",
                "val": "images/Validation",
                "names": CLASS_NAMES,
            },
        )
        results[name] = box_metrics(
            model.val(data=str(yaml_path), project=str(project), name=f"ttp-{name}", exist_ok=True)
        )
    return results
