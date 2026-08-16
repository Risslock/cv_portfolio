"""Test-time image preprocessing for `-seg` checkpoints: box+mask impact, not just box.

See docs/adr/0004-no-test-time-preprocessing.md — the detection track already tested
and rejected these candidates against its own trained distribution. Worth re-running
here rather than assuming the same verdict transfers: a mask head reads finer edge
detail than a box head does, so a contrast/lightness shift that was a wash for boxes
could plausibly help or hurt mask boundaries differently.

The actual pixel-level techniques (`TEST_TIME_PREPROCESSORS`, `build_preprocessors_from_spec`)
and the validation-split-copying step (`write_preprocessed_validation_split`) are
task-agnostic — imported from `detection/preprocessing_eval.py` rather than duplicated.
Only what's genuinely `-seg`-specific lives here: evaluating with a `-seg` model and
reading back box+mask metrics via `segmentation/yolo.py`'s `extract_metrics`.
"""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.utils import YAML

from poultry_monitoring.data.coco import CLASS_NAMES
from poultry_monitoring.detection.preprocessing_eval import (
    TEST_TIME_PREPROCESSORS,
    write_preprocessed_validation_split,
)
from poultry_monitoring.segmentation.yolo import extract_metrics


def evaluate_test_time_preprocessing(
    data_dir: Path,
    weights_path: Path,
    project: Path,
    preprocessors: dict[str, Callable[[np.ndarray], np.ndarray]] | None = None,
    output_path: Path | None = None,
) -> dict[str, dict[str, float]]:
    """Compare validation-set box+mask metrics of each preprocessed variant against the baseline.

    Each preprocessor runs over every validation image once, unconditionally — no
    retraining involved, inference/eval only. Mirrors
    `detection.preprocessing_eval.evaluate_test_time_preprocessing`'s structure; the
    only real difference is a `-seg` model and `extract_metrics`' box+mask shape
    instead of box-only.

    Args:
        data_dir: ChickenDet root (must already have `chickendet.yaml` from
            `data.coco.prepare_data`, with segments — i.e. `force_relabel` run at
            least once with `use_segments=True`, which `prepare_data` always does).
        weights_path: Trained `-seg` checkpoint to evaluate (unchanged across every
            variant).
        project: Local save dir for `model.val()` artifacts and each variant's image copy.
        preprocessors: Name -> image-array-in, image-array-out function. Defaults to
            `TEST_TIME_PREPROCESSORS`.
        output_path: If given, write the full results dict there as JSON.

    Returns:
        `{"baseline": {...}, <preprocessor_name>: {...}, ...}`, each value the same
        shape as `segmentation.yolo.TrainOutcome`'s metrics (box + mask).
    """
    data_dir = Path(data_dir).resolve()
    project = Path(project).resolve()
    preprocessors = preprocessors if preprocessors is not None else TEST_TIME_PREPROCESSORS
    model = YOLO(str(Path(weights_path).resolve()))

    results = {
        "baseline": extract_metrics(
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
        write_preprocessed_validation_split(data_dir, variant_dir, preprocess)
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
        results[name] = extract_metrics(
            model.val(data=str(yaml_path), project=str(project), name=f"ttp-{name}", exist_ok=True)
        )
    if output_path is not None:
        Path(output_path).write_text(json.dumps(results, indent=2))
    return results
