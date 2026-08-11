"""YOLO26 detection: train/tune wrappers around `ultralytics.YOLO`.

Framework class used natively per constitution Principle I — this module supplies the
project-specific glue (data prep, MLflow wiring, the augmentation hook), not a wrapper
class around YOLO itself. Productionized from `notebooks/02_yolo26_baseline.ipynb`;
see that notebook's Notes section for the fine-tuning rationale (freeze/lr0 choices,
mosaic/close_mosaic behavior) this module builds on.
"""

import argparse
import json
import random
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils import YAML

from poultry_monitoring.augmentation import detection as aug_detection
from poultry_monitoring.augmentation import shared as aug_shared
from poultry_monitoring.data.coco import (
    build_data_yaml,
    convert_coco_to_yolo_labels,
    fix_iscrowd_field,
)
from poultry_monitoring.mlflow_utils import (
    DETECTION_EXPERIMENT,
    configure_ultralytics_mlflow,
    finish_run,
    make_run_name,
)

SEED = 42
CLASS_NAMES = {0: "Chicken"}
# Precision/recall trade-off, picked off the F1 curve in notebook 02 — lower favors
# recall (fewer missed birds), higher favors precision (fewer double-counts).
DEFAULT_CONF_THRESHOLD = 0.36
DEFAULT_IOU_THRESHOLD = 0.5
# Freeze most of the backbone, fine-tune the neck/head — Ultralytics' own guidance for
# fine-tuning a pretrained checkpoint on a small dataset (see notebook 02's Notes).
# Not part of `model.tune()`'s search space (a structural choice, not a continuous
# hyperparameter), so it's applied as a fixed kwarg everywhere below.
FREEZE_LAYERS = 10

# Every custom Albumentations parameter across both augmentation modules — shared.py
# (image-only) and detection.py (bbox-aware, e.g. occlusion). Combined once here so
# train()/tune_augmentation_parameters() have one place to check "is this key a custom
# augmentation param, or a real YOLO training argument".
CUSTOM_AUGMENTATION_PARAM_RANGES = {**aug_shared.PARAM_RANGES, **aug_detection.PARAM_RANGES}

# Focused on augmentation + lr0 rather than Ultralytics' full default space (also sweeps
# momentum/weight_decay/warmup/loss weights — not this project's current focus). A
# custom `space=` *replaces* the default (not merged), so every built-in augmentation
# hyperparameter is listed explicitly. Excludes this project's own Albumentations
# parameters on purpose — see docs/adr/0001-custom-augmentation-search-separate-from-tuner.md.
AUGMENTATION_FOCUSED_SPACE = {
    "lr0": (1e-5, 1e-2),
    "hsv_h": (0.0, 0.1),
    "hsv_s": (0.0, 0.9),
    "hsv_v": (0.0, 0.9),
    "degrees": (0.0, 45.0),
    "translate": (0.0, 0.9),
    "scale": (0.0, 0.95),
    "shear": (0.0, 10.0),
    "perspective": (0.0, 0.001),
    "flipud": (0.0, 1.0),
    "fliplr": (0.0, 1.0),
    "bgr": (0.0, 1.0),
    "mosaic": (0.0, 1.0),
    "mixup": (0.0, 1.0),
    "cutmix": (0.0, 1.0),
    "copy_paste": (0.0, 1.0),
    "close_mosaic": (0.0, 10.0),
    # Added after the first tune/train pass (not in the original search) — per-batch
    # input-resolution jitter, distinct from `scale`'s per-image zoom/crop range.
    # Conservative range: ChickenDet is a fixed overhead rig, so extreme resizing is
    # less likely to help than for a dataset with real camera-distance variety.
    "multi_scale": (0.0, 0.3),
}


def prepare_data(data_dir: Path, force_relabel: bool = False) -> Path:
    """Fix `iscrowd`, convert COCO to YOLO labels (with segments), and write the data YAML.

    Segments are included even though this is the detection task: Ultralytics' native
    `copy_paste` density augmentation silently no-ops without them (see
    `data/coco.py`). A `detect`-task model trains on the derived boxes either way.

    Args:
        data_dir: Dataset root (contains `images/`, `annotations/`).
        force_relabel: Regenerate `labels/` even if it already exists. Needed the first
            time this runs against a `labels/` dir a box-only conversion already wrote
            (e.g. from `notebooks/02_yolo26_baseline.ipynb`) — otherwise the existing
            box-only labels are kept as-is and `copy_paste` silently has no segments.

    Returns:
        Path to the written `data.yaml`.
    """
    annotations_dir = data_dir / "annotations"
    for split in ("Train", "Validation", "Test"):
        fix_iscrowd_field(annotations_dir / f"instances_{split}.json", assume_yes=True)
    convert_coco_to_yolo_labels(data_dir, annotations_dir, force=force_relabel, use_segments=True)
    return build_data_yaml(data_dir, CLASS_NAMES, data_dir / "chickendet.yaml")


def _augmentation_kwargs_from(hyperparameters: dict) -> dict:
    """Pull every custom augmentation param out of a hyperparameters dict, if present.

    For logging (`train` tags each with `aug_` and passes them to `finish_run`) — see
    `_build_custom_augmentations` for turning these into an actual transform list.

    Args:
        hyperparameters: A dict that may contain any of
            `CUSTOM_AUGMENTATION_PARAM_RANGES`'s keys.

    Returns:
        Just the custom-augmentation entries of `hyperparameters`.
    """
    return {k: v for k, v in hyperparameters.items() if k in CUSTOM_AUGMENTATION_PARAM_RANGES}


def _build_custom_augmentations(hyperparameters: dict) -> list:
    """Build the full custom Albumentations transform list from a hyperparameters dict.

    Routes each key to whichever of `augmentation.shared`/`augmentation.detection`
    actually owns it, then concatenates both builders' output into one list — Ultralytics'
    own `Albumentations` wrapper checks `contains_spatial` across the whole list and
    threads bboxes through automatically when any transform needs it (see
    `augmentation/detection.py`), so no extra plumbing is needed here for that.

    Args:
        hyperparameters: A dict that may contain any of
            `CUSTOM_AUGMENTATION_PARAM_RANGES`'s keys. Missing keys fall back to each
            builder's own defaults.

    Returns:
        Combined transform list, for `model.train(augmentations=...)`.
    """
    shared_kwargs = {k: v for k, v in hyperparameters.items() if k in aug_shared.PARAM_RANGES}
    detection_kwargs = {k: v for k, v in hyperparameters.items() if k in aug_detection.PARAM_RANGES}
    return aug_shared.build_domain_transforms(
        **shared_kwargs
    ) + aug_detection.build_detection_transforms(**detection_kwargs)


def tune_hyperparameters(
    data_yaml: Path,
    project: Path,
    model_name: str = "yolo26n",
    iterations: int = 20,
    epochs: int = 15,
    fraction: float = 0.3,
    imgsz: int = 640,
    space: dict | None = None,
) -> dict:
    """Search YOLO26 hyperparameters with Ultralytics' native genetic-algorithm tuner.

    Runs `iterations` short training trials, each `epochs` epochs on `fraction` of the
    data, mutating hyperparameters toward better fitness. Each trial is its own MLflow
    run in `DETECTION_EXPERIMENT` (Ultralytics' native integration — `mlflow_utils.py`),
    with MLflow's own auto-generated run name.

    Doesn't cover this project's own Albumentations transforms — every trial gets
    Ultralytics' bare-minimum default touch instead (effectively off). See
    `tune_augmentation_parameters` for that search, and
    docs/adr/0001-custom-augmentation-search-separate-from-tuner.md for why they're split.

    Args:
        data_yaml: Path to the dataset YAML (`prepare_data`'s return value).
        project: Local save dir for tuning artifacts (`<project>/tune/`).
        model_name: Ultralytics checkpoint name to tune from, e.g. `"yolo26n"`.
        iterations: Number of genetic-algorithm generations.
        epochs: Epochs per trial — short by design; this is a search, not a final fit.
        fraction: Fraction of the training set per trial, for search speed.
        imgsz: Training image size.
        space: Hyperparameter search space, `{name: (min, max)}`. Defaults to
            `AUGMENTATION_FOCUSED_SPACE`.

    Returns:
        The best hyperparameters found, as a dict (from `best_hyperparameters.yaml`).
    """
    project = Path(project).resolve()
    configure_ultralytics_mlflow(DETECTION_EXPERIMENT)
    model = YOLO(f"{model_name}.pt")
    model.tune(
        data=str(Path(data_yaml).resolve()),
        iterations=iterations,
        epochs=epochs,
        fraction=fraction,
        imgsz=imgsz,
        seed=SEED,
        freeze=FREEZE_LAYERS,
        project=str(project),
        name="tune",
        exist_ok=True,
        plots=False,
        space=space if space is not None else AUGMENTATION_FOCUSED_SPACE,
    )
    return dict(YAML.load(project / "tune" / "best_hyperparameters.yaml"))


def tune_augmentation_parameters(
    data_yaml: Path,
    project: Path,
    model_name: str = "yolo26n",
    n_trials: int = 8,
    epochs: int = 15,
    fraction: float = 0.3,
    hyperparameters: dict | None = None,
    param_ranges: dict | None = None,
    seed: int = SEED,
) -> dict:
    """Random-search this project's custom Albumentations parameters, in-process.

    Runs `n_trials` random draws over `train` directly, scoring each by validation
    `box_map50_95`. Separate from `tune_hyperparameters`/`model.tune()`
    (docs/adr/0001-custom-augmentation-search-separate-from-tuner.md) and random rather
    than grid (docs/adr/0002-random-search-for-augmentation-parameters.md).

    Args:
        data_yaml: Path to the dataset YAML.
        project: Local save dir for trial artifacts.
        model_name: Ultralytics checkpoint name to search with, e.g. `"yolo26n"`.
        n_trials: Number of random parameter draws to try.
        epochs: Epochs per trial — short by design; this is a search.
        fraction: Fraction of the training set per trial.
        hyperparameters: Fixed hyperparameters (typically `tune_hyperparameters`'s
            result) applied to every trial, so the augmentation search happens under
            realistic conditions rather than untuned defaults.
        param_ranges: `{name: (min, max)}` to sample from. Defaults to
            `CUSTOM_AUGMENTATION_PARAM_RANGES` (both `augmentation.shared` and
            `augmentation.detection`).
        seed: Seed for the random draws, for a reproducible search.

    Returns:
        The best-scoring draw, as a `_build_custom_augmentations(...)`-ready dict.
    """
    ranges = param_ranges if param_ranges is not None else CUSTOM_AUGMENTATION_PARAM_RANGES
    rng = random.Random(seed)
    best_score = None
    best_choice = {name: rng.uniform(lo, hi) for name, (lo, hi) in ranges.items()}
    for i in range(n_trials):
        choice = {name: rng.uniform(lo, hi) for name, (lo, hi) in ranges.items()}
        outcome = train(
            data_yaml,
            project,
            model_name=model_name,
            variant=f"augsearch-{i}",
            hyperparameters={**(hyperparameters or {}), **choice},
            epochs=epochs,
            patience=epochs,  # no early stop mid-search — let every trial finish
            fraction=fraction,
        )
        if best_score is None or outcome.box_map50_95 > best_score:
            best_score = outcome.box_map50_95
            best_choice = choice
    return best_choice


@dataclass
class TrainOutcome:
    """Result of `train`: where the best checkpoint landed, and its validation numbers.

    Attributes:
        weights_path: Path to the reloaded best checkpoint (`weights/best.pt`).
        box_map50: Box mAP@50 of `weights_path` on the validation split.
        box_map50_95: Box mAP@50-95 of `weights_path` on the validation split.
        box_precision: Box precision of `weights_path` on the validation split.
        box_recall: Box recall of `weights_path` on the validation split.
    """

    weights_path: Path
    box_map50: float
    box_map50_95: float
    box_precision: float
    box_recall: float


def train(
    data_yaml: Path,
    project: Path,
    model_name: str,
    variant: str,
    hyperparameters: dict | None = None,
    epochs: int = 100,
    patience: int = 10,
    fraction: float = 1.0,
    imgsz: int = 640,
    resume_from: Path | None = None,
    weights_path: Path | None = None,
    freeze: int | None = None,
) -> TrainOutcome:
    """Fine-tune a YOLO26 checkpoint, validate the best epoch, and log both to MLflow.

    Mirrors notebook 02's established pattern: reload the *best* checkpoint after
    training (not whatever's left in `model` after the last epoch) and validate that
    explicitly. Ultralytics' native MLflow integration already logs per-epoch training
    metrics under its own key names, but not this project's `box_map50`/`box_map50_95`
    convention (`CLAUDE.md` § MLflow Conventions) — this validation pass supplies those,
    logged via `mlflow_utils.finish_run`'s `extra_metrics`.

    Args:
        data_yaml: Path to the dataset YAML.
        project: Local save dir for this run's artifacts.
        model_name: Ultralytics checkpoint name, e.g. `"yolo26n"`, `"yolo26s"`. Ignored
            (but still used for MLflow naming) if `weights_path` is given.
        variant: Short label for this run (e.g. `"tuned"`, `"baseline"`) — becomes
            part of the MLflow run name via `mlflow_utils.make_run_name`.
        hyperparameters: Extra `model.train()` kwargs, typically merged output from
            `tune_hyperparameters` and/or `tune_augmentation_parameters`. Any of
            `CUSTOM_AUGMENTATION_PARAM_RANGES`'s keys are pulled out and routed into
            `_build_custom_augmentations` instead of passed to `model.train()` directly
            — they aren't real YOLO training arguments and `get_cfg()` would reject
            them. `freeze`/`seed` are always applied regardless (see module-level
            `FREEZE_LAYERS`/`SEED`). Ignored when `resume_from` is given.
        epochs: Max training epochs. Ignored when `resume_from` is given.
        patience: Early-stopping patience. Ignored when `resume_from` is given.
        fraction: Fraction of the training set to use. Ignored when `resume_from` is given.
        imgsz: Training image size. Ignored when `resume_from` is given.
        resume_from: Path to an interrupted run's `weights/last.pt`. If given, resumes
            that run to completion (Ultralytics reads the rest of the training config
            from the same directory's `args.yaml`) instead of starting a fresh one.
            Logs as a new MLflow run — the interrupted run's own entry is left as-is.
            Takes priority over `weights_path`/`freeze` if both are given.
        weights_path: Start from these weights instead of a stock `f"{model_name}.pt"`
            pretrained checkpoint — e.g. a previous stage's `best.pt` in
            `progressive_unfreeze_train`. Unlike `resume_from`, this is a **fresh**
            `model.train()` call with the args below, not `resume=True` (which would
            keep the checkpoint's own stored freeze/lr0, defeating the point of a new
            stage with deliberately different ones).
        freeze: Override `FREEZE_LAYERS` for this call. See `progressive_unfreeze_train`.

    Returns:
        The best checkpoint's path and validation metrics.
    """
    project = Path(project).resolve()
    configure_ultralytics_mlflow(DETECTION_EXPERIMENT)
    hyperparameters = dict(hyperparameters or {})
    augmentation_kwargs = _augmentation_kwargs_from(hyperparameters)
    train_kwargs = {
        k: v for k, v in hyperparameters.items() if k not in CUSTOM_AUGMENTATION_PARAM_RANGES
    }

    if resume_from is not None:
        model = YOLO(str(Path(resume_from).resolve()))
        results = model.train(resume=True)
    else:
        checkpoint = (
            str(Path(weights_path).resolve()) if weights_path is not None else f"{model_name}.pt"
        )
        model = YOLO(checkpoint)
        results = model.train(
            data=str(Path(data_yaml).resolve()),
            epochs=epochs,
            patience=patience,
            fraction=fraction,
            imgsz=imgsz,
            seed=SEED,
            freeze=freeze if freeze is not None else FREEZE_LAYERS,
            project=str(project),
            name=f"{model_name}-{variant}",
            exist_ok=True,
            augmentations=_build_custom_augmentations(hyperparameters),  # documented kwarg
            **train_kwargs,
        )

    best_weights_path = Path(results.save_dir) / "weights" / "best.pt"
    best_model = YOLO(best_weights_path)
    val_metrics = best_model.val(
        data=str(Path(data_yaml).resolve()),
        project=str(project),
        name=f"{model_name}-{variant}-val",
        exist_ok=True,
    )

    scale = model_name.removeprefix("yolo26")
    make_run_name(model_family="yolo26", variant=f"{scale}-{variant}")
    finish_run(
        extra_params={
            "model_family": "yolo26",
            "variant": variant,
            "model_name": model_name,
            **{f"aug_{k}": v for k, v in augmentation_kwargs.items()},
        },
        extra_metrics={
            "box_map50": float(val_metrics.box.map50),
            "box_map50_95": float(val_metrics.box.map),
            "box_precision": float(val_metrics.box.mp),
            "box_recall": float(val_metrics.box.mr),
        },
    )
    return TrainOutcome(
        weights_path=best_weights_path,
        box_map50=float(val_metrics.box.map50),
        box_map50_95=float(val_metrics.box.map),
        box_precision=float(val_metrics.box.mp),
        box_recall=float(val_metrics.box.mr),
    )


# Default progressive-unfreezing schedule: freeze less, drop lr0, each stage — standard
# discriminative fine-tuning (ULMFiT-style gradual unfreezing). `optimizer: "AdamW"` is
# forced because Ultralytics' `optimizer="auto"` default picks its own lr independent of
# a configured lr0 (observed directly: the tuned lr0=0.01 got silently overridden to
# AdamW(lr=0.002) during the size-sweep run) — auto's heuristic has no notion of "this is
# a later refinement stage on an already-converged model," so a per-stage lr0 needs a
# fixed optimizer to actually take effect.
DEFAULT_UNFREEZE_STAGES = [
    {"freeze": 10, "lr0": 5e-4, "optimizer": "AdamW", "epochs": 30, "patience": 10},
    {"freeze": 5, "lr0": 1e-4, "optimizer": "AdamW", "epochs": 30, "patience": 10},
    {"freeze": 0, "lr0": 2e-5, "optimizer": "AdamW", "epochs": 30, "patience": 10},
]


def progressive_unfreeze_train(
    data_yaml: Path,
    project: Path,
    model_name: str,
    initial_weights: Path,
    hyperparameters: dict | None = None,
    stages: list[dict] | None = None,
    fraction: float = 1.0,
) -> list[TrainOutcome]:
    """Continue fine-tuning `initial_weights` through progressively less-frozen stages.

    Each stage reloads the *previous* stage's best checkpoint (or `initial_weights` for
    the first) and trains fresh with that stage's own `freeze`/`lr0` via `train`'s
    `weights_path`/`freeze` — not `resume_from` (`resume=True` would keep the earlier
    stage's freeze/lr0 from the checkpoint's own stored args, defeating the point).

    Args:
        data_yaml: Path to the dataset YAML.
        project: Local save dir for each stage's artifacts.
        model_name: Ultralytics model name for MLflow naming, e.g. `"yolo26n"`.
        initial_weights: Starting checkpoint — typically a prior `TrainOutcome.weights_path`.
        hyperparameters: Fixed hyperparameters (e.g. `tune_augmentation_parameters`'s
            result) applied to every stage; each stage's own `lr0`/`optimizer` override
            whatever this dict has for those two keys specifically.
        stages: List of `{"freeze", "lr0", "optimizer", "epochs", "patience"}` dicts, in
            order. Defaults to `DEFAULT_UNFREEZE_STAGES`.
        fraction: Fraction of the training set to use, every stage.

    Returns:
        One `TrainOutcome` per stage, in order (last one is the final result).
    """
    stages = stages if stages is not None else DEFAULT_UNFREEZE_STAGES
    outcomes = []
    weights = initial_weights
    for i, stage in enumerate(stages):
        stage_hyperparameters = {
            **(hyperparameters or {}),
            "lr0": stage["lr0"],
            "optimizer": stage["optimizer"],
        }
        outcome = train(
            data_yaml,
            project,
            model_name=model_name,
            variant=f"unfreeze-stage{i}",
            hyperparameters=stage_hyperparameters,
            epochs=stage["epochs"],
            patience=stage["patience"],
            fraction=fraction,
            weights_path=weights,
            freeze=stage["freeze"],
        )
        outcomes.append(outcome)
        weights = outcome.weights_path
    return outcomes


def run_size_sweep(
    data_dir: Path,
    sizes: tuple[str, ...] = ("n", "s"),
    tune_iterations: int = 20,
    tune_epochs: int = 15,
    tune_fraction: float = 0.3,
    aug_trials: int = 8,
    aug_epochs: int = 15,
    aug_fraction: float = 0.3,
    train_epochs: int = 100,
    train_patience: int = 10,
    train_fraction: float = 1.0,
    force_relabel: bool = False,
) -> dict[str, object]:
    """Tune once on `yolo26n`, then train every size in `sizes` with those hyperparameters.

    Two independent searches, both run once on `yolo26n` and then applied to every size
    in `sizes` — per the project's tuning-scope decision: one tuning pass is much
    cheaper than tuning per size, and hyperparameters/augmentation choices usually
    transfer reasonably well across scales of the same architecture:

    1. `tune_hyperparameters` — Ultralytics' native genetic search (lr0 + built-in
       augmentation magnitudes).
    2. `tune_augmentation_parameters` — this project's custom Albumentations parameters
       (color invariance, lighting/contrast), run *under* the hyperparameters (1) found,
       so the augmentation search reflects realistic training conditions.

    Args:
        data_dir: Dataset root (contains `images/`, `annotations/`).
        sizes: YOLO26 scale letters to train, e.g. `("n", "s")`.
        tune_iterations: Genetic-algorithm generations for `tune_hyperparameters`.
        tune_epochs: Epochs per hyperparameter-tuning trial.
        tune_fraction: Fraction of the training set per hyperparameter-tuning trial.
        aug_trials: Random draws for `tune_augmentation_parameters`.
        aug_epochs: Epochs per augmentation-search trial.
        aug_fraction: Fraction of the training set per augmentation-search trial.
        train_epochs: Max epochs for each final size's training run.
        train_patience: Early-stopping patience for each final size's training run.
        train_fraction: Fraction of the training set for each final size's training
            run. Defaults to the full dataset — override for a quick smoke test.
        force_relabel: See `prepare_data`.

    Returns:
        Mapping of size letter (e.g. `"n"`) to that size's `TrainOutcome`.
    """
    project = data_dir / "YOLO"
    data_yaml = prepare_data(data_dir, force_relabel=force_relabel)
    best_hyperparameters = tune_hyperparameters(
        data_yaml,
        project,
        model_name="yolo26n",
        iterations=tune_iterations,
        epochs=tune_epochs,
        fraction=tune_fraction,
    )
    best_augmentation = tune_augmentation_parameters(
        data_yaml,
        project,
        model_name="yolo26n",
        n_trials=aug_trials,
        epochs=aug_epochs,
        fraction=aug_fraction,
        hyperparameters=best_hyperparameters,
    )
    combined_hyperparameters = {**best_hyperparameters, **best_augmentation}
    return {
        size: train(
            data_yaml,
            project,
            model_name=f"yolo26{size}",
            variant="tuned",
            hyperparameters=combined_hyperparameters,
            epochs=train_epochs,
            fraction=train_fraction,
            patience=train_patience,
        )
        for size in sizes
    }


def predict(
    weights_path: Path,
    source: Path,
    conf: float = DEFAULT_CONF_THRESHOLD,
    iou: float = DEFAULT_IOU_THRESHOLD,
    save_dir: Path | None = None,
) -> list:
    """Run inference with a trained checkpoint on-demand, on arbitrary images.

    Not part of the training/tuning pipeline above — for trying a trained checkpoint
    (typically a `TrainOutcome.weights_path`) against new images: a single file, a
    directory, or a glob pattern, anything `ultralytics.YOLO.predict`'s own `source`
    accepts.

    Args:
        weights_path: Path to a trained `.pt` checkpoint.
        source: Image file, directory, or glob pattern to run inference on.
        conf: Minimum detection confidence to keep a box. See module-level
            `DEFAULT_CONF_THRESHOLD` for the precision/recall trade-off this controls.
        iou: NMS IoU threshold for de-duplicating overlapping boxes.
        save_dir: If given, save annotated prediction images (boxes drawn) here.
            If `None`, nothing is written to disk — results are only returned in-memory.

    Returns:
        One Ultralytics `Results` object per input image.
    """
    model = YOLO(str(Path(weights_path).resolve()))
    resolved_save_dir = Path(save_dir).resolve() if save_dir else None
    return model.predict(
        source=str(Path(source).resolve()),
        conf=conf,
        iou=iou,
        save=resolved_save_dir is not None,
        project=str(resolved_save_dir.parent) if resolved_save_dir else None,
        name=resolved_save_dir.name if resolved_save_dir else None,
        exist_ok=True,
    )


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


# Test-time-only preprocessing candidates — no retraining involved, see
# `evaluate_test_time_preprocessing`. Follows up on notebook 02's autocontrast observation.
TEST_TIME_PREPROCESSORS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "autocontrast": _autocontrast_image,
    "clahe": _clahe_image,
    "hist_eq": _histogram_equalize_image,
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
        data_dir: ChickenDet root (must already have `chickendet.yaml` from `prepare_data`).
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


def _add_dataset_args(subparser: argparse.ArgumentParser) -> None:
    """Add the `--data-dir`/`--force-relabel` args shared by the training subcommands."""
    subparser.add_argument("--data-dir", type=Path, required=True, help="ChickenDet dataset root.")
    subparser.add_argument(
        "--force-relabel",
        action="store_true",
        help="Regenerate labels/ with segments even if a box-only labels/ already exists.",
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the `python -m poultry_monitoring.detection.yolo` CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    tune_parser = subparsers.add_parser("tune", help="Hyperparameter search on yolo26n.")
    _add_dataset_args(tune_parser)
    tune_parser.add_argument("--iterations", type=int, default=20)
    tune_parser.add_argument("--epochs", type=int, default=15)
    tune_parser.add_argument("--fraction", type=float, default=0.3)

    augtune_parser = subparsers.add_parser(
        "augtune", help="Random-search custom Albumentations parameters on yolo26n."
    )
    _add_dataset_args(augtune_parser)
    augtune_parser.add_argument("--trials", type=int, default=8)
    augtune_parser.add_argument("--epochs", type=int, default=15)
    augtune_parser.add_argument("--fraction", type=float, default=0.3)

    train_parser = subparsers.add_parser("train", help="Fine-tune one size, fixed hyperparameters.")
    _add_dataset_args(train_parser)
    train_parser.add_argument("--model-name", default="yolo26n")
    train_parser.add_argument("--variant", default="baseline")
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--patience", type=int, default=10)
    train_parser.add_argument("--fraction", type=float, default=1.0)
    train_parser.add_argument(
        "--resume-from", type=Path, default=None, help="Resume an interrupted weights/last.pt."
    )
    train_parser.add_argument(
        "--hyperparameters-json",
        type=str,
        default="{}",
        help="JSON dict of fixed hyperparameters, e.g. a prior tune_hyperparameters result.",
    )

    unfreeze_parser = subparsers.add_parser(
        "unfreeze", help="Progressive-unfreezing refinement from a trained checkpoint."
    )
    _add_dataset_args(unfreeze_parser)
    unfreeze_parser.add_argument("--model-name", default="yolo26n")
    unfreeze_parser.add_argument("--initial-weights", type=Path, required=True)
    unfreeze_parser.add_argument(
        "--hyperparameters-json",
        type=str,
        default="{}",
        help="JSON dict of fixed hyperparameters applied to every stage.",
    )
    unfreeze_parser.add_argument("--fraction", type=float, default=1.0)

    sweep_parser = subparsers.add_parser("sweep", help="Tune once, train every size in --sizes.")
    _add_dataset_args(sweep_parser)
    sweep_parser.add_argument("--sizes", nargs="+", default=["n", "s"])
    sweep_parser.add_argument("--tune-iterations", type=int, default=20)
    sweep_parser.add_argument("--tune-epochs", type=int, default=15)
    sweep_parser.add_argument("--tune-fraction", type=float, default=0.3)
    sweep_parser.add_argument("--aug-trials", type=int, default=8)
    sweep_parser.add_argument("--aug-epochs", type=int, default=15)
    sweep_parser.add_argument("--aug-fraction", type=float, default=0.3)
    sweep_parser.add_argument("--train-epochs", type=int, default=100)
    sweep_parser.add_argument("--train-patience", type=int, default=10)
    sweep_parser.add_argument("--train-fraction", type=float, default=1.0)

    predict_parser = subparsers.add_parser("predict", help="Run a trained checkpoint on images.")
    predict_parser.add_argument("--weights", type=Path, required=True, help="Trained .pt file.")
    predict_parser.add_argument("--source", type=Path, required=True, help="Image, dir, or glob.")
    predict_parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD)
    predict_parser.add_argument("--iou", type=float, default=DEFAULT_IOU_THRESHOLD)
    predict_parser.add_argument("--save-dir", type=Path, default=None)

    ttp_parser = subparsers.add_parser(
        "ttp", help="Compare test-time-only preprocessing (autocontrast/CLAHE/hist-eq) on val."
    )
    ttp_parser.add_argument("--data-dir", type=Path, required=True, help="ChickenDet dataset root.")
    ttp_parser.add_argument("--weights", type=Path, required=True, help="Trained .pt file.")

    return parser


def main() -> None:
    """CLI entry point — see `_build_arg_parser` for `--help` on each subcommand."""
    args = _build_arg_parser().parse_args()

    if args.command == "predict":
        results = predict(
            args.weights, args.source, conf=args.conf, iou=args.iou, save_dir=args.save_dir
        )
        for r in results:
            print(f"{r.path}: {len(r.boxes)} boxes")
        return

    data_dir = args.data_dir.resolve()
    project = data_dir / "YOLO"

    if args.command == "tune":
        data_yaml = prepare_data(data_dir, force_relabel=args.force_relabel)
        best = tune_hyperparameters(
            data_yaml,
            project,
            iterations=args.iterations,
            epochs=args.epochs,
            fraction=args.fraction,
        )
        print(best)
    elif args.command == "augtune":
        data_yaml = prepare_data(data_dir, force_relabel=args.force_relabel)
        best = tune_augmentation_parameters(
            data_yaml,
            project,
            n_trials=args.trials,
            epochs=args.epochs,
            fraction=args.fraction,
        )
        print(best)
    elif args.command == "train":
        data_yaml = prepare_data(data_dir, force_relabel=args.force_relabel)
        outcome = train(
            data_yaml,
            project,
            model_name=args.model_name,
            variant=args.variant,
            hyperparameters=json.loads(args.hyperparameters_json),
            epochs=args.epochs,
            patience=args.patience,
            fraction=args.fraction,
            resume_from=args.resume_from,
        )
        print(outcome)
    elif args.command == "unfreeze":
        data_yaml = prepare_data(data_dir, force_relabel=args.force_relabel)
        outcomes = progressive_unfreeze_train(
            data_yaml,
            project,
            model_name=args.model_name,
            initial_weights=args.initial_weights,
            hyperparameters=json.loads(args.hyperparameters_json),
            fraction=args.fraction,
        )
        print(outcomes)
    elif args.command == "sweep":
        outcomes = run_size_sweep(
            data_dir,
            sizes=tuple(args.sizes),
            tune_iterations=args.tune_iterations,
            tune_epochs=args.tune_epochs,
            tune_fraction=args.tune_fraction,
            aug_trials=args.aug_trials,
            aug_epochs=args.aug_epochs,
            aug_fraction=args.aug_fraction,
            train_epochs=args.train_epochs,
            train_patience=args.train_patience,
            train_fraction=args.train_fraction,
            force_relabel=args.force_relabel,
        )
        print(outcomes)
    elif args.command == "ttp":
        results = evaluate_test_time_preprocessing(data_dir, args.weights, project)
        print(results)


if __name__ == "__main__":
    main()
