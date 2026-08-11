"""YOLO26 detection: train/tune wrappers around `ultralytics.YOLO`.

Framework class used natively per constitution Principle I — this module supplies the
project-specific glue (data prep, MLflow wiring, the augmentation hook), not a wrapper
class around YOLO itself. Productionized from `notebooks/02_yolo26_baseline.ipynb`;
see that notebook's Notes section for the fine-tuning rationale (freeze/lr0 choices,
mosaic/close_mosaic behavior) this module builds on.
"""

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import YAML

from poultry_monitoring.augmentation.shared import PARAM_RANGES, build_domain_transforms
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
    """Pull `build_domain_transforms` kwargs out of a hyperparameters dict, if present.

    `train` needs this to keep `augmentation.shared.PARAM_RANGES` keys (e.g. from
    `tune_augmentation_parameters`'s result) out of the raw kwargs handed to
    `model.train()` — those aren't real YOLO training arguments and `get_cfg()` would
    reject them outright — while still routing them into `build_domain_transforms`.
    Falls back to that function's own defaults for any key not present (e.g. a plain
    `train()` call with no tuned augmentation parameters at all).

    Args:
        hyperparameters: A dict that may contain any of `PARAM_RANGES`'s keys.

    Returns:
        A kwargs dict suitable for `build_domain_transforms(**kwargs)`.
    """
    return {k: hyperparameters[k] for k in PARAM_RANGES if k in hyperparameters}


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
            `augmentation.shared.PARAM_RANGES`.
        seed: Seed for the random draws, for a reproducible search.

    Returns:
        The best-scoring draw, as a `build_domain_transforms(**...)`-ready dict.
    """
    ranges = param_ranges if param_ranges is not None else PARAM_RANGES
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
        model_name: Ultralytics checkpoint name, e.g. `"yolo26n"`, `"yolo26s"`.
        variant: Short label for this run (e.g. `"tuned"`, `"baseline"`) — becomes
            part of the MLflow run name via `mlflow_utils.make_run_name`.
        hyperparameters: Extra `model.train()` kwargs, typically merged output from
            `tune_hyperparameters` and/or `tune_augmentation_parameters`. Any of
            `augmentation.shared.PARAM_RANGES`'s keys are pulled out and routed into
            `build_domain_transforms` instead of passed to `model.train()` directly —
            they aren't real YOLO training arguments and `get_cfg()` would reject them.
            `freeze`/`seed` are always applied regardless (see module-level
            `FREEZE_LAYERS`/`SEED`). Ignored when `resume_from` is given.
        epochs: Max training epochs. Ignored when `resume_from` is given.
        patience: Early-stopping patience. Ignored when `resume_from` is given.
        fraction: Fraction of the training set to use. Ignored when `resume_from` is given.
        imgsz: Training image size. Ignored when `resume_from` is given.
        resume_from: Path to an interrupted run's `weights/last.pt`. If given, resumes
            that run to completion (Ultralytics reads the rest of the training config
            from the same directory's `args.yaml`) instead of starting a fresh one.
            Logs as a new MLflow run — the interrupted run's own entry is left as-is.

    Returns:
        The best checkpoint's path and validation metrics.
    """
    project = Path(project).resolve()
    configure_ultralytics_mlflow(DETECTION_EXPERIMENT)
    hyperparameters = dict(hyperparameters or {})
    augmentation_kwargs = _augmentation_kwargs_from(hyperparameters)
    train_kwargs = {k: v for k, v in hyperparameters.items() if k not in PARAM_RANGES}

    if resume_from is not None:
        model = YOLO(str(Path(resume_from).resolve()))
        results = model.train(resume=True)
    else:
        model = YOLO(f"{model_name}.pt")
        results = model.train(
            data=str(Path(data_yaml).resolve()),
            epochs=epochs,
            patience=patience,
            fraction=fraction,
            imgsz=imgsz,
            seed=SEED,
            freeze=FREEZE_LAYERS,
            project=str(project),
            name=f"{model_name}-{variant}",
            exist_ok=True,
            augmentations=build_domain_transforms(**augmentation_kwargs),  # documented kwarg
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
            epochs=args.epochs,
            patience=args.patience,
            fraction=args.fraction,
            resume_from=args.resume_from,
        )
        print(outcome)
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


if __name__ == "__main__":
    main()
