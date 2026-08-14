"""YOLO26 detection core: single-run train/predict wrappers around `ultralytics.YOLO`.

Framework class used natively (constitution Principle I) — this module supplies the
project-specific glue (data prep, MLflow wiring, the augmentation hook), not a wrapper
around YOLO itself. Productionized from `notebooks/02_yolo26_baseline.ipynb`.

Multi-run strategies (hyperparameter/augmentation search, progressive unfreezing, size
sweeps) live in `detection/tuning.py`; test-time preprocessing comparisons live in
`detection/preprocessing_eval.py` — both built on `train()` here, and both import core
pieces (`train`, `SEED`, `CLASS_NAMES`, `CUSTOM_AUGMENTATION_PARAM_RANGES`) from this
module at their own top level. This module's CLI (`main`, below) still wires up every
subcommand from all three — `python -m poultry_monitoring.detection.yolo <subcommand>`
is unchanged — but imports `tuning`/`preprocessing_eval` *inside* `main()`, not at this
module's top level, specifically to avoid a circular import (see docs/adr/0010-split-
yolo-py-by-responsibility.md).
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO

from poultry_monitoring.augmentation import detection as aug_detection
from poultry_monitoring.augmentation import shared as aug_shared
from poultry_monitoring.data.coco import CLASS_NAMES, prepare_data
from poultry_monitoring.mlflow_utils import (
    DETECTION_EXPERIMENT,
    configure_ultralytics_mlflow,
    finish_run,
    make_run_name,
)

SEED = 42
# Precision/recall trade-off, picked off the F1 curve in notebook 02 — lower favors
# recall (fewer missed birds), higher favors precision (fewer double-counts).
DEFAULT_CONF_THRESHOLD = 0.36
DEFAULT_IOU_THRESHOLD = 0.5
# Freeze most of the backbone, fine-tune the neck/head — standard practice for
# fine-tuning a pretrained checkpoint on a small dataset. Fixed everywhere below, not
# searched.
FREEZE_LAYERS = 10

# Every custom Albumentations parameter across shared.py/detection.py, combined once so
# train() knows which hyperparameter keys are custom augmentation params vs. real YOLO
# training arguments.
CUSTOM_AUGMENTATION_PARAM_RANGES = {**aug_shared.PARAM_RANGES, **aug_detection.PARAM_RANGES}


def _augmentation_kwargs_from(hyperparameters: dict) -> dict:
    """Pull just the custom-augmentation entries out of a hyperparameters dict.

    Args:
        hyperparameters: A dict that may contain any of
            `CUSTOM_AUGMENTATION_PARAM_RANGES`'s keys.

    Returns:
        Only the entries that are custom-augmentation keys.
    """
    return {k: v for k, v in hyperparameters.items() if k in CUSTOM_AUGMENTATION_PARAM_RANGES}


def _build_custom_augmentations(hyperparameters: dict) -> list:
    """Build the full custom Albumentations transform list from a hyperparameters dict.

    Routes each key to whichever of `augmentation.shared`/`augmentation.detection`
    owns it, then concatenates both builders' output — Ultralytics' own
    `Albumentations` wrapper threads bboxes through automatically when any transform
    in the list needs it, so no extra plumbing is needed here.

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


def default_close_mosaic(epochs: int, fraction: float = 0.15) -> int:
    """Calculate the number of final epochs to run with mosaic disabled.

    Turning mosaic off for a tail of training lets the model see un-composited images
    before it's done learning, avoiding the sharp overfitting/train-val gap jumps a
    fixed epoch count causes on runs of very different lengths (see
    docs/adr/0008-conservative-hyperparameter-space.md).

    Args:
        epochs: Training epochs for this specific run/trial.
        fraction: Proportion of `epochs` to spend with mosaic disabled, once `epochs`
            clears the 20-epoch floor below.

    Returns:
        `0` if `epochs < 20` (too short a run for a mosaic-free tail to help), else
        `max(1, round(epochs * fraction))`.
    """
    return 0 if epochs < 20 else max(1, round(epochs * fraction))


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

    Reloads the *best* checkpoint after training (not whatever's left in `model` after
    the last epoch) and validates that explicitly, logging `box_map50`/`box_map50_95`/
    `box_precision`/`box_recall` via `mlflow_utils.finish_run` — Ultralytics' own MLflow
    integration logs per-epoch metrics under different key names, not this project's
    convention (CLAUDE.md § MLflow Conventions).

    Args:
        data_yaml: Path to the dataset YAML.
        project: Local save dir for this run's artifacts.
        model_name: Ultralytics checkpoint name, e.g. `"yolo26n"`. Ignored (but still
            used for MLflow naming) if `weights_path` is given.
        variant: Short label for this run (e.g. `"tuned"`) — part of the MLflow run
            name via `mlflow_utils.make_run_name`.
        hyperparameters: Extra `model.train()` kwargs. Any `CUSTOM_AUGMENTATION_PARAM_RANGES`
            key is routed into `_build_custom_augmentations` instead of passed to
            `model.train()` directly. Ignored when `resume_from` is given.
        epochs: Max training epochs. Ignored when `resume_from` is given.
        patience: Early-stopping patience. Ignored when `resume_from` is given.
        fraction: Fraction of the training set to use. Ignored when `resume_from` is given.
        imgsz: Training image size. Ignored when `resume_from` is given.
        resume_from: Path to an interrupted run's `weights/last.pt` — resumes it to
            completion instead of starting fresh. Takes priority over `weights_path`/`freeze`.
        weights_path: Start from these weights instead of a stock `f"{model_name}.pt"`
            checkpoint. Unlike `resume_from`, this is a fresh `model.train()` call, not
            `resume=True` — see `tuning.progressive_unfreeze_train`.
        freeze: Override `FREEZE_LAYERS` for this call.

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
    train_kwargs.setdefault("close_mosaic", default_close_mosaic(epochs))

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
            augmentations=_build_custom_augmentations(hyperparameters),
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


def predict(
    weights_path: Path,
    source: Path,
    conf: float = DEFAULT_CONF_THRESHOLD,
    iou: float = DEFAULT_IOU_THRESHOLD,
    save_dir: Path | None = None,
) -> list:
    """Run inference with a trained checkpoint on arbitrary images.

    Args:
        weights_path: Path to a trained `.pt` checkpoint.
        source: Image, directory, or glob pattern — anything `YOLO.predict`'s own
            `source` accepts.
        conf: Minimum detection confidence to keep a box (see `DEFAULT_CONF_THRESHOLD`).
        iou: NMS IoU threshold for de-duplicating overlapping boxes.
        save_dir: If given, save annotated prediction images here; otherwise results
            are only returned in-memory.

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


def _load_json_arg(json_str: str, file_path: Path | None) -> dict:
    """Load a `--<x>-json`/`--<x>-file` CLI pair into one dict.

    `file_path` wins when both are given — meant for a file written by `tune`'s
    `--output`, sidestepping having to shell-quote a JSON blob by hand (PowerShell in
    particular mangles quotes in an inline JSON argument passed to a native exe).
    Shared by `train`/`unfreeze`'s `--hyperparameters-*` and `ttp`'s `--variants-*` —
    all three are just "a JSON dict, inline or from a file."

    Args:
        json_str: Raw JSON string, e.g. `args.hyperparameters_json`.
        file_path: Path to a JSON file, e.g. `args.hyperparameters_file`.

    Returns:
        The parsed dict.
    """
    if file_path is not None:
        return json.loads(Path(file_path).read_text())
    return json.loads(json_str)


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
    tune_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the best hyperparameters here as JSON, e.g. for train/unfreeze's "
        "--hyperparameters-file. Defaults to <project>/best_hyperparameters.json.",
    )

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
    train_parser.add_argument("--imgsz", type=int, default=640)
    train_parser.add_argument(
        "--resume-from", type=Path, default=None, help="Resume an interrupted weights/last.pt."
    )
    train_parser.add_argument(
        "--hyperparameters-json",
        type=str,
        default="{}",
        help="JSON dict of fixed hyperparameters, e.g. a prior tune_hyperparameters result.",
    )
    train_parser.add_argument(
        "--hyperparameters-file",
        type=Path,
        default=None,
        help="JSON file of fixed hyperparameters (e.g. tune's --output). Wins over "
        "--hyperparameters-json if both are given.",
    )
    train_parser.add_argument(
        "--freeze",
        type=int,
        default=None,
        help="Override FREEZE_LAYERS (default 10) — e.g. --freeze 0 to unfreeze the "
        "whole backbone from the start instead of progressive_unfreeze_train's staged "
        "approach.",
    )

    unfreeze_parser = subparsers.add_parser(
        "unfreeze", help="Progressive-unfreezing refinement from a trained checkpoint."
    )
    _add_dataset_args(unfreeze_parser)
    unfreeze_parser.add_argument("--model-name", default="yolo26n")
    unfreeze_parser.add_argument("--initial-weights", type=Path, required=True)
    unfreeze_parser.add_argument(
        "--run-name",
        default=None,
        help="Distinguishes this run's folders/MLflow names, e.g. 'isolated' -> "
        "unfreeze-isolated-stage0. Default: a random 8-char id, printed at the start.",
    )
    unfreeze_parser.add_argument(
        "--hyperparameters-json",
        type=str,
        default="{}",
        help="JSON dict of fixed hyperparameters applied to every stage.",
    )
    unfreeze_parser.add_argument(
        "--hyperparameters-file",
        type=Path,
        default=None,
        help="JSON file of fixed hyperparameters (e.g. tune's --output). Wins over "
        "--hyperparameters-json if both are given.",
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
        "ttp",
        help="Compare test-time-only preprocessing (autocontrast/CLAHE/hist-eq/"
        "brightness-contrast) on val.",
    )
    ttp_parser.add_argument("--data-dir", type=Path, required=True, help="ChickenDet dataset root.")
    ttp_parser.add_argument("--weights", type=Path, required=True, help="Trained .pt file.")
    ttp_parser.add_argument(
        "--variants-json",
        type=str,
        default="{}",
        help='JSON spec of extra variants to test, e.g. \'{"bc_b30_c18": {"technique": '
        '"brightness_contrast", "brightness": -30, "contrast": 1.8}}\'. "technique" must '
        "be one of TEST_TIME_PREPROCESSORS' keys (autocontrast/clahe/hist_eq/"
        "brightness_contrast); everything else is passed through as that technique's "
        "kwargs. Replaces the default candidate set rather than adding to it.",
    )
    ttp_parser.add_argument(
        "--variants-file",
        type=Path,
        default=None,
        help="JSON file with the same shape as --variants-json. Wins over "
        "--variants-json if both are given — PowerShell mangles quotes in an inline "
        "JSON argument passed to a native exe, so this is the reliable option there.",
    )
    ttp_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the results dict here as JSON. Defaults to <project>/ttp_results.json.",
    )

    return parser


def main() -> None:
    """CLI entry point — see `_build_arg_parser` for `--help` on each subcommand.

    Imports from `tuning`/`preprocessing_eval` are local to this function, not at module
    top level — both of those modules import core pieces (`train`, `SEED`, `CLASS_NAMES`,
    `CUSTOM_AUGMENTATION_PARAM_RANGES`) from *this* module, so a top-level import here
    would be circular. See docs/adr/0010-split-yolo-py-by-responsibility.md.
    """
    from poultry_monitoring.detection.preprocessing_eval import (
        build_preprocessors_from_spec,
        evaluate_test_time_preprocessing,
    )
    from poultry_monitoring.detection.tuning import (
        progressive_unfreeze_train,
        run_size_sweep,
        tune_augmentation_parameters,
        tune_hyperparameters,
    )

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
        data_yaml = prepare_data(data_dir, CLASS_NAMES, force_relabel=args.force_relabel)
        output_path = (
            args.output if args.output is not None else project / "best_hyperparameters.json"
        )
        best = tune_hyperparameters(
            data_yaml,
            project,
            iterations=args.iterations,
            epochs=args.epochs,
            fraction=args.fraction,
            output_path=output_path,
        )
        print(json.dumps(best))
        print(f"Saved to {output_path}")
    elif args.command == "augtune":
        data_yaml = prepare_data(data_dir, CLASS_NAMES, force_relabel=args.force_relabel)
        best = tune_augmentation_parameters(
            data_yaml,
            project,
            n_trials=args.trials,
            epochs=args.epochs,
            fraction=args.fraction,
        )
        print(best)
    elif args.command == "train":
        data_yaml = prepare_data(data_dir, CLASS_NAMES, force_relabel=args.force_relabel)
        hyperparameters = _load_json_arg(args.hyperparameters_json, args.hyperparameters_file)
        outcome = train(
            data_yaml,
            project,
            model_name=args.model_name,
            variant=args.variant,
            hyperparameters=hyperparameters,
            epochs=args.epochs,
            patience=args.patience,
            fraction=args.fraction,
            imgsz=args.imgsz,
            resume_from=args.resume_from,
            freeze=args.freeze,
        )
        print(outcome)
    elif args.command == "unfreeze":
        data_yaml = prepare_data(data_dir, CLASS_NAMES, force_relabel=args.force_relabel)
        hyperparameters = _load_json_arg(args.hyperparameters_json, args.hyperparameters_file)
        outcomes = progressive_unfreeze_train(
            data_yaml,
            project,
            model_name=args.model_name,
            initial_weights=args.initial_weights,
            hyperparameters=hyperparameters,
            fraction=args.fraction,
            run_name=args.run_name,
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
        variants = _load_json_arg(args.variants_json, args.variants_file)
        output_path = args.output if args.output is not None else project / "ttp_results.json"
        results = evaluate_test_time_preprocessing(
            data_dir,
            args.weights,
            project,
            preprocessors=build_preprocessors_from_spec(variants) if variants else None,
            output_path=output_path,
        )
        print(results)
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
