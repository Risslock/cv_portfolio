"""YOLO instance segmentation: train and predict wrappers around `ultralytics.YOLO`.

Single-run training and inference for YOLO segmentation models.
Related tools: `segmentation/preprocessing_eval.py` for test-time preprocessing,
`segmentation/visualize.py` for training/label/prediction visualizations,
`segmentation/synthetic_data.py` for building a Phase 3 Stage B data yaml (pass it to
`train`'s `--data-yaml`).
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from poultry_monitoring.data.coco import CLASS_NAMES, prepare_data
from poultry_monitoring.mlflow_utils import (
    SEGMENTATION_EXPERIMENT,
    configure_ultralytics_mlflow,
    finish_run,
    make_run_name,
)

SEED = 42

DEFAULT_FREEZE = 0

# Pinned explicitly instead of leaving Ultralytics' `optimizer="auto"`, which selects on
# iteration count computed against `nbs` (64) rather than `batch` -- ceil(len(train)/64)
# * epochs -- and flips MuSGD -> AdamW below 10k iterations. On ChickenDet (5,219 train
# images) that threshold lands at `epochs > 122`, so the `epochs` cap alone silently
# decided the optimizer: the 200-epoch baseline trained on MuSGD/lr0=0.01 while the
# 100-epoch copy-paste run trained on AdamW/lr0=0.002, wrecking an ablation meant to
# differ only in copy-paste. These four values reproduce what `auto` actually chose for
# the 100-epoch run -- note `auto` *ignores* args' own lr0/momentum/warmup_bias_lr, so
# a run's saved args.yaml does not describe the optimizer it trained with.
# See docs/adr/0018-pin-segmentation-optimizer.md.
DEFAULT_OPTIMIZER = "AdamW"
DEFAULT_LR0 = 0.002  # == auto's lr_fit, round(0.002 * 5 / (4 + nc), 6), at nc=1
DEFAULT_MOMENTUM = 0.9  # auto's value, not args' own 0.937 default
DEFAULT_WARMUP_BIAS_LR = 0.0  # auto zeroes this for Adam-family optimizers
# ChickenVerse doesn't report an F1-curve-derived operating point for segmentation (or
# use one) -- unlike detection/yolo.py's DEFAULT_CONF_THRESHOLD, these are just
# Ultralytics' own predict() defaults, not re-derived for this model. Revisit once
# there's an actual segmentation model to eyeball predictions from.
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.7


@dataclass
class TrainOutcome:
    """Result of `train`: where the best checkpoint landed, and its validation numbers.

    Attributes:
        weights_path: Path to the reloaded best checkpoint (`weights/best.pt`).
        box_map50: Box mAP@50 of `weights_path` on the validation split.
        box_map50_95: Box mAP@50-95 of `weights_path` on the validation split.
        box_precision: Box precision of `weights_path` on the validation split.
        box_recall: Box recall of `weights_path` on the validation split.
        mask_map50: Mask mAP@50 of `weights_path` on the validation split.
        mask_map50_95: Mask mAP@50-95 of `weights_path` on the validation split.
        mask_map75: Mask mAP@75 -- not part of `CLAUDE.md`'s logged-metric convention,
            included anyway since it's free from the same result object and directly
            comparable to ChickenVerse's own published segmentation table.
        mask_precision: Mask precision of `weights_path` on the validation split.
        mask_recall: Mask recall of `weights_path` on the validation split.
    """

    weights_path: Path
    box_map50: float
    box_map50_95: float
    box_precision: float
    box_recall: float
    mask_map50: float
    mask_map50_95: float
    mask_map75: float
    mask_precision: float
    mask_recall: float


def extract_metrics(val_metrics) -> dict[str, float]:
    """Flatten an Ultralytics `-seg` validation result into this project's metric names.

    A small, reusable step so `train` doesn't build the same metrics dict twice (once
    for MLflow, once for the return value) — one source of truth for the mapping
    between Ultralytics' result object and this project's logged metric names. Not
    module-private (unlike `detection/yolo.py`'s per-call-site metric dicts) because
    `segmentation/preprocessing_eval.py` reuses it too, for the same box+mask shape.

    Args:
        val_metrics: Return value of `YOLO(...).val(...)` for a `-seg` model —
            exposes both `.box` and `.seg` (mask) metric groups.

    Returns:
        Flat dict with `box_map50`/`box_map50_95`/`box_precision`/`box_recall` and
        `mask_map50`/`mask_map50_95`/`mask_map75`/`mask_precision`/`mask_recall`.
    """
    return {
        "box_map50": float(val_metrics.box.map50),
        "box_map50_95": float(val_metrics.box.map),
        "box_precision": float(val_metrics.box.mp),
        "box_recall": float(val_metrics.box.mr),
        "mask_map50": float(val_metrics.seg.map50),
        "mask_map50_95": float(val_metrics.seg.map),
        "mask_map75": float(val_metrics.seg.map75),
        "mask_precision": float(val_metrics.seg.mp),
        "mask_recall": float(val_metrics.seg.mr),
    }


def train(
    data_yaml: Path,
    project: Path,
    model_name: str = "yolo26n-seg",
    variant: str = "baseline",
    epochs: int = 20,
    patience: int = 10,
    batch: int = 16,
    fraction: float = 1.0,
    imgsz: int = 640,
    workers: int | None = None,
    freeze: int = DEFAULT_FREEZE,
    data_source: str = "real",
    hyperparameters: dict | None = None,
    resume_from: Path | None = None,
    weights_path: Path | None = None,
    copy_paste_bank: Path | None = None,
    copy_paste_p: float = 0.3,
    copy_paste_max_donors: int = 5,
) -> TrainOutcome:
    """Fine-tune a YOLO26-seg checkpoint, validate the best epoch, and log both to MLflow.

    Stock hyperparameters throughout, matching ChickenVerse's own segmentation recipe
    (see module docstring): one `model.train()` call, no custom augmentation passed to
    `model.train()` (Ultralytics' own stock augmentation, unmodified), and
    `freeze=DEFAULT_FREEZE` (0, all layers trainable from the start) unless overridden.

    Args:
        data_yaml: Path to the dataset YAML.
        project: Local save dir for this run's artifacts.
        model_name: Ultralytics `-seg` checkpoint name, e.g. `"yolo26n-seg"`.
        variant: Short label for this run (e.g. `"baseline"`, `"synthetic"`) — part of
            the MLflow run name via `mlflow_utils.make_run_name`.
        epochs: Max training epochs.
        patience: Early-stopping patience.
        batch: Fixed batch size — ChickenVerse used a fixed batch, not Ultralytics'
            auto-batch, so this matches their recipe rather than detection/yolo.py's
            own auto-batch choice.
        fraction: Fraction of the training set to use.
        imgsz: Training image size.
        workers: `DataLoader` worker count. `None` leaves Ultralytics' own default —
            exposed as a parameter (not hardcoded either way) because this project's
            own detection track hit a real Windows `DataLoader`-worker RAM leak under
            heavy repeated training (docs/adr/0007-subprocess-per-optuna-trial.md);
            worth an escape hatch here too if the same symptom ever shows up.
        freeze: Number of leading layers to freeze. `DEFAULT_FREEZE` (0) matches
            ChickenVerse's own recipe — exposed as a real parameter anyway (not
            hardcoded away) for the same reason `detection/yolo.py`'s `FREEZE_LAYERS`
            is overridable: a frozen-backbone comparison may be worth running here too.
        data_source: `"real"` (Stage A) or `"synthetic"` (Stage B) — logged as an
            MLflow tag so the two stages are directly filterable/comparable within the
            same `poultry_segmentation` experiment (constitution Principle IV/V
            disclosure). Doesn't change training behavior itself; whether `data_yaml`
            actually points at synthetic-augmented data is the caller's job (Stage B's
            data pipeline, not yet built).
        hyperparameters: Extra `model.train()` kwargs, merged over the pinned optimizer
            defaults (`DEFAULT_OPTIMIZER`/`DEFAULT_LR0`/`DEFAULT_MOMENTUM`/
            `DEFAULT_WARMUP_BIAS_LR`) so a caller can override them — e.g. `mosaic=0.0`
            for a mosaic-off continuation, or a lower `lr0` when starting from an
            already-converged checkpoint. Ignored when `resume_from` is given.
        resume_from: Path to an interrupted run's `weights/last.pt` — resumes it to
            completion instead of starting fresh (e.g. after an OOM crash). Takes
            priority over every other training-shape arg above (`epochs`, `patience`,
            `batch`, `fraction`, `imgsz`, `workers`, `freeze`), which Ultralytics
            re-reads from the interrupted run's own saved args instead.
        weights_path: Start from these weights instead of a stock `f"{model_name}.pt"`
            (`model_name` then only feeds MLflow naming). Unlike `resume_from`, this is a
            *fresh* run — new output dir, new MLflow run, and this call's own args apply —
            so the source run's artifacts are never reopened for writing. The intended use
            is a continuation with deliberately different args, e.g. a mosaic-off tail
            (`hyperparameters={"mosaic": 0.0, "close_mosaic": 0, "warmup_epochs": 0.0}`)
            off an already-converged checkpoint, which `resume=True` cannot express.
        copy_paste_bank: Curated donor bank directory. When given, training samples get
            synthetic instances pasted in on the fly (`segmentation.copy_paste_training`)
            — the Phase 3 Stage B treatment. `None` trains on real data only, byte-for-byte
            the stock path.
        copy_paste_p: Probability of pasting into a given training sample.
        copy_paste_max_donors: Inclusive upper bound on donors pasted per sample.

    Returns:
        The best checkpoint's path and validation metrics (box + mask).
    """
    project = Path(project).resolve()
    configure_ultralytics_mlflow(SEGMENTATION_EXPERIMENT)

    if resume_from is not None:
        model = YOLO(str(Path(resume_from).resolve()))
        results = model.train(resume=True)
    else:
        train_kwargs: dict[str, Any] = {
            "optimizer": DEFAULT_OPTIMIZER,
            "lr0": DEFAULT_LR0,
            "momentum": DEFAULT_MOMENTUM,
            "warmup_bias_lr": DEFAULT_WARMUP_BIAS_LR,
            **(hyperparameters or {}),
        }
        if workers is not None:
            train_kwargs["workers"] = workers
        if copy_paste_bank is not None:
            # Custom trainer, not a model.train() kwarg: the bank settings can't be cfg
            # overrides (get_cfg rejects unknown keys) -- see copy_paste_training.py.
            from poultry_monitoring.segmentation.copy_paste_training import (
                make_donor_bank_trainer,
            )

            train_kwargs["trainer"] = make_donor_bank_trainer(
                Path(copy_paste_bank).resolve(), p=copy_paste_p, max_donors=copy_paste_max_donors
            )
        model = YOLO(
            str(Path(weights_path).resolve()) if weights_path is not None else f"{model_name}.pt"
        )
        results = model.train(
            data=str(Path(data_yaml).resolve()),
            epochs=epochs,
            patience=patience,
            batch=batch,
            fraction=fraction,
            imgsz=imgsz,
            seed=SEED,
            freeze=freeze,
            project=str(project),
            name=f"{model_name}-{variant}",
            exist_ok=True,
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
    metrics = extract_metrics(val_metrics)

    scale = model_name.removeprefix("yolo26").removesuffix("-seg")
    make_run_name(model_family="yolo26-seg", variant=f"{scale}-{variant}")
    extra_params = {"model_family": "yolo26-seg", "variant": variant, "model_name": model_name}
    if copy_paste_bank is not None:
        # Logged so a Stage B run records the augmentation strength it actually used --
        # these aren't Ultralytics args, so its native MLflow integration can't see them.
        extra_params |= {
            "copy_paste_bank": str(copy_paste_bank),
            "copy_paste_p": copy_paste_p,
            "copy_paste_max_donors": copy_paste_max_donors,
        }
    finish_run(
        extra_params=extra_params,
        extra_tags={"data_source": data_source},
        extra_metrics=metrics,
    )
    return TrainOutcome(weights_path=best_weights_path, **metrics)


def predict(
    weights_path: Path,
    source: Path,
    conf: float = DEFAULT_CONF_THRESHOLD,
    iou: float = DEFAULT_IOU_THRESHOLD,
    save_dir: Path | None = None,
    show_boxes: bool = True,
) -> list:
    """Run inference with a trained `-seg` checkpoint on arbitrary images.

    Renders and writes each saved image manually (`Results.plot` + `cv2.imwrite`)
    rather than `model.predict(save=True, project=..., name=...)` — the latter's
    `project`/`name` resolution goes through Ultralytics' own `runs/<task>` default
    dir for any relative path, which has silently written output outside this
    project before. Manual save keeps `save_dir` exactly what's given.

    Args:
        weights_path: Path to a trained `.pt` checkpoint.
        source: Image, directory, or glob pattern — anything `YOLO.predict`'s own
            `source` accepts.
        conf: Minimum detection confidence to keep a box/mask.
        iou: NMS IoU threshold for de-duplicating overlapping detections.
        save_dir: If given, save annotated prediction images here (one file per
            input, same basename); otherwise results are only returned in-memory.
        show_boxes: If `True` (default), draw Ultralytics' standard box + label +
            confidence overlay alongside the masks, one color per class. If `False`,
            draw masks only — no boxes/labels/confidence text — with each instance
            in its own random color (`color_mode="instance"`), which reads better
            on a single-class, high-density scene where a repeated "Chicken" label
            on every one of ~50 boxes is noise, not signal.

    Returns:
        One Ultralytics `Results` object per input image (each carries both `.boxes`
        and `.masks` for a `-seg` checkpoint).
    """
    model = YOLO(str(Path(weights_path).resolve()))
    results = model.predict(source=str(Path(source).resolve()), conf=conf, iou=iou, save=False)

    if save_dir is not None:
        resolved_save_dir = Path(save_dir).resolve()
        resolved_save_dir.mkdir(parents=True, exist_ok=True)
        for r in results:
            plotted = r.plot(
                boxes=show_boxes,
                labels=show_boxes,
                conf=show_boxes,
                masks=True,
                color_mode="class" if show_boxes else "instance",
            )
            cv2.imwrite(str(resolved_save_dir / Path(r.path).name), plotted)

    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the `python -m poultry_monitoring.segmentation.yolo` CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train", help="Fine-tune one size, ChickenVerse-matched stock hyperparameters."
    )
    train_parser.add_argument(
        "--data-dir", type=Path, required=True, help="ChickenDet dataset root."
    )
    train_parser.add_argument(
        "--force-relabel",
        action="store_true",
        help="Regenerate labels/ with segments even if a box-only labels/ already exists.",
    )
    train_parser.add_argument("--model-name", default="yolo26n-seg")
    train_parser.add_argument("--variant", default="baseline")
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--patience", type=int, default=10)
    train_parser.add_argument("--batch", type=int, default=16)
    train_parser.add_argument("--fraction", type=float, default=1.0)
    train_parser.add_argument("--imgsz", type=int, default=640)
    train_parser.add_argument("--workers", type=int, default=None)
    train_parser.add_argument(
        "--resume-from", type=Path, default=None, help="Resume an interrupted weights/last.pt."
    )
    train_parser.add_argument(
        "--initial-weights",
        type=Path,
        default=None,
        help="Start from this checkpoint as a FRESH run (new dir, new MLflow run) rather "
        "than a stock yolo26*-seg.pt -- unlike --resume-from, this call's own args apply, "
        "so it can continue a converged run with e.g. mosaic off.",
    )
    train_parser.add_argument(
        "--hyperparameters-json",
        type=str,
        default="{}",
        help="JSON dict of extra model.train() kwargs, merged over the pinned optimizer "
        'defaults -- e.g. \'{"mosaic": 0.0, "close_mosaic": 0, "lr0": 0.0007}\'.',
    )
    train_parser.add_argument(
        "--hyperparameters-file",
        type=Path,
        default=None,
        help="JSON file with the same shape as --hyperparameters-json. Wins over it if both.",
    )
    train_parser.add_argument(
        "--freeze",
        type=int,
        default=DEFAULT_FREEZE,
        help="Leading layers to freeze (default 0, matching ChickenVerse's own "
        "recipe -- see detection/yolo.py's --freeze for the frozen-backbone use case).",
    )
    train_parser.add_argument(
        "--data-source",
        choices=["real", "synthetic"],
        default="real",
        help="Tags the MLflow run -- Phase 3 Stage A (real) vs. Stage B (synthetic).",
    )
    train_parser.add_argument(
        "--copy-paste-bank",
        type=Path,
        default=None,
        help="Curated donor bank dir -- enables on-the-fly synthetic copy-paste "
        "(Phase 3 Stage B). Omit for a real-data-only run.",
    )
    train_parser.add_argument("--copy-paste-p", type=float, default=0.3)
    train_parser.add_argument("--copy-paste-max-donors", type=int, default=5)
    train_parser.add_argument(
        "--data-yaml",
        type=Path,
        default=None,
        help="Use this data yaml directly instead of regenerating labels/chickendet.yaml "
        "from --data-dir's raw COCO annotations -- e.g. a Stage B chickendet_stage_b.yaml "
        "from `segmentation.synthetic_data`'s `generate` CLI.",
    )

    predict_parser = subparsers.add_parser(
        "predict", help="Run a trained -seg checkpoint on images."
    )
    predict_parser.add_argument("--weights", type=Path, required=True, help="Trained .pt file.")
    predict_parser.add_argument("--source", type=Path, required=True, help="Image, dir, or glob.")
    predict_parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD)
    predict_parser.add_argument("--iou", type=float, default=DEFAULT_IOU_THRESHOLD)
    predict_parser.add_argument("--save-dir", type=Path, default=None)
    predict_parser.add_argument(
        "--masks-only",
        action="store_true",
        help="Draw masks only, no boxes/labels/confidence -- each instance a distinct "
        "random color (color_mode='instance'). Default draws Ultralytics' standard "
        "box+label+mask overlay.",
    )

    ttp_parser = subparsers.add_parser(
        "ttp",
        help="Compare test-time-only preprocessing (autocontrast/CLAHE/hist-eq/"
        "brightness-contrast) on val, box+mask metrics.",
    )
    ttp_parser.add_argument("--data-dir", type=Path, required=True, help="ChickenDet dataset root.")
    ttp_parser.add_argument("--weights", type=Path, required=True, help="Trained -seg .pt file.")
    ttp_parser.add_argument(
        "--variants-json",
        type=str,
        default="{}",
        help="JSON spec of extra variants to test -- same shape as detection/yolo.py's "
        "ttp --variants-json. Replaces the default candidate set rather than adding to it.",
    )
    ttp_parser.add_argument(
        "--variants-file",
        type=Path,
        default=None,
        help="JSON file with the same shape as --variants-json. Wins over "
        "--variants-json if both are given.",
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

    `preprocessing_eval`'s import is local to this function, not at module top level —
    that module imports `extract_metrics` from *this* module, so a top-level import
    here would be circular. `_load_json_arg`/`build_preprocessors_from_spec` are
    reused from `detection/yolo.py`/`detection/preprocessing_eval.py` rather than
    duplicated — both are task-agnostic (JSON-arg parsing, pixel-only transforms).
    See docs/adr/0010-split-yolo-py-by-responsibility.md.
    """
    from poultry_monitoring.detection.preprocessing_eval import build_preprocessors_from_spec
    from poultry_monitoring.detection.yolo import _load_json_arg
    from poultry_monitoring.segmentation.preprocessing_eval import evaluate_test_time_preprocessing

    args = _build_arg_parser().parse_args()

    if args.command == "predict":
        results = predict(
            args.weights,
            args.source,
            conf=args.conf,
            iou=args.iou,
            save_dir=args.save_dir,
            show_boxes=not args.masks_only,
        )
        for r in results:
            n_masks = 0 if r.masks is None else len(r.masks)
            print(f"{r.path}: {len(r.boxes)} boxes, {n_masks} masks")
        return

    if args.command == "ttp":
        data_dir = args.data_dir.resolve()
        project = data_dir / "YOLO"
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
        return

    data_dir = args.data_dir.resolve()
    project = data_dir / "YOLO"
    data_yaml = (
        args.data_yaml.resolve()
        if args.data_yaml is not None
        else prepare_data(data_dir, CLASS_NAMES, force_relabel=args.force_relabel)
    )
    outcome = train(
        data_yaml,
        project,
        model_name=args.model_name,
        variant=args.variant,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        fraction=args.fraction,
        imgsz=args.imgsz,
        workers=args.workers,
        freeze=args.freeze,
        data_source=args.data_source,
        hyperparameters=_load_json_arg(args.hyperparameters_json, args.hyperparameters_file),
        resume_from=args.resume_from,
        weights_path=args.initial_weights,
        copy_paste_bank=args.copy_paste_bank,
        copy_paste_p=args.copy_paste_p,
        copy_paste_max_donors=args.copy_paste_max_donors,
    )
    print(outcome)


if __name__ == "__main__":
    main()
