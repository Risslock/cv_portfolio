"""YOLO26 multi-run training strategies: search, progressive unfreezing, size sweeps.

Everything here runs `yolo.py`'s `train()` repeatedly under some strategy, as opposed to
`yolo.py` itself, which only defines the single-run train/predict core. Imports `train`
and a few constants from `yolo.py` — see that module's docstring for why `yolo.py`'s own
CLI imports back from here lazily (inside `main()`), not at module top level.
"""

import json
import random
import uuid
from pathlib import Path

import optuna
import torch

from poultry_monitoring.data.coco import prepare_data
from poultry_monitoring.detection.yolo import (
    CLASS_NAMES,
    CUSTOM_AUGMENTATION_PARAM_RANGES,
    SEED,
    TrainOutcome,
    train,
)

# Conservative search space centered on Ultralytics' shipped defaults — see
# docs/adr/0008-conservative-hyperparameter-space.md for the per-parameter rationale.
# `close_mosaic` isn't searched at all (see `yolo.default_close_mosaic`); `lr0` is held
# fixed here too, deliberately, so this pass tunes the rest of the space first.
AUGMENTATION_FOCUSED_SPACE = {
    "perspective": (0.0, 0.0005),
    "shear": (0.0, 5.0),
    "flipud": (0.0, 1.0),
    "mixup": (0.0, 1.0),
    "cutmix": (0.0, 1.0),
    "copy_paste": (0.0, 1.0),
    "multi_scale": (0.0, 0.25),
}


def tune_hyperparameters(
    data_yaml: Path,
    project: Path,
    model_name: str = "yolo26n",
    iterations: int = 20,
    epochs: int = 15,
    fraction: float = 0.3,
    imgsz: int = 640,
    space: dict | None = None,
    output_path: Path | None = None,
) -> dict:
    """Search hyperparameters with Optuna, scoring each trial by validation `box_map50_95`.

    Each trial is a plain `train()` call, in-process. Not Ultralytics' native
    `model.tune()` — see docs/adr/0005-genetic-tuner-undersearches-from-a-fixed-start.md.

    Args:
        data_yaml: Path to the dataset YAML (`prepare_data`'s return value).
        project: Local save dir for tuning artifacts (`<project>/<model_name>-tune-trial<N>/`).
        model_name: Ultralytics checkpoint name to tune from, e.g. `"yolo26n"`.
        iterations: Number of Optuna trials.
        epochs: Epochs per trial — short by design; this is a search, not a final fit.
        fraction: Fraction of the training set per trial, for search speed.
        imgsz: Training image size.
        space: Hyperparameter search space, `{name: (min, max)}`. Defaults to
            `AUGMENTATION_FOCUSED_SPACE` merged with `CUSTOM_AUGMENTATION_PARAM_RANGES`.
        output_path: If given, write the best hyperparameters there as JSON — meant to be
            passed straight to `train`/`unfreeze`'s `--hyperparameters-file`, sidestepping
            shell-quoting a JSON blob by hand.

    Returns:
        The best hyperparameters found, as a dict.
    """
    search_space = (
        space
        if space is not None
        else {**AUGMENTATION_FOCUSED_SPACE, **CUSTOM_AUGMENTATION_PARAM_RANGES}
    )

    def objective(trial: optuna.Trial) -> float:
        sampled = {k: trial.suggest_float(k, lo, hi) for k, (lo, hi) in search_space.items()}
        try:
            outcome = train(
                data_yaml,
                project,
                model_name=model_name,
                variant=f"tune-trial{trial.number}",
                hyperparameters=sampled,
                epochs=epochs,
                patience=epochs,
                fraction=fraction,
                imgsz=imgsz,
            )
        except (MemoryError, torch.OutOfMemoryError) as e:
            print(f"Trial {trial.number} ran out of memory, stopping the study: {e}")
            trial.study.stop()
            return 0.0
        except Exception as e:
            print(f"Trial {trial.number} failed, stopping the study: {e}")
            trial.study.stop()
            return 0.0
        return outcome.box_map50_95

    study = optuna.create_study(
        study_name=f"tune-{model_name}",
        storage=f"sqlite:///{Path('optuna.db').resolve()}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=iterations)
    best = dict(study.best_params)
    if output_path is not None:
        Path(output_path).write_text(json.dumps(best, indent=2))
    return best


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
    """Random-search this project's custom Albumentations parameters.

    Runs `n_trials` random draws through `train`, scoring each by validation
    `box_map50_95`. Random rather than grid — see
    docs/adr/0002-random-search-for-augmentation-parameters.md.

    Args:
        data_yaml: Path to the dataset YAML.
        project: Local save dir for trial artifacts.
        model_name: Ultralytics checkpoint name to search with, e.g. `"yolo26n"`.
        n_trials: Number of random parameter draws to try.
        epochs: Epochs per trial — short by design; this is a search.
        fraction: Fraction of the training set per trial.
        hyperparameters: Fixed hyperparameters applied to every trial (typically
            `tune_hyperparameters`'s result), so this search runs under realistic
            conditions rather than untuned defaults.
        param_ranges: `{name: (min, max)}` to sample from. Defaults to
            `CUSTOM_AUGMENTATION_PARAM_RANGES`.
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


# Progressive-unfreezing schedule: freeze less, drop lr0, each stage — see
# docs/adr/0006-progressive-unfreeze-stage-config.md for the optimizer/close_mosaic/
# patience rationale.
DEFAULT_UNFREEZE_STAGES = [
    {
        "freeze": 10,
        "lr0": 5e-4,
        "optimizer": "AdamW",
        "epochs": 30,
        "patience": 3,
        "close_mosaic": 4,
    },
    {
        "freeze": 5,
        "lr0": 1e-4,
        "optimizer": "AdamW",
        "epochs": 30,
        "patience": 3,
        "close_mosaic": 4,
    },
    {
        "freeze": 0,
        "lr0": 2e-5,
        "optimizer": "AdamW",
        "epochs": 30,
        "patience": 3,
        "close_mosaic": 4,
    },
]


def progressive_unfreeze_train(
    data_yaml: Path,
    project: Path,
    model_name: str,
    initial_weights: Path,
    hyperparameters: dict | None = None,
    stages: list[dict] | None = None,
    fraction: float = 1.0,
    run_name: str | None = None,
) -> list[TrainOutcome]:
    """Continue fine-tuning `initial_weights` through progressively less-frozen stages.

    Each stage reloads the *previous* stage's best checkpoint (or `initial_weights` for
    the first) via `train`'s `weights_path`/`freeze` — not `resume_from`, since
    `resume=True` would keep the earlier stage's own stored freeze/lr0 instead of this
    stage's.

    Args:
        data_yaml: Path to the dataset YAML.
        project: Local save dir for each stage's artifacts.
        model_name: Ultralytics model name for MLflow naming, e.g. `"yolo26n"`.
        initial_weights: Starting checkpoint — typically a prior `TrainOutcome.weights_path`.
        hyperparameters: Fixed hyperparameters applied to every stage; each stage's own
            `lr0`/`optimizer`/`close_mosaic` (if given) override this dict for those keys.
        stages: List of `{"freeze", "lr0", "optimizer", "epochs", "patience"}` dicts
            (`"close_mosaic"` optional), in order. Defaults to `DEFAULT_UNFREEZE_STAGES`.
        fraction: Fraction of the training set to use, every stage.
        run_name: Distinguishes this run's artifacts/MLflow names from other unfreeze
            runs (e.g. `"isolated"` -> `unfreeze-isolated-stage0`) so a re-run doesn't
            silently overwrite a previous one's local folder. Defaults to a random
            8-char id, printed once at the start — same convention as the `run_id[:8]`
            suffix `mlflow_utils.make_run_name` already appends to every MLflow run name.

    Returns:
        One `TrainOutcome` per stage, in order (last one is the final result).
    """
    stages = stages if stages is not None else DEFAULT_UNFREEZE_STAGES
    run_name = run_name if run_name is not None else uuid.uuid4().hex[:8]
    print(f"Unfreeze run name: {run_name}")
    variant_prefix = f"unfreeze-{run_name}"
    outcomes = []
    weights = initial_weights
    for i, stage in enumerate(stages):
        stage_hyperparameters = {
            **(hyperparameters or {}),
            "lr0": stage["lr0"],
            "optimizer": stage["optimizer"],
        }
        if "close_mosaic" in stage:
            stage_hyperparameters["close_mosaic"] = stage["close_mosaic"]
        outcome = train(
            data_yaml,
            project,
            model_name=model_name,
            variant=f"{variant_prefix}-stage{i}",
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

    Both searches run once on `yolo26n`; the winning config is applied to every size in
    `sizes`, not re-tuned per size.

    Args:
        data_dir: Dataset root (contains `images/`, `annotations/`).
        sizes: YOLO26 scale letters to train, e.g. `("n", "s")`.
        tune_iterations: Optuna trials for `tune_hyperparameters`.
        tune_epochs: Epochs per hyperparameter-tuning trial.
        tune_fraction: Fraction of the training set per hyperparameter-tuning trial.
        aug_trials: Random draws for `tune_augmentation_parameters`.
        aug_epochs: Epochs per augmentation-search trial.
        aug_fraction: Fraction of the training set per augmentation-search trial.
        train_epochs: Max epochs for each final size's training run.
        train_patience: Early-stopping patience for each final size's training run.
        train_fraction: Fraction of the training set for each final size's training run.
        force_relabel: See `data.coco.prepare_data`.

    Returns:
        Mapping of size letter (e.g. `"n"`) to that size's `TrainOutcome`.
    """
    project = data_dir / "YOLO"
    data_yaml = prepare_data(data_dir, CLASS_NAMES, force_relabel=force_relabel)
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
