# ADR 0010: Split `detection/yolo.py` by responsibility, keep the CLI unified

## Status

Accepted

## Context

`detection/yolo.py` grew to ~1000 lines and three genuinely different responsibilities:
single-run train/predict (the actual core), multi-run search/orchestration strategies
(`tune_hyperparameters`, `tune_augmentation_parameters`, `progressive_unfreeze_train`,
`run_size_sweep`), and a test-time image-preprocessing comparison harness
(`evaluate_test_time_preprocessing` and its four candidate transforms). None of the
latter two are "train/test/predict/validate/fine-tune the model" in any direct sense —
they're strategies and experiments *built on* that core, not the core itself.

Splitting them out hits an ordering problem: the multi-run strategies and the
preprocessing harness both need core pieces from `yolo.py` (`train`, `SEED`,
`CLASS_NAMES`, `CUSTOM_AUGMENTATION_PARAM_RANGES`), but the project's CLI convention
(`python -m poultry_monitoring.detection.yolo <subcommand>`, documented throughout
`README.md`/`CLAUDE.md` and every command from this session) means `yolo.py` also needs
to import *from* those modules to wire up their subcommands (`tune`, `augtune`,
`unfreeze`, `sweep`, `ttp`). A plain top-level import each way is a circular import.

## Decision

Three-way split, plus `prepare_data` moved into `data/coco.py` (it was already a thin
wrapper around functions already living there):

- **`detection/yolo.py`** — core only: `TrainOutcome`, `train()`, `predict()`, the
  augmentation-kwargs helpers `train()` needs internally, and the shared constants
  (`SEED`, `CLASS_NAMES`, `FREEZE_LAYERS`, `CUSTOM_AUGMENTATION_PARAM_RANGES`). Also
  still owns the CLI (`main`, `_build_arg_parser`) — the command surface is unchanged.
- **`detection/tuning.py`** — `AUGMENTATION_FOCUSED_SPACE`, `tune_hyperparameters`,
  `tune_augmentation_parameters`, `DEFAULT_UNFREEZE_STAGES`,
  `progressive_unfreeze_train`, `run_size_sweep`. Imports `train`/`SEED`/`CLASS_NAMES`/
  `CUSTOM_AUGMENTATION_PARAM_RANGES` from `yolo.py` at normal module top level.
- **`detection/preprocessing_eval.py`** — the four candidate transforms,
  `TEST_TIME_PREPROCESSORS`, `evaluate_test_time_preprocessing`. Imports `CLASS_NAMES`
  from `yolo.py` at normal module top level.
- **The circular-import fix**: `yolo.py`'s `main()` imports from `tuning`/
  `preprocessing_eval` *inside the function body*, not at module top level. By the time
  `main()` actually runs, `yolo.py` is already fully loaded, so `tuning.py`/
  `preprocessing_eval.py`'s own top-level `from ...yolo import ...` (triggered when
  Python resolves `main()`'s local import) finds a complete module, not a partial one.
  `_build_arg_parser()` doesn't need the local imports — it never calls the moved
  functions, only builds the argparse tree.

## Consequences

- `python -m poultry_monitoring.detection.yolo <subcommand>` is completely unchanged —
  no README/CLAUDE.md Commands-section rewrite, no muscle-memory broken.
- `tests/test_yolo.py` now covers only `yolo.py`'s own pieces
  (`_augmentation_kwargs_from`/`_build_custom_augmentations`); `progressive_unfreeze_train`'s
  tests moved to `tests/test_tuning.py`, mocking `poultry_monitoring.detection.tuning.train`
  (not `...yolo.train` — `unittest.mock.patch` targets where a name is *looked up*, and
  `tuning.py` binds its own local `train` via `from yolo import train`).
- `data/coco.py`'s `prepare_data` gained a `class_names` parameter instead of reaching for
  `yolo.py`'s `CLASS_NAMES` constant directly — keeps `data/coco.py` task-agnostic
  (its own docstring already claims "shared between the detection and segmentation
  tasks"), and avoids yet another cross-module import to manage.
- The local-import pattern in `main()` is a real, if standard, piece of fragility:
  moving that import back to module top level would silently reintroduce the circular
  import (Python raises `ImportError: cannot import name ... from partially initialized
  module` at whichever import happens second) — the comment in `yolo.py`'s module
  docstring and `main()`'s own docstring both point here so it isn't "cleaned up" by
  someone unaware of the constraint.
