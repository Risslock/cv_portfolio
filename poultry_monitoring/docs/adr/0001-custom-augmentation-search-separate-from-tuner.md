# ADR 0001: Custom Albumentations search runs separately from `model.tune()`

## Status

Accepted

## Context

`detection/yolo.py` needs to search two different kinds of augmentation parameters:
Ultralytics' own built-in augmentation hyperparameters (`mosaic`, `hsv_*`, `copy_paste`,
...) and this project's custom Albumentations parameters (`p_color_invariance`,
`p_lighting`, `brightness_limit`, `contrast_limit`, `autocontrast_cutoff` — see
`augmentation/shared.py`). The obvious approach was one search covering both, via
Ultralytics' native `model.tune()`.

Two things tried first, both confirmed broken by direct testing rather than assumed:

- Passing `model.tune(augmentations=[...])` directly: fails with `'str' object has no
  attribute 'available_keys'`. Root cause, found by reading `Tuner.__call__`'s source:
  every tuning trial runs as a **separate subprocess**
  (`python -m ultralytics.cfg.__init__ train key=value ...`), so `train_args` gets
  stringified into a CLI command. A live Python object (the transform list) cannot
  survive that round-trip.
- Registering an `on_pretrain_routine_start` callback to set `trainer.args.augmentations`
  as a live attribute (works for a plain `model.train()` call — confirmed): still fails
  under `tune()`, because the callback is registered on the parent-process `YOLO`
  instance and never reaches the child subprocess's trainer at all. Confirmed by writing
  a marker file inside the callback and observing it never gets created during a
  `tune()` run.
- A related dead end: adding custom keys (`p_color_invariance`, etc.) to `model.tune()`'s
  `space=` dict crashes immediately with `'p_color_invariance' is not a valid YOLO
  argument` — `Tuner` runs every trial's args through the same `get_cfg()` validation as
  a normal CLI invocation, which only recognizes Ultralytics' own argument names.

## Decision

Two independent search functions in `detection/yolo.py`:

- `tune_hyperparameters` — Ultralytics' native `model.tune()`, `AUGMENTATION_FOCUSED_SPACE`
  (lr0 + built-in augmentation hyperparameters only, no custom keys).
- `tune_augmentation_parameters` — this project's own random search, run **in-process**
  via repeated `train()` calls (no subprocess involved, so the plain `augmentations=`
  kwarg genuinely works here), over `augmentation.shared.PARAM_RANGES`.

`run_size_sweep` runs the first, then the second under the first's winning
hyperparameters, then merges both into the final per-size training calls.

## Consequences

- Two search loops instead of one, with their own budgets (`tune_iterations` vs.
  `aug_trials`) to reason about separately.
- Random search (not grid) for the custom parameters specifically because grid search
  doesn't scale as more parameters get added (5 dimensions already, per the
  `autocontrast_cutoff`/`brightness_limit`/`contrast_limit` additions) — random search's
  trial count stays fixed regardless of dimensionality.
- If Ultralytics ever changes `Tuner` to run in-process instead of via subprocess, this
  split becomes unnecessary — worth re-checking before adding a third search dimension
  that would benefit from a genetic/joint search instead of two separate ones.
