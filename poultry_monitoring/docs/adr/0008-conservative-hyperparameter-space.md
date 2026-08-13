# ADR 0008: Conservative, hand-curated search space instead of Ultralytics' wide default bounds

## Status

Accepted

## Context

`AUGMENTATION_FOCUSED_SPACE` (searched by `tune_hyperparameters`, ADR 0005) turned out to
be nearly a verbatim copy of Ultralytics' own `Tuner.space` (`ultralytics/engine/tuner.py`)
— e.g. `degrees: (0.0, 45.0)`, `shear: (0.0, 10.0)`, `mixup/cutmix/copy_paste: (0.0, 1.0)`
for every built-in param. Those bounds are sized for tuning a model from scratch across an
unknown dataset; this project fine-tunes a pretrained checkpoint with most of the backbone
frozen (`FREEZE_LAYERS`), where a wide search mostly re-explores territory Ultralytics
already tuned on COCO.

`plan.md`'s own tuning history backs this: the original 20-iteration search's winning
config was iteration 1 — Ultralytics' own unmutated defaults — never beaten across the
whole budget. ADR 0005 traced part of that to the genetic tuner's mutation mechanism, but
even after switching to Optuna's `TPESampler` (genuine independent draws, not the tuner's
bug), re-running the same wide space at real scale would spend real GPU-time most likely
re-confirming that near-default values already work, rather than finding real improvement.

## Decision

Replaced the mechanical "copy Ultralytics' wide bounds" approach with a hand-curated space,
one bucket per parameter based on this project's own dataset and deployment context (dense,
high-occlusion, near-nadir overhead imagery; production cameras that will vary in mounting
angle):

- **Fixed at Ultralytics' native default, removed from the space entirely**: `bgr`,
  `degrees`, `hsv_h`, `hsv_s`, `hsv_v`, `translate`, `scale`, `fliplr`, `mosaic`. Because
  each value equals what `model.train()` already defaults to unprompted, omitting them
  from `AUGMENTATION_FOCUSED_SPACE` is sufficient — no need to pass them explicitly.
- **Fixed for a different reason — coordinate search, not "no rationale to move it"**:
  `lr0`. Rather than searching it jointly with everything else, it's held fixed at
  Ultralytics' default for this pass so the augmentation-side search finds the best other
  hyperparameters against a stable learning rate. A separate follow-up pass, searching
  `lr0` alone with these results fixed, is planned once this round has real results.
- **Fixed, but computed per-run rather than a literal constant**: `close_mosaic`. Its
  Ultralytics default (10) is sized for ~100+ epoch runs; this project trains everything
  from 15-epoch tuning trials to 300-epoch final runs. `default_close_mosaic(epochs)`
  computes a proportional value instead (`0` below 20 epochs, else 15% of `epochs`,
  rounded), applied in `train()` via `setdefault` whenever a caller doesn't already supply
  one.
- **Still searched, narrowed**: `perspective` (`0.0`–`0.0005`) and `shear` (`0.0`–`5.0`)
  kept searched rather than fixed, specifically because production camera installs will
  vary in angle unlike this dataset's own near-nadir rig; `flipud`/`mixup`/`cutmix`/
  `copy_paste` kept at their full original `(0.0, 1.0)` range — exactly the
  density/occlusion/orientation axes this project's dataset stresses; `multi_scale`
  halved to `(0.0, 0.25)`.

## Consequences

- `AUGMENTATION_FOCUSED_SPACE` shrinks from 18 keys to 7 — a re-run at real scale is a much
  smaller GPU-time commitment than the original wide space would have been.
- The historic "iteration 1 never beaten" result isn't directly comparable to a run over
  this new space — different shape entirely. A fresh run is the only way to validate it.
- `train()`'s `close_mosaic` behavior changed for every caller that doesn't explicitly pass
  one, not just tuning trials — including plain `train` CLI calls, which previously fell
  through to Ultralytics' flat default of 10 regardless of `epochs`.
- A real `lr0`-only follow-up search doesn't exist yet — `tune_hyperparameters`'s `space`
  parameter can already take a `{"lr0": (lo, hi)}`-only dict for that pass once this
  round's augmentation results are in, but nothing wires that up automatically yet.
