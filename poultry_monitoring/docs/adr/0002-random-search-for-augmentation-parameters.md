# ADR 0002: Random search, not grid, for custom augmentation parameters

## Status

Accepted

## Context

`tune_augmentation_parameters` (see ADR 0001 for why it exists as a separate search at
all) started as a grid search over 2 parameters (`p_color_invariance`, `p_lighting`):
`len(choices_a) * len(choices_b)` trials, each a fixed list of candidate values.

Mid-session, the search grew to 5 parameters (`brightness_limit`, `contrast_limit`,
`autocontrast_cutoff` added alongside the original two — see `augmentation/shared.py`).
A grid over 5 dimensions, even with just 3 candidate values each, is 3^5 = 243 trials —
each one a real training run. Not viable at any reasonable epoch/fraction budget.

## Decision

Switched to random sampling: draw `n_trials` random points from
`augmentation.shared.PARAM_RANGES` (uniform, seeded for reproducibility), train each,
keep the best by validation `box_map50_95`. Trial count is a direct parameter
(`n_trials`) independent of how many dimensions `PARAM_RANGES` has.

## Consequences

- Coverage of the parameter space per trial is worse than a fine grid at the same trial
  count — acceptable given the budget constraint, not free.
- Adding a 6th parameter to `PARAM_RANGES` costs nothing extra in trial count (unlike
  grid, which would need `n_trials` re-derived from a growing product). No code change
  needed in `tune_augmentation_parameters` itself when `PARAM_RANGES` grows.
- No adaptive component (not Bayesian, not a genetic search) — if the search needs to
  get smarter later (more trials than budget allows justifies it), that's a bigger
  change, not a tweak to this function.
