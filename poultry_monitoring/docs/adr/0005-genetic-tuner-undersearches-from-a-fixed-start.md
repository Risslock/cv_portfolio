# ADR 0005: Ultralytics' genetic tuner doesn't meaningfully explore parameters starting from a fixed value

## Status

Accepted

## Context

A `multi_scale` (0.0–0.3) follow-up tune (`tune-2`, 25 iterations) was watched live via the
MLflow UI. Every observed value was on the order of `1e-4`–`1e-5` — nowhere near spanning
the given range — after 7+ iterations. Traced the cause directly in
`ultralytics/engine/tuner.py`'s `Tuner._mutate`/`_crossover`:

- **`_mutate`**: `new_value = parent_gene * exp(gaussian_noise)`, factor clipped to
  `[0.25, 4.0]` — purely **multiplicative**. A gene that is exactly `0.0` can never move
  (`0 * anything = 0`).
- **`_crossover`**: computes `span = max(parents) - min(parents)` per parameter across the
  top-k population. When every parent so far shares the same value (`span == 0` — true for
  *any* parameter on the very first mutation, since iteration 1 has only one data point to
  build a population from, not just parameters stuck at zero), it falls back to:
  ```python
  span = np.where(span == 0, np.random.uniform(0.01, 0.1, span.shape), span)
  return np.random.uniform(lo - alpha * span, hi + alpha * span)  # alpha = 0.2
  ```
  With `lo = hi = 0`, the next candidate is drawn from roughly `±0.002` to `±0.02` before
  clipping to bounds — exactly the `1e-4`–`1e-5` values observed. The mechanism only ever
  nudges a *small* amount around the current cluster; it has no way to "jump" to a
  meaningfully different region of a wide range like `(0.0, 0.3)`.

**Broader implication, not just `multi_scale`**: iteration 1 of any `model.tune()` run
always uses pure, unmutated defaults (no history to mutate from yet). By iteration 2, the
population still has only one data point, so *every* parameter — not only the ones that
happened to default to `0.0` (`degrees`, `shear`, `perspective`, `flipud`, `bgr`, `mixup`,
`cutmix`, `copy_paste`, and now `multi_scale`) — hits the same degenerate-span fallback and
only ever receives small nudges. This project's original 20-iteration `AUGMENTATION_FOCUSED_SPACE`
search had iteration 1 (unmutated defaults) win and never get beaten across all 20
iterations — consistent with the tuner under-exploring the space in that budget, not
necessarily with defaults being genuinely optimal for this dataset.

## Decision

Stopped the `multi_scale` tune run (7/25 iterations, GPU processes killed directly —
`bash` termination didn't cascade to the detached Python subprocess on Windows). Not
resuming it or trusting its partial result.

Going forward, built-in Ultralytics augmentation hyperparameters that need genuine,
range-spanning exploration should use this project's own random search
(`tune_augmentation_parameters`, ADR 0001/0002's pattern — independent `Uniform(min, max)`
draw per trial, no dependence on a starting population) instead of `model.tune()`'s genetic
algorithm, which is better suited to *local refinement* around an already-reasonable
starting point than to global, from-scratch exploration in a small iteration budget.

## Consequences

- The original `AUGMENTATION_FOCUSED_SPACE` tune result (still the base config every
  training run in this project uses) cannot be credited as a genuine search outcome for the
  parameters that landed at or near their defaults — it's closer to "Ultralytics' defaults,
  lightly perturbed" for those. The actual trained models' numbers are still real and still
  competitive (see README § Results) — this only undercuts the "found via hyperparameter
  search" claim for specific parameters, not the results themselves.
- Any future built-in-augmentation search should either (a) move to this project's own
  random-search function, or (b) if `model.tune()` is kept, pre-seed its population with
  several genuinely diverse random draws before letting mutation take over — a plain call
  with a fresh model provides neither today.
- Re-running the full search this way is a real GPU-time commitment, not a quick fix —
  flagged in `plan.md`, not launched automatically.
