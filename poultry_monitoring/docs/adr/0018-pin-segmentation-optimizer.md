# ADR 0018: Pin the segmentation optimizer instead of `optimizer="auto"`

## Status

Accepted

## Context

The Phase 3 segmentation track was deliberately kept on stock Ultralytics settings so
that copy-paste augmentation would be the *only* changed variable between Stage A
(real data) and Stage B (copy-paste). Both runs were launched from the same `train()`
with the same seed, batch, imgsz and dataset; the only knob that differed was the
`epochs` cap — 200 for the Stage A baseline, 100 for Stage B, both under `patience=10`
early stopping, on the assumption that an unreached cap is inert.

It is not inert. Ultralytics' `optimizer="auto"` selects on an *iteration count* that
is computed against `nbs` (nominal batch size, 64) rather than the actual `batch`:

```
iterations = ceil(len(train_dataset) / max(batch, nbs)) * epochs
name, lr, momentum = ("MuSGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
```

On ChickenDet (5,219 training images) that is `82 * epochs`, so the branch flips at
**`epochs > 122`** — regardless of how many epochs actually run. The two runs landed on
opposite sides of it:

| | Stage A baseline | Stage B copy-paste |
|---|---|---|
| `epochs` cap | 200 | 100 |
| `iterations` | 16,400 | 8,200 |
| Optimizer chosen | MuSGD, `lr0=0.01` | AdamW, `lr0=0.002` |
| Optimizer param groups | 8 | 3 |
| LR at final epoch | 0.0212 | 0.000535 |

The param-group counts confirm it independently (MuSGD splits muon/sgd groups; AdamW
uses the standard three), as does the LR trace: AdamW at `lr0=0.002` with `lrf=0.01`
predicts 0.000515 at epoch 75 against 0.000535 observed.

So the "only changed variable" ablation in fact changed the optimizer *and* ran the two
arms at learning rates ~20–40× apart. The Stage B result (every box and mask metric up
by +0.1 to +0.9 points) could not be attributed to copy-paste at all.

A second, compounding problem: `auto` **ignores** the run's own `lr0`, `momentum` and
`warmup_bias_lr` and substitutes its own. Those args are still written to the run's
`args.yaml` verbatim, so both runs' saved args claim `lr0: 0.01`, `momentum: 0.937`,
`warmup_bias_lr: 0.1` — none of which describe what actually trained. The saved args are
not a reproducible record of the run while `auto` is in play.

## Decision

Pin the optimizer explicitly in `segmentation/yolo.py` rather than leaving it to `auto`,
as four module constants passed on every `model.train()` call:

```python
DEFAULT_OPTIMIZER = "AdamW"
DEFAULT_LR0 = 0.002            # == auto's lr_fit, round(0.002 * 5 / (4 + nc), 6), at nc=1
DEFAULT_MOMENTUM = 0.9         # auto's value, not args' own 0.937 default
DEFAULT_WARMUP_BIAS_LR = 0.0   # auto zeroes this for Adam-family optimizers
```

The values reproduce exactly what `auto` chose for the 100-epoch Stage B run, so that
run stays valid and only the baseline needs re-running (at `epochs=100`, matching Stage
B's LR *schedule* length as well as its optimizer). All four are overridable through
`train()`'s new `hyperparameters` dict.

### Rejected alternatives

- **Leave `auto`, and just document the `epochs > 122` threshold.** Faithful to
  ChickenVerse's own stock recipe, which is what the segmentation track was trying to
  match. Rejected because the threshold is a function of dataset size and `nbs`, not a
  fixed number — it silently moves if the train split, `batch`, or `nbs` ever changes,
  so a comment recording "122" would itself go stale and re-introduce the same class of
  bug at a different boundary.
- **Match the arms by setting both `epochs` caps to 200.** Would have put both on MuSGD
  and been a valid comparison too, but costs a full re-run of *both* arms rather than
  one, and leaves the selection still implicit.
- **Re-run the baseline at `epochs=200` with the optimizer pinned.** Keeps the original
  budget, but `lrf` decays over the `epochs` cap, so a 200-epoch arm and a 100-epoch arm
  still follow different LR schedules even on an identical optimizer. Matching the cap
  is what makes the schedules identical.

## Consequences

- The segmentation track no longer matches ChickenVerse's recipe *byte for byte* — it
  now specifies an optimizer where they left it automatic. This is a deliberate
  divergence in favour of the ablation being real; the published-baseline comparison
  rows in the README remain a cross-recipe comparison either way.
- Any segmentation result produced before this ADR is suspect if its `epochs` cap
  crossed 122. The Stage A baseline (`yolo26n-seg-baseline`, cap 200) is affected and is
  being re-run; the `yolo26s-seg` baselines (cap 200) are affected too and their
  comparison against a future `s` copy-paste arm must not reuse them.
- `args.yaml` becomes a truthful record again: with the optimizer pinned, the values
  written are the values used.
- Watch for the same class of bug wherever an argument is passed as a *cap* or *budget*
  and assumed inert — `auto`-style heuristics elsewhere in Ultralytics (auto-batch, AMP
  selection) key off similar derived quantities.
