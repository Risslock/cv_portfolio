# ADR 0006: Progressive-unfreezing stage configuration — forced optimizer, tight close_mosaic/patience

## Status

Accepted

## Context

Two problems surfaced while running the first progressive-unfreezing pass
(`DEFAULT_UNFREEZE_STAGES`, three stages of decreasing `freeze`/`lr0`):

1. **`optimizer="auto"` silently overrides a configured `lr0`.** Observed directly in
   training logs: the tuned `lr0=0.01` got overridden to `AdamW(lr=0.002)` during an
   earlier size-sweep run. Ultralytics' `auto` heuristic picks its own learning rate with
   no notion of "this is a later refinement stage on an already-converged model" — it
   just sees a fresh `model.train()` call.
2. **`close_mosaic=10` was inherited unchanged from the 300-epoch tuned config** and
   reused as-is on 30-epoch unfreezing stages. That disabled mosaic for the last 33% of
   each stage instead of the ~3% it was sized for. Visible in stage 2's per-epoch curves
   as a sharp train-loss drop plus a transient val-loss/mAP50-95 dip right at the
   transition epoch (`epoch == epochs - close_mosaic`) — and likely why that stage's
   early stopping locked in a near-transition checkpoint instead of one that had time to
   re-stabilize under the new (no-mosaic) regime.

## Decision

- Force `optimizer="AdamW"` explicitly on every stage, so a per-stage `lr0` actually
  takes effect instead of being silently discarded.
- Size `close_mosaic`/`patience` per two rules of thumb instead of reusing values tuned
  for a different run length:
  - `epochs >> close_mosaic` — close_mosaic should be a small fraction of *this* stage's
    epochs, not sized for a different (much longer) run.
  - `patience < close_mosaic` — since Ultralytics' `best.pt` always tracks the
    best-ever-observed fitness regardless of when training stops, a patience shorter
    than close_mosaic acts as a bounded trial window: if training without mosaic hurts
    validation, the run stops (keeping the last good, likely pre-transition checkpoint)
    before the rest of the close_mosaic tail is spent on a regime that isn't working.
- Each stage: `close_mosaic=4`, `patience=3` (was `close_mosaic=10`, `patience=10`).

## Consequences

- The already-banked `yolo26n` unfreeze result predates this fix (used the old
  `close_mosaic=10`/`patience=10`) — not re-run, since it already lands within noise of
  ChickenVerse's published baseline. This schedule fix applies to any *future* unfreeze
  pass (e.g. a queued `yolo26s` unfreeze run).
- `DEFAULT_UNFREEZE_STAGES` now carries `close_mosaic` per stage; `progressive_unfreeze_train`
  only overrides it when a stage dict actually specifies one, so custom stage lists
  (e.g. in tests) that omit it still work unchanged.
