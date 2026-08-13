# ADR 0011: The ADR 0008 search's winning hyperparameters underperform baseline — not adopted

## Status

Accepted

## Context

ADR 0008 defined a conservative, hand-curated search space (7 keys: `perspective`,
`shear`, `flipud`, `mixup`, `cutmix`, `copy_paste`, `multi_scale`, plus the custom
Albumentations params) to replace the earlier wide brute-force space. A real 16-trial
run (ADR 0009 Addendum 2) completed cleanly and picked a winner — trial 2,
`box_map50_95=0.8166` on the 15-epoch/`fraction=0.3` search proxy. That result was never
validated at real scale until now.

Two real-scale validations, both against the pre-session baseline (near-Ultralytics-default
augmentation, per ADR 0005's "iteration 1 never beaten" finding):

1. **Warm-start** (`progressive_unfreeze_train`, continuing from the already-converged
   `yolo26n-tuned` checkpoint), schedule matched exactly to the baseline
   (`epochs=30/patience=3/close_mosaic=4` per stage, confirmed via `git diff` against the
   committed `DEFAULT_UNFREEZE_STAGES` after an earlier confound was found and reverted):
   final `box_map50_95=0.8750` vs. baseline `0.8919` — **Δ −0.0169**.
2. **Cold-start** (`train()` from stock `yolo26n.pt`, the exact regime `tune_hyperparameters`
   itself searches in), config matched to the baseline in every respect that matters
   (`epochs=300`, `patience=15`, `fraction=1.0`, `freeze=10`, `lr0=0.01`) except
   `close_mosaic` (baseline manually set `10`; this run used the current codebase's
   `default_close_mosaic(300)=45` — more mosaic-free tail than the baseline had, which if
   anything should help convergence, not hurt it): final `box_map50_95=0.8482` vs.
   baseline `0.8768` — **Δ −0.0286**, the larger of the two regressions.

The cold-start run's per-epoch curve rules out the mechanism from the earlier warm-start
investigation (rapid rise then destabilize/oscillate): it converged completely normally —
`0.70 → 0.81 → 0.83 → 0.84` over ~45 epochs, patience-stopped cleanly at epoch 62. It just
converged to a real ceiling ~3 points below the baseline's. Two different regimes, two
different failure shapes (destabilization vs. a simply-worse optimum), same direction and
similar magnitude — this is the search's winning hyperparameters underperforming, not an
artifact of how they were applied.

## Decision

Don't adopt the ADR 0008 search's winning hyperparameters (`best_hyperparameters.json`)
anywhere — not for cold-start `train()`, not for `progressive_unfreeze_train`. The
production baseline stays the near-Ultralytics-default augmentation config from before
this session's tuning work.

## Consequences

- This is a real negative result for *this specific 16-trial run's winner*, not proof the
  conservative-space approach (ADR 0008) is wrong in principle — 16 trials is a small
  budget for a 7-dimensional space, and nothing here re-runs the search with a larger
  budget to see if a better point exists in the same space.
- ADR 0008's own rationale for narrowing the space (avoiding a wide brute-force search
  that mostly re-explores territory Ultralytics already tuned) is unaffected by this
  result — the space design and the specific trial that won it are separate questions.
- The deferred `lr0`-only follow-up search (ADR 0008 Consequences) is now lower priority:
  searching `lr0` alone against a fixed augmentation config only makes sense once that
  fixed config is known to be good, which this one isn't.
- `best_hyperparameters.json` on disk is a validated-bad config — kept for the record
  (and because the file-based `--hyperparameters-file` plumbing built for it is still
  useful infrastructure) but should not be reused as "the tuned config" default anywhere.
- Re-running the search with a bigger iteration budget, or reconsidering the space itself,
  is a real option — not pursued automatically here, since it's another real GPU-time
  commitment with no guarantee of a better outcome.
