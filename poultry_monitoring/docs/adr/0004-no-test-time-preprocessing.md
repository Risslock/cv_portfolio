# ADR 0004: No deterministic preprocessing at inference/validation time

## Status

Accepted

## Context

Notebook 02's own exploration flagged a "washed-out white chicken" contrast problem and
noted autocontrast looked like it might help — that observation became `augmentation/
shared.py`'s trained-time `_RandomCutoffAutoContrast`, but left open a separate question:
would applying a deterministic enhancement **at inference/validation time only** (no
retraining, no augmentation randomness) on top of an already-trained checkpoint help
further? Three candidates were tested via a new `evaluate_test_time_preprocessing`
function (`detection/yolo.py`'s `ttp` CLI): per-channel percentile autocontrast, CLAHE
(on LAB's L channel), and global histogram equalization (on YCrCb's Y channel) — each
applied unconditionally to every validation image, compared against the unmodified
baseline, same checkpoint (`yolo26n`'s final progressively-unfrozen weights) throughout.

Results (val box mAP50-95, baseline 0.8919):

| Variant | mAP50-95 | Δ |
|---|---|---|
| autocontrast | 0.8912 | −0.0007 |
| CLAHE | 0.8792 | **−0.0128** |
| histogram equalization | 0.8782 | **−0.0138** |

Every variant was flat-to-negative. `autocontrast` was a wash (every metric within
±0.001–0.002 of baseline) — plausible, since the model already saw similarly mild
autocontrast during training via `_RandomCutoffAutoContrast`, so it's not a meaningfully
new input distribution. `CLAHE` and `hist_eq` are both much stronger, fully deterministic
global transforms than anything the model trained under; applying them only at inference
creates a real train/inference distribution mismatch instead of correcting one, and both
show a clear, non-noise-level drop concentrated on the harder mAP50-95 metric.

## Decision

Do not apply any deterministic image preprocessing at inference/validation time in this
project's serving or evaluation path. The trained-time randomized autocontrast
(`_RandomCutoffAutoContrast`, ADR-adjacent fix logged in `plan.md` Phase 2) is where
contrast/lighting robustness is handled — baking a *matching* enhancement into training
augmentation, not bolting a mismatched one onto inference, is what actually helped.

## Consequences

- `evaluate_test_time_preprocessing`/`ttp` CLI stays in the codebase as a reusable
  comparison harness (useful if this gets re-tested against `yolo26s` or a future
  segmentation model) but isn't part of any prediction/export path.
- If a genuinely different domain shift shows up later (e.g. a new facility with much
  worse native lighting than anything in ChickenDet), this result doesn't rule out
  test-time enhancement in general — it rules out these three specific transforms against
  *this* trained distribution. Re-test rather than assume the conclusion generalizes.
- Test split was not used for this comparison, consistent with `plan.md`'s "test split is
  off-limits for any decision-making" rule — the numbers above are all validation-set.
