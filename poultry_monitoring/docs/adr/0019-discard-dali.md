# ADR 0019: Discard the DALI phase — the training loop is GPU-bound, not input-bound

## Status

Accepted

## Context

Phase 5 (`plan.md`) proposed replacing the standard PyTorch `DataLoader` + Albumentations
input pipeline with NVIDIA DALI, to move JPEG decode and augmentation onto the GPU. It was
never an unconditional commitment: both the constitution's § Technology Stack entry and the
phase's own first checklist item gated it on *"profiling showing the data path — not the
augmentation step — is the actual bottleneck."* That profile was never run, so the phase sat
at 🔲 with the gate unevaluated.

Two things make the gate answerable without building a profiling harness.

**1. Adding substantial per-sample CPU work barely moved wall-clock.** Phase 3 Stage B's
copy-paste augmentation runs entirely on the CPU, per training sample, inside the
`DataLoader` workers (ADR 0017): donor sampling, polygon rasterization, scene-relative
resize, LAB-space color transfer, and alpha compositing of up to 5–10 donors. If the input
pipeline were saturating its 8 workers, that work would land close to 1:1 in epoch time.
Mean seconds/epoch, from each run's own `results.csv` (`time` column deltas):

| Run | batch | ms/img | Δ vs. its baseline |
|---|---|---|---|
| `yolo26n-seg-baseline-adamw` | 16 | 29.1 | — |
| `yolo26n-seg-synth_copy_paste` | 16 | 31.0 | **+1.9 (+6.5%)** |
| `yolo26s-seg-baseline-adamw` | 8 | 40.8 | — |
| `yolo26s-seg-synth_copy_paste` | 8 | 41.7 | **+0.9 (+2.2%)** |

All four ran `workers: 8`, `cache: false`, `imgsz: 640`, `amp: true` — so every epoch
decodes all 5,222 training JPEGs from disk. The compositing is nearly free in wall-clock
terms, which is the signature of workers with idle headroom, not a saturated input path.

**2. Wall-clock tracks model size.** Holding the dataset and pipeline fixed, going from
`yolo26n-seg` to `yolo26s-seg` costs +40% per image (29.1 → 40.8 ms). Time scaling with the
*model* rather than the *data* is what a GPU-bound loop looks like.

Supporting, weaker: Ultralytics' own validation-pass speed lines across these runs report
0.4–1.4 ms/img preprocess against 1.6–5.7 ms/img inference.

## Decision

**Do not build the DALI pipeline.** Phase 5 is closed as ❌ superseded — kept in `plan.md`
rather than renumbered away, the same treatment Phase 4 got. `data/dali_pipeline.py` is
removed from `CLAUDE.md`'s package layout. Effort moves to Phase 6 (export & optimization),
where the measured constraint — GPU compute at inference time — is what actually gets
addressed.

## Consequences

- The evidence above is **observational, not a profile.** It is derived from epoch-time
  deltas across runs that differ in one variable, not from an instrumented measurement of
  data-wait vs. compute time inside the loop. It is strong enough to rule out "the input
  pipeline is the bottleneck" as this project's next optimization target; it is not a
  per-batch attribution and shouldn't be cited as one.
- Two confounds are worth stating rather than hiding. The `n`-vs-`s` comparison co-varies
  model size with batch size (16 vs. 8 — `s` plus the donor-bank trainer OOMs an 8 GB card),
  the same confound `plan.md` Phase 3 Stage C already flags. And epoch time includes the
  per-epoch validation pass, so these are whole-epoch costs, not pure training-step costs.
  Neither affects the direction of the conclusion.
- This does not say DALI is useless in general — it says it's not warranted **here**, on
  this dataset (5.2k images at 640px), this hardware (RTX 2060 SUPER, 6c/12t CPU), and with
  `cache: false`. A larger dataset, a faster GPU, or a many-GPU setup would shift the
  balance and deserve a fresh measurement, not this conclusion carried forward.
- The unresolved packaging question is now moot rather than answered: DALI ships no native
  Windows wheels, which the constitution's Principle IX flagged as unverified. Closing the
  phase on measured grounds means that never had to be resolved.
- Removes the last motivation for the Phase 1 "sanity-check DALI availability" notebook item.
