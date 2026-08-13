# ADR 0009: Revert to in-process Optuna trials, accepting the RAM-leak risk from ADR 0007

## Status

Accepted

## Context

`tune_hyperparameters` (ADR 0007) ran each trial as its own subprocess — `subprocess.run([sys.executable,
"-m", ..., "train", ...])`, sampled hyperparameters passed as a `--hyperparameters-json`
blob, fitness read back by regex-matching `box_map50_95=...` out of the child's captured
stdout. That fixed a real bug (orphaned DataLoader worker processes leaking system RAM
across in-process trials), but at the cost of real complexity in this module's most
central function: process spawning, stdout parsing, and an extra CLI round-trip standing
between the objective function and the training call it's actually scoring.

Asked to clean up `detection/yolo.py` as this project's main script, explicitly ruling out
`sys.executable`/`subprocess.run` and asking for a direct Optuna objective instead —
accepting the reintroduced RAM-leak risk rather than mitigating it (`workers=0` and a
`multiprocessing.Process`-per-trial middle ground were both considered and declined).

## Decision

`tune_hyperparameters`'s objective now calls `train()` directly, in-process, once per
trial — no subprocess, no stdout parsing, no `--hyperparameters-json` round-trip. No
mitigation for the ADR 0007 leak was added: no forced `workers=0`, no per-trial
`gc.collect()`/`torch.cuda.empty_cache()`, no `try`/`except` around the `train()` call.

## Consequences

- Re-exposes the exact failure mode ADR 0007 fixed: DataLoader worker processes
  (Windows `spawn`) accumulate system RAM across repeated in-process training calls. A
  long enough run can start failing partway through, same as the 40-trial run that
  motivated ADR 0007 (trials 0-2 fine, every trial from 3 onward failing in ~1.3s).
- No `try`/`except` in the objective means a trial's exception now stops
  `study.optimize()` outright, rather than recording `0.0` and continuing. Already-
  completed trials stay in the SQLite study storage regardless (Optuna persists
  per-trial), so a crash costs only the rest of that run's iteration budget, not the
  study's history — restarting `tune` picks up a fresh `n_trials`, not a resume of the
  exact interrupted count.
- No cleanup of a possibly-stuck MLflow run if `train()` dies mid-training — check the
  MLflow UI for a run left in `RUNNING` state after a crashed tuning session.
- `AUGMENTATION_FOCUSED_SPACE` is much smaller now (ADR 0008, 7 params vs. the original
  18), so fewer trials are likely needed per run — this may reduce how often the leak
  actually surfaces in practice, but the trigger itself (repeated `model.train()` calls
  in one process) is unchanged, so it isn't fixed, just less exercised.

**Addendum 1**: exception handling was added around the `train()` call after all, in two
layers — `except (MemoryError, torch.OutOfMemoryError)` prints a memory-specific message,
`except Exception` catches anything else with a generic one. Both branches print the real
exception, call `trial.study.stop()`, and return `0.0` for that trial rather than letting
it raise — once a trial fails for *any* reason, all subsequent trials in the same process
are assumed likely to fail the same way, so the study stops instead of burning through the
rest of its budget on `0.0`s. This is closer to ADR 0007's original crash-resilience than
the first version of this ADR assumed, just without the GPU-memory cleanup ADR 0007 also
did (`del model`/`gc.collect()`/`torch.cuda.empty_cache()`) — not reinstated here.

**Addendum 2 — real-scale result**: a 16-trial run (15 epochs, `fraction=0.3`, the current
7-key `AUGMENTATION_FOCUSED_SPACE`) completed entirely in-process with zero crashes —
best trial (`#2`) scored `box_map50_95=0.8166`, every trial landed in a plausible
`0.73`–`0.82` range (see `plan.md` for the full breakdown). No RAM/process monitoring was
done during the run, so this isn't a re-diagnosis of ADR 0007's root cause — but it's real
evidence the specific failure mode doesn't reliably reproduce under the current code and
this trial budget/space size. Notably, the *same* `optuna.db`/study also held one leftover
`FAIL`ed trial from an earlier, unrelated attempt (identifiable by its params: the old,
pre-ADR-0008 18-key search space) that ran ~14 minutes before failing — nothing like ADR
0007's fast-cascading-from-trial-3 signature — and it sat there harmlessly across every
later run without affecting `study.best_params`, exactly as the SQLite-persistence design
intended.
