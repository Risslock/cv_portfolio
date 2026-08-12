# ADR 0007: Run each Optuna trial as its own subprocess

## Status

Accepted

## Context

The Optuna-based `tune_hyperparameters` (ADR 0005) ran all 40 trials in-process — one
long-lived Python process, a fresh `YOLO()` + `model.train()` call per trial. The first
real-scale run appeared to succeed (`study.optimize()` didn't crash, "BEST PARAMS"
printed, "ALL 40 TRIALS COMPLETE") but was actually silent garbage: trials 0-2 took their
normal ~10-13 minutes each, then every trial from 3 onward failed in ~1.3 seconds,
caught by a broad `except Exception: return 0.0` with no logging.

Root-caused in stages, each an assumption tested and found wrong rather than guessed at:

1. Added `traceback.print_exc()` to the except branch and a real repro — the actual
   error was a `MemoryError` (system RAM, not GPU VRAM) inside a DataLoader worker
   process, allocating a large segment-coordinate array.
2. First fix attempt: `del model; gc.collect(); torch.cuda.empty_cache()` after every
   trial. Wrong target — VRAM was never the constrained resource; live system-monitoring
   during a real run showed GPU usage fine, system RAM nearly exhausted (Windows
   "Available" memory dropped to ~860 MB during trial 2, having started around 6+ GB
   free).
3. Tried `workers=0` (no DataLoader worker processes at all) — genuinely stopped the
   RAM growth, but made each trial ~5-6x slower (single-process data loading, GPU
   starved). Impractical for a 40-trial budget.
4. Tried `workers=4` as a middle ground — RAM still climbed (total across all
   `python.exe` processes: ~3.7 GB after trial 0, ~7 GB after trial 1), just slower
   than with the default worker count.
5. Isolated the mechanism with a bare repro: repeated `model.train()` calls with no
   MLflow, no Optuna, no custom augmentations, tracking the *main* process's own RSS via
   `psutil`. Confirmed a ~2 GB one-time jump on trial 0 (CUDA/cuDNN/PyTorch allocator
   initialization — normal, not a leak) and only ~59 MB growth on trial 1 in the main
   process itself — ruling out the main process as the growth source.
6. Cross-checked with `Get-Process`/`Get-CimInstance Win32_Process`: a cluster of ~20
   child processes, all with `ParentProcessId` equal to the training process, spawned
   sequentially (~3 seconds apart) during a single trial, each 160-460 MB and growing —
   consistent with DataLoader worker processes re-importing torch/numpy/cv2/albumentations
   on Windows' `spawn`-based multiprocessing (heavier than Linux `fork`) each time
   Ultralytics recreates a DataLoader (training vs. periodic validation vs. the final
   re-validation `train()` does), without the previous pool being fully torn down first.

None of the levers available from the parent process — `del`, `gc.collect()`,
`torch.cuda.empty_cache()`, reducing `workers` — reach child *processes*; the OS only
reliably reclaims them when the process that spawned them exits. The original
Ultralytics `model.tune()` (ADR 0005's subject) never hit this because it isolates every
trial in a fresh subprocess by design — a benefit of that architecture not credited when
choosing in-process Optuna to fix the exploration problem.

## Decision

Each Optuna trial now runs as its own subprocess: `subprocess.run([sys.executable, "-m",
"poultry_monitoring.detection.yolo", "train", ...])`, passing the trial's sampled
hyperparameters via the existing `--hyperparameters-json` flag and reading the fitness
back by regex-matching `box_map50_95=...` out of the child's captured stdout (the
`TrainOutcome` repr `train`'s CLI already prints).

This doesn't reintroduce ADR 0001's blocker (`model.tune()` couldn't carry a live
Albumentations transform *object* across its own subprocess boundary) because this
subprocess call is ours to define — only the sampled hyperparameter *values* (plain
floats/ints) cross the boundary as JSON, and the child process rebuilds the actual
transform objects itself via `_build_custom_augmentations`, exactly like every other
entry point in this module already does.

Added `--imgsz` to the `train` CLI (previously missing) so `tune_hyperparameters`'s
`imgsz` parameter still has somewhere to go.

## Consequences

- The in-process GPU/exception-handling code (`del model`, `gc.collect()`,
  `torch.cuda.empty_cache()`, the broad `except Exception`, manual `mlflow.end_run()`
  cleanup) is gone — none of it applies once training happens in a child process.
  `train()`'s own MLflow handling (already correct) runs per-subprocess, so the earlier
  "next trial reuses the still-open run" bug (also fixed in-process, now moot) can't
  recur either — each trial is a fully separate process with its own MLflow run.
- Each trial pays fresh Python/CUDA/cuDNN startup cost (a few seconds), on top of the
  ~9-10 min training time — real but small relative to trial length.
- `tune_hyperparameters`'s `project` parameter is no longer honored — the subprocess
  always uses the `train` CLI's own `data_dir / "YOLO"` default. Not a behavior change
  in practice (every call site in this project already passes exactly that value), but a
  real API gap if a future caller passed something else.
- Regex-parsing stdout for the fitness value is more brittle than a structured return
  value would be; acceptable since `train`'s printed `TrainOutcome` format is a small,
  already-stable surface, but worth revisiting if that format ever changes.
