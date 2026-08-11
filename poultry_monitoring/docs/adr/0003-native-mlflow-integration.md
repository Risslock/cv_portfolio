# ADR 0003: Native Ultralytics MLflow integration, not hand-rolled logging

## Status

Accepted

## Context

`mlflow_utils.py` was first written by hand: `mlflow.start_run()`, then explicit
`mlflow.log_params`/`log_metric` calls for every training arg and per-epoch metric,
mirroring what `CLAUDE.md` § MLflow Conventions originally specified — including
starting each run unnamed and renaming it to `f"{model}-{variant}-{mlflow_auto_name}"`
once MLflow assigned its own readable adjective-animal name (e.g. `capable-shrike-728`).

Ultralytics already ships this: `ultralytics/utils/callbacks/mlflow.py` auto-logs every
`model.train()`/`model.tune()` call's params, per-epoch metrics, and artifacts once the
`mlflow` setting is on. Told to use that instead of re-implementing it.

Two things the native integration doesn't do the way the original hand-rolled version
assumed:

- It logs metrics under its own key names (e.g. `metrics/mAP50(B)`, parens stripped by
  its own `sanitize_dict`), not this project's `box_map50`/`box_map50_95` convention.
- `mlflow.start_run(run_name=...)` is called with `trainer.args.name`, which is
  basically always a real string (Ultralytics resolves a default like `"train"` even
  when the caller doesn't pass one) — so a genuinely unset `run_name` (the condition
  needed for MLflow to auto-generate its own adjective-animal name) isn't reliably
  reachable through the public `model.train()` kwargs.

## Decision

- Use the native integration for what it already does well (param/metric/artifact
  logging during training) via `configure_ultralytics_mlflow` (sets the env vars/
  setting it reads).
- `make_run_name` renames the run to `f"{model_family}-{variant}-{run_id[:8]}"` —
  `run_id[:8]` instead of MLflow's own auto-name, since that's reliably available
  (every run has a real UUID) without depending on `run_name` staying unset.
- `finish_run(extra_metrics=...)` supplies the canonically-named final metrics
  (`box_map50` etc.) that the native integration's own key names don't provide,
  typically from an explicit re-validation of the best checkpoint (see
  `detection/yolo.py`'s `train`).

## Consequences

- `CLAUDE.md` § MLflow Conventions' run-naming description (mlflow-auto-name suffix) is
  stale against this — `run_id[:8]` is what's actually implemented. Worth reconciling
  the doc text next time that section is touched.
- Per-epoch metrics in MLflow use Ultralytics' key names, not this project's convention
  — only the *final* numbers get the canonical names, via `finish_run`. A dashboard/query
  comparing per-epoch curves across runs needs to know this.
