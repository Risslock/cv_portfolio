# Architecture Decision Records

Non-obvious design decisions for `poultry_monitoring`, especially ones arrived at by
testing an assumption and finding it wrong. Code should point here with a short comment
instead of inlining the full rationale — see `docs/adr/template.md` for the format.

| ADR | Decision |
|---|---|
| [0001](0001-custom-augmentation-search-separate-from-tuner.md) | Custom Albumentations search runs separately from `model.tune()` |
| [0002](0002-random-search-for-augmentation-parameters.md) | Random search, not grid, for custom augmentation parameters |
| [0003](0003-native-mlflow-integration.md) | Native Ultralytics MLflow integration, not hand-rolled logging |
| [0004](0004-no-test-time-preprocessing.md) | No deterministic preprocessing (autocontrast/CLAHE/hist-eq) at inference time — tested, made it worse |
| [0005](0005-genetic-tuner-undersearches-from-a-fixed-start.md) | Ultralytics' genetic tuner doesn't meaningfully explore from a fixed start — move built-in augmentation search to this project's own random search |
| [0006](0006-progressive-unfreeze-stage-config.md) | Progressive-unfreeze stages need a forced optimizer and a `close_mosaic`/`patience` sized for the stage, not inherited from a longer run |
| [0007](0007-subprocess-per-optuna-trial.md) | Each Optuna trial runs as its own subprocess — orphaned DataLoader worker processes (Windows `spawn`) leak system RAM across in-process trials, and only the OS reclaiming a dead process actually frees them |
| [0008](0008-conservative-hyperparameter-space.md) | Hand-curated, conservative search space around Ultralytics' shipped defaults, replacing bounds nearly copied verbatim from `Tuner.space` |
| [0009](0009-revert-to-in-process-optuna-trials.md) | Reverted ADR 0007's subprocess-per-trial mechanism for a plain in-process Optuna objective, deliberately accepting the RAM-leak risk it had fixed |
| [0010](0010-split-yolo-py-by-responsibility.md) | Split `detection/yolo.py` into core/`tuning.py`/`preprocessing_eval.py` by responsibility, keeping the CLI unified via a local (not top-level) import to avoid a circular dependency |
| [0011](0011-conservative-search-result-not-adopted.md) | ADR 0008's 16-trial search winner underperforms baseline at real scale in both cold-start and warm-start regimes — not adopted |
| [0012](0012-directional-brightness-contrast-augmentation.md) | Directional (asymmetric) `brightness_limit`/`contrast_limit` as a training-time augmentation — new best `yolo26n` result, adopted |
| [0013](0013-rle-to-polygon-preprocessing-for-yolo-seg-conversion.md) | Preprocess RLE segmentations to polygons before `convert_coco`, cache persistently — `convert_coco` silently bbox-fills RLE-only masks otherwise |
| [0014](0014-copy-paste-donor-bank-design.md) | Curated, disk-cached (PNG pair) donor bank for synthetic copy-paste, with scene-relative Normal(mean, std) resizing |
| [0015](0015-color-aware-donor-compositing.md) | LAB-space statistical color transfer (Reinhard-style) to match a pasted donor's color to the target scene's own birds |
| [0016](0016-fingerprinted-label-cache-invalidation.md) | Fingerprint (name/size/mtime + format version) the polygon annotation cache and `labels/`, not existence-only — an existence check silently served a stale `labels/` past a conversion-logic fix |
