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
