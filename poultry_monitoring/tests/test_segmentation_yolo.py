"""Smoke tests for `poultry_monitoring.segmentation.yolo`.

Constitution Principle VIII scope: deterministic, non-ML code only -- the
Ultralytics-result-to-metrics-dict mapping, not training/tuning itself.
"""

from types import SimpleNamespace

import numpy as np

from poultry_monitoring.segmentation.yolo import extract_metrics


def _make_val_metrics(box_vals: dict, seg_vals: dict) -> SimpleNamespace:
    """Build a minimal stand-in for `YOLO(...).val()`'s return value.

    Only the `.box`/`.seg` attribute shape `extract_metrics` actually reads --
    `map50`, `map`, `mp`, `mr` (+ `map75` for `.seg`) -- not a real Ultralytics object.
    """
    return SimpleNamespace(box=SimpleNamespace(**box_vals), seg=SimpleNamespace(**seg_vals))


class TestExtractMetrics:
    def test_maps_box_and_mask_fields_to_project_names(self):
        val_metrics = _make_val_metrics(
            box_vals={"map50": 0.9, "map": 0.8, "mp": 0.95, "mr": 0.85},
            seg_vals={"map50": 0.7, "map": 0.6, "map75": 0.65, "mp": 0.75, "mr": 0.55},
        )

        result = extract_metrics(val_metrics)

        assert result == {
            "box_map50": 0.9,
            "box_map50_95": 0.8,
            "box_precision": 0.95,
            "box_recall": 0.85,
            "mask_map50": 0.7,
            "mask_map50_95": 0.6,
            "mask_map75": 0.65,
            "mask_precision": 0.75,
            "mask_recall": 0.55,
        }

    def test_casts_to_plain_float(self):
        """Ultralytics' own metric fields are numpy scalars, not plain floats.

        MLflow (and `TrainOutcome`) expect plain Python floats, not `np.float64`.
        """
        vals = {
            "map50": np.float64(0.9),
            "map": np.float64(0.8),
            "mp": np.float64(0.95),
            "mr": np.float64(0.85),
        }
        val_metrics = _make_val_metrics(box_vals=vals, seg_vals={**vals, "map75": np.float64(0.65)})

        result = extract_metrics(val_metrics)

        assert all(type(v) is float for v in result.values())
