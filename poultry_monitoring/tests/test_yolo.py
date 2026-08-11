"""Smoke tests for `poultry_monitoring.detection.yolo`.

Constitution Principle VIII scope: deterministic, non-ML code only — the
hyperparameters-dict/augmentation-kwargs split, not training/tuning itself.
"""

from poultry_monitoring.augmentation.shared import PARAM_RANGES
from poultry_monitoring.detection.yolo import _augmentation_kwargs_from


class TestAugmentationKwargsFrom:
    def test_empty_dict_returns_empty(self):
        assert _augmentation_kwargs_from({}) == {}

    def test_pulls_only_param_ranges_keys(self):
        hyperparameters = {
            "lr0": 0.01,
            "p_color_invariance": 0.15,
            "p_lighting": 0.35,
            "brightness_limit": 0.2,
        }

        result = _augmentation_kwargs_from(hyperparameters)

        assert result == {
            "p_color_invariance": 0.15,
            "p_lighting": 0.35,
            "brightness_limit": 0.2,
        }
        assert "lr0" not in result

    def test_covers_every_param_ranges_key(self):
        hyperparameters = dict.fromkeys(PARAM_RANGES, 0.1)

        result = _augmentation_kwargs_from(hyperparameters)

        assert set(result) == set(PARAM_RANGES)
