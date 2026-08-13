"""Smoke tests for `poultry_monitoring.detection.yolo`.

Constitution Principle VIII scope: deterministic, non-ML code only — the
hyperparameters-dict/augmentation-kwargs split and transform-list assembly, not
training/tuning itself. `progressive_unfreeze_train`'s tests live in `test_tuning.py`,
mirroring the `detection/tuning.py` split.
"""

from poultry_monitoring.detection.yolo import (
    CUSTOM_AUGMENTATION_PARAM_RANGES,
    _augmentation_kwargs_from,
    _build_custom_augmentations,
)


class TestAugmentationKwargsFrom:
    def test_empty_dict_returns_empty(self):
        assert _augmentation_kwargs_from({}) == {}

    def test_pulls_only_custom_augmentation_keys(self):
        hyperparameters = {
            "lr0": 0.01,
            "p_color_invariance": 0.15,
            "p_lighting": 0.35,
            "brightness_limit": 0.2,
            "p_occlusion": 0.25,
        }

        result = _augmentation_kwargs_from(hyperparameters)

        assert result == {
            "p_color_invariance": 0.15,
            "p_lighting": 0.35,
            "brightness_limit": 0.2,
            "p_occlusion": 0.25,
        }
        assert "lr0" not in result

    def test_covers_every_param_ranges_key(self):
        hyperparameters = dict.fromkeys(CUSTOM_AUGMENTATION_PARAM_RANGES, 0.1)

        result = _augmentation_kwargs_from(hyperparameters)

        assert set(result) == set(CUSTOM_AUGMENTATION_PARAM_RANGES)


class TestBuildCustomAugmentations:
    def test_no_hyperparameters_uses_each_builder_defaults(self):
        transforms = _build_custom_augmentations({})

        # 2 from shared.build_domain_transforms (OneOf groups) + 1 from
        # detection.build_detection_transforms (CoarseDropout).
        assert len(transforms) == 3

    def test_routes_shared_and_detection_keys_to_their_own_builder(self):
        transforms = _build_custom_augmentations(
            {"p_color_invariance": 1.0, "p_occlusion": 1.0, "lr0": 0.01}
        )

        assert len(transforms) == 3
        probabilities = [t.p for t in transforms]
        assert 1.0 in probabilities
