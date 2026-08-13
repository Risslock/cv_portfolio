"""Smoke tests for `poultry_monitoring.augmentation.shared`.

Constitution Principle VIII scope: deterministic, non-ML code — that the transform
list is well-formed and runnable, not anything about training/convergence.
"""

import albumentations as A
import numpy as np

from poultry_monitoring.augmentation.shared import build_domain_transforms


class TestBuildDomainTransforms:
    def test_returns_list_of_transforms(self):
        transforms = build_domain_transforms()
        assert len(transforms) == 2
        # Each entry is a `OneOf` group (a composition), not a leaf transform.
        assert all(isinstance(t, (A.BasicTransform, A.BaseCompose)) for t in transforms)

    def test_composes_and_runs_on_an_image(self):
        transforms = build_domain_transforms(p_color_invariance=1.0, p_lighting=1.0)
        pipeline = A.Compose(transforms)
        image = np.zeros((32, 32, 3), dtype=np.uint8)

        result = pipeline(image=image)["image"]

        assert result.shape == image.shape
        assert result.dtype == image.dtype

    def test_zero_probability_is_a_noop(self):
        transforms = build_domain_transforms(p_color_invariance=0.0, p_lighting=0.0)
        pipeline = A.Compose(transforms)
        image = np.random.default_rng(0).integers(0, 255, (16, 16, 3), dtype=np.uint8)

        result = pipeline(image=image)["image"]

        assert np.array_equal(result, image)
