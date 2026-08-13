"""Smoke tests for `poultry_monitoring.augmentation.detection`.

Constitution Principle VIII scope: deterministic, non-ML code — that the transform
list is well-formed, runnable, and (this module's whole point) leaves bounding boxes
untouched — not anything about training/convergence.
"""

import albumentations as A
import numpy as np

from poultry_monitoring.augmentation.detection import build_detection_transforms


class TestBuildDetectionTransforms:
    def test_returns_list_of_transforms(self):
        transforms = build_detection_transforms()
        assert len(transforms) == 1
        assert isinstance(transforms[0], A.CoarseDropout)

    def test_composes_and_runs_on_an_image(self):
        transforms = build_detection_transforms(p_occlusion=1.0)
        pipeline = A.Compose(transforms)
        image = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)

        result = pipeline(image=image)["image"]

        assert result.shape == image.shape
        assert result.dtype == image.dtype

    def test_zero_probability_is_a_noop(self):
        transforms = build_detection_transforms(p_occlusion=0.0)
        pipeline = A.Compose(transforms)
        image = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)

        result = pipeline(image=image)["image"]

        assert np.array_equal(result, image)

    def test_boxes_survive_even_full_occlusion(self):
        """Boxes must survive full coverage.

        The whole point of `CoarseDropout` here: pixel-level, not geometric — a fully
        covered box keeps its coordinates, forcing the model to use partial-visibility
        cues instead of the label silently vanishing.
        """
        transforms = build_detection_transforms(p_occlusion=1.0)
        pipeline = A.Compose(
            transforms, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])
        )
        image = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        box = (0.5, 0.5, 1.0, 1.0)  # covers the whole image

        result = pipeline(image=image, bboxes=[box], class_labels=[0])

        assert [tuple(b) for b in result["bboxes"]] == [box]
