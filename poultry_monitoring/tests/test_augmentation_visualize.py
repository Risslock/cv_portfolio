"""Smoke tests for `poultry_monitoring.augmentation.visualize`.

Constitution Principle VIII scope: `generate_augmented_samples` and
`load_yolo_label_boxes` only — the plotting half (`plot_augmented_grid`) opens a
matplotlib window and isn't meaningfully testable without image-diffing.
"""

import numpy as np
import pytest

from poultry_monitoring.augmentation.shared import build_domain_transforms
from poultry_monitoring.augmentation.visualize import (
    generate_augmented_samples,
    load_yolo_label_boxes,
)


class TestGenerateAugmentedSamples:
    def test_returns_requested_count_and_shape(self):
        image = np.random.default_rng(0).integers(0, 255, (32, 32, 3), dtype=np.uint8)
        transforms = build_domain_transforms()

        samples = generate_augmented_samples(image, transforms, n_samples=4)

        assert len(samples) == 4
        assert all(sample["image"].shape == image.shape for sample in samples)
        assert all("bboxes" not in sample for sample in samples)

    def test_seed_gives_reproducible_draws(self):
        image = np.random.default_rng(1).integers(0, 255, (32, 32, 3), dtype=np.uint8)
        transforms = build_domain_transforms(p_color_invariance=1.0, p_lighting=1.0)

        first = generate_augmented_samples(image, transforms, n_samples=3, seed=42)
        second = generate_augmented_samples(image, transforms, n_samples=3, seed=42)

        for a, b in zip(first, second):
            assert np.array_equal(a["image"], b["image"])

    def test_bboxes_pass_through_unmoved_by_image_only_transforms(self):
        image = np.random.default_rng(2).integers(0, 255, (32, 32, 3), dtype=np.uint8)
        transforms = build_domain_transforms(p_color_invariance=1.0, p_lighting=1.0)
        bboxes = [(0.5, 0.5, 0.2, 0.3)]

        samples = generate_augmented_samples(image, transforms, bboxes=bboxes, n_samples=2)

        for sample in samples:
            assert len(sample["bboxes"]) == 1
            assert sample["bboxes"][0] == pytest.approx(bboxes[0], abs=1e-4)


class TestLoadYoloLabelBoxes:
    def test_missing_file_returns_empty(self, tmp_path):
        assert load_yolo_label_boxes(tmp_path / "missing.txt") == []

    def test_plain_box_format(self, tmp_path):
        label_path = tmp_path / "img.txt"
        label_path.write_text("0 0.5 0.5 0.2 0.3\n")

        boxes = load_yolo_label_boxes(label_path)

        assert boxes == [(0.5, 0.5, 0.2, 0.3)]

    def test_segment_polygon_format_derives_bbox(self, tmp_path):
        # Square polygon (0.4, 0.4) -> (0.6, 0.6): center (0.5, 0.5), size (0.2, 0.2).
        label_path = tmp_path / "img.txt"
        label_path.write_text("0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6\n")

        boxes = load_yolo_label_boxes(label_path)

        assert len(boxes) == 1
        assert boxes[0] == pytest.approx((0.5, 0.5, 0.2, 0.2), abs=1e-6)
