"""Smoke tests for `poultry_monitoring.segmentation.copy_paste_training`.

Constitution Principle VIII scope: deterministic, non-ML code — that the transform
produces a *structurally valid* Ultralytics sample, not that it improves training.

The core risk this guards is silent label corruption: the transform has to append
instances whose segments stack with the scene's fixed-width `(N, P, 2)` array, whose
class ids land in `cls`, and whose bbox/normalization convention matches what the next
transform in the chain expects. Any of those going wrong trains the model on wrong
labels without raising.
"""

import random

import numpy as np
import pytest
from ultralytics.utils.instance import Instances
from ultralytics.utils.ops import resample_segments

from poultry_monitoring.segmentation.copy_paste_training import (
    BankCopyPaste,
    remap_manifest_to_yolo_class_ids,
)

N_POINTS = 1000


def _make_labels(n_instances: int = 4, size: int = 640, seed: int = 0) -> dict:
    """Build a minimal Ultralytics sample dict: BGR image, cls, and normalized Instances."""
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)

    polygons, bboxes = [], []
    for i in range(n_instances):
        x0, y0, side = 40 + i * 90, 60, 50
        square = np.array(
            [[x0, y0], [x0 + side, y0], [x0 + side, y0 + side], [x0, y0 + side]], dtype=np.float32
        )
        polygons.append(resample_segments([square], n=N_POINTS)[0].astype(np.float32) / size)
        bboxes.append([x0 / size, y0 / size, (x0 + side) / size, (y0 + side) / size])

    instances = Instances(
        np.array(bboxes, dtype=np.float32),
        np.stack(polygons, axis=0),
        None,
        bbox_format="xyxy",
        normalized=True,
    )
    return {
        "img": image,
        "cls": np.zeros((n_instances, 1), dtype=np.float32),
        "instances": instances,
    }


class TestRemapManifestToYoloClassIds:
    """Regression guard for a bug a real smoke train caught and the unit tests missed.

    `build_donor_bank` stores raw COCO category ids (ChickenDet's is 1), but a single-class
    model has only class 0. Feeding the raw id to training indexes a class column that
    doesn't exist and blows up as an opaque CUDA device-side assert inside the loss.
    """

    def test_remaps_raw_coco_ids_to_zero_indexed(self):
        manifest = [{"donor_id": "ann1", "category_id": 1}, {"donor_id": "ann2", "category_id": 1}]

        remapped = remap_manifest_to_yolo_class_ids(manifest, nc=1)

        assert [e["category_id"] for e in remapped] == [0, 0]

    def test_is_a_noop_on_already_remapped_manifest(self):
        manifest = [{"donor_id": "ann1", "category_id": 0}]

        assert remap_manifest_to_yolo_class_ids(manifest, nc=1)[0]["category_id"] == 0

    def test_raises_when_bank_has_more_classes_than_the_model(self):
        manifest = [{"donor_id": "a", "category_id": 1}, {"donor_id": "b", "category_id": 7}]

        with pytest.raises(ValueError, match="outside the model's nc"):
            remap_manifest_to_yolo_class_ids(manifest, nc=1)


class TestBankCopyPaste:
    def test_zero_probability_is_a_noop(self, donor_bank):
        bank_dir, manifest = donor_bank
        labels = _make_labels()
        original_image = labels["img"].copy()
        transform = BankCopyPaste(bank_dir, manifest, p=0.0, max_donors=3)

        result = transform(labels)

        assert len(result["instances"]) == 4
        assert np.array_equal(result["img"], original_image)

    def test_appends_instances_cls_and_segments(self, donor_bank):
        bank_dir, manifest = donor_bank
        labels = _make_labels()
        transform = BankCopyPaste(bank_dir, manifest, p=1.0, max_donors=3)
        random.seed(0)

        result = transform(labels)

        instances = result["instances"]
        assert len(instances) > 4, "no donors were pasted"
        # cls and instances must stay in lockstep or the loss pairs the wrong label
        assert result["cls"].shape[0] == len(instances)
        # segments must remain a stackable fixed-width array
        assert instances.segments.shape[1:] == (N_POINTS, 2)
        assert instances.segments.shape[0] == len(instances)

    def test_preserves_normalization_and_bbox_convention(self, donor_bank):
        bank_dir, manifest = donor_bank
        labels = _make_labels()
        transform = BankCopyPaste(bank_dir, manifest, p=1.0, max_donors=3)
        random.seed(0)

        result = transform(labels)

        assert result["instances"].normalized is True
        assert result["instances"]._bboxes.format == "xyxy"
        # normalized coords must stay in [0, 1] -- a donor pasted with pixel coords left
        # in would sail past 1.0 and silently corrupt the label
        assert result["instances"].segments.max() <= 1.0 + 1e-6
        assert result["instances"].segments.min() >= -1e-6

    def test_pasted_class_ids_come_from_the_manifest(self, donor_bank):
        bank_dir, manifest = donor_bank
        labels = _make_labels()
        transform = BankCopyPaste(bank_dir, manifest, p=1.0, max_donors=3)
        random.seed(0)

        result = transform(labels)

        assert set(np.unique(result["cls"])) == {0.0}

    def test_image_is_actually_modified(self, donor_bank):
        bank_dir, manifest = donor_bank
        labels = _make_labels()
        original_image = labels["img"].copy()
        transform = BankCopyPaste(bank_dir, manifest, p=1.0, max_donors=3)
        random.seed(0)

        result = transform(labels)

        assert not np.array_equal(result["img"], original_image)

    def test_scene_with_no_segments_is_a_noop(self, donor_bank):
        bank_dir, manifest = donor_bank
        labels = _make_labels()
        labels["instances"] = Instances(
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0, N_POINTS, 2), dtype=np.float32),
            None,
            bbox_format="xyxy",
            normalized=True,
        )
        labels["cls"] = np.zeros((0, 1), dtype=np.float32)
        transform = BankCopyPaste(bank_dir, manifest, p=1.0, max_donors=3)

        result = transform(labels)

        assert len(result["instances"]) == 0

    def test_bboxes_enclose_their_segments(self, donor_bank):
        """A donor's appended bbox must actually bound its polygon."""
        bank_dir, manifest = donor_bank
        labels = _make_labels()
        transform = BankCopyPaste(bank_dir, manifest, p=1.0, max_donors=3)
        random.seed(0)

        result = transform(labels)

        instances = result["instances"]
        for bbox, segment in zip(instances.bboxes, instances.segments):
            x0, y0, x1, y1 = bbox
            assert segment[:, 0].min() == pytest.approx(x0, abs=1e-3)
            assert segment[:, 1].min() == pytest.approx(y0, abs=1e-3)
            assert segment[:, 0].max() == pytest.approx(x1, abs=1e-3)
            assert segment[:, 1].max() == pytest.approx(y1, abs=1e-3)
