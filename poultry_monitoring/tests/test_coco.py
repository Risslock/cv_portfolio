"""Smoke tests for `poultry_monitoring.data.coco`.

Constitution Principle VIII scope: deterministic, non-ML code only — COCO
parsing/conversion, not model behavior.
"""

import json
import os
import time

import numpy as np
import pycocotools.mask as mask_utils
import pytest
import yaml
from pycocotools.coco import COCO

from poultry_monitoring.data.coco import (
    _convert_rle_to_polygons_in_json,
    _decode_rle_mask,
    build_data_yaml,
    cache_polygon_annotations,
    coco_category_id_to_yolo_class_id,
    convert_coco_to_yolo_labels,
    fix_iscrowd_field,
)


def _write_synthetic_coco(annotations_dir, split: str, iscrowd_values: list[int]) -> None:
    """Write a minimal COCO instances JSON with one image and one annotation per `iscrowd` value."""
    coco = {
        "images": [{"id": 1, "file_name": "img1.jpg", "height": 100, "width": 100}],
        "annotations": [
            {
                "id": i,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 20, 20],
                "area": 400,
                "iscrowd": iscrowd,
                "segmentation": [],
            }
            for i, iscrowd in enumerate(iscrowd_values, start=1)
        ],
        "categories": [{"id": 1, "name": "Chicken"}],
    }
    annotations_dir.mkdir(parents=True, exist_ok=True)
    (annotations_dir / f"instances_{split}.json").write_text(json.dumps(coco))


def _bump_mtime_forward(path) -> None:
    """Set `path`'s mtime a few seconds into the future.

    Rewriting a file in-place within the same test can land on the same mtime (some
    filesystems' mtime resolution is coarser than a fast test run), which would hide a
    real fingerprint change -- bumping explicitly makes the "source changed" signal
    deterministic instead of flaky.
    """
    future = time.time() + 5
    os.utime(path, (future, future))


def _make_rle_segmentation(mask: np.ndarray) -> dict:
    """Encode a binary mask (H, W) as an *uncompressed* COCO RLE segmentation dict.

    Deliberately not `pycocotools.mask.encode()`, which produces the *compressed*
    variant (`counts` as a byte string) -- ChickenDet's real annotations use the
    uncompressed variant instead (`counts` as a plain list of run-lengths), which
    `frPyObjects` needs to correctly convert. Passing a compressed dict's string
    `counts` straight into `frPyObjects` raises `ValueError: invalid literal for int()`,
    which is how this mismatch was actually caught.
    """
    flat = np.asfortranarray(mask).flatten(order="F")
    counts = []
    current, prev = 0, 0
    for v in flat:
        if v == prev:
            current += 1
        else:
            counts.append(current)
            current, prev = 1, v
    counts.append(current)
    return {"size": list(mask.shape), "counts": counts}


def _write_synthetic_coco_with_square_mask(annotations_dir, split: str = "Train") -> None:
    """Write a synthetic COCO JSON with one instance: a 20x20 filled square mask (RLE-encoded)."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:30, 10:30] = 1
    _write_synthetic_coco_with_segmentation(annotations_dir, split, _make_rle_segmentation(mask))


def _write_synthetic_coco_with_segmentation(annotations_dir, split: str, segmentation) -> None:
    """Write a minimal COCO instances JSON with one image, one annotation, given segmentation."""
    coco = {
        "images": [{"id": 1, "file_name": "img1.jpg", "height": 100, "width": 100}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 20, 20],
                "area": 400,
                "iscrowd": 0,
                "segmentation": segmentation,
            }
        ],
        "categories": [{"id": 1, "name": "Chicken"}],
    }
    annotations_dir.mkdir(parents=True, exist_ok=True)
    (annotations_dir / f"instances_{split}.json").write_text(json.dumps(coco))


class TestFixIscrowdField:
    def test_noop_when_already_zero(self, tmp_path):
        annotations_dir = tmp_path / "annotations"
        _write_synthetic_coco(annotations_dir, "Train", iscrowd_values=[0, 0])
        coco_path = annotations_dir / "instances_Train.json"
        original_bytes = coco_path.read_bytes()

        fix_iscrowd_field(coco_path, assume_yes=True)

        assert coco_path.read_bytes() == original_bytes
        assert not coco_path.with_suffix(".json.bak").exists()

    def test_zeroes_iscrowd_and_backs_up(self, tmp_path):
        annotations_dir = tmp_path / "annotations"
        _write_synthetic_coco(annotations_dir, "Train", iscrowd_values=[1, 0])
        coco_path = annotations_dir / "instances_Train.json"

        fix_iscrowd_field(coco_path, assume_yes=True)

        coco = json.loads(coco_path.read_text())
        assert all(ann["iscrowd"] == 0 for ann in coco["annotations"])
        assert coco_path.with_suffix(".json.bak").exists()


class TestDecodeRleMask:
    def test_decodes_uncompressed_rle(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:30] = 1
        seg = _make_rle_segmentation(mask)  # ChickenDet's actual variant: counts as a list

        decoded = _decode_rle_mask(seg)

        assert np.array_equal(decoded, mask)

    def test_decodes_compressed_rle(self):
        """Not ChickenDet's own format (verified 100% uncompressed across all real annotations).

        `pycocotools.mask.encode()`'s default output shape though -- worth handling
        explicitly rather than assuming only one variant ever shows up.
        """
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:30] = 1
        encoded = mask_utils.encode(np.asfortranarray(mask))
        seg = {"size": list(encoded["size"]), "counts": encoded["counts"].decode("ascii")}

        decoded = _decode_rle_mask(seg)

        assert np.array_equal(decoded, mask)


class TestConvertRleToPolygonsInJson:
    def test_converts_filled_mask_to_polygon(self, tmp_path):
        annotations_dir = tmp_path / "annotations"
        _write_synthetic_coco_with_square_mask(annotations_dir)
        json_path = annotations_dir / "instances_Train.json"
        output_path = tmp_path / "instances_Train_poly.json"

        unconverted = _convert_rle_to_polygons_in_json(json_path, output_path)

        assert unconverted == []
        seg = json.loads(output_path.read_text())["annotations"][0]["segmentation"]
        assert isinstance(seg, list)  # polygon, not the original RLE dict
        assert len(seg[0]) >= 6  # at least 3 (x, y) points for one contour

    def test_reports_unconvertible_empty_mask(self, tmp_path):
        empty_mask = np.zeros((100, 100), dtype=np.uint8)  # no foreground pixels at all
        segmentation = _make_rle_segmentation(empty_mask)
        annotations_dir = tmp_path / "annotations"
        _write_synthetic_coco_with_segmentation(annotations_dir, "Train", segmentation)
        json_path = annotations_dir / "instances_Train.json"
        output_path = tmp_path / "instances_Train_poly.json"

        unconverted = _convert_rle_to_polygons_in_json(json_path, output_path)

        assert unconverted == [1]
        # left as the original RLE, unchanged, since no contour could be extracted --
        # `cache_polygon_annotations` relies on this to report it rather than lose it
        seg = json.loads(output_path.read_text())["annotations"][0]["segmentation"]
        assert seg == segmentation


class TestCachePolygonAnnotations:
    def test_builds_cache_from_every_split_json(self, tmp_path):
        annotations_dir = tmp_path / "annotations"
        _write_synthetic_coco_with_square_mask(annotations_dir)
        cache_dir = tmp_path / "cache"

        result = cache_polygon_annotations(annotations_dir, cache_dir)

        assert result == cache_dir
        cached = json.loads((cache_dir / "instances_Train.json").read_text())
        assert isinstance(cached["annotations"][0]["segmentation"], list)

    def test_skips_existing_cache_unless_forced(self, tmp_path):
        annotations_dir = tmp_path / "annotations"
        _write_synthetic_coco_with_square_mask(annotations_dir)
        cache_dir = tmp_path / "cache"
        cache_polygon_annotations(annotations_dir, cache_dir)
        cached_path = cache_dir / "instances_Train.json"
        original_bytes = cached_path.read_bytes()

        cache_polygon_annotations(annotations_dir, cache_dir)  # no force -- should no-op
        assert cached_path.read_bytes() == original_bytes

        cache_polygon_annotations(annotations_dir, cache_dir, force=True)
        assert cached_path.exists()  # rebuilt, not just left alone

    def test_regenerates_when_source_annotation_changes_without_force(self, tmp_path):
        """Re-converts on a genuine source change even without `force=True`.

        Existence of a same-named cached file is no longer sufficient, see
        docs/adr/0016-fingerprinted-label-cache-invalidation.md.
        """
        annotations_dir = tmp_path / "annotations"
        mask_a = np.zeros((100, 100), dtype=np.uint8)
        mask_a[10:30, 10:30] = 1
        _write_synthetic_coco_with_segmentation(
            annotations_dir, "Train", _make_rle_segmentation(mask_a)
        )
        cache_dir = tmp_path / "cache"
        cache_polygon_annotations(annotations_dir, cache_dir)

        mask_b = np.zeros((100, 100), dtype=np.uint8)
        mask_b[40:80, 40:80] = 1  # a different, larger square
        _write_synthetic_coco_with_segmentation(
            annotations_dir, "Train", _make_rle_segmentation(mask_b)
        )
        _bump_mtime_forward(annotations_dir / "instances_Train.json")

        cache_polygon_annotations(annotations_dir, cache_dir)  # still no force

        cached_poly = json.loads((cache_dir / "instances_Train.json").read_text())
        xs = cached_poly["annotations"][0]["segmentation"][0][0::2]
        assert min(xs) >= 40  # picked up mask_b's region, not the stale mask_a cache


class TestConvertCocoToYoloLabels:
    def test_regenerates_stale_labels_dir_with_no_manifest(self, tmp_path):
        """A pre-existing `labels/` with no fingerprint manifest is rebuilt, not trusted.

        e.g. built before fingerprinting existed -- exactly the real stale-cache bug
        that motivated it, see docs/adr/0016-...md.
        """
        data_dir = tmp_path / "ChickenDet"
        (data_dir / "labels" / "Train").mkdir(parents=True)
        (data_dir / "labels" / "Train" / "stale.txt").write_text("0 0 0 1 1")
        annotations_dir = data_dir / "annotations"
        _write_synthetic_coco(annotations_dir, "Train", iscrowd_values=[0])

        labels_dir = convert_coco_to_yolo_labels(data_dir, annotations_dir)

        assert not (labels_dir / "Train" / "stale.txt").exists()  # old content discarded
        assert (labels_dir / "Train" / "img1.txt").exists()  # rebuilt from real inputs

    def test_skips_when_fingerprint_matches(self, tmp_path):
        data_dir = tmp_path / "ChickenDet"
        annotations_dir = data_dir / "annotations"
        _write_synthetic_coco(annotations_dir, "Train", iscrowd_values=[0])
        convert_coco_to_yolo_labels(data_dir, annotations_dir)
        label_path = data_dir / "labels" / "Train" / "img1.txt"
        original_bytes = label_path.read_bytes()

        convert_coco_to_yolo_labels(data_dir, annotations_dir)  # unchanged inputs -- no-op

        assert label_path.read_bytes() == original_bytes

    def test_regenerates_when_source_annotations_change_without_force(self, tmp_path):
        data_dir = tmp_path / "ChickenDet"
        annotations_dir = data_dir / "annotations"
        _write_synthetic_coco(annotations_dir, "Train", iscrowd_values=[0])
        convert_coco_to_yolo_labels(data_dir, annotations_dir)

        coco_path = annotations_dir / "instances_Train.json"
        coco = json.loads(coco_path.read_text())
        coco["annotations"][0]["bbox"] = [0, 0, 40, 40]  # was [10, 10, 20, 20]
        coco_path.write_text(json.dumps(coco))
        _bump_mtime_forward(coco_path)

        convert_coco_to_yolo_labels(data_dir, annotations_dir)  # still no force

        label_path = data_dir / "labels" / "Train" / "img1.txt"
        _cls, _x_c, _y_c, w, _h = map(float, label_path.read_text().split())
        assert w == pytest.approx(0.4)  # picked up the new bbox, not the stale cache

    def test_converts_and_lays_out_alongside_images(self, tmp_path):
        data_dir = tmp_path / "ChickenDet"
        annotations_dir = data_dir / "annotations"
        _write_synthetic_coco(annotations_dir, "Train", iscrowd_values=[0])

        labels_dir = convert_coco_to_yolo_labels(data_dir, annotations_dir)

        assert labels_dir == data_dir / "labels"
        label_file = labels_dir / "Train" / "img1.txt"
        assert label_file.exists()
        cls, x_c, y_c, w, h = map(float, label_file.read_text().split())
        assert cls == 0
        # bbox [10, 10, 20, 20] on a 100x100 image -> center (20, 20)/100, size (20, 20)/100
        assert x_c == pytest.approx(0.2)
        assert y_c == pytest.approx(0.2)
        assert w == pytest.approx(0.2)
        assert h == pytest.approx(0.2)

    def test_use_segments_converts_rle_to_real_polygon_labels(self, tmp_path):
        data_dir = tmp_path / "ChickenDet"
        annotations_dir = data_dir / "annotations"
        _write_synthetic_coco_with_square_mask(annotations_dir)

        labels_dir = convert_coco_to_yolo_labels(data_dir, annotations_dir, use_segments=True)

        assert (data_dir / "annotations_polygon_cache" / "instances_Train.json").exists()
        values = (labels_dir / "Train" / "img1.txt").read_text().split()
        # class + at least 3 (x, y) polygon points -- more than a box-only line's 5 values
        assert len(values) > 5


class TestBuildDataYaml:
    def test_writes_expected_schema(self, tmp_path):
        data_dir = tmp_path / "ChickenDet"
        yaml_path = tmp_path / "chickendet.yaml"

        result = build_data_yaml(data_dir, {0: "Chicken"}, yaml_path)

        assert result == yaml_path
        written = yaml.safe_load(yaml_path.read_text())
        assert written == {
            "path": str(data_dir),
            "train": "images/Train",
            "val": "images/Validation",
            "test": "images/Test",
            "names": {0: "Chicken"},
        }

    def test_multiple_train_splits_writes_a_list(self, tmp_path):
        data_dir = tmp_path / "ChickenDet"
        yaml_path = tmp_path / "chickendet_stage_b.yaml"

        build_data_yaml(
            data_dir, {0: "Chicken"}, yaml_path, train_splits=("Train", "TrainSynthetic")
        )

        written = yaml.safe_load(yaml_path.read_text())
        assert written["train"] == ["images/Train", "images/TrainSynthetic"]


class TestCocoCategoryIdToYoloClassId:
    def test_maps_sorted_category_ids_to_sequential_yolo_indices(self, tmp_path):
        coco_json = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "Chicken"}],
        }
        annotations_path = tmp_path / "instances.json"
        annotations_path.write_text(json.dumps(coco_json))

        mapping = coco_category_id_to_yolo_class_id(COCO(str(annotations_path)))

        assert mapping == {1: 0}
