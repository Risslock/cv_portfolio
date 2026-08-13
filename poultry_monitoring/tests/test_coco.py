"""Smoke tests for `poultry_monitoring.data.coco`.

Constitution Principle VIII scope: deterministic, non-ML code only — COCO
parsing/conversion, not model behavior.
"""

import json

import pytest
import yaml

from poultry_monitoring.data.coco import (
    build_data_yaml,
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


class TestConvertCocoToYoloLabels:
    def test_skips_when_labels_dir_exists(self, tmp_path):
        data_dir = tmp_path / "ChickenDet"
        (data_dir / "labels").mkdir(parents=True)
        annotations_dir = data_dir / "annotations"

        result = convert_coco_to_yolo_labels(data_dir, annotations_dir)

        assert result == data_dir / "labels"

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
            "names": {0: "Chicken"},
        }
