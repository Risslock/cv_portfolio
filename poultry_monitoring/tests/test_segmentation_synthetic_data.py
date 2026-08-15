"""Smoke tests for `poultry_monitoring.segmentation.synthetic_data`.

Constitution Principle VIII scope: deterministic, non-ML code — file I/O, label
formatting, and fingerprint/caching behavior, not training/convergence. The
category-id remap tests are the regression check for a real bug found while designing
this module: the donor bank stores raw COCO category ids, real YOLO-seg label files use
0-indexed class ids, and the two must not be mixed unremapped in a written label file.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image
from pycocotools.coco import COCO

from poultry_monitoring.augmentation.segmentation import build_donor_bank, mask_to_yolo_polygon
from poultry_monitoring.segmentation.synthetic_data import (
    SYNTHETIC_MANIFEST_SUFFIX,
    _remap_manifest_category_ids,
    build_stage_b_data_yaml,
    generate_synthetic_split,
    remove_synthetic_split,
)


def _square_mask(shape: tuple[int, int], y0: int, x0: int, size: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[y0 : y0 + size, x0 : x0 + size] = 1
    return mask


def _make_rle_segmentation(mask: np.ndarray) -> dict:
    """Encode a binary mask as an uncompressed COCO RLE dict (ChickenDet's own variant)."""
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


def _write_categories_only_annotations(tmp_path) -> Path:
    """Write a COCO json with just the single ChickenDet category.

    All this module needs annotations for is `coco_category_id_to_yolo_class_id`'s
    category list.
    """
    annotations_path = tmp_path / "instances.json"
    annotations_path.write_text(
        json.dumps({"images": [], "annotations": [], "categories": [{"id": 1, "name": "Chicken"}]})
    )
    return annotations_path


def _build_donor_bank(tmp_path, bank_dir) -> None:
    """Curate a tiny one-donor bank from a standalone donor-source image, category_id=1."""
    donor_img_dir = tmp_path / "donor_images"
    donor_img_dir.mkdir()
    rng = np.random.default_rng(1)
    donor_image = rng.integers(0, 255, (600, 600, 3), dtype=np.uint8)
    Image.fromarray(donor_image).save(donor_img_dir / "donor.png")

    mask = _square_mask((600, 600), 100, 100, 60)
    donor_coco_json = {
        "images": [{"id": 1, "file_name": "donor.png", "height": 600, "width": 600}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 100, 60, 60],
                "area": 3600,
                "iscrowd": 0,
                "segmentation": _make_rle_segmentation(mask),
            }
        ],
        "categories": [{"id": 1, "name": "Chicken"}],
    }
    donor_annotations_path = tmp_path / "donor_instances.json"
    donor_annotations_path.write_text(json.dumps(donor_coco_json))

    donor_coco = COCO(str(donor_annotations_path))
    build_donor_bank(donor_coco, donor_img_dir, bank_dir, max_donors=1, seed=0)


def _build_training_split(data_dir, img_size: int = 500) -> None:
    """Write one real Train image + its YOLO-seg label (one real instance, class 0)."""
    images_dir = data_dir / "images" / "Train"
    labels_dir = data_dir / "labels" / "Train"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (img_size, img_size, 3), dtype=np.uint8)
    Image.fromarray(image).save(images_dir / "img1.jpeg", format="JPEG")

    real_mask = _square_mask((img_size, img_size), 50, 50, 40)
    line = mask_to_yolo_polygon(real_mask, category_id=0, img_w=img_size, img_h=img_size)
    (labels_dir / "img1.txt").write_text(line)


class TestRemapManifestCategoryIds:
    def test_rewrites_category_id_via_map(self):
        manifest = [{"donor_id": "ann1", "category_id": 1, "h": 10, "w": 10}]

        remapped = _remap_manifest_category_ids(manifest, {1: 0})

        assert remapped == [{"donor_id": "ann1", "category_id": 0, "h": 10, "w": 10}]
        assert manifest[0]["category_id"] == 1  # original untouched


class TestGenerateSyntheticSplit:
    def test_pasted_donor_gets_remapped_yolo_class_id(self, tmp_path):
        data_dir = tmp_path / "ChickenDet"
        _build_training_split(data_dir)
        annotations_path = _write_categories_only_annotations(tmp_path)
        bank_dir = tmp_path / "bank"
        _build_donor_bank(tmp_path, bank_dir)

        output_images_dir = generate_synthetic_split(
            data_dir,
            annotations_path,
            bank_dir,
            source_split="Train",
            output_split="TrainSynthetic",
            p_copy_paste=1.0,
            max_donors=1,
            seed=0,
        )

        assert (output_images_dir / "img1.jpeg").exists()
        label_path = data_dir / "labels" / "TrainSynthetic" / "img1.txt"
        lines = label_path.read_text().splitlines()
        assert len(lines) == 2  # 1 original real instance + 1 pasted donor
        # Both must read class 0 -- the donor bank's raw category_id=1 must not leak
        # through unremapped (that would silently mislabel every pasted donor).
        assert {int(line.split()[0]) for line in lines} == {0}

    def test_real_split_is_never_modified(self, tmp_path):
        data_dir = tmp_path / "ChickenDet"
        _build_training_split(data_dir)
        annotations_path = _write_categories_only_annotations(tmp_path)
        bank_dir = tmp_path / "bank"
        _build_donor_bank(tmp_path, bank_dir)
        original_image_bytes = (data_dir / "images" / "Train" / "img1.jpeg").read_bytes()
        original_label_text = (data_dir / "labels" / "Train" / "img1.txt").read_text()

        generate_synthetic_split(
            data_dir, annotations_path, bank_dir, p_copy_paste=1.0, max_donors=1, seed=0
        )

        assert (data_dir / "images" / "Train" / "img1.jpeg").read_bytes() == original_image_bytes
        assert (data_dir / "labels" / "Train" / "img1.txt").read_text() == original_label_text

    def test_zero_probability_writes_nothing(self, tmp_path):
        data_dir = tmp_path / "ChickenDet"
        _build_training_split(data_dir)
        annotations_path = _write_categories_only_annotations(tmp_path)
        bank_dir = tmp_path / "bank"
        _build_donor_bank(tmp_path, bank_dir)

        output_images_dir = generate_synthetic_split(
            data_dir, annotations_path, bank_dir, p_copy_paste=0.0, max_donors=1, seed=0
        )

        assert list(output_images_dir.iterdir()) == []

    def test_mismatched_rerun_without_force_raises(self, tmp_path):
        data_dir = tmp_path / "ChickenDet"
        _build_training_split(data_dir)
        annotations_path = _write_categories_only_annotations(tmp_path)
        bank_dir = tmp_path / "bank"
        _build_donor_bank(tmp_path, bank_dir)
        generate_synthetic_split(
            data_dir, annotations_path, bank_dir, p_copy_paste=1.0, max_donors=1, seed=0
        )

        with pytest.raises(FileExistsError):
            generate_synthetic_split(
                data_dir, annotations_path, bank_dir, p_copy_paste=0.5, max_donors=1, seed=0
            )

    def test_reuses_matching_split_without_error(self, tmp_path):
        data_dir = tmp_path / "ChickenDet"
        _build_training_split(data_dir)
        annotations_path = _write_categories_only_annotations(tmp_path)
        bank_dir = tmp_path / "bank"
        _build_donor_bank(tmp_path, bank_dir)
        first = generate_synthetic_split(
            data_dir, annotations_path, bank_dir, p_copy_paste=1.0, max_donors=1, seed=0
        )

        second = generate_synthetic_split(
            data_dir, annotations_path, bank_dir, p_copy_paste=1.0, max_donors=1, seed=0
        )

        assert second == first


class TestRemoveSyntheticSplit:
    def test_deletes_images_labels_and_fingerprint(self, tmp_path):
        data_dir = tmp_path / "ChickenDet"
        _build_training_split(data_dir)
        annotations_path = _write_categories_only_annotations(tmp_path)
        bank_dir = tmp_path / "bank"
        _build_donor_bank(tmp_path, bank_dir)
        generate_synthetic_split(
            data_dir, annotations_path, bank_dir, p_copy_paste=1.0, max_donors=1, seed=0
        )
        fingerprint_path = data_dir / f"TrainSynthetic{SYNTHETIC_MANIFEST_SUFFIX}"
        assert fingerprint_path.exists()

        remove_synthetic_split(data_dir, "TrainSynthetic", assume_yes=True)

        assert not (data_dir / "images" / "TrainSynthetic").exists()
        assert not (data_dir / "labels" / "TrainSynthetic").exists()
        assert not fingerprint_path.exists()

    def test_noop_on_missing_split(self, tmp_path):
        data_dir = tmp_path / "ChickenDet"
        remove_synthetic_split(data_dir, "TrainSynthetic", assume_yes=True)  # should not raise


class TestBuildStageBDataYaml:
    def test_combines_real_and_synthetic_splits(self, tmp_path):
        data_dir = tmp_path / "ChickenDet"
        data_dir.mkdir(parents=True)

        yaml_path = build_stage_b_data_yaml(data_dir)

        written = yaml.safe_load(yaml_path.read_text())
        assert written["train"] == ["images/Train", "images/TrainSynthetic"]
        assert written["val"] == "images/Validation"
