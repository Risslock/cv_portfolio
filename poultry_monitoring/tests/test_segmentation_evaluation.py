"""Smoke tests for `segmentation/evaluation.py`'s density binning.

Only the pure, deterministic part is covered — `evaluate_split`/`evaluate_by_density` run a
real model and are outside the gate (constitution Principle VIII).
"""

import json

import pytest

from poultry_monitoring.segmentation.evaluation import TERCILE_LABELS, density_bins


def write_coco(path, counts_by_name: dict[str, int]) -> None:
    """Write a COCO instances JSON where each image has the requested annotation count."""
    images, annotations = [], []
    for image_id, (file_name, count) in enumerate(counts_by_name.items(), start=1):
        images.append({"id": image_id, "file_name": file_name, "height": 100, "width": 100})
        for _ in range(count):
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [0, 0, 10, 10],
                    "area": 100,
                    "iscrowd": 0,
                }
            )
    path.write_text(
        json.dumps(
            {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "C"}]}
        )
    )


class TestDensityBins:
    def test_splits_into_equal_size_bins_sorted_by_density(self, tmp_path):
        path = tmp_path / "instances_Test.json"
        write_coco(path, {f"img{i}.jpg": i for i in range(1, 10)})  # counts 1..9

        bins = density_bins(path, n_bins=3)

        assert list(bins) == list(TERCILE_LABELS)
        assert [b["images"] for b in bins.values()] == [3, 3, 3]
        assert bins["sparse"]["file_names"] == ["img1.jpg", "img2.jpg", "img3.jpg"]
        assert bins["dense"]["file_names"] == ["img7.jpg", "img8.jpg", "img9.jpg"]

    def test_reports_per_bin_density_summary(self, tmp_path):
        path = tmp_path / "instances_Test.json"
        write_coco(path, {f"img{i}.jpg": i for i in range(1, 10)})

        bins = density_bins(path, n_bins=3)

        assert bins["sparse"] == {
            "file_names": ["img1.jpg", "img2.jpg", "img3.jpg"],
            "images": 3,
            "instances": 6,
            "min": 1,
            "median": 2,
            "max": 3,
        }

    def test_uneven_split_differs_by_at_most_one_image(self, tmp_path):
        path = tmp_path / "instances_Test.json"
        write_coco(path, {f"img{i}.jpg": i for i in range(1, 12)})  # 11 images, 3 bins

        sizes = [b["images"] for b in density_bins(path, n_bins=3).values()]

        assert sum(sizes) == 11
        assert max(sizes) - min(sizes) <= 1

    def test_images_with_no_annotations_land_in_the_sparsest_bin(self, tmp_path):
        path = tmp_path / "instances_Test.json"
        write_coco(path, {"empty.jpg": 0, "a.jpg": 5, "b.jpg": 9})

        bins = density_bins(path, n_bins=3)

        assert bins["sparse"]["file_names"] == ["empty.jpg"]
        assert bins["sparse"]["instances"] == 0

    def test_non_tercile_bin_counts_get_generic_labels(self, tmp_path):
        path = tmp_path / "instances_Test.json"
        write_coco(path, {f"img{i}.jpg": i for i in range(1, 9)})

        assert list(density_bins(path, n_bins=4)) == ["bin1", "bin2", "bin3", "bin4"]

    @pytest.mark.parametrize("n_bins", [0, -1, 99])
    def test_rejects_out_of_range_bin_counts(self, tmp_path, n_bins):
        path = tmp_path / "instances_Test.json"
        write_coco(path, {f"img{i}.jpg": i for i in range(1, 5)})

        with pytest.raises(ValueError, match="n_bins"):
            density_bins(path, n_bins=n_bins)
