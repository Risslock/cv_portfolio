"""Smoke tests for `poultry_monitoring.augmentation.segmentation`.

Constitution Principle VIII scope: deterministic, non-ML code — array shapes/values and
label-line formatting, not training/convergence.
"""

import json

import cv2
import numpy as np
import pytest
from PIL import Image
from pycocotools.coco import COCO

from poultry_monitoring.augmentation.segmentation import (
    BUILD_FINGERPRINT_FILENAME,
    add_synthetic_donors,
    build_donor_bank,
    build_or_reuse_donor_bank,
    copy_paste_compose,
    crop_to_mask_bbox,
    find_non_overlapping_offset,
    find_unoccluded_untruncated_donor_instances,
    instance_sizes,
    load_donor_bank,
    local_color_stats,
    local_sizes,
    mask_to_yolo_polygon,
    masked_pixel_stats,
    match_color_to_target,
    polygon_areas,
    polygon_centers,
    polygon_sizes,
    rasterize_polygons,
    remove_donor_bank,
    resize_donor,
    sample_domain_scale_factor,
    sample_donor_from_bank,
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


def _write_coco_json_with_n_donors(
    tmp_path, n: int = 1, img_size: int = 600, square_size: int = 60
) -> tuple:
    """Write a one-image, n-instance COCO json + matching image file.

    Returns (annotations_path, img_dir) -- a path, not a loaded `COCO` object; callers
    that need one build it themselves (see `_write_coco_with_one_donor` below).

    Instances are laid out in a row, each comfortably inside the default 80px
    `border_margin` and with a full mask/box area ratio, so all of them pass donor
    curation as-is.
    """
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (img_size, img_size, 3), dtype=np.uint8)
    Image.fromarray(image).save(img_dir / "img1.png")

    annotations = []
    for i in range(n):
        x = 100 + i * (square_size + 90)
        mask = _square_mask((img_size, img_size), 100, x, square_size)
        annotations.append(
            {
                "id": i + 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [x, 100, square_size, square_size],
                "area": square_size**2,
                "iscrowd": 0,
                "segmentation": _make_rle_segmentation(mask),
            }
        )

    coco_json = {
        "images": [{"id": 1, "file_name": "img1.png", "height": img_size, "width": img_size}],
        "annotations": annotations,
        "categories": [{"id": 1, "name": "Chicken"}],
    }
    annotations_path = tmp_path / "instances.json"
    annotations_path.write_text(json.dumps(coco_json))
    return annotations_path, img_dir


def _write_coco_with_one_donor(tmp_path, img_size: int = 300, square_size: int = 100) -> tuple:
    """Write a one-image, one-instance COCO json + matching image file; return (COCO, img_dir)."""
    annotations_path, img_dir = _write_coco_json_with_n_donors(
        tmp_path, n=1, img_size=img_size, square_size=square_size
    )
    return COCO(str(annotations_path)), img_dir


class TestCropToMaskBbox:
    def test_trims_to_nonzero_region(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        mask = _square_mask((20, 20), 5, 8, 4)

        cropped_image, cropped_mask = crop_to_mask_bbox(image, mask)

        assert cropped_image.shape == (4, 4, 3)
        assert cropped_mask.shape == (4, 4)

    def test_empty_mask_returns_unchanged(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        mask = np.zeros((20, 20), dtype=np.uint8)

        cropped_image, cropped_mask = crop_to_mask_bbox(image, mask)

        assert cropped_image.shape == image.shape
        assert cropped_mask.shape == mask.shape


def _square_polygon(x0: float, y0: float, side: float) -> np.ndarray:
    return np.array(
        [[x0, y0], [x0 + side, y0], [x0 + side, y0 + side], [x0, y0 + side]], dtype=np.float64
    )


class TestPolygonPrimitives:
    def test_shoelace_area_matches_rasterized_area(self):
        """The whole point of going polygon-native is that it agrees with rasterizing."""
        polygon = _square_polygon(10, 10, 40)

        shoelace = polygon_areas([polygon])[0]
        rasterized = rasterize_polygons([polygon], 100, 100).sum()

        assert shoelace == pytest.approx(1600.0)
        assert shoelace == pytest.approx(rasterized, rel=0.05)

    def test_areas_vectorize_over_a_stacked_array(self):
        stacked = np.stack([_square_polygon(0, 0, 10), _square_polygon(0, 0, 20)])

        areas = polygon_areas(stacked)

        assert areas == pytest.approx([100.0, 400.0])

    def test_sizes_are_sqrt_of_area(self):
        sizes = polygon_sizes([_square_polygon(0, 0, 30)])

        assert sizes[0] == pytest.approx(30.0)

    def test_degenerate_polygon_has_zero_area(self):
        assert polygon_areas([np.array([[0.0, 0.0], [1.0, 1.0]])])[0] == 0.0

    def test_centers_are_bbox_centers(self):
        centers = polygon_centers([_square_polygon(10, 20, 40)])

        assert centers[0] == pytest.approx([30.0, 40.0])

    def test_rasterize_unions_overlapping_polygons(self):
        """Overlaps must union, not accumulate — the canvas is used as a boolean mask."""
        first, second = _square_polygon(0, 0, 20), _square_polygon(10, 0, 20)

        combined = rasterize_polygons([first, second], 50, 50)
        expected = np.logical_or(
            rasterize_polygons([first], 50, 50), rasterize_polygons([second], 50, 50)
        )

        assert set(np.unique(combined)) <= {0, 1}
        assert np.array_equal(combined.astype(bool), expected)


class TestLocalSizes:
    def test_returns_only_nearby_instances(self):
        sizes = np.array([10.0, 11.0, 12.0, 99.0])
        centers = np.array([[10, 10], [20, 20], [30, 30], [500, 500]])

        nearby = local_sizes(sizes, centers, point=(20, 20), radius=50)

        assert 99.0 not in nearby
        assert len(nearby) == 3

    def test_falls_back_to_all_when_too_few_neighbors(self):
        sizes = np.array([10.0, 11.0, 12.0, 99.0])
        centers = np.array([[10, 10], [20, 20], [30, 30], [500, 500]])

        nearby = local_sizes(sizes, centers, point=(500, 500), radius=10, min_neighbors=3)

        assert len(nearby) == 4  # the lone neighbour isn't a trustworthy sample


class TestLocalColorStats:
    def test_prefers_the_local_neighborhood(self):
        """A donor near the dark cluster should match it, not the frame-wide average."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        image[:100] = 40  # dark half
        image[100:] = 220  # bright half
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        instance_mask = np.zeros((200, 200), dtype=np.uint8)
        instance_mask[10:60, 10:60] = 1  # instance in the dark half
        instance_mask[140:190, 140:190] = 1  # instance in the bright half

        dark_mean, _ = local_color_stats(lab, instance_mask, point=(35, 35), radius=60)
        bright_mean, _ = local_color_stats(lab, instance_mask, point=(165, 165), radius=60)

        assert dark_mean[0] < bright_mean[0]  # L channel separates the two neighbourhoods

    def test_falls_back_to_global_when_window_is_empty(self):
        image = np.full((200, 200, 3), 128, dtype=np.uint8)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        instance_mask = np.zeros((200, 200), dtype=np.uint8)
        instance_mask[10:60, 10:60] = 1

        mean, _ = local_color_stats(lab, instance_mask, point=(190, 190), radius=5)

        assert mean[0] > 0  # used the global instance pixels rather than returning nothing

    def test_no_instances_returns_no_signal_default(self):
        lab = np.zeros((50, 50, 3), dtype=np.float32)
        mean, std = local_color_stats(lab, np.zeros((50, 50), np.uint8), point=(25, 25), radius=10)

        assert np.array_equal(mean, np.zeros(3))
        assert np.array_equal(std, np.ones(3))


class TestInstanceSizes:
    def test_computes_sqrt_area_and_skips_empty_masks(self):
        masks = [_square_mask((10, 10), 0, 0, 4), np.zeros((10, 10), dtype=np.uint8)]

        sizes = instance_sizes(masks)

        assert sizes.shape == (1,)
        assert sizes[0] == pytest.approx(4.0)  # sqrt(16)


class TestSampleDomainScaleFactor:
    def test_empty_target_sizes_returns_one(self):
        donor_mask = _square_mask((10, 10), 0, 0, 4)
        rng = np.random.default_rng(0)

        assert sample_domain_scale_factor(donor_mask, np.array([]), rng) == 1.0

    def test_empty_donor_mask_returns_one(self):
        donor_mask = np.zeros((10, 10), dtype=np.uint8)
        rng = np.random.default_rng(0)

        assert sample_domain_scale_factor(donor_mask, np.array([10.0]), rng) == 1.0

    def test_scales_toward_target_mean_on_average(self):
        donor_mask = _square_mask((10, 10), 0, 0, 4)  # donor size = sqrt(16) = 4
        target_sizes = np.full(50, 8.0)  # tightly clustered target, mean/std = 8/0
        rng = np.random.default_rng(0)

        factors = [sample_domain_scale_factor(donor_mask, target_sizes, rng) for _ in range(200)]

        # reference size ~ Normal(8, floor std) / donor size 4 -> factor should center near 2
        assert np.mean(factors) == pytest.approx(2.0, rel=0.15)


class TestResizeDonor:
    def test_scales_shape_by_factor(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        mask = np.ones((10, 10), dtype=np.uint8)

        resized_image, resized_mask = resize_donor(image, mask, scale_factor=2.0)

        assert resized_image.shape == (20, 20, 3)
        assert resized_mask.shape == (20, 20)

    def test_floors_pathologically_small_factor(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        mask = np.ones((10, 10), dtype=np.uint8)

        resized_image, _ = resize_donor(image, mask, scale_factor=0.001)

        assert min(resized_image.shape[:2]) >= 4  # floored, not collapsed to ~0px


class TestMaskedPixelStats:
    def test_no_masks_returns_default_zero_one(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)

        mean, std = masked_pixel_stats(image, [])

        assert np.array_equal(mean, np.zeros(3))
        assert np.array_equal(std, np.ones(3))

    def test_pools_pixels_across_masks(self):
        image = np.full((10, 10, 3), 128, dtype=np.uint8)
        masks = [_square_mask((10, 10), 0, 0, 5), _square_mask((10, 10), 5, 5, 5)]

        mean, std = masked_pixel_stats(image, masks)

        assert mean.shape == (3,)
        assert std.shape == (3,)


class TestMatchColorToTarget:
    def test_shifts_mean_toward_target(self):
        donor_image = np.full((10, 10, 3), 50, dtype=np.uint8)
        donor_mask = np.ones((10, 10), dtype=np.uint8)
        target_image = np.full((10, 10, 3), 200, dtype=np.uint8)
        target_mean, target_std = masked_pixel_stats(target_image, [donor_mask])

        matched = match_color_to_target(donor_image, donor_mask, target_mean, target_std)

        assert matched.mean() > donor_image.mean()

    def test_empty_mask_returns_input_unchanged(self):
        donor_image = np.full((10, 10, 3), 50, dtype=np.uint8)
        donor_mask = np.zeros((10, 10), dtype=np.uint8)

        matched = match_color_to_target(donor_image, donor_mask, np.zeros(3), np.ones(3))

        assert matched is donor_image


class TestFindNonOverlappingOffset:
    def test_finds_offset_on_empty_canvas(self):
        occupied = np.zeros((50, 50), dtype=bool)
        donor_mask = np.ones((10, 10), dtype=bool)
        rng = np.random.default_rng(0)

        offset = find_non_overlapping_offset(occupied, donor_mask, rng)

        assert offset is not None
        y, x = offset
        assert 0 <= y <= 40 and 0 <= x <= 40

    def test_returns_none_when_fully_occupied(self):
        occupied = np.ones((50, 50), dtype=bool)
        donor_mask = np.ones((10, 10), dtype=bool)
        rng = np.random.default_rng(0)

        assert find_non_overlapping_offset(occupied, donor_mask, rng, max_attempts=5) is None


class TestCopyPasteCompose:
    def test_pastes_donors_and_carries_category_ids(self):
        base_image = np.zeros((100, 100, 3), dtype=np.uint8)
        donor_crop = np.full((10, 10, 3), 255, dtype=np.uint8)
        donor_mask = np.ones((10, 10), dtype=np.uint8)
        rng = np.random.default_rng(0)

        composed, pasted_masks, pasted_category_ids, skipped = copy_paste_compose(
            base_image,
            base_masks=[],
            donor_fn=lambda: (donor_crop, donor_mask, 0),
            rng=rng,
            n_donors=2,
        )

        assert skipped == 0
        assert len(pasted_masks) == 2
        assert pasted_category_ids == [0, 0]
        assert composed[pasted_masks[0]].mean() == pytest.approx(255)

    def test_skips_donor_with_no_valid_placement(self):
        base_image = np.zeros((10, 10, 3), dtype=np.uint8)
        donor_crop = np.full((10, 10, 3), 255, dtype=np.uint8)
        donor_mask = np.ones((10, 10), dtype=np.uint8)  # covers the whole tiny canvas
        base_masks = [np.ones((10, 10), dtype=bool)]  # already fully occupied
        rng = np.random.default_rng(0)

        _, pasted_masks, pasted_category_ids, skipped = copy_paste_compose(
            base_image,
            base_masks=base_masks,
            donor_fn=lambda: (donor_crop, donor_mask, 0),
            rng=rng,
            n_donors=1,
            max_attempts=5,
        )

        assert skipped == 1
        assert pasted_masks == []
        assert pasted_category_ids == []


class TestMaskToYoloPolygon:
    def test_converts_square_mask_to_normalized_polygon(self):
        mask = _square_mask((100, 100), 10, 10, 20)

        line = mask_to_yolo_polygon(mask, category_id=0, img_w=100, img_h=100)

        assert line is not None
        parts = line.split()
        assert parts[0] == "0"
        coords = list(map(float, parts[1:]))
        assert len(coords) >= 6  # at least 3 (x, y) points
        assert all(0.0 <= v <= 1.0 for v in coords)

    def test_empty_mask_returns_none(self):
        mask = np.zeros((100, 100), dtype=np.uint8)

        assert mask_to_yolo_polygon(mask, category_id=0, img_w=100, img_h=100) is None


class TestFindUnoccludedUntruncatedDonorInstances:
    def test_accepts_a_well_clear_instance(self, tmp_path):
        coco, _ = _write_coco_with_one_donor(tmp_path)

        candidates = find_unoccluded_untruncated_donor_instances(coco)

        assert candidates == [{"image_id": 1, "ann_id": 1}]

    def test_rejects_instance_too_close_to_border(self, tmp_path):
        coco, _ = _write_coco_with_one_donor(tmp_path)

        candidates = find_unoccluded_untruncated_donor_instances(coco, border_margin=500)

        assert candidates == []


class TestDonorBankRoundTrip:
    def test_build_load_and_sample(self, tmp_path):
        coco, img_dir = _write_coco_with_one_donor(tmp_path)
        bank_dir = tmp_path / "bank"

        manifest = build_donor_bank(coco, img_dir, bank_dir, max_donors=5, seed=0)

        assert len(manifest) == 1
        assert manifest[0]["category_id"] == 1

        loaded_manifest = load_donor_bank(bank_dir)
        assert loaded_manifest == manifest

        rng = np.random.default_rng(0)
        crop, mask, category_id = sample_donor_from_bank(bank_dir, loaded_manifest, rng)
        assert crop.shape == (100, 100, 3)
        assert mask.shape == (100, 100)
        assert category_id == 1

    def test_load_missing_bank_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_donor_bank(tmp_path / "nonexistent")


class TestRemoveDonorBank:
    def test_deletes_bank_directory(self, tmp_path):
        bank_dir = tmp_path / "bank"
        bank_dir.mkdir()
        (bank_dir / "manifest.json").write_text("[]")

        remove_donor_bank(bank_dir, assume_yes=True)

        assert not bank_dir.exists()

    def test_noop_on_missing_directory(self, tmp_path):
        remove_donor_bank(tmp_path / "nonexistent", assume_yes=True)  # should not raise


class TestBuildOrReuseDonorBank:
    def test_builds_fresh_when_no_existing_bank(self, tmp_path):
        annotations_path, img_dir = _write_coco_json_with_n_donors(tmp_path, n=3)
        bank_dir = tmp_path / "bank"

        manifest = build_or_reuse_donor_bank(
            annotations_path, img_dir, bank_dir, max_donors=3, seed=0
        )

        assert len(manifest) == 3
        assert (bank_dir / BUILD_FINGERPRINT_FILENAME).exists()

    def test_reuses_matching_bank_without_error(self, tmp_path):
        annotations_path, img_dir = _write_coco_json_with_n_donors(tmp_path, n=3)
        bank_dir = tmp_path / "bank"
        first = build_or_reuse_donor_bank(annotations_path, img_dir, bank_dir, max_donors=3, seed=0)

        second = build_or_reuse_donor_bank(
            annotations_path, img_dir, bank_dir, max_donors=3, seed=0
        )

        assert second == first

    def test_mismatched_params_without_force_raises(self, tmp_path):
        annotations_path, img_dir = _write_coco_json_with_n_donors(tmp_path, n=3)
        bank_dir = tmp_path / "bank"
        build_or_reuse_donor_bank(annotations_path, img_dir, bank_dir, max_donors=3, seed=0)

        with pytest.raises(FileExistsError):
            build_or_reuse_donor_bank(annotations_path, img_dir, bank_dir, max_donors=1, seed=0)

    def test_unrecognized_existing_directory_without_force_raises(self, tmp_path):
        """Content with no fingerprint file is treated as unverifiable, not trusted."""
        annotations_path, img_dir = _write_coco_json_with_n_donors(tmp_path, n=1)
        bank_dir = tmp_path / "bank"
        bank_dir.mkdir()
        (bank_dir / "leftover.png").write_bytes(b"not a real bank")

        with pytest.raises(FileExistsError):
            build_or_reuse_donor_bank(annotations_path, img_dir, bank_dir, max_donors=1, seed=0)

    def test_force_rebuild_leaves_no_stale_donor_files(self, tmp_path):
        annotations_path, img_dir = _write_coco_json_with_n_donors(tmp_path, n=3)
        bank_dir = tmp_path / "bank"
        build_or_reuse_donor_bank(annotations_path, img_dir, bank_dir, max_donors=3, seed=0)

        manifest = build_or_reuse_donor_bank(
            annotations_path, img_dir, bank_dir, max_donors=1, seed=0, force=True
        )

        assert len(manifest) == 1
        # 1 crop + 1 mask for the surviving donor -- no leftovers from the 3-donor build
        assert len(list((bank_dir / "images").glob("*.png"))) == 1
        assert len(list((bank_dir / "masks").glob("*.png"))) == 1


class TestAddSyntheticDonors:
    def test_zero_probability_is_a_noop(self, tmp_path):
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        masks = [_square_mask((50, 50), 5, 5, 5)]
        rng = np.random.default_rng(0)

        result_image, result_masks, result_category_ids = add_synthetic_donors(
            image, masks, [0], tmp_path, manifest=[], rng=rng, p_copy_paste=0.0, max_donors=3
        )

        assert np.array_equal(result_image, image)
        assert len(result_masks) == 1
        assert result_category_ids == [0]

    def test_empty_bank_is_a_noop_even_at_full_probability(self, tmp_path):
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        rng = np.random.default_rng(0)

        _, result_masks, result_category_ids = add_synthetic_donors(
            image, [], [], tmp_path, manifest=[], rng=rng, p_copy_paste=1.0, max_donors=3
        )

        assert result_masks == []
        assert result_category_ids == []

    def test_pastes_at_least_one_donor_when_activated(self, tmp_path):
        coco, img_dir = _write_coco_with_one_donor(tmp_path, img_size=300, square_size=100)
        bank_dir = tmp_path / "bank"
        manifest = build_donor_bank(coco, img_dir, bank_dir, max_donors=5, seed=0)

        # Large, mostly-empty base scene with one existing "real" instance, so
        # target size/color matching has something real to reference.
        base_image = np.random.default_rng(1).integers(0, 255, (400, 400, 3), dtype=np.uint8)
        base_masks = [_square_mask((400, 400), 300, 300, 50)]
        rng = np.random.default_rng(0)

        result_image, result_masks, result_category_ids = add_synthetic_donors(
            base_image,
            base_masks,
            [0],
            bank_dir,
            manifest,
            rng=rng,
            p_copy_paste=1.0,
            max_donors=2,
        )

        assert result_image.shape == base_image.shape
        assert len(result_masks) > len(base_masks)
        assert len(result_masks) == len(result_category_ids)

    def test_max_donors_below_one_raises(self, tmp_path):
        rng = np.random.default_rng(0)

        with pytest.raises(ValueError):
            add_synthetic_donors(
                np.zeros((10, 10, 3), dtype=np.uint8),
                [],
                [],
                tmp_path,
                manifest=[{"donor_id": "x", "category_id": 0, "h": 1, "w": 1}],
                rng=rng,
                p_copy_paste=1.0,
                max_donors=0,
            )
