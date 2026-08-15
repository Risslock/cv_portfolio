"""Shared pytest helpers for the copy-paste tests.

A donor bank is needed by both the augmentation tests and the training-transform tests,
and building one takes a small COCO fixture — so it lives here once rather than being
re-derived per test module.
"""

import json

import numpy as np
import pytest
from PIL import Image
from pycocotools.coco import COCO

from poultry_monitoring.augmentation.segmentation import build_donor_bank


def square_mask(shape: tuple[int, int], y0: int, x0: int, size: int) -> np.ndarray:
    """Binary mask with one filled square."""
    mask = np.zeros(shape, dtype=np.uint8)
    mask[y0 : y0 + size, x0 : x0 + size] = 1
    return mask


def uncompressed_rle(mask: np.ndarray) -> dict:
    """Encode a binary mask as an uncompressed COCO RLE dict (ChickenDet's own variant)."""
    flat = np.asfortranarray(mask).flatten(order="F")
    counts: list[int] = []
    current, prev = 0, 0
    for v in flat:
        if v == prev:
            current += 1
        else:
            counts.append(current)
            current, prev = 1, v
    counts.append(current)
    return {"size": list(mask.shape), "counts": counts}


@pytest.fixture
def donor_bank(tmp_path):
    """Build a small on-disk donor bank.

    Returns:
        Tuple of (bank_dir, manifest) where the manifest's `category_id` is already in
        YOLO class-id space (0), as the training transform requires.
    """
    img_size, square_size, n_donors = 600, 60, 3
    img_dir = tmp_path / "donor_images"
    img_dir.mkdir()
    rng = np.random.default_rng(1)
    Image.fromarray(rng.integers(0, 255, (img_size, img_size, 3), dtype=np.uint8)).save(
        img_dir / "donor.png"
    )

    annotations = []
    for i in range(n_donors):
        x = 100 + i * (square_size + 90)
        donor_mask = square_mask((img_size, img_size), 100, x, square_size)
        annotations.append(
            {
                "id": i + 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [x, 100, square_size, square_size],
                "area": square_size**2,
                "iscrowd": 0,
                "segmentation": uncompressed_rle(donor_mask),
            }
        )
    annotations_path = tmp_path / "donor_instances.json"
    annotations_path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "donor.png", "height": img_size, "width": img_size}
                ],
                "annotations": annotations,
                "categories": [{"id": 1, "name": "Chicken"}],
            }
        )
    )

    bank_dir = tmp_path / "bank"
    manifest = build_donor_bank(
        COCO(str(annotations_path)), img_dir, bank_dir, max_donors=n_donors, seed=0
    )
    return bank_dir, [{**entry, "category_id": 0} for entry in manifest]
