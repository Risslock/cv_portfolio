"""Mask-aware augmentation: curated-donor-bank copy-paste for synthetic training data.

Ported from `notebooks/03_explore_segmentation.ipynb`. Design rationale is in
docs/adr/0014-copy-paste-donor-bank-design.md (donor curation, disk-cached bank,
scene-relative resize) and docs/adr/0015-color-aware-donor-compositing.md (LAB-space
color matching) — not repeated here.

Layout:
  - Atomic building blocks (`crop_to_mask_bbox` through `copy_paste_compose`, plus the
    donor-bank functions): plain functions over arrays/COCO objects.
  - `add_synthetic_donors`: the training-usable entry point. `mask_to_yolo_polygon`/
    `read_yolo_seg_masks` convert between its masks and YOLO-seg label lines.
  - `build_or_reuse_donor_bank`/`remove_donor_bank`: safe bank lifecycle management,
    also exposed as a CLI (`python -m poultry_monitoring.augmentation.segmentation
    build-bank`/`remove-bank`) — see `_build_arg_parser` for the full flag list.

This module augments one in-memory sample; `segmentation.synthetic_data` wires it into
an offline synthetic dataset (images + labels on disk).
"""

import argparse
import hashlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pycocotools.mask as mask_utils
from PIL import Image
from pycocotools.coco import COCO

# Donor-side orientation, fixed per ADR 0014: overhead imagery has no canonical "up",
# so any rotation is plausible. Scale is handled separately (sample_domain_scale_factor
# below), not folded into this Affine. fit_output=True stops a near-45-degree rotation
# from clipping at the crop's original edges; crop_to_mask_bbox trims the grown canvas
# back down afterward.
DONOR_GEOMETRIC_AUGMENT = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(rotate=(-180, 180), fit_output=True, p=1.0),
    ]
)


def crop_to_mask_bbox(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trim `image`/`mask` to the tight bounding box of `mask`'s nonzero pixels.

    Args:
        image: RGB image, same H, W as `mask`.
        mask: Binary mask, nonzero where the donor actually is.

    Returns:
        Tuple of (trimmed image, trimmed mask). Unchanged if `mask` is entirely empty.
    """
    ys, xs = np.where(mask)
    if ys.size == 0:
        return image, mask
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def polygon_areas(polygons: np.ndarray | list[np.ndarray]) -> np.ndarray:
    """Shoelace area of each polygon, in pixel units.

    The polygon-native replacement for rasterize-then-count. Rasterizing every instance
    just to measure it is O(N x H x W); this is O(N x points), which is what makes
    per-sample use at training time affordable — see docs/adr/0017.

    Args:
        polygons: Either a stacked `(N, P, 2)` array or a list of `(P, 2)` arrays, in
            pixel coordinates.

    Returns:
        1D array of absolute areas, one per polygon.
    """
    if isinstance(polygons, np.ndarray) and polygons.ndim == 3:
        x, y = polygons[..., 0], polygons[..., 1]
        cross = x * np.roll(y, -1, axis=-1) - np.roll(x, -1, axis=-1) * y
        return 0.5 * np.abs(cross.sum(axis=-1))

    areas = []
    for poly in polygons:
        poly = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        if len(poly) < 3:
            areas.append(0.0)
            continue
        x, y = poly[:, 0], poly[:, 1]
        areas.append(0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))))
    return np.array(areas, dtype=np.float64)


def polygon_sizes(polygons: np.ndarray | list[np.ndarray]) -> np.ndarray:
    """Linear size metric (sqrt of polygon area) per instance — `instance_sizes`' fast path.

    Args:
        polygons: Pixel-coordinate polygons, see `polygon_areas`.

    Returns:
        1D array of sqrt(area) values, one per polygon (0.0 for degenerate ones).
    """
    return np.sqrt(polygon_areas(polygons))


def polygon_centers(polygons: np.ndarray | list[np.ndarray]) -> np.ndarray:
    """Bounding-box center of each polygon, for spatial-locality filtering.

    Args:
        polygons: Pixel-coordinate polygons, see `polygon_areas`.

    Returns:
        `(N, 2)` array of `(x, y)` centers.
    """
    if isinstance(polygons, np.ndarray) and polygons.ndim == 3:
        return (polygons.min(axis=1) + polygons.max(axis=1)) / 2.0
    return np.array(
        [
            (np.asarray(p).reshape(-1, 2).min(axis=0) + np.asarray(p).reshape(-1, 2).max(axis=0))
            / 2.0
            for p in polygons
        ],
        dtype=np.float64,
    ).reshape(-1, 2)


def rasterize_polygons(
    polygons: np.ndarray | list[np.ndarray], height: int, width: int
) -> np.ndarray:
    """Fill every polygon into one shared binary canvas.

    One union canvas serves as both the placement-occupancy map and the colour-sampling
    mask, instead of one full-size mask per instance.

    Polygons are drawn one call at a time on purpose. Handing `cv2.fillPoly` the whole
    list at once applies an even-odd winding rule across contours, so two *overlapping*
    instances cancel and leave a hole exactly where the scene is densest — which on
    high-occlusion ChickenVerse frames would both invite placements on top of a crowd and
    drop those pixels from the colour statistics. Verified directly: two overlapping
    squares sum to 480 px in one call vs. 651 looped. Per-contour still costs
    O(total instance area), not the O(N x H x W) of a mask-per-instance approach.

    Args:
        polygons: Pixel-coordinate polygons, see `polygon_areas`.
        height: Canvas height.
        width: Canvas width.

    Returns:
        `(height, width)` uint8 canvas, 1 wherever any polygon covers.
    """
    canvas = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        contour = np.asarray(poly, dtype=np.float64).reshape(-1, 2).astype(np.int32)
        if len(contour) >= 3:
            cv2.fillPoly(canvas, [contour], 1)
    return canvas


def _pooled_lab_stats(
    lab_image: np.ndarray, region: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-channel LAB mean/std over `region`'s pixels, or `None` if the region is empty."""
    if not region.any():
        return None
    pixels = lab_image[region]
    return pixels.mean(axis=0), pixels.std(axis=0)


def local_color_stats(
    lab_image: np.ndarray,
    instance_mask: np.ndarray,
    point: tuple[float, float],
    radius: float,
    min_pixels: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """LAB mean/std of real-instance pixels *near* `point`, falling back to the whole frame.

    Locality matters because a mosaic canvas stitches 4 different scenes (and so 4
    lighting conditions) into one frame — matching a donor against the whole canvas would
    aim it at an average of facilities rather than its actual neighbourhood. This also
    closes the local-lighting limitation docs/adr/0015 recorded as known-but-unfixed.

    Implemented by slicing a window out of the pre-computed LAB image and union mask, so
    it costs O(radius^2) and needs no extra rasterization per donor.

    Args:
        lab_image: The scene already converted to LAB (converted once per sample).
        instance_mask: Union of the scene's *real* instances, from `rasterize_polygons`.
        point: `(x, y)` location the donor is going to.
        radius: Half-width of the neighbourhood window, in pixels.
        min_pixels: If the window holds fewer instance pixels than this, fall back to the
            whole frame rather than trust a tiny sample.

    Returns:
        Tuple of (LAB mean, LAB std), each length-3. Defaults to `(0, 1)` when the frame
        has no instance pixels at all — treat as "no signal", same as `masked_pixel_stats`.
    """
    h, w = instance_mask.shape
    x, y = point
    x0, x1 = max(0, int(x - radius)), min(w, int(x + radius))
    y0, y1 = max(0, int(y - radius)), min(h, int(y + radius))

    window = instance_mask[y0:y1, x0:x1].astype(bool)
    if window.sum() >= min_pixels:
        stats = _pooled_lab_stats(lab_image[y0:y1, x0:x1], window)
        if stats is not None:
            return stats

    stats = _pooled_lab_stats(lab_image, instance_mask.astype(bool))
    return stats if stats is not None else (np.zeros(3, np.float32), np.ones(3, np.float32))


def local_sizes(
    sizes: np.ndarray,
    centers: np.ndarray,
    point: tuple[float, float],
    radius: float,
    min_neighbors: int = 3,
) -> np.ndarray:
    """Instance sizes within `radius` of `point`, falling back to all of them.

    Size counterpart to `local_color_stats` — a donor should be scaled like the birds
    actually around it, which on a mosaic canvas is not the same as the canvas average.

    Args:
        sizes: Per-instance sizes (see `polygon_sizes`), computed once per sample.
        centers: Matching `(N, 2)` instance centers (see `polygon_centers`).
        point: `(x, y)` location the donor is going to.
        radius: Neighbourhood radius in pixels.
        min_neighbors: Below this many neighbours the local sample isn't trustworthy, so
            all instances are returned instead.

    Returns:
        1D array of sizes — the local subset, or all of `sizes` if too few neighbours.
    """
    if sizes.size == 0 or centers.size == 0:
        return sizes
    distances = np.linalg.norm(centers - np.asarray(point, dtype=np.float64), axis=1)
    nearby = sizes[distances <= radius]
    return nearby if nearby.size >= min_neighbors else sizes


def instance_sizes(masks: list[np.ndarray]) -> np.ndarray:
    """Linear size metric (sqrt of mask area) for each instance mask.

    Mask-native, so inherently O(N x H x W) — fine for the offline path, but training-time
    callers should use `polygon_sizes` instead.

    Args:
        masks: List of binary instance masks.

    Returns:
        1D array of sqrt(area) values, one per non-empty mask.
    """
    return np.array([np.sqrt(area) for m in masks if (area := m.astype(bool).sum()) > 0])


def sample_domain_scale_factor(
    donor_mask: np.ndarray,
    target_sizes: np.ndarray,
    rng: np.random.Generator,
    jitter: float = 0.15,
) -> float:
    """Sample a resize factor bringing a donor's size in line with the target scene's own sizes.

    Draws a reference size from `Normal(mean(target_sizes), std(target_sizes))` — see
    ADR 0014 for why (target scene's own spread, not the donor's native scale).

    Args:
        donor_mask: The donor's current binary mask (before resizing).
        target_sizes: Reference sizes (sqrt(area), see `instance_sizes`) from the target
            scene's own instances.
        rng: Random generator for the reference draw.
        jitter: Minimum reference-size spread, as a fraction of the mean — a floor under
            `target_sizes`'s own std, for scenes with too few instances to estimate it.

    Returns:
        Scale factor to resize the donor by (1.0 = no resize). Returns 1.0 if
        `target_sizes` is empty or the donor mask is empty.
    """
    donor_size = np.sqrt(donor_mask.astype(bool).sum())
    if donor_size == 0 or target_sizes.size == 0:
        return 1.0

    mean_size = float(np.mean(target_sizes))
    std_size = max(float(np.std(target_sizes)), jitter * mean_size)
    reference_size = max(rng.normal(mean_size, std_size), 0.25 * mean_size)
    return float(reference_size / donor_size)


def resize_donor(
    image: np.ndarray, mask: np.ndarray, scale_factor: float
) -> tuple[np.ndarray, np.ndarray]:
    """Resize a donor crop + mask by `scale_factor`.

    Args:
        image: Donor RGB crop.
        mask: Donor binary mask, same H, W as `image`.
        scale_factor: Multiplicative resize factor (1.0 = unchanged). Floored so a
            pathological factor can't collapse the donor below a few pixels.

    Returns:
        Tuple of (resized image, resized mask). Mask uses nearest-neighbor
        interpolation to stay binary; image uses area interpolation when shrinking,
        linear when enlarging.
    """
    h, w = mask.shape
    scale_factor = max(scale_factor, 4 / max(h, w, 1))
    new_w, new_h = max(1, round(w * scale_factor)), max(1, round(h * scale_factor))

    interp = cv2.INTER_AREA if scale_factor < 1 else cv2.INTER_LINEAR
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=interp)
    resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return resized_image, resized_mask


def masked_pixel_stats(image: np.ndarray, masks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel LAB mean and std of an image's pixels, pooled across instance masks.

    Args:
        image: RGB image the masks index into.
        masks: One binary mask per instance to pool pixels from.

    Returns:
        Tuple of (per-channel LAB mean, per-channel LAB std), each length-3. Defaults to
        `(0, 1)` if no mask has any nonzero pixels — treat as "no signal."

    Notes:
        Pixels are pooled through a single union mask rather than indexed per instance.
        Overlapping pixels therefore count once instead of twice, which is a negligible
        (arguably more correct) difference for a pooled mean/std, and turns an
        O(N x H x W) pass into O(H x W).
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    if not masks:
        return np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)
    union = np.logical_or.reduce([m.astype(bool) for m in masks])
    stats = _pooled_lab_stats(lab, union)
    return stats if stats is not None else (np.zeros(3, np.float32), np.ones(3, np.float32))


def match_color_to_target(
    donor_image: np.ndarray,
    donor_mask: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    strength: float = 1.0,
    std_floor: float = 5.0,
    to_lab: int = cv2.COLOR_RGB2LAB,
    from_lab: int = cv2.COLOR_LAB2RGB,
) -> np.ndarray:
    """Shift/scale a donor's colors toward a target LAB mean+std (Reinhard-style transfer).

    Args:
        donor_image: Donor crop (whole crop — only mask pixels get pasted downstream, so
            recoloring the rest is harmless), in the colorspace `to_lab` expects.
        donor_mask: The donor's own binary mask, used to compute *its* current stats.
        target_mean: Target per-channel LAB mean, from `masked_pixel_stats`.
        target_std: Target per-channel LAB std, from `masked_pixel_stats`.
        strength: 1.0 = full statistical match, 0.0 = no change, in between blends.
        std_floor: Minimum donor std per channel before dividing by it — guards a
            near-solid-color donor crop from an exploding scale factor.
        to_lab: cv2 conversion code into LAB. Defaults to RGB; pass
            `cv2.COLOR_BGR2LAB` when working inside Ultralytics' pipeline, which is BGR
            until `Format` flips it at the very end.
        from_lab: cv2 conversion code back out of LAB — must be `to_lab`'s inverse.

    Returns:
        A recolored copy of `donor_image`, same shape, dtype (uint8) and colorspace.
    """
    region = donor_mask.astype(bool)
    if not region.any():
        return donor_image

    lab = cv2.cvtColor(donor_image, to_lab).astype(np.float32)
    donor_mean = lab[region].mean(axis=0)
    donor_std = np.maximum(lab[region].std(axis=0), std_floor)

    matched = (lab - donor_mean) * (target_std / donor_std) + target_mean
    blended = lab + strength * (matched - lab)
    blended_uint8 = np.clip(blended, 0, 255).astype(np.uint8)
    return cv2.cvtColor(blended_uint8, from_lab)


def find_non_overlapping_offset(
    occupied_mask: np.ndarray,
    donor_mask: np.ndarray,
    rng: np.random.Generator,
    max_overlap_ratio: float = 0.15,
    max_attempts: int = 30,
) -> tuple[int, int] | None:
    """Rejection-sample a (y, x) top-left offset for `donor_mask` that avoids `occupied_mask`.

    Args:
        occupied_mask: Binary canvas (base image's H, W) marking already-covered pixels
            (the base image's own instances, or donors already pasted this call).
        donor_mask: Binary mask of the donor crop (donor crop's own H, W).
        rng: Random generator to draw candidate offsets from.
        max_overlap_ratio: Largest fraction of the donor's own mask area allowed to land
            on already-occupied pixels before an offset is rejected.
        max_attempts: How many random offsets to try before giving up.

    Returns:
        (y, x) top-left offset, or `None` if no attempt met the overlap budget within
        `max_attempts` — caller decides whether to skip this donor.
    """
    canvas_h, canvas_w = occupied_mask.shape
    donor_h, donor_w = donor_mask.shape
    donor_area = donor_mask.sum()
    if donor_area == 0 or donor_h > canvas_h or donor_w > canvas_w:
        return None

    for _ in range(max_attempts):
        y = int(rng.integers(0, canvas_h - donor_h + 1))
        x = int(rng.integers(0, canvas_w - donor_w + 1))
        overlap = np.logical_and(occupied_mask[y : y + donor_h, x : x + donor_w], donor_mask).sum()
        if overlap / donor_area <= max_overlap_ratio:
            return y, x
    return None


def copy_paste_compose(
    base_image: np.ndarray,
    base_masks: list[np.ndarray],
    donor_fn: Callable[[], tuple[np.ndarray, np.ndarray, int]],
    rng: np.random.Generator,
    n_donors: int,
    donor_augment: A.Compose | None = None,
    target_sizes: np.ndarray | None = None,
    size_jitter: float = 0.15,
    target_color_stats: tuple[np.ndarray, np.ndarray] | None = None,
    color_strength: float = 1.0,
    max_overlap_ratio: float = 0.15,
    max_attempts: int = 30,
) -> tuple[np.ndarray, list[np.ndarray], list[int], int]:
    """Paste up to `n_donors` donors (from `donor_fn`) onto `base_image`, avoiding heavy overlap.

    Each donor goes through, in order: `donor_augment` (orientation) -> color match
    toward `target_color_stats` -> resize toward `target_sizes` (scale) ->
    `find_non_overlapping_offset` (placement).

    Args:
        base_image: RGB image to paste onto.
        base_masks: The base image's own instance masks — the starting occupied region.
        donor_fn: Zero-arg callable returning one (donor crop, donor mask, category id)
            triple per call — e.g. `sample_donor_from_bank` bound to a bank/rng.
        rng: Random generator, passed through to `find_non_overlapping_offset` and
            `sample_domain_scale_factor`.
        n_donors: How many donors to attempt to paste.
        donor_augment: Optional Albumentations pipeline (`image`/`mask` targets)
            applied to each donor before color/resizing/placement — e.g.
            `DONOR_GEOMETRIC_AUGMENT`. Each donor is re-cropped to its transformed
            mask's bounding box afterward via `crop_to_mask_bbox`. `None` skips it.
        target_sizes: Optional reference sizes (sqrt(area), see `instance_sizes`) to
            resize each donor toward via `sample_domain_scale_factor`. `None` skips
            resizing.
        size_jitter: See `sample_domain_scale_factor`.
        target_color_stats: Optional `(mean, std)` LAB pair (see `masked_pixel_stats`)
            to color-match each donor toward via `match_color_to_target`. `None` skips
            color matching.
        color_strength: See `match_color_to_target`.
        max_overlap_ratio: See `find_non_overlapping_offset`.
        max_attempts: See `find_non_overlapping_offset`.

    Returns:
        Tuple of (composed image, list of pasted donor masks, list of pasted donor
        category ids [same order as the masks], number of donors skipped because no
        placement met the overlap budget within `max_attempts`).
    """
    occupied = np.zeros(base_image.shape[:2], dtype=bool)
    for m in base_masks:
        occupied |= m.astype(bool)

    composed_image = base_image.copy()
    pasted_masks: list[np.ndarray] = []
    pasted_category_ids: list[int] = []
    skipped = 0

    for _ in range(n_donors):
        donor_crop, donor_mask, donor_category_id = donor_fn()

        if donor_augment is not None:
            augmented = donor_augment(image=donor_crop, mask=donor_mask)
            donor_crop, donor_mask = crop_to_mask_bbox(augmented["image"], augmented["mask"])

        if target_color_stats is not None:
            target_mean, target_std = target_color_stats
            donor_crop = match_color_to_target(
                donor_crop, donor_mask, target_mean, target_std, strength=color_strength
            )

        if target_sizes is not None and target_sizes.size > 0:
            scale_factor = sample_domain_scale_factor(
                donor_mask, target_sizes, rng, jitter=size_jitter
            )
            donor_crop, donor_mask = resize_donor(donor_crop, donor_mask, scale_factor)

        donor_mask = donor_mask.astype(bool)
        offset = find_non_overlapping_offset(
            occupied, donor_mask, rng, max_overlap_ratio, max_attempts
        )
        if offset is None:
            skipped += 1
            continue

        y, x = offset
        h, w = donor_mask.shape
        region = (slice(y, y + h), slice(x, x + w))
        composed_image[region][donor_mask] = donor_crop[donor_mask]

        full_mask = np.zeros(base_image.shape[:2], dtype=bool)
        full_mask[region] = donor_mask
        occupied |= full_mask
        pasted_masks.append(full_mask)
        pasted_category_ids.append(donor_category_id)

    return composed_image, pasted_masks, pasted_category_ids, skipped


def find_unoccluded_untruncated_donor_instances(
    coco: COCO,
    min_area_ratio: float = 0.6,
    min_mask_area: int = 70,
    border_margin: int = 80,
) -> list[dict]:
    """Find COCO annotations that make good copy-paste donors.

    Not occluded, not truncated at the image border, not a tiny sliver. Uses
    `mask_utils.area` directly on the RLE (no full pixel decode) so scanning a whole
    split stays fast. Threshold rationale is in ADR 0014.

    Args:
        coco: COCO object to scan.
        min_area_ratio: Minimum mask-area / box-area ratio to accept.
        min_mask_area: Minimum mask area in pixels, to skip near-invisible slivers.
        border_margin: Pixels of tolerance before a bbox touching the image edge is
            treated as truncated and rejected.

    Returns:
        List of `{"image_id", "ann_id"}` dicts for each accepted instance.
    """
    candidates = []
    for img_id in coco.getImgIds():
        img_info = coco.loadImgs(img_id)[0]
        img_w, img_h = img_info["width"], img_info["height"]

        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
            seg = ann.get("segmentation")
            if seg is None or (isinstance(seg, list) and len(seg) == 0):
                continue

            x, y, w, h = ann["bbox"]
            if x <= border_margin or y <= border_margin:
                continue
            if x + w >= img_w - border_margin or y + h >= img_h - border_margin:
                continue

            if isinstance(seg, list):
                rle = mask_utils.frPyObjects(seg, img_h, img_w)
                mask_area = float(mask_utils.area(rle).sum())
            elif isinstance(seg, dict):
                rle = mask_utils.frPyObjects([seg], seg["size"][0], seg["size"][1])
                mask_area = float(mask_utils.area(rle)[0])
            else:
                continue

            box_area = w * h
            if mask_area < min_mask_area or box_area <= 0 or mask_area / box_area < min_area_ratio:
                continue

            candidates.append({"image_id": img_id, "ann_id": ann["id"]})
    return candidates


def _decode_donor_mask(coco: COCO, ann: dict) -> np.ndarray:
    """Decode a COCO segmentation (polygon or compressed RLE) into a binary mask."""
    seg = ann["segmentation"]
    if isinstance(seg, list):
        return coco.annToMask(ann)
    rle = mask_utils.frPyObjects([seg], seg["size"][0], seg["size"][1])
    return mask_utils.decode(rle)[:, :, 0]


def build_donor_bank(
    coco: COCO,
    img_dir: Path,
    bank_dir: Path,
    max_donors: int = 500,
    min_area_ratio: float = 0.6,
    min_mask_area: int = 70,
    border_margin: int = 80,
    seed: int | None = 0,
) -> list[dict]:
    """Curate donor instances and cache their crop + mask to disk as a reusable bank.

    Scans `coco` via `find_unoccluded_untruncated_donor_instances`, takes a random
    subset (up to `max_donors`), and writes each donor's cropped image to
    `bank_dir/images/<donor_id>.png` and its mask to `bank_dir/masks/<donor_id>.png`
    (single-channel, 0/255) — separate image/mask directories with matching basenames,
    the layout most ML/CV tooling expects, rather than co-mingled in one flat directory.
    Also writes a `manifest.json`.

    Args:
        coco: COCO object to curate donors from.
        img_dir: Directory the raw images live in.
        bank_dir: Where to write the bank (created if missing).
        max_donors: Cap on how many curated donors to materialize to disk.
        min_area_ratio: See `find_unoccluded_untruncated_donor_instances`.
        min_mask_area: See `find_unoccluded_untruncated_donor_instances`.
        border_margin: See `find_unoccluded_untruncated_donor_instances`.
        seed: Seeds the random subset chosen from the full candidate pool.

    Returns:
        The manifest list (also written to `bank_dir/manifest.json`): one dict per
        donor with `donor_id`, `category_id`, `h`, `w`.
    """
    images_dir = bank_dir / "images"
    masks_dir = bank_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    candidates = find_unoccluded_untruncated_donor_instances(
        coco, min_area_ratio, min_mask_area, border_margin
    )
    print(
        f"{len(candidates)} unoccluded, untruncated candidate instance(s) found "
        f"(of {len(coco.getAnnIds())} total)."
    )

    rng = np.random.default_rng(seed)
    n_take = min(max_donors, len(candidates))
    chosen_idx = rng.choice(len(candidates), size=n_take, replace=False)

    manifest = []
    for i in chosen_idx:
        c = candidates[int(i)]
        img_info = coco.loadImgs(c["image_id"])[0]
        ann = coco.loadAnns([c["ann_id"]])[0]
        image = np.array(Image.open(img_dir / img_info["file_name"]).convert("RGB"))
        mask = _decode_donor_mask(coco, ann)
        x, y, w, h = (int(v) for v in ann["bbox"])
        crop, crop_mask = image[y : y + h, x : x + w], mask[y : y + h, x : x + w]

        donor_id = f"ann{c['ann_id']}"
        Image.fromarray(crop).save(images_dir / f"{donor_id}.png")
        Image.fromarray((crop_mask.astype(np.uint8)) * 255).save(masks_dir / f"{donor_id}.png")
        manifest.append({"donor_id": donor_id, "category_id": ann["category_id"], "h": h, "w": w})

    (bank_dir / "manifest.json").write_text(json.dumps(manifest))
    print(f"Donor bank: {len(manifest)} donor(s) written to {bank_dir}")
    return manifest


def load_donor_bank(bank_dir: Path) -> list[dict]:
    """Load a donor bank's manifest, written by `build_donor_bank`.

    Args:
        bank_dir: Directory `build_donor_bank` wrote the bank to.

    Returns:
        The manifest list.

    Raises:
        FileNotFoundError: No `manifest.json` at `bank_dir` — run `build_donor_bank` first.
    """
    manifest_path = bank_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No donor bank manifest at {manifest_path} -- run build_donor_bank first."
        )
    return json.loads(manifest_path.read_text())


# Bump whenever curation/build logic changes (e.g. a different PNG format, on-disk
# layout, or manifest shape) -- folded into every bank fingerprint below, so a logic
# change alone flags an old bank as stale even when its source annotations and build
# params didn't change. v2: split into images/+masks/ subfolders (previously flat,
# <id>.png + <id>_mask.png side by side).
BANK_FORMAT_VERSION = 2
BUILD_FINGERPRINT_FILENAME = "build_fingerprint.json"


def _annotations_signature(path: Path) -> str:
    """Cheap fingerprint of a source annotations file: name + size + mtime, not content."""
    stat = path.stat()
    return f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}"


def _bank_build_fingerprint(
    annotations_path: Path,
    min_area_ratio: float,
    min_mask_area: int,
    border_margin: int,
    max_donors: int,
    seed: int | None,
) -> str:
    """Fingerprint everything that determines a donor bank's contents."""
    payload = {
        "version": BANK_FORMAT_VERSION,
        "source": _annotations_signature(annotations_path),
        "min_area_ratio": min_area_ratio,
        "min_mask_area": min_mask_area,
        "border_margin": border_margin,
        "max_donors": max_donors,
        "seed": seed,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def remove_donor_bank(bank_dir: Path, assume_yes: bool = False) -> None:
    """Delete a donor bank directory entirely.

    Destructive and irreversible (re-run `build_donor_bank`/`build_or_reuse_donor_bank`
    to regenerate) — asks for confirmation unless `assume_yes=True`, same convention as
    `data.coco.fix_iscrowd_field`.

    Args:
        bank_dir: Directory to remove. No-op if it doesn't exist.
        assume_yes: Skip the interactive confirmation prompt (for non-interactive runs).
    """
    if not bank_dir.exists():
        return

    if not assume_yes:
        n_files = sum(1 for _ in bank_dir.iterdir())
        print(f"This will permanently delete {bank_dir} ({n_files} file(s)).")
        if input("Proceed? (y/n): ").lower() != "y":
            print("Operation cancelled by user.")
            return

    shutil.rmtree(bank_dir)


def build_or_reuse_donor_bank(
    annotations_path: Path,
    img_dir: Path,
    bank_dir: Path,
    max_donors: int = 500,
    min_area_ratio: float = 0.6,
    min_mask_area: int = 70,
    border_margin: int = 80,
    seed: int | None = 0,
    force: bool = False,
) -> list[dict]:
    """Build a donor bank if needed, or reuse an existing one that still matches its inputs.

    Guards against a mismatched bank — e.g. crop/mask PNGs left over from a build with
    different curation thresholds, or a `max_donors` that shrank since the last build —
    by fingerprinting the source annotations file plus every curation/build parameter
    (`_bank_build_fingerprint`). A bank directory that already has content but no
    matching fingerprint is never written into in place; `force=True` deletes it first
    (`remove_donor_bank`) so a rebuild always starts from an empty directory.

    Args:
        annotations_path: Path to the source COCO instances json to curate donors from.
        img_dir: Directory the raw images live in.
        bank_dir: Where the bank lives (or will be written).
        max_donors: See `build_donor_bank`.
        min_area_ratio: See `find_unoccluded_untruncated_donor_instances`.
        min_mask_area: See `find_unoccluded_untruncated_donor_instances`.
        border_margin: See `find_unoccluded_untruncated_donor_instances`.
        seed: See `build_donor_bank`.
        force: Wipe and rebuild from scratch even if a matching bank already exists.

    Returns:
        The bank manifest — freshly built, or reused unchanged.

    Raises:
        FileExistsError: `bank_dir` already has content that doesn't match the requested
            source/curation parameters (or has no recorded fingerprint to check against)
            and `force` is False. Call again with `force=True`, or `remove_donor_bank`
            it first, rather than risk silently mixing old and new donor files.
    """
    fingerprint = _bank_build_fingerprint(
        annotations_path, min_area_ratio, min_mask_area, border_margin, max_donors, seed
    )
    bank_has_content = bank_dir.exists() and any(bank_dir.iterdir())

    if bank_has_content and not force:
        fingerprint_path = bank_dir / BUILD_FINGERPRINT_FILENAME
        cached_fingerprint = (
            json.loads(fingerprint_path.read_text()).get("fingerprint")
            if fingerprint_path.exists()
            else None
        )
        if cached_fingerprint == fingerprint:
            return load_donor_bank(bank_dir)
        raise FileExistsError(
            f"{bank_dir} already holds a donor bank that doesn't match the requested "
            f"source/curation parameters (or has no recorded fingerprint at all). Call "
            f"again with force=True, or remove_donor_bank(bank_dir) first."
        )

    if bank_dir.exists():
        print(f"Rebuilding {bank_dir} from scratch (force=True).")
        remove_donor_bank(bank_dir, assume_yes=True)

    coco = COCO(str(annotations_path))
    manifest = build_donor_bank(
        coco, img_dir, bank_dir, max_donors, min_area_ratio, min_mask_area, border_margin, seed
    )
    (bank_dir / BUILD_FINGERPRINT_FILENAME).write_text(json.dumps({"fingerprint": fingerprint}))
    return manifest


def sample_donor_from_bank(
    bank_dir: Path, manifest: list[dict], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, int]:
    """Sample one donor crop + mask + category id from a pre-built bank on disk.

    Args:
        bank_dir: Directory `build_donor_bank` wrote the bank to.
        manifest: Bank manifest, from `build_donor_bank`/`load_donor_bank`.
        rng: Random generator to pick the donor with.

    Returns:
        Tuple of (donor crop, donor mask, category id) — `copy_paste_compose`'s
        `donor_fn` contract.
    """
    entry = manifest[int(rng.integers(0, len(manifest)))]
    crop = np.array(Image.open(bank_dir / "images" / f"{entry['donor_id']}.png").convert("RGB"))
    mask_image = Image.open(bank_dir / "masks" / f"{entry['donor_id']}.png").convert("L")
    mask = (np.array(mask_image) > 0).astype(np.uint8)
    return crop, mask, entry["category_id"]


def mask_to_yolo_polygon(mask: np.ndarray, category_id: int, img_w: int, img_h: int) -> str | None:
    """Convert one binary instance mask to a YOLO-seg label line.

    Same contour-extraction approach as `data.coco`'s RLE-to-polygon preprocessing
    (`RETR_EXTERNAL` + `approxPolyDP`, ~0.5px tolerance, docs/adr/0013), applied to a
    raw mask array instead of a COCO RLE dict.

    Args:
        mask: Binary instance mask, shape `(img_h, img_w)`.
        category_id: YOLO class index to prefix the line with.
        img_w: Image width, for normalizing coordinates to `[0, 1]`.
        img_h: Image height, for normalizing coordinates to `[0, 1]`.

    Returns:
        A `"class x1 y1 x2 y2 ..."` label line (normalized, space-separated), using the
        largest contour if the mask decomposes into more than one — a pasted donor is a
        single silhouette, not a multi-part object. `None` if the mask is empty or no
        contour with at least 3 points could be extracted.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    polygon = cv2.approxPolyDP(largest, 0.5, True).reshape(-1, 2)
    if len(polygon) < 3:
        return None

    normalized = polygon.astype(np.float64)
    normalized[:, 0] /= img_w
    normalized[:, 1] /= img_h
    coords = " ".join(f"{v:.6f}" for v in normalized.flatten())
    return f"{category_id} {coords}"


def yolo_polygon_line_to_mask(line: str, img_w: int, img_h: int) -> tuple[np.ndarray, int] | None:
    """Convert one YOLO-seg label line back into a binary mask + category id.

    Inverse of `mask_to_yolo_polygon`.

    Args:
        line: One `"class x1 y1 x2 y2 ..."` label line (normalized, space-separated).
        img_w: Image width, to de-normalize coordinates.
        img_h: Image height, to de-normalize coordinates.

    Returns:
        Tuple of (binary mask, category id), or `None` if the line doesn't have at
        least a class id and 3 (x, y) points.
    """
    parts = line.split()
    if len(parts) < 7:  # class + at least 3 (x, y) points
        return None

    category_id = int(float(parts[0]))
    coords = np.array(parts[1:], dtype=np.float64).reshape(-1, 2)
    coords[:, 0] *= img_w
    coords[:, 1] *= img_h

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, [coords.astype(np.int32)], 1)
    return mask, category_id


def read_yolo_seg_masks(
    label_path: Path, img_w: int, img_h: int
) -> tuple[list[np.ndarray], list[int]]:
    """Read a YOLO-seg label file into binary masks + category ids.

    Args:
        label_path: Path to a YOLO-seg `.txt` label file. Missing file returns nothing.
        img_w: Image width, to de-normalize coordinates.
        img_h: Image height, to de-normalize coordinates.

    Returns:
        Tuple of (masks, category ids), same order/length, one pair per label line.
    """
    if not label_path.exists():
        return [], []

    masks: list[np.ndarray] = []
    category_ids: list[int] = []
    for line in label_path.read_text().splitlines():
        parsed = yolo_polygon_line_to_mask(line, img_w, img_h)
        if parsed is None:
            continue
        mask, category_id = parsed
        masks.append(mask)
        category_ids.append(category_id)
    return masks, category_ids


def add_synthetic_donors(
    image: np.ndarray,
    masks: list[np.ndarray],
    category_ids: list[int],
    bank_dir: Path,
    manifest: list[dict],
    rng: np.random.Generator,
    p_copy_paste: float,
    max_donors: int,
    size_jitter: float = 0.15,
    color_strength: float = 1.0,
    max_overlap_ratio: float = 0.15,
    max_attempts: int = 30,
) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
    """Training-time entry point: probabilistically paste curated donors onto one sample.

    A single `Bernoulli(p_copy_paste)` draw decides whether this call pastes anything;
    if it fires, the donor count is drawn uniformly from `1..max_donors`. Donor
    orientation (`DONOR_GEOMETRIC_AUGMENT`) and scene-relative size/color matching (ADR
    0014/0015) are always applied together when donors are pasted — not optional here,
    since both are the whole reason a pasted donor looks plausible rather than obviously
    fake. Pair with `mask_to_yolo_polygon` to turn the returned masks into label lines.

    Args:
        image: RGB training image to (maybe) paste onto.
        masks: The image's own ground-truth instance masks.
        category_ids: Category id per entry of `masks` (same order/length) — together
            with `masks`, defines the target scale/color reference population.
        bank_dir: Curated donor bank directory, from `build_donor_bank`.
        manifest: Bank manifest, from `build_donor_bank`/`load_donor_bank`.
        rng: Random generator — draws the activation coin flip, donor count, donor
            identities, placement, and geometric augmentation, all from one source for a
            fully reproducible call.
        p_copy_paste: Probability this call pastes anything at all.
        max_donors: Inclusive upper bound on the donor count drawn when activated. Must
            be >= 1.
        size_jitter: See `sample_domain_scale_factor`.
        color_strength: See `match_color_to_target`.
        max_overlap_ratio: See `find_non_overlapping_offset`.
        max_attempts: See `find_non_overlapping_offset`.

    Returns:
        Tuple of `(image, masks, category_ids)`. Unchanged copies of the inputs if the
        Bernoulli draw doesn't fire or the bank is empty; base + pasted donors' masks/
        category ids appended (same order, `masks[i]` <-> `category_ids[i]` throughout)
        otherwise. A donor that couldn't find a non-overlapping placement is simply
        skipped, so pasting fewer than `max_donors` is expected, not an error.

    Raises:
        ValueError: `max_donors < 1`.
    """
    if max_donors < 1:
        raise ValueError(f"max_donors must be >= 1, got {max_donors}")

    if rng.random() >= p_copy_paste or not manifest:
        return image.copy(), list(masks), list(category_ids)

    n_donors = int(rng.integers(1, max_donors + 1))
    target_sizes = instance_sizes(masks)
    target_color_stats = masked_pixel_stats(image, masks)

    composed_image, pasted_masks, pasted_category_ids, _skipped = copy_paste_compose(
        image,
        masks,
        donor_fn=lambda: sample_donor_from_bank(bank_dir, manifest, rng),
        rng=rng,
        n_donors=n_donors,
        donor_augment=DONOR_GEOMETRIC_AUGMENT,
        target_sizes=target_sizes,
        size_jitter=size_jitter,
        target_color_stats=target_color_stats,
        color_strength=color_strength,
        max_overlap_ratio=max_overlap_ratio,
        max_attempts=max_attempts,
    )
    return composed_image, [*masks, *pasted_masks], [*category_ids, *pasted_category_ids]


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the `python -m poultry_monitoring.augmentation.segmentation` CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build-bank", help="Curate a donor bank from a COCO annotations file."
    )
    build_parser.add_argument(
        "--annotations", type=Path, required=True, help="Source COCO instances_*.json."
    )
    build_parser.add_argument(
        "--img-dir", type=Path, required=True, help="Directory the raw images live in."
    )
    build_parser.add_argument(
        "--bank-dir", type=Path, required=True, help="Where to write/read the bank."
    )
    build_parser.add_argument("--max-donors", type=int, default=500)
    build_parser.add_argument("--min-area-ratio", type=float, default=0.6)
    build_parser.add_argument("--min-mask-area", type=int, default=70)
    build_parser.add_argument("--border-margin", type=int, default=80)
    build_parser.add_argument("--seed", type=int, default=0)
    build_parser.add_argument(
        "--force",
        action="store_true",
        help="Wipe and rebuild even if a matching bank already exists.",
    )

    remove_parser = subparsers.add_parser("remove-bank", help="Delete a donor bank directory.")
    remove_parser.add_argument("--bank-dir", type=Path, required=True)
    remove_parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")

    return parser


def main() -> None:
    """CLI entry point — see `_build_arg_parser` for `--help` on each subcommand."""
    args = _build_arg_parser().parse_args()

    if args.command == "build-bank":
        manifest = build_or_reuse_donor_bank(
            args.annotations,
            args.img_dir,
            args.bank_dir,
            max_donors=args.max_donors,
            min_area_ratio=args.min_area_ratio,
            min_mask_area=args.min_mask_area,
            border_margin=args.border_margin,
            seed=args.seed,
            force=args.force,
        )
        print(f"Bank ready: {len(manifest)} donor(s) at {args.bank_dir}")
    elif args.command == "remove-bank":
        remove_donor_bank(args.bank_dir, assume_yes=args.yes)


if __name__ == "__main__":
    main()
