"""Training-time copy-paste from the curated donor bank, as an Ultralytics transform.

Pastes donors into each training sample **in memory**, so the donors differ every epoch
and nothing is materialized to disk. `segmentation/synthetic_data.py` is the offline
fallback for the same idea; see docs/adr/0017 for why this is the default and why the
other candidate hooks don't work.

This module owns all the Ultralytics coupling (`Instances`, `YOLODataset`,
`SegmentationTrainer`); the compositing primitives it calls are framework-free and live
in `augmentation/segmentation.py`.

Three pieces, each a thin override:
  - `BankCopyPaste` — the transform, inserted after mosaic/affine so it runs once per
    sample on the final canvas.
  - `DonorBankYOLODataset` — overrides `build_transforms` to insert it. Overriding the
    builder (rather than mutating `dataset.transforms` from a callback) is required:
    `close_mosaic` rebuilds the chain mid-training and would otherwise silently drop it.
  - `DonorBankSegmentationTrainer` — overrides `build_dataset`, since
    `build_yolo_dataset` hardcodes the dataset class.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.utils import colorstr
from ultralytics.utils.instance import Instances
from ultralytics.utils.ops import resample_segments
from ultralytics.utils.torch_utils import unwrap_model

from poultry_monitoring.augmentation.segmentation import (
    DONOR_GEOMETRIC_AUGMENT,
    crop_to_mask_bbox,
    find_non_overlapping_offset,
    load_donor_bank,
    local_color_stats,
    local_sizes,
    match_color_to_target,
    polygon_centers,
    polygon_sizes,
    rasterize_polygons,
    resize_donor,
    sample_domain_scale_factor,
)

# Neighbourhood radius (px, on the training canvas) for the local scene statistics. At
# the default 640 imgsz a mosaic quadrant is 320px, so 160 keeps a donor matched to its
# own quadrant rather than the 4-scene average -- see docs/adr/0017.
DEFAULT_LOCALITY_RADIUS = 160.0


class BankCopyPaste:
    """Paste curated donor-bank instances into a training sample, labels included.

    One `Bernoulli(p)` draw per sample; when it fires, `1..max_donors` donors are pasted
    and appended to `labels["instances"]`/`labels["cls"]` as real segmentation labels.

    Attributes:
        bank_dir: Donor bank directory.
        manifest: Bank manifest entries, with `category_id` already in YOLO class-id space.
        p: Probability of pasting anything into a given sample.
        max_donors: Inclusive upper bound on the per-sample donor count.
    """

    def __init__(
        self,
        bank_dir: Path,
        manifest: list[dict],
        p: float = 0.3,
        max_donors: int = 5,
        locality_radius: float = DEFAULT_LOCALITY_RADIUS,
        size_jitter: float = 0.15,
        color_strength: float = 1.0,
        max_overlap_ratio: float = 0.15,
        max_attempts: int = 30,
        donor_pool_size: int = 400,
    ) -> None:
        """Initialize the transform.

        Args:
            bank_dir: Donor bank directory, from `augmentation.segmentation.build_donor_bank`.
            manifest: Bank manifest whose `category_id`s are YOLO class ids (remap COCO
                ids first — see `data.coco.coco_category_id_to_yolo_class_id`).
            p: Probability of pasting anything into a given sample.
            max_donors: Inclusive upper bound on the donor count drawn when active.
            locality_radius: Neighbourhood radius for local size/colour stats.
            size_jitter: See `sample_domain_scale_factor`.
            color_strength: See `match_color_to_target`.
            max_overlap_ratio: See `find_non_overlapping_offset`.
            max_attempts: See `find_non_overlapping_offset`.
            donor_pool_size: How many donors each dataloader worker keeps resident in
                RAM. Reading a donor from disk costs ~9 ms here versus ~0.01 ms cached
                (Windows per-file-open overhead; warming the OS page cache does not help),
                which made uncached loading ~47% of the transform's runtime — so the pool
                is fully cached rather than partially. Caching the whole 2000-donor bank
                would cost ~135 MB *per worker*, so each worker instead draws its own
                random slice: the model still sees the full bank across workers, at
                `donor_pool_size / len(manifest)` of the memory.
        """
        self.bank_dir = Path(bank_dir)
        self.manifest = manifest
        self.p = p
        self.max_donors = max_donors
        self.locality_radius = locality_radius
        self.size_jitter = size_jitter
        self.color_strength = color_strength
        self.max_overlap_ratio = max_overlap_ratio
        self.max_attempts = max_attempts
        self.donor_pool_size = donor_pool_size
        self._pool: list[dict] | None = None
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _donor_pool(self) -> list[dict]:
        """This worker's random slice of the bank, drawn once on first use.

        Deliberately lazy: dataloader workers are spawned *after* the transform is
        constructed, and each seeds `random` differently, so every worker ends up with a
        different slice and the bank stays fully covered across them.
        """
        if self._pool is None:
            self._pool = random.sample(
                self.manifest, k=min(self.donor_pool_size, len(self.manifest))
            )
        return self._pool

    def _load_donor(self, entry: dict) -> tuple[np.ndarray, np.ndarray]:
        """Load one donor's crop + mask, BGR to match Ultralytics' in-pipeline colorspace."""
        donor_id = entry["donor_id"]
        cached = self._cache.get(donor_id)
        if cached is None:
            crop = cv2.imread(str(self.bank_dir / "images" / f"{donor_id}.png"), cv2.IMREAD_COLOR)
            mask = cv2.imread(
                str(self.bank_dir / "masks" / f"{donor_id}.png"), cv2.IMREAD_GRAYSCALE
            )
            if crop is None or mask is None:
                raise FileNotFoundError(f"Donor {donor_id} missing from bank at {self.bank_dir}")
            # Importing ultralytics monkeypatches cv2.imread globally with a version that
            # "always ensures 3 dimensions" (ultralytics/utils/patches.py), so
            # IMREAD_GRAYSCALE hands back (H, W, 1) rather than (H, W). Squeeze it back or
            # every downstream mask op silently works in the wrong rank -- see docs/adr/0017.
            if mask.ndim == 3:
                mask = mask[..., 0]
            cached = (crop, (mask > 0).astype(np.uint8))  # imread gives BGR already
            self._cache[donor_id] = cached  # bounded by the pool size, so no eviction needed
        return cached[0].copy(), cached[1].copy()

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        """Paste donors into one sample, appending their labels.

        Args:
            labels: Ultralytics sample dict with `img`, `cls` and `instances`.

        Returns:
            The same dict, with pasted donors composited into `img` and their bboxes,
            classes and 1000-point segments appended to `instances`/`cls`.
        """
        if not self.manifest or self.p <= 0 or random.random() >= self.p:
            return labels

        instances = labels["instances"]
        if instances.segments.size == 0:
            return labels  # nothing to measure the scene against

        image = labels["img"]
        h, w = image.shape[:2]

        # Remember the incoming convention so this transform is order-independent.
        was_normalized = instances.normalized
        original_format = instances._bboxes.format
        instances.convert_bbox(format="xyxy")
        instances.denormalize(w, h)

        n_donors = random.randint(1, self.max_donors)
        pasted, classes = self._paste_donors(image, instances, n_donors, h, w)

        if pasted:
            instances = Instances.concatenate([instances, *pasted], axis=0)
            labels["cls"] = np.concatenate(
                [
                    labels["cls"].reshape(-1, 1),
                    np.array(classes, dtype=labels["cls"].dtype).reshape(-1, 1),
                ],
                axis=0,
            )

        instances.convert_bbox(format=original_format)
        if was_normalized:
            instances.normalize(w, h)
        labels["instances"] = instances
        return labels

    def _paste_donors(
        self,
        image: np.ndarray,
        instances: Instances,
        n_donors: int,
        h: int,
        w: int,
    ) -> tuple[list[Instances], list[int]]:
        """Composite donors into `image` in place.

        Returns:
            Tuple of (one `Instances` per pasted donor, their class ids in the same order).
        """
        segments = instances.segments  # pixel coords, (N, P, 2)
        real_mask = rasterize_polygons(segments, h, w)
        occupied = real_mask.copy()  # mutable: also blocks previously pasted donors
        lab_scene = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        sizes = polygon_sizes(segments)
        centers = polygon_centers(segments)
        n_points = segments.shape[1]

        pasted: list[Instances] = []
        classes: list[int] = []

        pool = self._donor_pool()
        for _ in range(n_donors):
            entry = pool[random.randrange(len(pool))]
            crop, mask = self._load_donor(entry)

            augmented = DONOR_GEOMETRIC_AUGMENT(image=crop, mask=mask)
            crop, mask = crop_to_mask_bbox(augmented["image"], augmented["mask"])

            # Anchor the *size* draw on a provisional location, then place, then colour
            # match against wherever it actually landed (colour has no geometric effect).
            anchor = (random.uniform(0, w), random.uniform(0, h))
            scale = sample_domain_scale_factor(
                mask,
                local_sizes(sizes, centers, anchor, self.locality_radius),
                np.random.default_rng(random.randrange(2**32)),
                jitter=self.size_jitter,
            )
            crop, mask = resize_donor(crop, mask, scale)
            mask = mask.astype(bool)

            offset = find_non_overlapping_offset(
                occupied,
                mask,
                np.random.default_rng(random.randrange(2**32)),
                self.max_overlap_ratio,
                self.max_attempts,
            )
            if offset is None:
                continue

            y, x = offset
            dh, dw = mask.shape
            center = (x + dw / 2, y + dh / 2)
            target_mean, target_std = local_color_stats(
                lab_scene, real_mask, center, self.locality_radius
            )
            crop = match_color_to_target(
                crop,
                mask.astype(np.uint8),
                target_mean,
                target_std,
                strength=self.color_strength,
                to_lab=cv2.COLOR_BGR2LAB,
                from_lab=cv2.COLOR_LAB2BGR,
            )

            region = (slice(y, y + dh), slice(x, x + dw))
            image[region][mask] = crop[mask]
            occupied[region] |= mask

            donor_instances = self._instances_from_mask(mask, x, y, n_points)
            if donor_instances is None:
                continue
            pasted.append(donor_instances)
            classes.append(int(entry["category_id"]))

        return pasted, classes

    @staticmethod
    def _instances_from_mask(mask: np.ndarray, x: int, y: int, n_points: int) -> Instances | None:
        """Turn a pasted donor's mask into a one-instance `Instances` in canvas coordinates.

        The polygon is resampled to `n_points` so it can stack with the scene's own
        segments, which Ultralytics keeps as a fixed-width `(N, P, 2)` array.
        """
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        polygon = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
        if len(polygon) < 3:
            return None

        polygon += np.array([x, y], dtype=np.float32)  # crop-local -> canvas coords
        polygon = resample_segments([polygon], n=n_points)[0].astype(np.float32)

        bbox = np.array(
            [[polygon[:, 0].min(), polygon[:, 1].min(), polygon[:, 0].max(), polygon[:, 1].max()]],
            dtype=np.float32,
        )
        return Instances(bbox, polygon[None, ...], None, bbox_format="xyxy", normalized=False)


class DonorBankYOLODataset(YOLODataset):
    """`YOLODataset` that inserts `BankCopyPaste` into its training transform chain.

    The transform goes after the mosaic/affine `pre_transform` and before
    `Albumentations`, so it sees the final-size canvas exactly once per sample.
    """

    def __init__(self, *args, copy_paste_transform: BankCopyPaste | None = None, **kwargs):
        """Initialize, stashing the transform before `super().__init__` builds transforms."""
        self.copy_paste_transform = copy_paste_transform
        super().__init__(*args, **kwargs)

    def build_transforms(self, hyp: dict | None = None):
        """Build the stock chain, then splice the donor-bank paste in before Albumentations.

        Also runs on `close_mosaic`'s mid-training rebuild, which is exactly why the
        insertion lives here rather than in a callback.
        """
        transforms = super().build_transforms(hyp)
        if not self.augment or self.copy_paste_transform is None:
            return transforms

        names = [type(t).__name__ for t in transforms.transforms]
        # Fall back to "just before Format" (always last) rather than index 0 -- landing at
        # the front would put the paste *before* mosaic/affine and silently change what the
        # donor is scaled and colour-matched against.
        fallback = max(len(names) - 1, 0)
        index = names.index("Albumentations") if "Albumentations" in names else fallback
        transforms.insert(index, self.copy_paste_transform)
        return transforms


def remap_manifest_to_yolo_class_ids(manifest: list[dict], nc: int) -> list[dict]:
    """Map a donor bank's stored COCO category ids onto YOLO class indices.

    `build_donor_bank` records `ann["category_id"]` straight from COCO (ChickenDet's is
    `1`), but YOLO-seg labels are 0-indexed. Pasting a raw COCO id makes the loss index
    a class column that doesn't exist, which surfaces as an unreadable device-side assert
    deep inside the task-aligned assigner rather than a clear error — so this also
    validates the result. Mirrors `convert_coco`'s rule: categories sorted by id map to
    0, 1, 2, ... , and is a no-op on an already-remapped manifest.

    Args:
        manifest: Bank manifest from `load_donor_bank`.
        nc: Number of classes the model is training on.

    Returns:
        The manifest with `category_id` remapped to YOLO class indices.

    Raises:
        ValueError: A remapped id falls outside `[0, nc)`.
    """
    coco_ids = sorted({int(entry["category_id"]) for entry in manifest})
    mapping = {coco_id: index for index, coco_id in enumerate(coco_ids)}
    remapped = [{**e, "category_id": mapping[int(e["category_id"])]} for e in manifest]

    invalid = sorted({e["category_id"] for e in remapped if not 0 <= e["category_id"] < nc})
    if invalid:
        raise ValueError(
            f"Donor bank yields class id(s) {invalid} outside the model's nc={nc}. "
            f"The bank's COCO category ids {coco_ids} don't line up with the dataset's classes."
        )
    return remapped


class DonorBankSegmentationTrainer(SegmentationTrainer):
    """`SegmentationTrainer` whose training dataset pastes donor-bank instances.

    Settings live as **class** attributes, set by `make_donor_bank_trainer`, because
    `model.train()` constructs the trainer itself so there is no instance to configure
    first — and Ultralytics' `get_cfg` rejects unknown keys, so they can't ride in as
    `model.train()` overrides either.

    Attributes:
        copy_paste_bank: Donor bank directory, or `None` to behave exactly like stock.
        copy_paste_p: Probability of pasting into a given training sample.
        copy_paste_max_donors: Inclusive upper bound on donors per sample.
    """

    copy_paste_bank: Path | None = None
    copy_paste_p: float = 0.3
    copy_paste_max_donors: int = 5

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build the stock dataset, swapping in `DonorBankYOLODataset` for training.

        Validation is deliberately left untouched — synthetic instances belong in
        training only, never in the numbers a run is judged on.
        """
        if mode != "train" or not self.copy_paste_bank:
            return super().build_dataset(img_path, mode, batch)

        bank_dir = Path(self.copy_paste_bank)
        manifest = remap_manifest_to_yolo_class_ids(load_donor_bank(bank_dir), int(self.data["nc"]))
        transform = BankCopyPaste(
            bank_dir,
            manifest,
            p=self.copy_paste_p,
            max_donors=self.copy_paste_max_donors,
        )
        # Mirrors ultralytics' own build_yolo_dataset kwargs (data/build.py) -- it
        # hardcodes YOLODataset, so the class can't be injected through it.
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return DonorBankYOLODataset(
            copy_paste_transform=transform,
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=True,
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=gs,
            pad=0.0,
            prefix=colorstr("train: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction,
        )


def make_donor_bank_trainer(
    bank_dir: Path, p: float = 0.3, max_donors: int = 5
) -> type[DonorBankSegmentationTrainer]:
    """Build a trainer class with the donor-bank settings baked in.

    `model.train(trainer=...)` takes a *class* and instantiates it internally, so the
    configuration has to travel on the class rather than on an instance.

    Args:
        bank_dir: Curated donor bank directory.
        p: Probability of pasting into a given training sample.
        max_donors: Inclusive upper bound on donors per sample.

    Returns:
        A `DonorBankSegmentationTrainer` subclass ready to hand to `model.train`.
    """
    return type(
        "ConfiguredDonorBankSegmentationTrainer",
        (DonorBankSegmentationTrainer,),
        {
            "copy_paste_bank": Path(bank_dir),
            "copy_paste_p": p,
            "copy_paste_max_donors": max_donors,
        },
    )
