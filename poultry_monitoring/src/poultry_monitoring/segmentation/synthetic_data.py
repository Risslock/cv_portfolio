"""Materialize a synthetic-augmented training split to disk — the **fallback** path.

The default for Phase 3 Stage B is `segmentation/copy_paste_training.py`, which pastes
donors per-sample inside the training loop so they differ every epoch and nothing is
written to disk. This module is kept as the documented plan B: if on-the-fly compositing
ever turns into a dataloader bottleneck, generating one large synthetic dataset up front
and training on it normally is a perfectly good trade. See docs/adr/0017 for the
comparison. It is also handy for eyeballing a fixed, inspectable set of composites.

For each real image in a split it optionally pastes curated donors (`add_synthetic_donors`)
and, when that fires, writes the composited image + a full YOLO-seg label file (original
instances + pasted donors, via `mask_to_yolo_polygon`) into a *separate*
`images/<output_split>/` + `labels/<output_split>/` tree — the real split's own
images/labels are never touched. `data.coco.build_data_yaml` then combines real +
synthetic via Ultralytics' list-valued `train:`, so nothing is merged on disk either.

Compositing itself comes from `augmentation/segmentation.py`, shared with the on-the-fly
path rather than reimplemented — see
docs/adr/0014-copy-paste-donor-bank-design.md / docs/adr/0015-color-aware-donor-compositing.md.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO

from poultry_monitoring.augmentation.segmentation import (
    add_synthetic_donors,
    load_donor_bank,
    mask_to_yolo_polygon,
    read_yolo_seg_masks,
)
from poultry_monitoring.data.coco import (
    CLASS_NAMES,
    _directory_fingerprint,
    _file_signature,
    build_data_yaml,
    coco_category_id_to_yolo_class_id,
)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Bump whenever this module's generation logic changes -- folded into every split
# fingerprint below, so a logic change alone flags an old synthetic split as stale.
SYNTHETIC_FORMAT_VERSION = 1
# Sibling-file convention, matching data.coco's own cache-manifest naming
# (`<name><MANIFEST_FILENAME>`) rather than a file inside the output dirs themselves.
SYNTHETIC_MANIFEST_SUFFIX = ".cache_manifest.json"


def _images_dir(data_dir: Path, split: str) -> Path:
    return data_dir / "images" / split


def _labels_dir(data_dir: Path, split: str) -> Path:
    return data_dir / "labels" / split


def _remap_manifest_category_ids(
    manifest: list[dict], category_id_map: dict[int, int]
) -> list[dict]:
    """Rewrite a donor bank manifest's `category_id` from COCO ids to YOLO class ids.

    The donor bank stores each donor's raw COCO `category_id` (see
    `augmentation.segmentation.build_donor_bank`); real YOLO-seg label files use
    0-indexed class ids instead. Pasting a donor straight from the bank into a real
    label file without this remap silently mislabels it (checked directly against
    ChickenDet's own data: bank donors are `category_id=1`, real labels use class `0`).

    Args:
        manifest: Bank manifest, from `augmentation.segmentation.load_donor_bank`.
        category_id_map: `{coco_category_id: yolo_class_id}`, from
            `data.coco.coco_category_id_to_yolo_class_id`.

    Returns:
        A new manifest list with every entry's `category_id` remapped.
    """
    return [{**entry, "category_id": category_id_map[entry["category_id"]]} for entry in manifest]


def generate_synthetic_sample(
    image: np.ndarray,
    label_path: Path,
    bank_dir: Path,
    manifest: list[dict],
    rng: np.random.Generator,
    p_copy_paste: float,
    max_donors: int,
) -> tuple[np.ndarray, list[str], int] | None:
    """Maybe paste donors onto one real image; return the full label set if anything was pasted.

    Args:
        image: RGB training image.
        label_path: The image's existing YOLO-seg label file.
        bank_dir: Curated donor bank directory.
        manifest: Bank manifest with `category_id` already remapped to YOLO class ids
            (see `_remap_manifest_category_ids`) — passed straight through to
            `add_synthetic_donors`.
        rng: Random generator — see `add_synthetic_donors`.
        p_copy_paste: See `add_synthetic_donors`.
        max_donors: See `add_synthetic_donors`.

    Returns:
        `(composited image, label lines, donors pasted)` if at least one donor was
        pasted, where `label lines` covers every instance (original + pasted); `None`
        if nothing changed (the `Bernoulli(p_copy_paste)` draw didn't fire, or no
        placement succeeded) — the caller skips writing anything for this image.
    """
    img_h, img_w = image.shape[:2]
    masks, category_ids = read_yolo_seg_masks(label_path, img_w, img_h)

    composed_image, new_masks, new_category_ids = add_synthetic_donors(
        image, masks, category_ids, bank_dir, manifest, rng, p_copy_paste, max_donors
    )
    n_pasted = len(new_masks) - len(masks)
    if n_pasted == 0:
        return None

    lines = [
        line
        for mask, category_id in zip(new_masks, new_category_ids)
        if (line := mask_to_yolo_polygon(mask, category_id, img_w, img_h)) is not None
    ]
    return composed_image, lines, n_pasted


def _synthetic_split_fingerprint(
    data_dir: Path,
    annotations_path: Path,
    bank_dir: Path,
    source_split: str,
    p_copy_paste: float,
    max_donors: int,
    seed: int | None,
) -> str:
    """Fingerprint everything that determines a synthetic split's contents.

    Reuses `data.coco`'s own file/directory signature helpers (not duplicated a third
    time, after the donor bank's own fingerprinting) — see docs/adr/0010 for the
    precedent of reusing a leading-underscore helper across modules when it fits.
    """
    payload = {
        "version": SYNTHETIC_FORMAT_VERSION,
        "source_images": _directory_fingerprint(_images_dir(data_dir, source_split), pattern="*"),
        "source_labels": _directory_fingerprint(_labels_dir(data_dir, source_split), pattern="*"),
        "annotations": _file_signature(annotations_path),
        "bank_manifest": _file_signature(bank_dir / "manifest.json"),
        "p_copy_paste": p_copy_paste,
        "max_donors": max_donors,
        "seed": seed,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def remove_synthetic_split(data_dir: Path, output_split: str, assume_yes: bool = False) -> None:
    """Delete a synthetic split's images/labels directories (+ its fingerprint file).

    Destructive and irreversible (re-run `generate_synthetic_split` to regenerate) —
    same confirm-unless-`assume_yes` convention as
    `augmentation.segmentation.remove_donor_bank`.

    Args:
        data_dir: ChickenDet dataset root.
        output_split: Split name to remove, e.g. `"TrainSynthetic"`.
        assume_yes: Skip the interactive confirmation prompt (for non-interactive runs).
    """
    images_dir = _images_dir(data_dir, output_split)
    labels_dir = _labels_dir(data_dir, output_split)
    fingerprint_path = data_dir / f"{output_split}{SYNTHETIC_MANIFEST_SUFFIX}"

    if not images_dir.exists() and not labels_dir.exists():
        return

    if not assume_yes:
        n_files = sum(1 for d in (images_dir, labels_dir) if d.exists() for _ in d.iterdir())
        print(f"This will permanently delete synthetic split '{output_split}' ({n_files} file(s)).")
        if input("Proceed? (y/n): ").lower() != "y":
            print("Operation cancelled by user.")
            return

    shutil.rmtree(images_dir, ignore_errors=True)
    shutil.rmtree(labels_dir, ignore_errors=True)
    fingerprint_path.unlink(missing_ok=True)


def generate_synthetic_split(
    data_dir: Path,
    annotations_path: Path,
    bank_dir: Path,
    source_split: str = "Train",
    output_split: str = "TrainSynthetic",
    p_copy_paste: float = 0.3,
    max_donors: int = 5,
    seed: int | None = 0,
    force: bool = False,
) -> Path:
    """Build (or reuse) a synthetic-augmented split from `source_split`'s real images.

    Fingerprinted the same way `augmentation.segmentation.build_or_reuse_donor_bank`
    guards the donor bank itself: an existing output split that doesn't match the
    requested inputs is never written into in place — `force=True` wipes it first.

    Args:
        data_dir: ChickenDet dataset root.
        annotations_path: Source COCO instances json — used to build the
            COCO-category-id -> YOLO-class-id remap (`coco_category_id_to_yolo_class_id`)
            and folded into the fingerprint.
        bank_dir: Curated donor bank directory (`build_or_reuse_donor_bank`'s output).
        source_split: Real image split to augment (must already have `images/<split>` +
            `labels/<split>`).
        output_split: Name of the new split this writes — `images/<output_split>/` +
            `labels/<output_split>/`.
        p_copy_paste: See `add_synthetic_donors`.
        max_donors: See `add_synthetic_donors`.
        seed: Seeds one independent child generator per image
            (`np.random.default_rng(seed).spawn(n)`), so the whole run is reproducible
            without every image sharing one mutable rng state.
        force: Wipe and regenerate even if a matching output split already exists.

    Returns:
        Path to `data_dir/images/<output_split>` (labels live at the equivalent
        `labels/<output_split>` path).

    Raises:
        FileExistsError: The output split already has content that doesn't match the
            requested inputs and `force` is False.
    """
    output_images_dir = _images_dir(data_dir, output_split)
    output_labels_dir = _labels_dir(data_dir, output_split)
    fingerprint_path = data_dir / f"{output_split}{SYNTHETIC_MANIFEST_SUFFIX}"

    fingerprint = _synthetic_split_fingerprint(
        data_dir, annotations_path, bank_dir, source_split, p_copy_paste, max_donors, seed
    )
    already_built = output_images_dir.exists() and any(output_images_dir.iterdir())

    if already_built and not force:
        cached_fingerprint = (
            json.loads(fingerprint_path.read_text()).get("fingerprint")
            if fingerprint_path.exists()
            else None
        )
        if cached_fingerprint == fingerprint:
            print(f"Reusing existing synthetic split at {output_images_dir}.")
            return output_images_dir
        raise FileExistsError(
            f"{output_images_dir} already holds a synthetic split that doesn't match "
            f"the requested source/params (or has no recorded fingerprint at all). Call "
            f"again with force=True, or remove_synthetic_split(data_dir, output_split) first."
        )

    if output_images_dir.exists() or output_labels_dir.exists():
        print(f"Rebuilding '{output_split}' from scratch (force=True).")
        remove_synthetic_split(data_dir, output_split, assume_yes=True)
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    coco = COCO(str(annotations_path))
    category_id_map = coco_category_id_to_yolo_class_id(coco)
    manifest = _remap_manifest_category_ids(load_donor_bank(bank_dir), category_id_map)

    source_images_dir = _images_dir(data_dir, source_split)
    source_labels_dir = _labels_dir(data_dir, source_split)
    image_paths = sorted(
        p for p in source_images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    rngs = np.random.default_rng(seed).spawn(len(image_paths))

    n_augmented, n_instances_pasted = 0, 0
    for image_path, rng in zip(image_paths, rngs):
        label_path = source_labels_dir / f"{image_path.stem}.txt"
        image = np.array(Image.open(image_path).convert("RGB"))

        result = generate_synthetic_sample(
            image, label_path, bank_dir, manifest, rng, p_copy_paste, max_donors
        )
        if result is None:
            continue

        composed_image, lines, n_pasted = result
        Image.fromarray(composed_image).save(output_images_dir / image_path.name)
        (output_labels_dir / f"{image_path.stem}.txt").write_text("\n".join(lines))
        n_augmented += 1
        n_instances_pasted += n_pasted

    print(
        f"Synthetic split '{output_split}': {n_augmented}/{len(image_paths)} image(s) "
        f"augmented, {n_instances_pasted} donor instance(s) pasted, written to "
        f"{output_images_dir}."
    )
    fingerprint_path.write_text(json.dumps({"fingerprint": fingerprint}))
    return output_images_dir


def build_stage_b_data_yaml(
    data_dir: Path, output_split: str = "TrainSynthetic", yaml_path: Path | None = None
) -> Path:
    """Write a Stage B data.yaml combining the real Train split with a synthetic one.

    Args:
        data_dir: ChickenDet dataset root.
        output_split: Synthetic split name, from `generate_synthetic_split`.
        yaml_path: Destination — defaults to `data_dir/chickendet_stage_b.yaml`.

    Returns:
        The written yaml's path.
    """
    yaml_path = yaml_path if yaml_path is not None else data_dir / "chickendet_stage_b.yaml"
    return build_data_yaml(data_dir, CLASS_NAMES, yaml_path, train_splits=("Train", output_split))


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the `python -m poultry_monitoring.segmentation.synthetic_data` CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser(
        "generate", help="Build (or reuse) a synthetic-augmented split + Stage B data.yaml."
    )
    generate_parser.add_argument(
        "--data-dir", type=Path, required=True, help="ChickenDet dataset root."
    )
    generate_parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Source COCO instances json (for the COCO->YOLO category id remap).",
    )
    generate_parser.add_argument(
        "--bank-dir", type=Path, required=True, help="Curated donor bank directory."
    )
    generate_parser.add_argument("--source-split", default="Train")
    generate_parser.add_argument("--output-split", default="TrainSynthetic")
    generate_parser.add_argument("--p-copy-paste", type=float, default=0.3)
    generate_parser.add_argument("--max-donors", type=int, default=5)
    generate_parser.add_argument("--seed", type=int, default=0)
    generate_parser.add_argument(
        "--force",
        action="store_true",
        help="Wipe and regenerate even if a matching split already exists.",
    )

    remove_parser = subparsers.add_parser("remove", help="Delete a synthetic split.")
    remove_parser.add_argument("--data-dir", type=Path, required=True)
    remove_parser.add_argument("--output-split", default="TrainSynthetic")
    remove_parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")

    return parser


def main() -> None:
    """CLI entry point — see `_build_arg_parser` for `--help` on each subcommand."""
    args = _build_arg_parser().parse_args()

    if args.command == "generate":
        generate_synthetic_split(
            args.data_dir,
            args.annotations,
            args.bank_dir,
            source_split=args.source_split,
            output_split=args.output_split,
            p_copy_paste=args.p_copy_paste,
            max_donors=args.max_donors,
            seed=args.seed,
            force=args.force,
        )
        yaml_path = build_stage_b_data_yaml(args.data_dir, args.output_split)
        print(f"Stage B data.yaml written to {yaml_path}")
    elif args.command == "remove":
        remove_synthetic_split(args.data_dir, args.output_split, assume_yes=args.yes)


if __name__ == "__main__":
    main()
