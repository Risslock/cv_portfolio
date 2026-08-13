"""ChickenDet COCO annotation parsing and COCO -> YOLO label conversion.

Boxes and masks live in the same COCO-format annotation file per split, so this module
is shared between the detection and segmentation tasks (constitution Principle III)
rather than duplicated per task.

Productionized from `notebooks/02_yolo26_baseline.ipynb`'s exploration — see that
notebook's Notes section for the `convert_coco` gotchas this module works around — and
from `notebooks/03_explore_segmentation.ipynb`'s RLE-to-polygon preprocessing, see
`docs/adr/0013-rle-to-polygon-preprocessing-for-yolo-seg-conversion.md`.
"""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_utils
from ultralytics.data.converter import convert_coco


def fix_iscrowd_field(coco_path: Path, assume_yes: bool = False) -> None:
    """Zero out every annotation's `iscrowd` field in a COCO JSON, in place.

    `ultralytics.data.converter.convert_coco` silently drops any annotation with
    `iscrowd=1`. ChickenDet ships some regardless of whether the region is an actual
    crowd, so left as-is this quietly drops labels during COCO -> YOLO conversion. This
    is a destructive, irreversible edit to the source JSON (re-download from Zenodo to
    get the original back), so it is a no-op if every annotation is already
    `iscrowd=0`, writes a `.bak` copy alongside the original the first time it actually
    changes anything, and asks for confirmation before writing unless
    `assume_yes=True`.

    Args:
        coco_path: Path to a COCO-format instances JSON file.
        assume_yes: Skip the interactive confirmation prompt (for non-interactive runs).
    """
    with coco_path.open("r") as f:
        coco = json.load(f)

    already_fixed = all(ann["iscrowd"] == 0 for ann in coco["annotations"])
    if already_fixed:
        return

    if not assume_yes:
        n_crowd = sum(ann["iscrowd"] != 0 for ann in coco["annotations"])
        print(f"{coco_path.name}: {n_crowd} annotation(s) have iscrowd != 0.")
        print("This will overwrite them to 0, in place. Re-download from Zenodo to undo.")
        confirmation = input("Proceed? (y/n): ")
        if confirmation.lower() != "y":
            print("Operation cancelled by user.")
            return

    backup_path = coco_path.with_suffix(coco_path.suffix + ".bak")
    if not backup_path.exists():
        shutil.copy2(coco_path, backup_path)

    for ann in coco["annotations"]:
        ann["iscrowd"] = 0
    with coco_path.open("w") as f:
        json.dump(coco, f, indent=4)


def _decode_rle_mask(seg: dict) -> np.ndarray:
    """Decode a COCO RLE segmentation dict -- compressed or uncompressed -- to a binary mask.

    ChickenDet's own annotations are 100% *uncompressed* RLE (`counts` as a plain list
    of run-lengths, verified across all 153,764 real annotations, not just a sample) --
    but `frPyObjects` (the uncompressed -> compressed converter) raises on a
    *compressed* dict's string `counts` instead of handling it, so this branches
    explicitly rather than assuming only one variant ever shows up. Cheap to do, and
    turns a latent crash into correct behavior if this module ever runs against a
    different dataset.

    Args:
        seg: A COCO segmentation dict (`{"size": [h, w], "counts": ...}`).

    Returns:
        Binary mask, shape `(h, w)`.
    """
    if isinstance(seg["counts"], list):
        rle = mask_utils.frPyObjects([seg], seg["size"][0], seg["size"][1])
        return mask_utils.decode(rle)[:, :, 0]

    counts = seg["counts"]
    counts_bytes = counts.encode("ascii") if isinstance(counts, str) else counts
    compressed = {"size": seg["size"], "counts": counts_bytes}
    return mask_utils.decode(compressed)


def _convert_rle_to_polygons_in_json(json_path: Path, output_path: Path) -> list[int]:
    """Replace every RLE segmentation (compressed or uncompressed) in one COCO JSON with polygon(s).

    Args:
        json_path: Source COCO instances JSON.
        output_path: Where to write the polygon-converted copy.

    Returns:
        Annotation ids where OpenCV couldn't extract a usable contour — segmentation
        left as RLE, unchanged. `convert_coco` will still bbox-fill these silently, so
        the caller should log them rather than let them pass unnoticed (the same
        silent-fallback failure mode this whole preprocessing step exists to avoid,
        just at the one remaining point it could still happen).
    """
    with json_path.open() as f:
        data = json.load(f)

    unconverted = []
    for ann in data["annotations"]:
        seg = ann.get("segmentation")
        if not (isinstance(seg, dict) and "counts" in seg):
            continue  # already a polygon (or missing) -- nothing to do

        mask = _decode_rle_mask(seg)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for cnt in contours:
            # ~0.5px tolerance: cuts label-file size while keeping shape detail
            poly = cv2.approxPolyDP(cnt, 0.5, True).flatten().tolist()
            if len(poly) >= 6:  # x, y * 3 minimum for a valid polygon
                polygons.append(poly)

        if polygons:
            ann["segmentation"] = polygons
        else:
            unconverted.append(ann["id"])

    with output_path.open("w") as f:
        json.dump(data, f)

    return unconverted


def cache_polygon_annotations(annotations_dir: Path, cache_dir: Path, force: bool = False) -> Path:
    """Preprocess every split's COCO segmentation from compressed RLE to polygons, cached to disk.

    `ultralytics.data.converter.convert_coco` can't read compressed RLE — it silently
    substitutes a box-shaped segment instead of the real mask. ChickenDet ships
    segmentations as 100% RLE; left unpatched, every `-seg` label file is effectively
    box-only (confirmed via `notebooks/03_explore_segmentation.ipynb`: mean IoU against
    the true mask 0.63 vs. 0.97 once this preprocessing runs first — see
    `docs/adr/0013-rle-to-polygon-preprocessing-for-yolo-seg-conversion.md`).

    This is the expensive step (~1-2 min for ChickenDet's Train split) — cached
    persistently rather than recomputed on every call, since `convert_coco_to_yolo_labels`
    runs on every `train`/`tune`/`sweep` invocation, not just once per dataset download.

    Args:
        annotations_dir: Directory holding the source COCO instances JSONs (already
            `iscrowd`-fixed, if applicable — see `fix_iscrowd_field`).
        cache_dir: Where to write the polygon-converted JSONs, one per
            `instances_*.json` found in `annotations_dir`.
        force: Rebuild the cache even if it already exists (e.g. after the source
            annotations changed).

    Returns:
        `cache_dir`, unchanged, for convenient chaining.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    for json_path in sorted(annotations_dir.glob("instances_*.json")):
        output_path = cache_dir / json_path.name
        if output_path.exists() and not force:
            continue

        unconverted = _convert_rle_to_polygons_in_json(json_path, output_path)
        if unconverted:
            print(
                f"{json_path.name}: {len(unconverted)} annotation(s) had no usable "
                f"contour and kept their original RLE segmentation -- convert_coco "
                f"will bbox-fill these silently. ann_id(s): {unconverted[:10]}"
                f"{' ...' if len(unconverted) > 10 else ''}"
            )

    return cache_dir


def convert_coco_to_yolo_labels(
    data_dir: Path, annotations_dir: Path, force: bool = False, use_segments: bool = False
) -> Path:
    """Convert ChickenDet's COCO annotations to YOLO-format label files.

    Wraps `ultralytics.data.converter.convert_coco`, which has two gotchas handled here:
    it never overwrites an existing `save_dir` (a re-run silently writes to
    `<save_dir>-2` and everything downstream reads stale labels), and it writes into
    `<save_dir>/labels/<json_stem>/`, one level deeper than Ultralytics' training
    convention of `images/<split>` + `labels/<split>` side by side.

    Args:
        data_dir: Dataset root — must already contain `images/<split>/`.
        annotations_dir: Directory holding the (iscrowd-fixed) COCO instances JSONs.
        force: Delete and regenerate an existing `labels/` directory instead of
            skipping. Also forces a rebuild of the polygon annotation cache (see
            `cache_polygon_annotations`) when `use_segments=True`.
        use_segments: Include segmentation polygons in the label files, not just boxes.
            Needed even for a detection-task run if Ultralytics' native `copy_paste`
            augmentation is in use — it silently no-ops without segments (see
            `detection/yolo.py`'s density-augmentation notes). A `detect`-task model
            trains on the derived boxes either way; the polygons are otherwise unused.
            When True, `annotations_dir` is preprocessed through
            `cache_polygon_annotations` first — see that function for why.

    Returns:
        Path to the resulting `<data_dir>/labels/` directory.
    """
    labels_dir = data_dir / "labels"
    if labels_dir.exists():
        if not force:
            return labels_dir
        shutil.rmtree(labels_dir)

    source_dir = annotations_dir
    if use_segments:
        cache_dir = data_dir / "annotations_polygon_cache"
        source_dir = cache_polygon_annotations(annotations_dir, cache_dir, force=force)

    scratch_dir = data_dir / "yolo_labels_tmp"
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)  # leftover from an interrupted previous run

    convert_coco(
        labels_dir=str(source_dir),  # source COCO JSON dir (confusing param name)
        save_dir=str(scratch_dir),
        use_segments=use_segments,
        use_keypoints=False,
        cls91to80=False,
    )
    shutil.move(str(scratch_dir / "labels"), str(labels_dir))
    shutil.rmtree(scratch_dir)
    return labels_dir


def build_data_yaml(data_dir: Path, class_names: dict[int, str], yaml_path: Path) -> Path:
    """Write the Ultralytics dataset YAML (image dirs + class names).

    Args:
        data_dir: Dataset root containing `images/Train`, `images/Validation`, etc.
        class_names: Mapping of class index to name, e.g. `{0: "Chicken"}`.
        yaml_path: Destination path for the YAML file.

    Returns:
        `yaml_path`, unchanged, for convenient chaining.
    """
    import yaml

    data_yaml = {
        "path": str(data_dir),
        "train": "images/Train",
        "val": "images/Validation",
        "names": class_names,
    }
    with yaml_path.open("w") as f:
        yaml.safe_dump(data_yaml, f)
    return yaml_path


def prepare_data(data_dir: Path, class_names: dict[int, str], force_relabel: bool = False) -> Path:
    """Fix `iscrowd`, convert COCO to YOLO labels (with segments), and write the data YAML.

    Segments are included even for a detection-only run: Ultralytics' native `copy_paste`
    density augmentation silently no-ops without them. A `detect`-task model trains on
    the derived boxes either way.

    Args:
        data_dir: Dataset root (contains `images/`, `annotations/`).
        class_names: Mapping of class index to name, e.g. `{0: "Chicken"}`.
        force_relabel: Regenerate `labels/` even if it already exists. Needed the first
            time this runs against a `labels/` dir a box-only conversion already wrote
            — otherwise the existing box-only labels are kept as-is and `copy_paste`
            silently has no segments.

    Returns:
        Path to the written `data.yaml`.
    """
    annotations_dir = data_dir / "annotations"
    for split in ("Train", "Validation", "Test"):
        fix_iscrowd_field(annotations_dir / f"instances_{split}.json", assume_yes=True)
    convert_coco_to_yolo_labels(data_dir, annotations_dir, force=force_relabel, use_segments=True)
    return build_data_yaml(data_dir, class_names, data_dir / "chickendet.yaml")
