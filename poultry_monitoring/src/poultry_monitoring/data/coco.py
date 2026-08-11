"""ChickenDet COCO annotation parsing and COCO -> YOLO label conversion.

Boxes and (eventually) masks live in the same COCO-format annotation file per split, so
this module is shared between the detection and segmentation tasks (constitution
Principle III) rather than duplicated per task.

Productionized from `notebooks/02_yolo26_baseline.ipynb`'s exploration — see that
notebook's Notes section for the `convert_coco` gotchas this module works around.
"""

import json
import shutil
from pathlib import Path

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
        force: Delete and regenerate an existing `labels/` directory instead of skipping.
        use_segments: Include segmentation polygons in the label files, not just boxes.
            Needed even for a detection-task run if Ultralytics' native `copy_paste`
            augmentation is in use — it silently no-ops without segments (see
            `detection/yolo.py`'s density-augmentation notes). A `detect`-task model
            trains on the derived boxes either way; the polygons are otherwise unused.

    Returns:
        Path to the resulting `<data_dir>/labels/` directory.
    """
    labels_dir = data_dir / "labels"
    if labels_dir.exists():
        if not force:
            return labels_dir
        shutil.rmtree(labels_dir)

    scratch_dir = data_dir / "yolo_labels_tmp"
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)  # leftover from an interrupted previous run

    convert_coco(
        labels_dir=str(annotations_dir),  # source COCO JSON dir (confusing param name)
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
