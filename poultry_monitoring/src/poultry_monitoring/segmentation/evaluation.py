"""Evaluate a trained `-seg` checkpoint on a dataset split, optionally stratified by density.

Separate from `segmentation/yolo.py` for the same reason `preprocessing_eval.py` is —
evaluation-only harnesses don't belong in the train/predict core (docs/adr/0010). The
`val` CLI subcommand lives in `yolo.py` and imports this locally.

Density stratification exists because a single aggregate number per model hid the clearest
pattern in the copy-paste ablation: the effect scales with how crowded the scene is,
in opposite directions for `yolo26n-seg` and `yolo26s-seg`. See the README's
"The effect scales with scene density" and `docs/engineering-notes.md`.
"""

import json
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import YAML

from poultry_monitoring.data.coco import CLASS_NAMES
from poultry_monitoring.segmentation.yolo import extract_metrics

# Ultralytics' default (8) spawns DataLoader workers per `val()` call. Evaluating several
# checkpoints in one process that way hung indefinitely on Windows -- same family as
# docs/adr/0007's `spawn` trouble -- and parallel loading buys nothing on a few hundred
# images. In-process loading is the default here; override if a large split needs it.
DEFAULT_WORKERS = 0
DEFAULT_BATCH = 8

# Only meaningful for the 3-bin default, which is what the README publishes.
TERCILE_LABELS = ("sparse", "medium", "dense")


def density_bins(annotations_path: Path, n_bins: int = 3) -> dict[str, dict]:
    """Split one COCO split's images into equal-size bins by their own annotation count.

    Pure function over the annotation file — no model, no I/O beyond the read — so the
    binning itself is unit-testable independently of any evaluation.

    Args:
        annotations_path: COCO instances JSON for the split being evaluated.
        n_bins: How many equal-size bins to cut. Bins are labelled
            `sparse`/`medium`/`dense` at the default 3, `bin1`..`binN` otherwise.

    Returns:
        Ordered dict of `{label: {"file_names", "images", "instances", "min",
        "median", "max"}}`, sparsest bin first. Bin sizes differ by at most one image
        when the split doesn't divide evenly.

    Raises:
        ValueError: If `n_bins` is below 1 or exceeds the number of images in the split.
    """
    coco = json.loads(Path(annotations_path).read_text())
    counts: dict[int, int] = {image["id"]: 0 for image in coco["images"]}
    for annotation in coco["annotations"]:
        counts[annotation["image_id"]] = counts.get(annotation["image_id"], 0) + 1
    name_of = {image["id"]: image["file_name"] for image in coco["images"]}

    if n_bins < 1 or n_bins > len(name_of):
        raise ValueError(f"n_bins must be in 1..{len(name_of)}, got {n_bins}")

    ordered = sorted(((counts[i], name_of[i]) for i in name_of), key=lambda pair: pair[0])
    labels = TERCILE_LABELS if n_bins == 3 else tuple(f"bin{i + 1}" for i in range(n_bins))

    bins = {}
    total = len(ordered)
    for index, label in enumerate(labels):
        start, stop = index * total // n_bins, (index + 1) * total // n_bins
        chunk = ordered[start:stop]
        chunk_counts = [count for count, _ in chunk]
        bins[label] = {
            "file_names": [name for _, name in chunk],
            "images": len(chunk),
            "instances": sum(chunk_counts),
            "min": min(chunk_counts),
            "median": chunk_counts[len(chunk_counts) // 2],
            "max": max(chunk_counts),
        }
    return bins


def evaluate_split(
    weights_path: Path,
    data_yaml: Path,
    project: Path,
    split: str = "test",
    name: str | None = None,
    batch: int = DEFAULT_BATCH,
    imgsz: int = 640,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, float]:
    """Score one checkpoint on one split, returning this project's box+mask metric names.

    Args:
        weights_path: Trained `-seg` checkpoint.
        data_yaml: Dataset YAML declaring the split being asked for.
        project: Local save dir for `model.val()` artifacts.
        split: Which split key in `data_yaml` to evaluate — `"val"` or `"test"`.
        name: Run subdirectory under `project`. Defaults to `<weights' run dir>-<split>eval`.
        batch: Evaluation batch size.
        imgsz: Evaluation image size — must match training to be comparable.
        workers: `DataLoader` workers; see `DEFAULT_WORKERS` for why this is 0.

    Returns:
        Flat metrics dict, same shape as `segmentation.yolo.extract_metrics`.
    """
    weights_path = Path(weights_path).resolve()
    if name is None:
        name = f"{weights_path.parent.parent.name}-{split}eval"
    metrics = YOLO(str(weights_path)).val(
        data=str(Path(data_yaml).resolve()),
        split=split,
        project=str(Path(project).resolve()),
        name=name,
        exist_ok=True,
        verbose=False,
        plots=False,
        batch=batch,
        imgsz=imgsz,
        workers=workers,
    )
    return extract_metrics(metrics)


def evaluate_by_density(
    weights_path: Path,
    data_dir: Path,
    project: Path,
    split: str = "test",
    n_bins: int = 3,
    output_path: Path | None = None,
    batch: int = DEFAULT_BATCH,
    imgsz: int = 640,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, dict]:
    """Score one checkpoint separately on each density bin of a split.

    Writes a per-bin image listing and a matching dataset YAML under
    `project/density_<split>/`, then evaluates each. Ultralytics resolves label paths from
    image paths by swapping `images/` for `labels/`, so the listings can point straight at
    the real split without copying anything.

    Args:
        weights_path: Trained `-seg` checkpoint.
        data_dir: ChickenDet root, holding `annotations/` and `images/`.
        project: Local save dir for per-bin YAMLs, listings and `model.val()` artifacts.
        split: Split to stratify — `"Validation"` or `"Test"` (matching the directory and
            `instances_<split>.json` naming, not the Ultralytics `val`/`test` key).
        n_bins: Number of equal-size density bins.
        output_path: If given, write the full results dict there as JSON.
        batch: Evaluation batch size.
        imgsz: Evaluation image size.
        workers: `DataLoader` workers; see `DEFAULT_WORKERS`.

    Returns:
        `{"bins": {label: {images, instances, min, median, max}}, "metrics":
        {label: <extract_metrics dict>}}`.
    """
    data_dir, project = Path(data_dir).resolve(), Path(project).resolve()
    bins = density_bins(data_dir / "annotations" / f"instances_{split}.json", n_bins=n_bins)
    scratch = project / f"density_{split.lower()}"
    scratch.mkdir(parents=True, exist_ok=True)

    metrics = {}
    for label, info in bins.items():
        listing = scratch / f"{label}.txt"
        listing.write_text(
            "\n".join(str(data_dir / "images" / split / name) for name in info["file_names"])
        )
        bin_yaml = scratch / f"{label}.yaml"
        # `train`/`val` are required by the schema but never read for a val()-only call on
        # the `test` key -- only the active split's path is resolved.
        YAML.save(
            bin_yaml,
            {
                "path": str(data_dir),
                "train": f"images/{split}",
                "val": f"images/{split}",
                "test": str(listing),
                "names": CLASS_NAMES,
            },
        )
        metrics[label] = evaluate_split(
            weights_path,
            bin_yaml,
            project,
            split="test",
            name=f"{Path(weights_path).parent.parent.name}-density-{label}",
            batch=batch,
            imgsz=imgsz,
            workers=workers,
        )

    results = {
        "bins": {
            label: {k: v for k, v in info.items() if k != "file_names"}
            for label, info in bins.items()
        },
        "metrics": metrics,
    }
    if output_path is not None:
        Path(output_path).write_text(json.dumps(results, indent=2))
    return results
