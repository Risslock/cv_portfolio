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

from ultralytics.utils import YAML

from poultry_monitoring.data.coco import CLASS_NAMES
from poultry_monitoring.inference import load_model
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
    task: str = "segment",
    rect: bool | None = None,
) -> dict[str, float]:
    """Score one checkpoint or exported artifact on one split, in this project's metric names.

    Args:
        weights_path: Trained `-seg` checkpoint, or any exported artifact
            (`.onnx`/`.engine`/`*_openvino_model/`).
        data_yaml: Dataset YAML declaring the split being asked for.
        project: Local save dir for `model.val()` artifacts.
        split: Which split key in `data_yaml` to evaluate — `"val"` or `"test"`.
        name: Run subdirectory under `project`. Defaults to `<weights' run dir>-<split>eval`.
        batch: Evaluation batch size.
        imgsz: Evaluation image size — must match training to be comparable.
        workers: `DataLoader` workers; see `DEFAULT_WORKERS` for why this is 0.
        task: Ultralytics task. Must stay `"segment"` for exported artifacts: Ultralytics
            infers task from the *filename*, and an exported `.../weights/best.onnx` matches
            no `-seg` pattern, so it silently scores as `detect` and returns no mask metrics
            at all. See docs/adr/0020.
        rect: Letterbox mode. `None` uses Ultralytics' default (`True`, aspect-ratio
            padding). Pass `False` to force square padding, which is what a graph frozen at
            `imgsz x imgsz` is restricted to — a PyTorch baseline must match the artifact's
            mode or a ~0.7 mask-mAP50 padding difference is misread as quantization loss.

    Returns:
        Flat metrics dict, same shape as `segmentation.yolo.extract_metrics`.
    """
    weights_path = Path(weights_path).resolve()
    if name is None:
        name = f"{weights_path.parent.parent.name}-{split}eval"
    optional = {} if rect is None else {"rect": rect}
    metrics = load_model(weights_path, task=task).val(
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
        **optional,
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
    task: str = "segment",
    rect: bool | None = None,
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
        task: Ultralytics task; see `evaluate_split`.
        rect: Letterbox mode; see `evaluate_split`.

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
            task=task,
            rect=rect,
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


def score_manifest(
    manifest_path: Path,
    data_yaml: Path,
    project: Path,
    splits: tuple[str, ...] = ("Validation", "Test"),
    variants: list[str] | None = None,
    batch: int = DEFAULT_BATCH,
    imgsz: int = 640,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, dict]:
    """Score every exported artifact in the manifest, writing results back into it.

    Exploits the fact that **accuracy is a property of the artifact, not of the device it
    runs on**: an INT8 graph scores the same wherever it executes, so each artifact is
    scored once per split while the device/batch sweep is left entirely to the latency
    harness. That is what keeps this a ~30-cell job rather than a ~90-cell one.

    Each artifact is scored under its own letterbox mode (`rect` follows the entry's
    `dynamic` flag), so a delta against `export.baseline_for`'s matching baseline reflects
    quantization alone rather than padding.

    Args:
        manifest_path: Export manifest to read and enrich.
        data_yaml: Dataset YAML declaring the splits.
        project: Local save dir for `model.val()` artifacts.
        splits: Split directory names — mapped onto Ultralytics' `val`/`test` keys.
        variants: Manifest keys to score. Defaults to every entry without an `error`.
        batch: Evaluation batch size.
        imgsz: Evaluation image size.
        workers: `DataLoader` workers; see `DEFAULT_WORKERS`.

    Returns:
        `{manifest_key: {split: <extract_metrics dict>}}` for everything scored.
    """
    from poultry_monitoring.export import read_manifest, update_manifest_entry

    entries = read_manifest(manifest_path)
    keys = variants or sorted(k for k in entries if "error" not in entries[k])
    scored: dict[str, dict] = {}

    for key in keys:
        entry = entries[key]
        artifact = Path(entry["path"])
        if not artifact.exists():
            scored[key] = {"error": f"missing artifact: {artifact}"}
            continue
        # A frozen graph cannot do rectangular letterboxing; force the same square padding
        # on the PyTorch baselines so the two are actually comparable (docs/adr/0020).
        rect = bool(entry.get("dynamic"))
        per_split: dict[str, dict] = {}
        for split in splits:
            try:
                per_split[split] = evaluate_split(
                    artifact,
                    data_yaml,
                    project,
                    split="test" if split == "Test" else "val",
                    name=f"score-{key}-{split}".replace("__", "-"),
                    batch=batch,
                    imgsz=imgsz,
                    workers=workers,
                    task="segment",
                    rect=rect,
                )
            except Exception as error:  # noqa: BLE001 - one unscoreable artifact must not
                # abort the sweep; the failure is recorded against that entry instead.
                per_split[split] = {"error": f"{type(error).__name__}: {error}"}
        scored[key] = per_split
        # `accuracy` mirrors the primary split so `summarize_manifest` has one number to
        # show; `accuracy_by_split` keeps the full picture.
        primary = per_split.get("Test") or next(iter(per_split.values()))
        update_manifest_entry(
            manifest_path,
            key,
            {"accuracy": primary, "accuracy_by_split": per_split, "scored_rect": rect},
        )
    return scored


def accuracy_deltas(manifest_path: Path, split: str = "Test") -> dict[str, dict]:
    """Compare each variant against its shape-matched PyTorch baseline.

    Args:
        manifest_path: Manifest already populated by `score_manifest`.
        split: Which split's numbers to compare.

    Returns:
        `{manifest_key: {"baseline": key, metric: delta, ...}}`, skipping entries whose
        baseline wasn't scored.
    """
    from poultry_monitoring.export import baseline_for, entry_key, read_manifest

    entries = read_manifest(manifest_path)
    deltas: dict[str, dict] = {}
    for key, entry in entries.items():
        metrics = (entry.get("accuracy_by_split") or {}).get(split)
        if not metrics or "error" in metrics:
            continue
        base_key = entry_key(entry.get("model", ""), baseline_for(entry))
        base = (entries.get(base_key, {}).get("accuracy_by_split") or {}).get(split)
        if not base or "error" in base or base_key == key:
            continue
        deltas[key] = {"baseline": base_key} | {
            name: round(value - base[name], 4) for name, value in metrics.items() if name in base
        }
    return deltas
