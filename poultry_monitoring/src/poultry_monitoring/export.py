"""Export a trained checkpoint across the Phase 6 backend/precision matrix, and index it.

Two things here are less obvious than they look, both recorded in
docs/adr/0020-export-quantization-matrix.md:

**`quantize=`, not `int8=`/`half=`.** ultralytics 8.4.117 replaced the per-precision flags
with one `quantize` argument (`8`/`16`/`32`/`'w8a8'`/`'w8a16'`/`'w8a32'`); the old flags
still work but emit a deprecation warning. INT8 additionally requires `data=`, since static
quantization has to observe real activation ranges to choose its scales.

**Static and dynamic exports are not interchangeable.** A graph frozen at `1x3x640x640`
cannot do Ultralytics' default rectangular letterboxing, so it is scored under square
padding instead. That alone moves mask mAP50 by ~0.7 points on ChickenDet — before any
quantization — so a variant must be compared against a baseline sharing its input shape, and
`baseline_for` below exists to pick the right one. Accuracy artifacts are exported dynamic
where the backend allows; TensorRT engines are always frozen.
"""

import os

os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # before ultralytics; see inference.py

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, dataclass  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

from ultralytics import YOLO  # noqa: E402

from poultry_monitoring.inference import DEFAULT_IMGSZ, DEFAULT_TASK, configure_runtime

MANIFEST_SCHEMA = 1

# Placeholder for a metric no stage has filled in yet, so a freshly exported manifest still
# renders as a table instead of raising.
UNMEASURED = "—"

# Fraction of the training split used to calibrate INT8. Calibration only has to cover the
# activation range, not the data distribution, so a small sample is enough and keeps the
# TensorRT INT8 build inside an 8 GB card.
DEFAULT_CALIBRATION_FRACTION = 0.05


@dataclass(frozen=True)
class ExportSpec:
    """One cell of the export matrix.

    Attributes:
        variant: Short name, unique per source model (e.g. `onnx-int8`).
        fmt: Ultralytics export format (`onnx`, `openvino`, `engine`).
        precision: `fp32`, `fp16` or `int8` — the artifact's numerics.
        quantize: Value for `export(quantize=)`; `None` means FP32.
        device: Device the *export* runs on. FP16 ONNX and any TensorRT build need a GPU.
        dynamic: Whether input shape is left symbolic. See the module docstring.
        runtime_device: Where the artifact can actually run once built.
        post_quantize: A quantization step applied to the exported graph afterwards, for
            schemes Ultralytics has no route to. Only `"onnx_dynamic"` today; see
            `quantize_onnx_dynamic`.
    """

    variant: str
    fmt: str
    precision: str
    quantize: int | None
    device: str | int
    dynamic: bool
    runtime_device: str
    post_quantize: str | None = None


# TensorRT is always static: engines are built for a fixed profile, and a dynamic engine
# would defeat the point of comparing it as the fastest fixed-shape deployment target.
EXPORT_MATRIX: tuple[ExportSpec, ...] = (
    ExportSpec("onnx-fp32", "onnx", "fp32", None, "cpu", True, "cpu+cuda"),
    ExportSpec("onnx-fp16", "onnx", "fp16", 16, 0, True, "cuda"),
    ExportSpec("onnx-int8", "onnx", "int8", 8, "cpu", False, "cpu"),
    # Weight-only INT8: INT8 weights, FP32 activations, no calibration data. Named for the
    # scheme rather than "int8-dynamic" so it isn't confused with dynamic *shapes* -- this
    # one keeps symbolic shapes too, hence `dynamic=True`.
    ExportSpec("onnx-w8a32", "onnx", "w8a32", None, "cpu", True, "cpu", "onnx_dynamic"),
    ExportSpec("openvino-fp32", "openvino", "fp32", None, "cpu", True, "cpu"),
    ExportSpec("openvino-int8", "openvino", "int8", 8, "cpu", False, "cpu"),
    ExportSpec("engine-fp16", "engine", "fp16", 16, 0, False, "cuda"),
    ExportSpec("engine-int8", "engine", "int8", 8, 0, False, "cuda"),
)


@dataclass
class ExportOutcome:
    """Result of building one artifact."""

    model: str
    variant: str
    path: str
    backend: str
    precision: str
    dynamic: bool
    imgsz: int
    size_mb: float
    export_seconds: float
    source_weights: str


def register_baselines(manifest_path: Path, weights: Path, imgsz: int = DEFAULT_IMGSZ) -> list[str]:
    """Record the source checkpoint twice, as the two shape-matched PyTorch baselines.

    Every exported artifact's accuracy is reported as a delta, and the delta is only
    quantization if the baseline shares the artifact's letterboxing. A frozen graph is
    restricted to square padding while a dynamic one keeps the validator's aspect-ratio
    padding, and those differ by ~0.7 mask mAP50 on ChickenDet — so the same `.pt` is
    registered under both modes and `baseline_for` routes each variant to the right one.

    Args:
        manifest_path: Path to `manifest.json`.
        weights: Source `.pt` checkpoint.
        imgsz: Image size the baselines are scored at.

    Returns:
        The manifest keys written.
    """
    weights = Path(weights).resolve()
    model_name = weights.parent.parent.name
    keys = []
    for variant, dynamic in (("pytorch-fp32-dynamic", True), ("pytorch-fp32-static", False)):
        key = entry_key(model_name, variant)
        update_manifest_entry(
            manifest_path,
            key,
            {
                "model": model_name,
                "variant": variant,
                "path": str(weights),
                "backend": "pytorch",
                "precision": "fp32",
                "dynamic": dynamic,
                "imgsz": imgsz,
                "size_mb": round(_artifact_size_mb(weights), 2),
                "source_weights": str(weights),
            },
        )
        keys.append(key)
    return keys


def default_manifest_path(data_dir: Path | None = None) -> Path:
    """Locate the export manifest.

    Args:
        data_dir: ChickenDet root. Defaults to `./data/ChickenDet` relative to the cwd.

    Returns:
        Path to `manifest.json` under the export tree (which is gitignored).
    """
    data_dir = Path(data_dir) if data_dir is not None else Path("data/ChickenDet")
    return data_dir / "YOLO" / "export" / "manifest.json"


def read_manifest(manifest_path: Path) -> dict[str, dict]:
    """Read the manifest's entries, tolerating a missing file.

    Args:
        manifest_path: Path to `manifest.json`.

    Returns:
        `{entry_key: entry_dict}`, empty if the manifest doesn't exist yet.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text()).get("entries", {})


def write_manifest(manifest_path: Path, entries: dict[str, dict]) -> Path:
    """Write entries back to the manifest, preserving schema/timestamp fields.

    Args:
        manifest_path: Path to `manifest.json`.
        entries: Full entry mapping to persist.

    Returns:
        The manifest path written.
    """
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema": MANIFEST_SCHEMA,
                "updated": datetime.now(UTC).isoformat(timespec="seconds"),
                "entries": entries,
            },
            indent=2,
        )
    )
    return manifest_path


def update_manifest_entry(manifest_path: Path, key: str, fields: dict) -> dict[str, dict]:
    """Merge fields into one entry, leaving every other entry untouched.

    Lets the accuracy stage enrich the same manifest the export stage wrote, instead of
    emitting a parallel file that then has to be joined back. Accuracy fits here because it
    is one number per artifact per split; latency does not, being per
    (artifact x device x batch), and lives in `benchmark_results.json` keyed by cell.

    Args:
        manifest_path: Path to `manifest.json`.
        key: Entry key, as produced by `entry_key`.
        fields: Fields to merge into that entry.

    Returns:
        The full updated entry mapping.
    """
    entries = read_manifest(manifest_path)
    entries.setdefault(key, {}).update(fields)
    write_manifest(manifest_path, entries)
    return entries


def entry_key(model: str, variant: str) -> str:
    """Build the manifest key for a (model, variant) pair.

    Args:
        model: Source run name, e.g. `yolo26n-seg-baseline-adamw`.
        variant: Variant name, e.g. `onnx-int8`.

    Returns:
        `"{model}__{variant}"`.
    """
    return f"{model}__{variant}"


def baseline_for(entry: dict) -> str:
    """Name the PyTorch baseline a variant must be compared against.

    Static artifacts are letterboxed square and dynamic ones aren't, which is worth ~0.7
    mask mAP50 on ChickenDet on its own. Comparing across that boundary would bill a
    preprocessing difference as quantization damage.

    Args:
        entry: A manifest entry, which must carry a `dynamic` flag.

    Returns:
        Variant name of the matching baseline.
    """
    return "pytorch-fp32-dynamic" if entry.get("dynamic") else "pytorch-fp32-static"


def summarize_manifest(entries: dict[str, dict]) -> str:
    """Render the manifest as an aligned table for `inference --list`.

    Args:
        entries: Entry mapping from `read_manifest`.

    Returns:
        Printable table, or a hint if the manifest is empty.
    """
    if not entries:
        return "No exported artifacts yet — run `python -m poultry_monitoring.export`."
    # Sized from the data: keys are `{model}__{variant}` and run past any fixed width once
    # a model name is long, which silently misaligns every following column.
    width = max(len("variant"), max(len(key) for key in entries))
    header = (
        f"{'variant':<{width}} {'backend':<10} {'prec':<6} {'dyn':<4} {'MB':>7} {'mask50-95':>10}"
    )
    lines = [header, "-" * len(header)]
    for key in sorted(entries):
        e = entries[key]
        accuracy = (e.get("accuracy") or {}).get("mask_map50_95")
        # Formatted before interpolation, not inline: a format spec applied to a conditional
        # covers the whole expression, and `format(None, ">10")` raises TypeError -- which
        # would break the table for every variant that hasn't been scored yet.
        accuracy_text = UNMEASURED if accuracy is None else f"{accuracy:.4f}"
        error = e.get("error")
        lines.append(
            f"{key:<{width}} {e.get('backend', '?'):<10} {e.get('precision', '?'):<6} "
            f"{'yes' if e.get('dynamic') else 'no':<4} {e.get('size_mb', 0):>7.1f} "
            f"{accuracy_text:>10}" + (f"  FAILED: {error}" if error else "")
        )
    return "\n".join(lines)


def _artifact_size_mb(path: Path) -> float:
    """Total size of an artifact, which may be a file or an OpenVINO directory."""
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
    return path.stat().st_size / 1e6


def export_variant(
    weights: Path,
    spec: ExportSpec,
    output_dir: Path,
    data_yaml: Path | None = None,
    imgsz: int = DEFAULT_IMGSZ,
    batch: int = 1,
    calibration_fraction: float = DEFAULT_CALIBRATION_FRACTION,
    task: str = DEFAULT_TASK,
) -> ExportOutcome:
    """Build one artifact from one checkpoint.

    Ultralytics writes an export next to its *source* `.pt`, not into the working directory,
    and derives the artifact name from it — so each variant gets its own directory holding
    its own copy of the weights. Without that, variants silently overwrite each other.

    Args:
        weights: Source `.pt` checkpoint.
        spec: Matrix cell to build.
        output_dir: Root for exported artifacts; a per-variant subdirectory is created.
        data_yaml: Dataset YAML, required for INT8 calibration.
        imgsz: Export image size — must match training to keep metrics comparable.
        batch: Static batch dimension, ignored when `spec.dynamic`.
        calibration_fraction: Fraction of the split used to calibrate INT8.
        task: Ultralytics task for the source checkpoint.

    Returns:
        An `ExportOutcome` describing the built artifact.

    Raises:
        ValueError: If the spec needs INT8 calibration data and `data_yaml` is None.
    """
    configure_runtime()
    weights = Path(weights)
    if spec.precision == "int8" and data_yaml is None:
        raise ValueError(f"{spec.variant} is INT8 and needs data_yaml for calibration.")

    variant_dir = Path(output_dir) / spec.variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    source_copy = variant_dir / "model.pt"
    if not source_copy.exists():
        shutil.copy2(weights, source_copy)

    kwargs: dict[str, object] = {"format": spec.fmt, "imgsz": imgsz, "device": spec.device}
    if spec.quantize is not None:
        kwargs["quantize"] = spec.quantize
    if spec.dynamic:
        kwargs["dynamic"] = True
    else:
        kwargs["batch"] = batch
    if spec.precision == "int8":
        kwargs["data"] = str(data_yaml)
        kwargs["fraction"] = calibration_fraction

    started = time.time()
    # Trust the returned path: INT8 exports land as `model_int8.onnx`, OpenVINO as a
    # `*_openvino_model/` directory, so the name cannot be predicted from the spec alone.
    produced = Path(YOLO(str(source_copy), task=task).export(**kwargs))
    if spec.post_quantize == "onnx_dynamic":
        # Ultralytics exports FP32 here; the weight-only INT8 pass runs afterwards, against
        # ONNX Runtime directly, because `quantize='w8a32'` is LiteRT-only in the exporter.
        produced = quantize_onnx_dynamic(produced, produced.with_name("model_w8a32.onnx"))
    elif spec.post_quantize is not None:
        raise ValueError(f"Unknown post_quantize step {spec.post_quantize!r} on {spec.variant}")
    elapsed = time.time() - started

    return ExportOutcome(
        model=weights.parent.parent.name,
        variant=spec.variant,
        path=str(produced.resolve()),
        backend=spec.fmt if spec.fmt != "engine" else "tensorrt",
        precision=spec.precision,
        dynamic=spec.dynamic,
        imgsz=imgsz,
        size_mb=round(_artifact_size_mb(produced), 2),
        export_seconds=round(elapsed, 1),
        source_weights=str(weights.resolve()),
    )


def quantize_onnx_dynamic(onnx_path: Path, output_path: Path) -> Path:
    """Weight-only INT8 quantization of an ONNX graph, with no calibration data.

    The one quantization path Ultralytics offers no route to: `engine/exporter.py` restricts
    `quantize='w8a32'` to LiteRT (`W8A32_FORMATS = frozenset({"litert"})`), so ONNX dynamic
    INT8 has to be done directly against ONNX Runtime.

    Included in the matrix as a measured comparison rather than a serious candidate — on
    Conv-heavy graphs weight-only INT8 is frequently *slower* than FP32 on CPU, because
    weights are dequantized on the fly every forward pass while activations stay FP32.

    Only weighted ops are quantized, matching Ultralytics' own static-INT8 reasoning
    (`utils/export/onnx.py`): a single INT8 scale spanning box pixel coordinates (~0-640)
    and class probabilities (0-1) would round every score to zero.

    Args:
        onnx_path: Source FP32 ONNX graph.
        output_path: Where to write the quantized graph.

    Returns:
        The written path.
    """
    from onnxruntime.quantization import quantize_dynamic
    from onnxruntime.quantization.quant_utils import QuantType

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        str(onnx_path),
        str(output_path),
        # QUInt8, not QInt8, and the difference is pass/fail rather than a tuning choice.
        # Quantizing Conv dynamically emits `ConvInteger` nodes, and ONNX Runtime's CPU
        # kernel for those handles unsigned weights only -- a signed graph builds happily
        # and then dies at session run with "Could not find an implementation for
        # ConvInteger(10)". Excluding Conv instead avoids the crash but quantizes nothing
        # worth having: a CNN is almost entirely Conv, so the artifact stays FP32-sized.
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=["Conv", "Gemm", "MatMul"],
    )
    return output_path


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the `python -m poultry_monitoring.export` CLI."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=Path, required=True, help="ChickenDet root.")
    parser.add_argument(
        "--weights",
        type=Path,
        nargs="+",
        default=None,
        help="One or more trained -seg checkpoints to export. Required unless --list.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=None,
        help=f"Subset of {[s.variant for s in EXPORT_MATRIX]}. Default: all.",
    )
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--batch", type=int, default=1, help="Static batch for frozen graphs.")
    parser.add_argument("--fraction", type=float, default=DEFAULT_CALIBRATION_FRACTION)
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--list", action="store_true", help="Print the current manifest and exit.")
    return parser


def main() -> None:
    """CLI entry point — see `_build_arg_parser` for `--help`."""
    args = _build_arg_parser().parse_args()
    data_dir = args.data_dir.resolve()
    manifest_path = args.manifest or default_manifest_path(data_dir)

    if args.list:
        print(summarize_manifest(read_manifest(manifest_path)))
        return
    if not args.weights:
        raise SystemExit("Pass --weights <checkpoint> [...], or --list to show the manifest.")

    data_yaml = data_dir / "chickendet.yaml"
    output_root = data_dir / "YOLO" / "export"
    specs = [s for s in EXPORT_MATRIX if args.variants is None or s.variant in args.variants]

    for weights in args.weights:
        weights = weights.resolve()
        model_name = weights.parent.parent.name
        register_baselines(manifest_path, weights, imgsz=args.imgsz)
        for spec in specs:
            try:
                outcome = export_variant(
                    weights,
                    spec,
                    output_root / model_name,
                    data_yaml=data_yaml,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    calibration_fraction=args.fraction,
                    task=args.task,
                )
            except Exception as error:  # noqa: BLE001 - one failing backend must not
                # abort the rest of the matrix; the failure is recorded and reported.
                print(f"FAIL  {model_name:<32} {spec.variant:<16} {type(error).__name__}: {error}")
                update_manifest_entry(
                    manifest_path,
                    entry_key(model_name, spec.variant),
                    {"error": f"{type(error).__name__}: {error}"},
                )
                continue
            update_manifest_entry(
                manifest_path, entry_key(model_name, spec.variant), asdict(outcome)
            )
            print(
                f"OK    {model_name:<32} {spec.variant:<16} "
                f"{outcome.size_mb:>7.1f} MB  {outcome.export_seconds:>6.1f}s"
            )

    print(f"\nManifest: {manifest_path}\n")
    print(summarize_manifest(read_manifest(manifest_path)))


if __name__ == "__main__":
    main()
