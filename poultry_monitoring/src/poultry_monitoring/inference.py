"""The single model-load path for every backend, plus a runner CLI.

Export, accuracy scoring, benchmarking and prediction all open artifacts through
`load_model` here. That is deliberate: it makes "what you benchmark is what you serve"
structural rather than conventional, and it concentrates three non-obvious requirements in
one place instead of scattering them across four call sites. See
docs/adr/0020-export-quantization-matrix.md.

The three requirements, each of which fails *silently* when missed:

1. `task=` must be declared for exported artifacts. Ultralytics infers a model's task from
   the filename when it isn't told (`nn/tasks.py`'s `guess_model_task`), matching `-seg` in
   the stem or `segment` in the path parts. An exported `.../weights/best.onnx` matches
   neither, so it falls through to `"detect"` and scores with a `DetectionValidator` --
   producing a full metrics table with no mask columns at all.
2. On Windows, ONNX Runtime's CUDA provider needs CUDA/cuDNN DLLs on the process search
   path and does not find them unaided. `torch` registers its bundled `lib/` when imported,
   which holds exactly the CUDA 12 + cuDNN 9 build the pinned `onnxruntime-gpu` wants -- so
   import order alone decides whether the GPU works. `configure_runtime` registers it
   explicitly rather than leaning on that side effect.
3. Ultralytics' AutoUpdate pip-installs missing backends mid-run, which puts packages
   outside `uv.lock` (constitution Principle VII). Loading an ONNX model on CPU asks for
   `onnxruntime`, and since this project pins `onnxruntime-gpu`, AutoUpdate fetches a second
   build into the same module directory and breaks both. `YOLO_AUTOINSTALL` is read once at
   ultralytics import time, so it is set at the top of this module, before that import.
"""

import os

# Before any ultralytics import: `utils/__init__.py` reads this once, at import time.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402  -- must precede onnxruntime; see module docstring
from ultralytics import YOLO  # noqa: E402

DEFAULT_IMGSZ = 640
DEFAULT_TASK = "segment"

# Ultralytics' AutoBackend dispatches on the artifact itself; this map exists so the CLI and
# the manifest can name a backend without loading the model first.
BACKEND_BY_SUFFIX = {
    ".pt": "pytorch",
    ".onnx": "onnx",
    ".engine": "tensorrt",
    ".tflite": "litert",
    ".mlpackage": "coreml",
}

_runtime_configured = False


def configure_runtime(cpu_threads: int | None = None) -> dict[str, object]:
    """Make the process able to reach every backend, and pin what can be pinned.

    Idempotent — safe to call from each entry point. See the module docstring for why the
    DLL directory registration is needed rather than relying on `import torch` ordering.

    Args:
        cpu_threads: Torch intra-op thread count to pin. `None` leaves torch's default.
            Note this does **not** reach ONNX Runtime: `nn/backends/onnx.py` constructs its
            `InferenceSession` without a `SessionOptions`, so its thread count is not
            settable through Ultralytics and is reported rather than controlled.

    Returns:
        Dict describing what was configured, suitable for logging alongside a benchmark
        result (constitution Principle V).
    """
    global _runtime_configured
    if not _runtime_configured and os.name == "nt":
        torch_lib = Path(torch.__file__).parent / "lib"
        if torch_lib.is_dir():
            os.add_dll_directory(str(torch_lib))
        _runtime_configured = True
    if cpu_threads is not None:
        torch.set_num_threads(cpu_threads)
    return {
        "torch_num_threads": torch.get_num_threads(),
        "cpu_count": os.cpu_count(),
        "autoinstall_disabled": os.environ.get("YOLO_AUTOINSTALL") == "false",
    }


def resolve_backend(weights: Path) -> str:
    """Name the runtime an artifact will load under, from its path alone.

    Args:
        weights: Artifact path — a file (`.pt`/`.onnx`/`.engine`) or an OpenVINO model
            directory, which Ultralytics emits as a `*_openvino_model/` folder.

    Returns:
        One of `pytorch`, `onnx`, `tensorrt`, `openvino`, `litert`, `coreml`.

    Raises:
        ValueError: If the artifact's suffix maps to no known backend.
    """
    weights = Path(weights)
    if weights.is_dir() or weights.name.endswith("_openvino_model"):
        return "openvino"
    backend = BACKEND_BY_SUFFIX.get(weights.suffix.lower())
    if backend is None:
        raise ValueError(
            f"Unrecognized artifact {weights.name!r}; expected one of "
            f"{sorted(BACKEND_BY_SUFFIX)} or an OpenVINO model directory."
        )
    return backend


def load_model(
    weights: Path,
    task: str = DEFAULT_TASK,
    device: str | int | None = None,
    imgsz: int = DEFAULT_IMGSZ,
    warmup: bool = False,
    cpu_threads: int | None = None,
) -> YOLO:
    """Open any exported artifact as a ready-to-run model.

    The only place in this package that constructs a `YOLO`. Precision is deliberately not
    an argument: an exported artifact carries its own (an FP16 ONNX graph is FP16), while a
    PyTorch checkpoint takes `half=` at call time, so folding both into one flag here would
    misrepresent one of them.

    Args:
        weights: Artifact path — see `resolve_backend` for accepted forms.
        task: Ultralytics task. Must be passed explicitly for exported artifacts; see the
            module docstring for what goes wrong otherwise.
        device: `0`/`"cuda"` for GPU, `"cpu"` for CPU, `None` for Ultralytics' own choice.
        imgsz: Image size used for the warmup pass.
        warmup: Run one dummy forward pass so lazy graph/engine initialization and CUDA
            context creation are not charged to the first timed iteration.
        cpu_threads: Forwarded to `configure_runtime`.

    Returns:
        A loaded `YOLO`, its backend already materialized if `warmup` is set.

    Raises:
        FileNotFoundError: If `weights` does not exist.
    """
    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(f"No artifact at {weights}")
    configure_runtime(cpu_threads=cpu_threads)

    model = YOLO(str(weights), task=task)
    # Only PyTorch models move between devices; an exported artifact's device is fixed at
    # build time (a TensorRT engine is GPU-only, an OpenVINO IR CPU-only here), and is
    # instead selected per call via the `device=` kwarg that Ultralytics forwards.
    if device is not None and resolve_backend(weights) == "pytorch":
        model.to(device)
    if warmup:
        # A real predict call, not AutoBackend.warmup(), so the whole pre/post path is built.
        blank = torch.zeros(imgsz, imgsz, 3, dtype=torch.uint8).numpy()
        model.predict(blank, imgsz=imgsz, device=device, verbose=False)
    return model


def autobackend(model: YOLO):
    """Return the object that actually executes the graph, whatever the artifact type.

    Non-obvious, and the source of a bug worth not repeating: for anything other than a
    `.pt`, `YOLO.model` is the **path string**, not a backend — `engine/model.py`'s `_load`
    does `self.model, self.ckpt = weights, None`. The real `AutoBackend` is built lazily by
    the predictor, so it only exists at `model.predictor.model`, and only after a first
    `predict()` call. Reaching for `model.model` instead silently yields a `str`, which has
    no `.session` and is not callable -- so provider checks come back empty and forward-pass
    timing raises, for precisely the exported artifacts both were written to handle.

    Args:
        model: A model returned by `load_model`.

    Returns:
        The `AutoBackend` (or `nn.Module` for a `.pt`), or `None` if no predictor has been
        built yet and the artifact isn't a PyTorch module.
    """
    predictor = getattr(model, "predictor", None)
    backend = getattr(predictor, "model", None)
    if backend is not None:
        return backend
    inner = getattr(model, "model", None)
    return inner if isinstance(inner, torch.nn.Module) else None


def describe_runtime(model: YOLO) -> dict[str, object]:
    """Report what a loaded model is *actually* running on, not what was requested.

    Needed because ONNX Runtime fails soft: asking for `CUDAExecutionProvider` yields a
    working session with no exception even when the provider could not load, and Ultralytics
    makes this worse -- `nn/backends/onnx.py` gates on `get_available_providers()` (which
    lists CUDA EP even when it cannot initialize), appends `CPUExecutionProvider` as a
    fallback, and logs "Using ... CUDAExecutionProvider" *before* constructing the session.
    A benchmark that trusts the device argument can therefore publish CPU latency as GPU.

    Args:
        model: A model returned by `load_model`.

    Returns:
        Dict with the resolved `backend`, `device`, and — for ONNX Runtime — the session's
        real `providers` list. Keys are absent when a backend doesn't expose them.
    """
    backend_obj = autobackend(model)
    # Ask the backend first: `YOLO.device` is unset for an exported artifact (the wrapper
    # holds a path, not a module), so trusting it would report "None" for a model that is
    # demonstrably running on the GPU -- the opposite of what this function is for.
    device = getattr(backend_obj, "device", None) or getattr(model, "device", None)
    info: dict[str, object] = {
        "backend": type(backend_obj).__name__ if backend_obj is not None else None,
        "device": str(device) if device is not None else "unknown",
    }
    session = getattr(backend_obj, "session", None)
    if session is not None and hasattr(session, "get_providers"):
        info["providers"] = list(session.get_providers())
    return info


def run_inference(
    weights: Path,
    source: Path,
    task: str = DEFAULT_TASK,
    device: str | int | None = None,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int = DEFAULT_IMGSZ,
    save_dir: Path | None = None,
    model: YOLO | None = None,
) -> list:
    """Run an artifact over images and optionally save annotated copies.

    Images only — a single file, a directory, or a glob. Video and stream sources are
    deliberately out of scope for this entry point.

    Args:
        weights: Artifact path.
        source: Image file, directory, or glob.
        task: Ultralytics task; see `load_model`.
        device: Inference device.
        conf: Confidence threshold.
        iou: IoU threshold.
        imgsz: Inference image size.
        save_dir: If given, write annotated images here, one per input.
        model: An already-loaded model to reuse. Loading is not free — a TensorRT engine
            deserializes and builds a CUDA context — so a caller that already has one (to
            inspect its runtime, say) should pass it rather than pay for a second load.

    Returns:
        List of Ultralytics `Results`, one per input image.
    """
    if model is None:
        model = load_model(weights, task=task, device=device, imgsz=imgsz)
    results = model.predict(
        str(source), conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False
    )
    if save_dir is not None:
        # Rendered and written by hand rather than via `save=True`: Ultralytics' own
        # project/name resolution has written outside the project dir before, same reason
        # segmentation/yolo.py's predict() does it this way.
        import cv2

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for result in results:
            cv2.imwrite(str(save_dir / Path(result.path).name), result.plot())
    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the `python -m poultry_monitoring.inference` CLI."""
    parser = argparse.ArgumentParser(description="Run any exported artifact on images.")
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant name from the export manifest (see --list). Alternative to --weights.",
    )
    parser.add_argument(
        "--weights", type=Path, default=None, help="Artifact path, instead of --variant."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Export manifest. Defaults to <data-dir>/YOLO/export/manifest.json.",
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="ChickenDet root.")
    parser.add_argument("--source", type=Path, default=None, help="Image, directory, or glob.")
    parser.add_argument("--device", type=str, default=None, help="'cpu', '0', 'cuda'.")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the manifest's variants with their measured accuracy/latency, then exit.",
    )
    return parser


def main() -> None:
    """CLI entry point — see `_build_arg_parser` for `--help`.

    `export`'s import is local to this function: `export.py` imports `load_model` from this
    module at import time, so a top-level import here would be circular. Same pattern as
    `detection/yolo.py`'s CLI (docs/adr/0010).
    """
    from poultry_monitoring.export import default_manifest_path, read_manifest, summarize_manifest

    args = _build_arg_parser().parse_args()
    manifest_path = args.manifest or default_manifest_path(args.data_dir)

    if args.list:
        print(summarize_manifest(read_manifest(manifest_path)))
        return

    if args.weights is None and args.variant is None:
        raise SystemExit("Pass --weights <path> or --variant <name> (see --list).")
    if args.source is None:
        raise SystemExit("Pass --source <image|dir|glob>.")

    weights = args.weights
    if weights is None:
        entries = read_manifest(manifest_path)
        if args.variant not in entries:
            raise SystemExit(f"Unknown variant {args.variant!r}. Available: {sorted(entries)}")
        weights = Path(entries[args.variant]["path"])

    # warmup=True builds the predictor, which is what actually holds the backend --
    # without it `describe_runtime` has nothing to inspect and reports nulls, defeating
    # the device verification this print exists for (see `autobackend`).
    model = load_model(weights, task=args.task, device=args.device, imgsz=args.imgsz, warmup=True)
    print(json.dumps(describe_runtime(model), indent=2))

    results = run_inference(
        weights,
        args.source,
        task=args.task,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        save_dir=args.save_dir,
        model=model,
    )
    for result in results:
        n_masks = 0 if result.masks is None else len(result.masks)
        print(f"{result.path}: {len(result.boxes)} boxes, {n_masks} masks")
    if args.save_dir is not None:
        print(f"Saved annotated images to {args.save_dir}")


if __name__ == "__main__":
    main()
