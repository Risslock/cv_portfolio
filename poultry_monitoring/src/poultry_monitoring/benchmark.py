"""Latency/throughput measurement for exported artifacts, under constitution Principle V.

Principle V asks for hardware, batch size, precision and device disclosed; warmup excluded;
averaged over multiple runs. Three choices here follow from that, and from what Phase 6's
exploration found (docs/adr/0020):

**Two numbers per cell, not one.** `forward` times the backend's forward pass with the
input tensor already resident; `end_to_end` times `predict()` on a real image, including
letterbox, normalization, and — for segmentation — prototype-mask assembly. Those diverge a
lot: mask assembly is largely CPU-bound and untouched by quantization, so reporting only the
forward pass overstates what a deployment gains, while reporting only end-to-end hides what
the quantization actually did.

**`forward` is a fixed-shape microbenchmark and is NOT a component of `end_to_end`.** It
always runs a square `imgsz x imgsz` tensor so every backend is compared on identical work.
`predict()`, by contrast, letterboxes to the source aspect ratio — 640x384 for ChickenDet's
1280x720 — which is ~40% fewer pixels. On CPU that is large enough that `forward` can exceed
`end_to_end` for a dynamic-shape artifact. Compare `forward` across backends; compare
`end_to_end` across deployments; do not subtract one from the other.

**Everything is measured through Ultralytics' `AutoBackend`**, via `inference.load_model`,
so pre/post-processing is identical across backends and only the graph differs.

**The device is verified, not assumed.** ONNX Runtime returns a working session that has
silently fallen back to CPU, and Ultralytics logs the requested provider before the session
exists — so a run records `providers` from the live session and flags any cell whose
execution didn't happen where it was asked to.
"""

import os

os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # before ultralytics; see inference.py

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable  # noqa: E402
from dataclasses import asdict, dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from poultry_monitoring.inference import (  # noqa: E402
    DEFAULT_IMGSZ,
    DEFAULT_TASK,
    autobackend,
    configure_runtime,
    describe_runtime,
    load_model,
    resolve_backend,
)

# Warmup covers lazy graph building, CUDA context creation, cuDNN algorithm selection and
# TensorRT's first-call tactic setup -- all one-off costs that would otherwise be charged
# to the first timed iteration and inflate the mean.
DEFAULT_WARMUP = 10
DEFAULT_ITERATIONS_GPU = 100
DEFAULT_ITERATIONS_CPU = 30  # CPU FP32 runs ~250 ms/image; 30 keeps a cell near 10 s


@dataclass
class LatencyStats:
    """Timing summary for one measured configuration.

    Attributes:
        mean_ms: Mean wall time per call.
        median_ms: Median — reported alongside the mean because a single scheduling stall
            skews the mean but not the median, and the gap between them is itself a signal.
        p95_ms: 95th percentile, the tail a deployment actually feels.
        std_ms: Standard deviation across timed iterations.
        min_ms: Fastest iteration observed.
        throughput_img_per_sec: Images per second, accounting for batch size.
        iterations: Timed iterations (warmup excluded).
        warmup: Discarded warmup iterations.
        batch: Images per call.
    """

    mean_ms: float
    median_ms: float
    p95_ms: float
    std_ms: float
    min_ms: float
    throughput_img_per_sec: float
    iterations: int
    warmup: int
    batch: int


def hardware_fingerprint() -> dict[str, object]:
    """Capture everything Principle V requires disclosed, automatically.

    Recorded per benchmark run rather than written into prose once, so a number can never
    drift away from the machine that produced it.

    Returns:
        Dict of CPU/GPU/OS/library details. GPU keys are absent when CUDA is unavailable.
    """
    info: dict[str, object] = {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or platform.machine(),
        "cpu_count_logical": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    try:
        import psutil

        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        pass

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_compute_capability"] = f"{major}.{minor}"
        info["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        info["cuda_torch_built_for"] = torch.version.cuda
        # Turing (7.5) has FP16 and INT8 tensor cores but neither BF16 nor FP8; recorded so
        # a reader knows which precisions were unavailable rather than merely untested.
        info["supports_bf16"] = major >= 8

    for module_name, key in [
        ("onnxruntime", "onnxruntime"),
        ("openvino", "openvino"),
        ("tensorrt", "tensorrt"),
        ("ultralytics", "ultralytics"),
    ]:
        try:
            info[key] = __import__(module_name).__version__
        except Exception:  # noqa: BLE001 - a missing backend is a fact to record, not raise
            info[key] = None
    return info


def measure_latency(
    call: Callable[[], object],
    batch: int,
    warmup: int = DEFAULT_WARMUP,
    iterations: int = DEFAULT_ITERATIONS_GPU,
    synchronize: bool = False,
) -> LatencyStats:
    """Time a callable, discarding warmup and summarizing the distribution.

    Args:
        call: Zero-argument callable performing exactly one inference.
        batch: Images per call, used for the throughput figure.
        warmup: Iterations to run and discard before timing.
        iterations: Timed iterations.
        synchronize: Call `torch.cuda.synchronize()` after each iteration. Required on GPU:
            CUDA work is queued asynchronously, so without it the timer measures how long
            the *launch* took, not the work. It covers PyTorch and any backend driving a
            torch CUDA stream; ONNX Runtime's `session.run` is already synchronous, where
            it is a harmless no-op.

    Returns:
        A `LatencyStats` over the timed iterations.

    Raises:
        ValueError: If `iterations` is below 1.
    """
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")

    for _ in range(warmup):
        call()
    if synchronize and torch.cuda.is_available():
        torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        call()
        if synchronize and torch.cuda.is_available():
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)

    mean_ms = statistics.fmean(samples)
    ordered = sorted(samples)
    return LatencyStats(
        mean_ms=round(mean_ms, 3),
        median_ms=round(statistics.median(samples), 3),
        p95_ms=round(ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)], 3),
        std_ms=round(statistics.stdev(samples) if len(samples) > 1 else 0.0, 3),
        min_ms=round(ordered[0], 3),
        # Computed from the unrounded mean, and guarded: a clock too coarse to resolve the
        # call would otherwise divide by zero rather than report an unmeasurably fast one.
        throughput_img_per_sec=round(batch * 1000.0 / mean_ms, 2) if mean_ms > 0 else float("inf"),
        iterations=iterations,
        warmup=warmup,
        batch=batch,
    )


def _forward_input(model, batch: int, imgsz: int, device: str | int | None) -> torch.Tensor:
    """Build a preprocessed input tensor matching what the backend expects.

    Reads the dtype off the live backend rather than the requested precision: an FP16 graph
    rejects an FP32 tensor, and `AutoBackend.fp16` is the only thing that knows which it is.
    """
    backend = autobackend(model)
    dtype = torch.float16 if getattr(backend, "fp16", False) else torch.float32
    target = "cuda" if (device not in (None, "cpu") and torch.cuda.is_available()) else "cpu"
    return torch.rand(batch, 3, imgsz, imgsz, dtype=dtype, device=target)


def benchmark_variant(
    weights: Path,
    source_image: Path,
    device: str | int | None = None,
    batch: int = 1,
    imgsz: int = DEFAULT_IMGSZ,
    task: str = DEFAULT_TASK,
    warmup: int = DEFAULT_WARMUP,
    iterations: int | None = None,
    cpu_threads: int | None = None,
    entry_is_dynamic: bool = True,
) -> dict[str, object]:
    """Measure one (artifact x device x batch) cell, both forward and end-to-end.

    Args:
        weights: Artifact path.
        source_image: A real image for the end-to-end measurement — synthetic noise would
            produce an unrepresentative number of detections and so unrepresentative
            postprocessing cost.
        device: `"cpu"`, `0`/`"cuda"`, or None for Ultralytics' choice.
        batch: Images per call.
        imgsz: Inference image size.
        task: Ultralytics task; must be explicit for exported artifacts.
        warmup: Warmup iterations, discarded.
        iterations: Timed iterations. Defaults by device, since CPU cells are ~10x slower.
        cpu_threads: Torch thread count to pin, for reproducible CPU numbers.
        entry_is_dynamic: Whether the artifact has a symbolic input shape. Recorded so the
            end-to-end column can be read within its shape class rather than across it.

    Returns:
        Dict with `runtime` (observed providers/device), `forward` and `end_to_end` stats,
        and any `forward_error` if the raw forward pass could not be timed.
    """
    on_gpu = device not in (None, "cpu")
    iterations = iterations or (DEFAULT_ITERATIONS_GPU if on_gpu else DEFAULT_ITERATIONS_CPU)
    configure_runtime(cpu_threads=cpu_threads)

    model = load_model(weights, task=task, device=device, imgsz=imgsz, warmup=True)
    runtime = describe_runtime(model)
    runtime["backend_family"] = resolve_backend(Path(weights))
    runtime["requested_device"] = str(device)
    # ORT can silently bind CPU after a CUDA request; record it rather than trust the ask.
    providers = runtime.get("providers")
    if on_gpu and providers is not None:
        runtime["device_verified"] = any("CUDA" in p or "Tensorrt" in p for p in providers)

    result: dict[str, object] = {"runtime": runtime}

    try:
        backend = autobackend(model)
        if backend is None:
            raise RuntimeError("no AutoBackend available; load_model(warmup=True) is required")
        tensor = _forward_input(model, batch, imgsz, device)
        result["forward"] = asdict(
            measure_latency(
                lambda: backend(tensor),
                batch=batch,
                warmup=warmup,
                iterations=iterations,
                synchronize=on_gpu,
            )
        )
    except Exception as error:  # noqa: BLE001 - a static graph may reject this batch size;
        # that is worth recording per-cell rather than aborting the whole sweep.
        result["forward_error"] = f"{type(error).__name__}: {error}"

    image = np.ascontiguousarray(
        np.repeat(np.expand_dims(_load_image(source_image, imgsz), 0), batch, axis=0)
    )
    batch_images = [image[i] for i in range(batch)]
    result["end_to_end"] = asdict(
        measure_latency(
            lambda: model.predict(batch_images, imgsz=imgsz, device=device, verbose=False),
            batch=batch,
            warmup=max(warmup // 2, 1),
            iterations=max(iterations // 4, 5),  # end-to-end is far slower than raw forward
            synchronize=on_gpu,
        )
    )

    # Ultralytics' own per-stage accounting, plus the shape it actually ran. Both are needed
    # to read the end-to-end column honestly: `predict` letterboxes a dynamic graph to the
    # source aspect ratio (640x384 here) but must pad a frozen graph to a full square
    # (640x640, ~1.7x the pixels). Two artifacts therefore do different amounts of work
    # end-to-end, so an end-to-end comparison is only fair *within* a shape class -- the
    # same confound that `export.baseline_for` handles on the accuracy side. `forward` is
    # immune, since it pins every backend to the same square tensor.
    probe = model.predict(batch_images[:1], imgsz=imgsz, device=device, verbose=False)[0]
    result["stage_ms"] = {k: round(v, 3) for k, v in probe.speed.items()}
    result["inference_shape"] = list(getattr(probe, "orig_shape", ()) or ())
    result["letterbox"] = "rect" if entry_is_dynamic else "square"
    return result


def _load_image(path: Path, imgsz: int) -> np.ndarray:
    """Read an image as BGR uint8, falling back to noise if it can't be read."""
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        return np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    return image


def _rel(path: str, root: Path) -> str:
    """Render an absolute artifact path relative to the repo, for a committed document."""
    try:
        return Path(path).resolve().relative_to(root).as_posix()
    except ValueError:
        return Path(path).name


def build_summary(manifest_path: Path, results_path: Path, repo_root: Path) -> str:
    """Join accuracy (manifest) and latency (benchmark results) into a committable report.

    The two live in separate files on purpose — accuracy is per artifact, latency is per
    (artifact x device x batch) — so this is where they are brought back together. Paths are
    emitted repo-relative and machine-specific fields are confined to one disclosed block,
    so the output is meaningful after a clone even though the JSON inputs are gitignored.

    Args:
        manifest_path: Export manifest, already populated by the accuracy sweep.
        results_path: `benchmark_results.json` from the latency sweep.
        repo_root: Directory paths are made relative to.

    Returns:
        Markdown document.
    """
    # Local import: `export` imports `inference` which this module also imports, and the CLI
    # below already follows this pattern (docs/adr/0010).
    from poultry_monitoring.export import baseline_for, entry_key, read_manifest

    entries = read_manifest(manifest_path)
    payload = json.loads(Path(results_path).read_text()) if Path(results_path).exists() else {}
    hardware, cells = payload.get("hardware", {}), payload.get("results", {})

    out: list[str] = [
        "# Phase 6 — Export & Optimization Results",
        "",
        "Generated by `python -m poultry_monitoring.benchmark --summary docs/export_results.md`.",
        "Regenerate after any re-export, re-score or re-benchmark; do not edit by hand.",
        "",
        "The underlying `manifest.json` / `accuracy_results.json` / `benchmark_results.json`",
        "live under the gitignored `data/` tree, so this file is the durable record.",
        "",
        "## Measurement conditions",
        "",
        "Constitution Principle V requires hardware, batch size, precision and device stated",
        "with every number.",
        "",
    ]
    for label, key in [
        ("GPU", "gpu"),
        ("Compute capability", "gpu_compute_capability"),
        ("VRAM (GB)", "gpu_vram_gb"),
        ("CPU", "cpu"),
        ("Physical cores", "cpu_count_physical"),
        ("Logical cores", "cpu_count_logical"),
        ("RAM (GB)", "ram_gb"),
        ("OS", "os"),
        ("Python", "python"),
        ("torch", "torch"),
        ("ONNX Runtime", "onnxruntime"),
        ("OpenVINO", "openvino"),
        ("TensorRT", "tensorrt"),
        ("ultralytics", "ultralytics"),
        ("torch threads", "torch_num_threads"),
    ]:
        if hardware.get(key) is not None:
            out.append(f"- **{label}**: {hardware[key]}")
    out += [
        "",
        "Warmup iterations are discarded; every figure is averaged over multiple timed runs.",
        "BF16 and FP8 are absent from the matrix because Turing (7.5) has neither.",
        "",
        "## Accuracy",
        "",
        "`Delta` is against the **shape-matched** PyTorch FP32 baseline: a frozen graph is",
        "letterboxed square while a dynamic one keeps the source aspect ratio, and that",
        "difference alone is worth ~0.7 mask mAP50 here. Comparing across it would bill",
        "padding to quantization (see `docs/optimization_playbook.md` section 5.2).",
        "",
    ]
    models = sorted({e.get("model", "?") for e in entries.values()})
    for model in models:
        out += [
            f"### `{model}`",
            "",
            "| Variant | Backend | Precision | Letterbox | Val box | Val mask | Test box | "
            "Test mask | d box | d mask |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for key in sorted(k for k in entries if entries[k].get("model") == model):
            e = entries[key]
            by = e.get("accuracy_by_split", {})

            def cell(split, metric):
                v = by.get(split, {})
                return f"{v[metric]:.4f}" if metric in v else "—"

            base = entries.get(entry_key(model, baseline_for(e)), {}).get("accuracy_by_split", {})

            def delta(metric):
                t, b = by.get("Test", {}), base.get("Test", {})
                if metric not in t or metric not in b or key.endswith(baseline_for(e)):
                    return "—"
                return f"{t[metric] - b[metric]:+.4f}"

            out.append(
                f"| `{e.get('variant')}` | {e.get('backend')} | {e.get('precision')} | "
                f"{'rect' if e.get('dynamic') else 'square'} | "
                f"{cell('Validation', 'box_map50_95')} | {cell('Validation', 'mask_map50_95')} | "
                f"{cell('Test', 'box_map50_95')} | {cell('Test', 'mask_map50_95')} | "
                f"{delta('box_map50_95')} | {delta('mask_map50_95')} |"
            )
        out.append("")

    if cells:
        out += [
            "## Latency",
            "",
            "`forward` pins every backend to the same square tensor, so it is the fair",
            "cross-backend comparison. `end-to-end` runs each artifact at its own shape and",
            "includes preprocessing and mask assembly, so compare it only *within* a letterbox",
            "class. The two are not subtractable.",
            "",
            "| Cell | Letterbox | forward mean | median | p95 | end-to-end | img/s | "
            "pre / inf / post |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for name in sorted(cells):
            c = cells[name]
            # Cell keys are `artifact|device|batch`; a bare pipe is a column separator in
            # markdown even inside a code span, so it has to be escaped.
            label = name.replace("|", r"\|")
            if "error" in c:
                out.append(f"| `{label}` | — | FAILED: {c['error']} | | | | | |")
                continue
            f, e2e = c.get("forward", {}), c["end_to_end"]
            st = c.get("stage_ms", {})
            fwd = f"{f['mean_ms']:.2f}" if "mean_ms" in f else "—"
            med = f"{f['median_ms']:.2f}" if "median_ms" in f else "—"
            p95 = f"{f['p95_ms']:.2f}" if "p95_ms" in f else "—"
            out.append(
                f"| `{label}` | {c.get('letterbox', '—')} | {fwd} | {med} | {p95} | "
                f"{e2e['mean_ms']:.2f} | {e2e['throughput_img_per_sec']:.1f} | "
                f"{st.get('preprocess', 0):.2f} / {st.get('inference', 0):.2f} / "
                f"{st.get('postprocess', 0):.2f} |"
            )
        out.append("")

    out += ["## Artifacts", "", "| Variant | Size (MB) | Build (s) | Path |", "|---|---|---|---|"]
    for key in sorted(entries):
        e = entries[key]
        out.append(
            f"| `{key}` | {e.get('size_mb', '—')} | {e.get('export_seconds', '—')} | "
            f"`{_rel(e.get('path', ''), repo_root)}` |"
        )
    return "\n".join(out) + "\n"


def devices_for(entry: dict) -> list[str | int | None]:
    """List the devices an artifact can actually run on.

    An exported artifact's device set is fixed when it is built, so the sweep asks each
    backend only for what it can serve rather than discovering the rest as failures:

    - **TensorRT** engines are CUDA-only by construction.
    - **OpenVINO** is the CPU target here (its GPU plugin means Intel graphics, which this
      machine does not have).
    - **ONNX** runs on both, except INT8 — the CUDA provider has no INT8 kernels for this
      graph and would silently fall back to CPU, producing a duplicate CPU measurement
      mislabelled as GPU.
    - **PyTorch** runs on both.

    Args:
        entry: A manifest entry carrying `backend` and `precision`.

    Returns:
        Devices to sweep, in `load_model`/`predict` form (`0` for CUDA, `"cpu"` for CPU).
    """
    backend = entry.get("backend")
    if backend == "tensorrt":
        return [0]
    if backend == "openvino":
        return ["cpu"]
    if backend == "onnx":
        # Both INT8 schemes are CPU-only here: the CUDA provider has no INT8 kernels for
        # this graph and would fall back to CPU, producing a duplicate CPU measurement
        # mislabelled as GPU.
        return ["cpu"] if entry.get("precision") in {"int8", "w8a32"} else [0, "cpu"]
    if backend == "pytorch":
        return [0, "cpu"]
    return [None]


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the `python -m poultry_monitoring.benchmark` CLI."""
    parser = argparse.ArgumentParser(description="Benchmark exported artifacts (Principle V).")
    parser.add_argument("--data-dir", type=Path, required=True, help="ChickenDet root.")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--variants", type=str, nargs="+", default=None, help="Manifest keys. Default: all."
    )
    parser.add_argument("--batch", type=int, nargs="+", default=[1], help="Batch sizes to sweep.")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--cpu-threads", type=int, default=None)
    parser.add_argument("--source", type=Path, default=None, help="Image for end-to-end timing.")
    parser.add_argument("--output", type=Path, default=None, help="Write results JSON here.")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Join the existing manifest and benchmark results into a committable markdown "
        "report at this path, then exit without measuring anything.",
    )
    return parser


def main() -> None:
    """CLI entry point — see `_build_arg_parser` for `--help`."""
    from poultry_monitoring.export import default_manifest_path, read_manifest
    from poultry_monitoring.mlflow_utils import finish_run, start_benchmark_run

    args = _build_arg_parser().parse_args()
    data_dir = args.data_dir.resolve()
    manifest_path = args.manifest or default_manifest_path(data_dir)

    if args.summary is not None:
        # Repo root: this file is src/poultry_monitoring/benchmark.py.
        repo_root = Path(__file__).resolve().parents[2]
        summary = build_summary(
            manifest_path, data_dir / "YOLO" / "export" / "benchmark_results.json", repo_root
        )
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(summary, encoding="utf-8")
        print(f"Wrote {args.summary} ({len(summary.splitlines())} lines)")
        return

    entries = read_manifest(manifest_path)
    if not entries:
        raise SystemExit("No manifest entries — run `python -m poultry_monitoring.export` first.")

    source = args.source or next((data_dir / "images" / "Test").glob("*.jpg"))
    hardware = hardware_fingerprint()
    print(json.dumps(hardware, indent=2))

    keys = args.variants or sorted(k for k in entries if "error" not in entries[k])
    results: dict[str, dict] = {}
    for key in keys:
        entry = entries[key]
        for device in devices_for(entry):
            for batch in args.batch:
                label = f"{key}|{'cuda' if device not in (None, 'cpu') else 'cpu'}|b{batch}"
                try:
                    measured = benchmark_variant(
                        Path(entry["path"]),
                        source,
                        device=device,
                        batch=batch,
                        imgsz=args.imgsz,
                        task=args.task,
                        warmup=args.warmup,
                        iterations=args.iterations,
                        cpu_threads=args.cpu_threads,
                        entry_is_dynamic=bool(entry.get("dynamic")),
                    )
                except Exception as error:  # noqa: BLE001 - keep sweeping other cells
                    print(f"FAIL  {label}  {type(error).__name__}: {error}")
                    results[label] = {"error": f"{type(error).__name__}: {error}"}
                    continue

                results[label] = measured
                e2e = measured["end_to_end"]
                fwd = measured.get("forward", {})
                print(
                    f"OK    {label:<58} fwd {fwd.get('mean_ms', float('nan')):>8.2f} ms   "
                    f"e2e {e2e['mean_ms']:>8.2f} ms   {e2e['throughput_img_per_sec']:>7.1f} img/s"
                )

                if not args.no_mlflow:
                    start_benchmark_run(run_name=label.replace("|", "-"))
                    finish_run(
                        extra_params={**hardware, "artifact": entry["path"]},
                        extra_tags={
                            "export_target": str(entry.get("backend")),
                            "precision": str(entry.get("precision")),
                            "device": "cuda" if device not in (None, "cpu") else "cpu",
                            "batch": str(batch),
                            "dynamic": str(entry.get("dynamic")),
                        },
                        extra_metrics={
                            "latency_ms_mean": e2e["mean_ms"],
                            "latency_ms_median": e2e["median_ms"],
                            "latency_ms_p95": e2e["p95_ms"],
                            "throughput_img_per_sec": e2e["throughput_img_per_sec"],
                            **({"forward_ms_mean": fwd["mean_ms"]} if "mean_ms" in fwd else {}),
                        },
                    )

    output = args.output or data_dir / "YOLO" / "export" / "benchmark_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    # Merge, don't replace: a `--variants` run measures a subset, and overwriting would
    # discard every cell it didn't touch. Re-measuring one contaminated cell is a normal
    # thing to want, so it must not cost the rest of the sweep.
    merged: dict[str, dict] = {}
    if output.exists():
        merged = json.loads(output.read_text()).get("results", {})
    merged.update(results)
    output.write_text(json.dumps({"hardware": hardware, "results": merged}, indent=2))
    print(f"\nSaved to {output} ({len(results)} cells measured, {len(merged)} total)")


if __name__ == "__main__":
    main()
