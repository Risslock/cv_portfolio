# ADR 0020: Export/quantization matrix — one load path, a manifest, and six silent failures

## Status

Accepted

## Context

Phase 6 needs a credible answer to "what does this model cost to run, and what does making
it cheaper cost in accuracy" — on CPU and GPU, across FP32/FP16/INT8, on this machine
(RTX 2060 SUPER, Turing cc 7.5; Ryzen 5 3600XT; Python 3.13, native Windows, torch cu124).

The design below is shaped almost entirely by things that turned out to be false when
tested. Each was verified against the installed `ultralytics 8.4.117`, not assumed, and each
fails **silently** — producing plausible numbers rather than an error.

### 1. Exported artifacts silently score as detection models

`nn/tasks.py`'s `guess_model_task` infers task from the *filename*: `-seg` in the stem, or
`segment` in the path parts. A `.pt` carries its task in the checkpoint, so this never
surfaces during training. An exported `.../yolo26n-seg-baseline-adamw/weights/best.onnx` has
the stem `best` and no path part named `segment`, so the match fails and it returns
`"detect"`. Scoring then runs a `DetectionValidator` and yields a complete, believable
metrics table with **no mask columns at all** — the entire segmentation result, silently
absent. Measured on a 60-image validation subset:

| Load | task | box_map50 | mask_map50 |
|---|---|---|---|
| `YOLO(onnx)` | `detect` (guessed) | 0.9779 | *absent* |
| `YOLO(onnx, task="segment")` | `segment` | 0.9779 | 0.9772 |

### 2. A static export appears to lose accuracy, and doesn't

The obvious reading of the numbers below is "ONNX export costs ~0.7 mask mAP50":

| Configuration | box_map50 | mask_map50 |
|---|---|---|
| PyTorch `.pt`, `rect=True` | 0.9849 | 0.9847 |
| ONNX **dynamic** | 0.9849 | 0.9847 |
| PyTorch `.pt`, `rect=False` | 0.9779 | 0.9772 |
| ONNX **static** (`1x3x640x640`) | 0.9779 | 0.9772 |

Forcing the *PyTorch* model to `rect=False` reproduces the static export exactly, and
`rect=True` reproduces the dynamic export exactly. ONNX FP32 export is numerically faithful
to four decimal places; the gap is entirely **letterboxing**. Ultralytics' validator defaults
to `rect=True`, padding each batch to its own aspect ratio (roughly 640x384 for ChickenDet's
1280x720), which a graph frozen at 640x640 cannot do — so it gets square padding instead.

Left unexamined, that 0.7-point preprocessing artifact would have been reported as
quantization damage in every static variant's row.

### 3. "Using CUDAExecutionProvider" is not evidence of GPU execution

ONNX Runtime fails soft: request `CUDAExecutionProvider`, and if it cannot load you still
get a working session, no exception, running on CPU. Only `session.get_providers()` reveals
it. This is not hypothetical here — `onnxruntime-gpu` 1.29 links against CUDA 13
(`cublasLt64_13.dll`) while this project's torch is cu124, so the provider listed but never
loaded.

Ultralytics makes it worse rather than better. `nn/backends/onnx.py` gates on
`get_available_providers()` — which *lists* CUDA EP even when it cannot initialize — appends
`CPUExecutionProvider` as a fallback, and logs `Using ONNX Runtime ... with
CUDAExecutionProvider` **before** constructing the session. Its explicit "CUDA requested but
not available" warning only fires when the provider isn't listed at all, which is a different
failure. So the log asserts GPU while the work happens on CPU.

### 4. Ultralytics installs packages behind the lockfile

`check_requirements(("onnx", "onnxruntime"))` runs when an ONNX model is loaded on CPU. This
project pins `onnxruntime-gpu`, so the distribution metadata for plain `onnxruntime` is
absent and AutoUpdate pip-installs it — outside `uv.lock`, violating constitution Principle
VII. Both distributions install into the same `onnxruntime/` package directory, so the two
builds collide; removing the stray one then deleted shared files and left `onnxruntime`
importable but without `InferenceSession` until the GPU build was force-reinstalled. This
happened during Phase 6 development, not in theory.

### 5. The letterbox confound corrupts latency too — and there it is harder to see

The static/dynamic padding difference was caught early on the accuracy side and handled by
`baseline_for`. The same discipline was **not** applied to latency, and it produced a
confidently wrong conclusion: `openvino-int8` measured 1.44x *slower* end-to-end than
`openvino-fp32`, which read as "INT8 doesn't pay off on this CPU".

It does. The INT8 artifact is static, so `predict()` pads it to a full 640x640 while the
dynamic FP32 artifact keeps 640x384 — about 1.7x the pixels, in both the inference stage and
the mask assembly that follows it. Measured at an identical square shape via the forward
pass, OpenVINO INT8 is **1.24-1.30x faster** than FP32 (medians, both model sizes).

Two hypotheses were tested and rejected before finding this: that degraded INT8 produced
more detections and so more mask work (detection counts are identical, 55.3 vs 55.5 per
image), and that Zen 2's lack of VNNI made INT8 slow (true that this CPU reports only AVX2,
but OpenVINO gets its speedup regardless).

### 6. "INT8" hides an enormous spread between toolchains

Same FP32 graph, same calibration data, same nominal precision — measured against
shape-matched baselines on the held-out test split:

| Toolchain | Δ box mAP50-95 | Δ mask mAP50-95 | Forward vs. its FP32 |
|---|---|---|---|
| TensorRT | −0.009 | −0.008 | ~1.0x (small model), faster (large) |
| OpenVINO (NNCF) | −0.014 | −0.008 | **1.24-1.30x faster** |
| ONNX Runtime static | **−0.113** | −0.036 | **1.68-1.89x slower** |

ONNX Runtime's static INT8 is worse on both axes at once — an order of magnitude more
accuracy damage *and* slower than the FP32 it replaced. Nothing about the label "INT8"
predicts that.

## Decision

**One load path.** `inference.load_model` is the only place this package constructs a
`YOLO`. Export, accuracy scoring, benchmarking and prediction all go through it. This makes
"what you benchmark is what you serve" structural rather than conventional, and puts the
`task=` requirement, the DLL registration and the AutoUpdate guard in one place instead of
four. `describe_runtime` reports the session's *actual* providers, so a benchmark asserts on
observed execution rather than the requested device.

**`YOLO_AUTOINSTALL=false`, set before importing ultralytics** at the top of `inference.py`
and `export.py` — `AUTOINSTALL` is read once, at ultralytics import time, so a later
assignment is too late.

**Explicit `os.add_dll_directory` for torch's `lib/`**, rather than relying on `import torch`
preceding `import onnxruntime`. Torch's bundled CUDA 12 + cuDNN 9 libraries are exactly what
the pinned `onnxruntime-gpu` build needs, but depending on import order for that is a trap
for the next person to reorder imports.

**Two upper-bound pins**, both from running the backends rather than trusting resolution:
`onnxruntime-gpu>=1.22,<1.23` (1.23+ needs CUDA 13) and `tensorrt>=10.7,<11` (TRT 11 removed
`BuilderFlag.FP16`/`INT8`/`OBEY_PRECISION_CONSTRAINTS`, which `ultralytics/utils/export/
engine.py` still sets, so engine export raises `AttributeError`).

**Static and dynamic are separate exports, and each compares against its own baseline.**
`export.baseline_for` maps a variant to `pytorch-fp32-dynamic` or `pytorch-fp32-static` by
its `dynamic` flag, so no quantization delta ever spans the letterboxing boundary. TensorRT
engines are always static — a dynamic engine gives up the fixed-profile optimization that is
the only reason TensorRT is in the matrix.

**A manifest is the registry, and results are stored at their own grain.** `export.py`
writes one entry per (model, variant) with path, backend, precision, `dynamic`, size and
build time. The accuracy stage merges its scores into those same entries, because **accuracy
is a property of the artifact** — one number per artifact per split.

Latency is deliberately *not* merged in. It is a property of **(artifact x device x batch)**,
a different cardinality: one artifact yields several latency cells, which would have to be
nested as a list inside an entry and would no longer be keyed by anything the manifest is
organized around. It goes to `benchmark_results.json`, keyed by cell, alongside the hardware
fingerprint the numbers are only meaningful with. The two files join on the manifest key. It is also what makes `inference --variant <name>` possible instead of typing
paths to one of fourteen artifacts.

**Weight-only INT8 is in the matrix as `onnx-w8a32`, and it earns its place by losing.**
It needed first-party code (`W8A32_FORMATS = frozenset({"litert"})` blocks the native
route) and a non-obvious flag: quantizing Conv dynamically emits `ConvInteger`, whose ONNX
Runtime CPU kernel takes unsigned weights only, so a `QInt8` graph saves fine and then fails
at session creation with `NOT_IMPLEMENTED`. With `QUInt8` it runs — and measures **1.8x
slower than the FP32 it replaced** (65.6 -> 120.0 ms forward) at 3x smaller and ~3 points of
box mAP. Notably it still beats ONNX Runtime's *static* INT8 on both speed and accuracy,
which is a comment on that implementation rather than a recommendation for this one. Kept as
a measured negative result: the matrix should be able to say "we tried it, here is the cost".

**`quantize=`, not `int8=`/`half=`** — the latter are deprecated in 8.4.117 and forwarded
with a warning. ONNX **dynamic** INT8 is the one path with no Ultralytics route
(`W8A32_FORMATS = frozenset({"litert"})`), so `quantize_onnx_dynamic` calls ONNX Runtime
directly; it is in the matrix as a measured comparison, not a serious candidate, since
weight-only INT8 is frequently slower than FP32 on Conv-heavy CPU graphs.

**LiteRT is dropped**, not attempted-and-fixed. Its `tensorflow` + `onnx2tf` toolchain is the
least likely to install on Python 3.13 / native Windows, and the matrix already covers CPU
INT8 twice (ONNX Runtime static, OpenVINO NNCF).

## Consequences

- Every accuracy number in Phase 6 is stated against a shape-matched baseline. Static
  variants therefore carry two separately-reported costs — letterboxing and quantization —
  rather than one conflated number. This is more honest and more useful, but it means the
  results table has more columns than a naive one would.
- `describe_runtime` must actually be *called* for the provider assertion to be worth
  anything. A benchmark that loads through `load_model` but never checks the result can still
  publish CPU numbers as GPU ones; the check is available, not automatic.
- Pinning below the current release of two backends means neither gets security or
  performance updates without revisiting this. Both pins have a concrete unblocking
  condition: a cu13 torch for ONNX Runtime, and Ultralytics adopting TensorRT's
  strongly-typed builder API for TensorRT.
- Ultralytics writes an export next to its *source* `.pt` and derives the artifact name from
  it, so `export_variant` copies the checkpoint into a per-variant directory. Without that,
  variants overwrite each other — both TensorRT engines are named `model.engine`. Artifact
  paths are read from `export()`'s return value rather than predicted, since INT8 lands as
  `model_int8.onnx` and OpenVINO as a `*_openvino_model/` directory.
- A TensorRT INT8 engine (7.8 MB) came out no smaller than its FP16 counterpart (7.7 MB).
  Engine size does not disclose the precision actually used — TensorRT selects per-layer
  tactics and will keep layers in higher precision where INT8 would be slower. "INT8 built"
  is therefore not "INT8 ran", and only the latency and accuracy measurements can tell them
  apart. Worth watching when those numbers land.
- **Two latency numbers per cell, and they are not subtractable.** `forward` pins every
  backend to the same square tensor, so it is the only fair cross-backend comparison;
  `end_to_end` runs each artifact at its natural shape, so it is the deployment number but
  is only comparable *within* a shape class. Each cell records its letterbox class and
  Ultralytics' per-stage split to keep that readable.
- **After TensorRT, the model stops being the bottleneck.** `engine-fp16` on `yolo26n-seg`:
  3.07 ms forward against 9.32 ms end-to-end, split 2.13 preprocess / 3.85 inference / 3.77
  postprocess. TensorRT is ~4.3x faster than PyTorch on the inference stage but only ~1.9x
  end-to-end, because roughly two-thirds of the frame budget is letterboxing and mask
  assembly — neither of which quantization touches. Further gains have to come from the
  pipeline, not the network.
- **CPU measurements need their dispersion reported, not just a mean.** ONNX and PyTorch
  cells are stable (1.4-3.4% coefficient of variation) but OpenVINO's run 14.5-20.5%, and
  one cell was contaminated by outside load badly enough to shift its mean 39% while its
  median moved 15%. Conclusions here are checked against median and min, not mean alone.
- Disabling AutoUpdate means a genuinely missing backend now fails loudly instead of being
  installed mid-run. That is the intent, but it does mean a fresh clone must
  `uv sync --extra export` before any export works.
