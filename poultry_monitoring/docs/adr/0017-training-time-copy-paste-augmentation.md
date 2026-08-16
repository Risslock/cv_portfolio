# ADR 0017: Copy-paste as a training-time transform, not a materialized dataset

## Status

Accepted

## Context

[ADR 0014](0014-copy-paste-donor-bank-design.md)/[0015](0015-color-aware-donor-compositing.md)
established *what* a good synthetic instance looks like (curated donor bank, scene-relative
size, LAB colour transfer). They left open *where* the compositing runs.

The first implementation was an offline materializer (`segmentation/synthetic_data.py`):
walk the Train split once, write composited images + labels to
`images/TrainSynthetic/`, train on real + synthetic combined. That works, but every image
gets exactly one frozen set of donors for the whole run — the augmentation is baked in at
generation time rather than resampled per epoch, which is the opposite of how every other
augmentation in this project behaves.

## Decision

Paste donors **per-sample inside the training loop**, in memory, via
`segmentation/copy_paste_training.py`. Donors, orientation, scale, colour and placement
are redrawn every epoch; nothing is written to disk. The offline materializer is kept and
documented as the fallback (see Consequences).

Three thin overrides, all reusing framework-free primitives from
`augmentation/segmentation.py`:

- `BankCopyPaste` — a transform over Ultralytics' `labels` dict, appending pasted donors
  to `Instances`/`cls`. Inserted **after** the mosaic/affine `pre_transform` and before
  `Albumentations`, so it runs once per sample on the final-size canvas.
- `DonorBankYOLODataset` — overrides `build_transforms` to splice it in.
- `DonorBankSegmentationTrainer` — overrides `build_dataset`, since `build_yolo_dataset`
  hardcodes the dataset class. Configuration travels as *class* attributes
  (`make_donor_bank_trainer`) because `model.train()` instantiates the trainer itself and
  `get_cfg` rejects unknown keys, so the settings cannot ride in as `model.train()`
  overrides.

**Scene statistics are spatially local, not canvas-global.** Colour and size references
are pooled from real instances within `DEFAULT_LOCALITY_RADIUS` (160 px) of the paste
location, falling back to the whole frame when too few neighbours are nearby. With mosaic
at its stock default a canvas stitches 4 different scenes — and therefore 4 lighting
conditions — so a canvas-wide match would aim a donor at an average of facilities instead
of its actual surroundings. This keeps mosaic (and so plan.md's "stock defaults in both
stages" commitment) intact while closing the local-lighting limitation ADR 0015 recorded
as known-but-unfixed.

### Rejected alternatives

- **A custom Albumentations transform**, the natural guess given `shared.py`/`detection.py`
  already plug in that way. Ultralytics' `Albumentations` wrapper does thread segments
  through, but it only *maps existing* instances and filters them by `idx`
  (`data/augment.py:2202-2273`) — there is no path to append one. Copy-paste adds
  instances, so this is structurally impossible, not merely awkward.
- **A custom DataLoader** (Ultralytics' or a plain `torch.utils.data.DataLoader`).
  Wrong layer: `BaseDataset.__getitem__` is `return self.transforms(self.get_image_and_label(index))`
  (`data/base.py:395`), so augmentation happens *inside* the Dataset and a loader only
  ever sees transformed samples. Worker parallelism is already ours regardless, since
  `InfiniteDataLoader` subclasses torch's. (`ultralytics/data/loaders.py` is unrelated —
  it holds inference sources for `model.predict`.)
- **Injecting from a callback** by mutating `dataset.transforms` after construction. Looks
  lighter than subclassing but is silently broken: `close_mosaic`
  (`data/dataset.py:363-373`) does `self.transforms = self.build_transforms(hyp)`, a full
  rebuild fired mid-training (`engine/trainer.py:454-456`), which drops the injected
  transform for the final epochs with no error. Overriding `build_transforms` survives it
  because it *is* the rebuild path.
- **A `torch.utils.data.Dataset` written from scratch.** Right layer, but it means
  reimplementing the label cache and corrupt-image verification, `load_image` with the
  mosaic buffer, image caching, rect batching, `update_labels_info`, `close_mosaic`,
  `collate_fn`'s `batch_idx` assembly and the whole `v8_transforms`/`Format` chain, then
  re-verifying all of it on every Ultralytics upgrade. Subclassing *is* the custom-dataset
  route, with one method overridden instead of a dozen reimplemented.
- **Compositing on GPU in `preprocess_batch`.** Masks are already rasterized and
  `mask_ratio`-downsampled there, so `batch_idx`/`cls`/`bboxes` would all need surgery,
  and it would contend for GPU while the worker CPUs sit idle.

## Consequences

- **Three bugs found by building this:**
  0. The donor bank stores raw COCO `category_id` (ChickenDet's is `1`), but a
     single-class model only has class `0`. The trainer passed the manifest through
     unmapped, so training indexed a class column that doesn't exist and died with an
     opaque `CUDA error: device-side assert triggered` inside the task-aligned assigner.
     Notable because the **unit tests could not catch it** — their fixture handed the
     transform an already-remapped manifest, so only an end-to-end run on real data
     exercised the path. Fixed by `remap_manifest_to_yolo_class_ids`, which also
     validates against `nc` so the failure is a readable `ValueError` rather than a CUDA
     assert. Its counterpart in the offline path had the remap from the start; the
     on-the-fly path simply didn't inherit it.
  The other two are silent-corruption class — nothing raises, the labels are just wrong:
  1. `cv2.fillPoly(canvas, [a, b], 1)` applies an **even-odd winding rule across
     contours**, so two *overlapping* instances cancel and leave a hole — measured 480 px
     for one call vs. 651 looped per-contour. On high-occlusion ChickenVerse frames that
     would mark the densest regions as free space for placement and drop them from the
     colour statistics. `rasterize_polygons` therefore fills one contour per call.
  2. Importing `ultralytics` **monkeypatches `cv2.imread` globally**
     (`ultralytics/utils/patches.py`, "Always ensure 3 dimensions"), so
     `IMREAD_GRAYSCALE` returns `(H, W, 1)` instead of `(H, W)`. Donor masks are squeezed
     back explicitly; this only reproduces once ultralytics is imported, which is why it
     surfaced under pytest and not in an isolated check.
- **Measured cost** (640x640 canvas, RTX 2060 SUPER, `p=1.0`, `max_donors=5`):

  | instances | naive mask-per-instance stats | polygon-native | speedup |
  |---|---|---|---|
  | 50 | 106 ms | 12.3 ms | 8.7x |
  | 200 (mosaic-scale) | 434 ms | 27.9 ms | 15.5x |

  Full transform, steady state: **18 ms** (50 instances) / **30 ms** (200). At the default
  `p=0.3` that averages ~7 ms/sample, i.e. ~14 ms of CPU per batch of 16 across 8 workers
  against a GPU step an order of magnitude longer — hidden by prefetch. The naive variant
  at 200 instances would have been ~870 ms/batch and dataloader-bound.
- **Donor I/O dominated until cached**: reading a donor costs ~9 ms versus ~0.01 ms cached
  (Windows per-file-open overhead; warming the OS page cache does *not* help), and was 47%
  of transform runtime under profiling. Caching the whole 2000-donor bank would cost
  ~135 MB *per worker*, so each worker instead draws its own random `donor_pool_size`
  (400) slice and caches it fully — the bank stays covered across workers at ~27 MB each.
- Validation data is deliberately left untouched: synthetic instances belong in training
  only, never in the numbers a run is judged on.
- **Copy-paste keeps running after `close_mosaic`, deliberately.** At
  `epoch == epochs - close_mosaic` (default 10) Ultralytics rebuilds the transform chain
  and retires mosaic/mixup/cutmix/its own copy-paste, the usual "let the model settle on
  clean data" step. Because the insertion lives in `build_transforms`, `BankCopyPaste` is
  re-inserted on that rebuild and stays active to the last epoch. Kept on rather than
  retired with the rest: it *is* the Stage B treatment, so switching it off for the final
  10% would dilute the intervention being measured, and a pasted donor is real,
  colour/size-matched bird pixels rather than a 4-way stitched canvas — much closer to the
  true distribution than the augmentations Ultralytics is retiring at that point. Noted
  here because the behaviour otherwise reads as an accident of where the hook lives; a
  `copy_paste_close` flag exposes it as a switch, defaulting to off (keep pasting).
- **Known limitation, not fixed**: placement only avoids *other instances*. Nothing stops
  a donor landing on a feeder or drinker, which the visual check confirmed happens
  occasionally. Real birds do stand next to equipment so it is not obviously wrong, but a
  scene-furniture mask would be the fix if it ever looks like it matters.
