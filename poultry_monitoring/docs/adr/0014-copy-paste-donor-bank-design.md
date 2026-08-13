# ADR 0014: Curated, disk-cached donor bank for synthetic copy-paste, with scene-relative resizing

## Status

Accepted

## Context

Prototyping synthetic training-data generation via copy-paste in `notebooks/03_explore_segmentation.
ipynb` — `plan.md`'s pre-existing "Synthetic scene generation via cutout compositing" Future Work
item, now a working notebook prototype rather than only a plan. Three sub-decisions needed real
design, not defaults: which instances make good donors, how to store them, and how big to paste
them.

A first pass sampled donors uniformly at random across all COCO instances. On a densely-occluded
overhead dataset, that frequently grabbed a half-visible bird as a donor — pasting a partially
occluded fragment onto a new image looks obviously wrong, not like added density.

## Decision

1. **Curate donors, don't sample uniformly.** `find_unoccluded_untruncated_donor_instances` filters
   to instances whose mask fills most of its own bounding box and whose bbox doesn't touch the image
   border, reusing the same area-ratio signal the Mask Quality Checks section already computes for
   the whole dataset. Two thresholds, both retuned once against real data rather than left at their
   initial guesses:
   - `min_area_ratio` (mask/box area) landed at **0.6**, deliberately *below* the dataset's own
     observed median (0.64–0.68 across splits) — moderate occlusion is the typical case for a
     ChickenVerse instance, not an exception, so a threshold above the median would reject most of
     the population instead of just the genuinely badly-occluded tail. (Started at 0.85, revised
     down to 0.7 after the first real run, then to 0.6 once the median was measured directly.)
   - `border_margin` landed at **80px** (~8% of this dataset's 1024px frame width) — deliberately
     large, a quality-over-quantity choice. A margin of just a few pixels is already enough to catch
     *genuinely* truncated instances (a cut-off bird's visible silhouette naturally extends to the
     frame edge), so 80px isn't needed for truncation detection — it's insurance against fully-visible
     but near-edge instances that could still catch lens vignetting/distortion. Checked empirically
     before committing to it: the candidate pool stays abundant even at this size (tens of thousands
     of instances vs. the 500 the bank needs), so there's no real cost to being conservative here,
     unlike `min_area_ratio` above. (Started at 2px.)
2. **Store the curated bank as one PNG image + one PNG mask per donor**, on disk
   (`data/ChickenDet/copy_paste_donor_bank/`, persistent cache + `manifest.json`), not `.npz`, COCO
   json, or YOLO polygon labels.
3. **Resize each pasted donor toward the *target scene's own* instance-size distribution**, not a
   fixed jitter range or the donor's native pixel scale. `sample_domain_scale_factor` draws a
   reference size from `Normal(mean, std)` fit to the target image's own instances
   (`instance_sizes`), with `std` floored against an unreliably small sample.

### Rejected alternatives

- **Storage — `.npz`.** Functionally similar (single-file, lossless), but not browsable in an image
  viewer. That mattered directly: the curation thresholds (`min_area_ratio`, `min_mask_area`) were
  tuned by eyeballing the bank's actual contents, not just trusting a printed count.
- **Storage — COCO json.** Wrong grain: COCO's per-image/per-annotation structure is built for full
  scenes with many annotations, not a flat bank of independent single-instance crops.
- **Storage — YOLO polygon labels.** Would reintroduce the exact lossy RLE→polygon round-trip
  [ADR 0013](0013-rle-to-polygon-preprocessing-for-yolo-seg-conversion.md) fixed, for a mask already
  held as exact pixels. Storing the raster mask directly is strictly better here — no reason to
  re-encode it through a lossier format just to decode it straight back on load.
- **Sizing — bootstrap one random real instance's size + fixed multiplicative jitter.** The original
  prototype. Anchors every pasted donor's size to whichever single individual happened to get
  picked — an arbitrary extra source of variance unrelated to the scene's actual population, on top
  of the jitter meant to represent real variance. Domain motivation for the Normal(mean, std)
  approach instead: poultry flocks arrive roughly age/weight-matched and fan out into a real weight
  (and so apparent-size) distribution as they grow — a tightly-clustered young flock's synthetic
  birds should stay tightly sized, a size-diverse later-stage flock's should stay size-diverse. The
  scene's own mean and std capture that; a single bootstrapped individual doesn't.

## Consequences

- Bank built once — 500 curated donors (`max_donors` cap reached; even the final, more conservative
  0.6/70/80 thresholds leave a healthy candidate pool, ~27k instances measured directly before
  committing to them) — reused across the multi-donor/augmented/scale-matched demo cells without
  re-scanning the full COCO index each time.
- Per-image std is a small-sample estimate (often single-to-low-double-digit birds per image) —
  worth revisiting if production realism ever demands it: track std per camera installation, or
  condition it on flock age/production stage, instead of recomputing from one image's own noisy
  sample every time. Not needed yet.
- **Color/lighting harmonization was flagged here as not yet handled — since fixed.** ChickenVerse
  spans 5 facilities with different lighting, and a donor pulled from one facility's conditions could
  visibly clash with a target scene from another. Addressed by a LAB-space statistical color transfer
  — see [ADR 0015](0015-color-aware-donor-compositing.md). Both geometric (orientation, scale) and
  photometric (color) realism are now handled for pasted donors.
- Everything built as small, single-purpose functions — `instance_sizes`, `sample_domain_scale_
  factor`, `resize_donor`, `crop_to_mask_bbox` all take/return plain arrays, with no COCO or
  Albumentations dependency — specifically so they can move into `augmentation/segmentation.py`
  close to unchanged, per constitution Principle II (notebook proves it works; `src/` may redesign
  but doesn't have to reinvent).
