# ADR 0013: Preprocess RLE segmentations to polygons before `convert_coco`, cache persistently

## Status

Accepted

## Context

Built while verifying `data/coco.py`'s existing `convert_coco_to_yolo_labels(..., use_segments=True)`
call in `notebooks/03_explore_segmentation.ipynb` (Phase 3 exploration), not writing new conversion
code from scratch.

`check_mask_quality` (same notebook) confirmed ChickenDet's segmentation annotations are 100%
compressed RLE across all three splits — zero native polygons. `ultralytics.data.converter.
convert_coco`, which `data/coco.py` wraps, can't read that: whenever a segmentation "is missing,
empty, or not a point list such as an RLE mask" (its own warning text), it silently substitutes a
box-shaped segment instead of the real mask.

Confirmed empirically, not just from the warning text: rasterizing a converted label file's
polygons back to pixel space and comparing against the original COCO mask (via IoU) on a sample
image measured **mean IoU 0.63, min 0.29** — the converter really was producing bbox-shaped masks,
not segmentation, for every RLE-only instance.

## Decision

Preprocess every RLE segmentation to a polygon (or several, for disjoint contours) with OpenCV
(`cv2.findContours` + `cv2.approxPolyDP`, ~0.5px tolerance) before handing the annotation JSON to
`convert_coco`. Any annotation OpenCV can't extract a usable contour from is logged by id, not
silently left as RLE — closing the one remaining point where the original silent-bbox-fallback
failure mode could still happen unnoticed.

Cache the polygon-converted JSON to disk persistently (`data/ChickenDet/annotations_polygon_cache/`)
rather than regenerating it every run — the RLE-decode + contour-extraction pass costs roughly
1–2 minutes for Train's 116k annotations.

Verified: the same IoU check against the polygon-cache-based conversion measured **mean IoU 0.97,
min 0.91** on the same sample image.

### Rejected alternatives

- **A custom COCO→YOLO-seg converter, bypassing `convert_coco` entirely.** Would throw away
  `convert_coco`'s already-tested class-remapping and train/val/test directory-layout logic for no
  real gain — RLE input is the only actual gap, nothing else about the conversion needs replacing.
  Reuse-with-a-patch over reinvention (constitution Principle I).
- **Recomputing the RLE→polygon pass on every run instead of caching.** Acceptable for a one-off
  notebook cell, not for a production path invoked on every `train`/`tune`/`sweep` run.

## Consequences

- `RETR_EXTERNAL` (used for contour extraction) only keeps outer contours — an instance mask with a
  genuine hole (occluded through its middle by another bird) gets silently filled in the polygon
  version. Not fixable by a different OpenCV flag: YOLO-seg's own polygon format has no hole
  representation at all. A small, permanent fidelity gap specific to this dataset's occlusion
  pattern, not something the IoU check above will flag as wrong per se.
- `data/coco.py`'s `convert_coco_to_yolo_labels` should adopt this same disk-cache pattern before
  Phase 3 training runs at real scale — not yet done; notebook-only so far.
- The cache directory lives under the already-gitignored `/data/` tree — no new `.gitignore` entry
  needed.
