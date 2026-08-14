# ADR 0016: Fingerprinted cache invalidation for the polygon annotation cache and `labels/`

## Status

Accepted

## Context

ADR 0013 introduced a persistent, on-disk cache for the RLE→polygon-converted COCO
annotations (`annotations_polygon_cache/`), and `convert_coco_to_yolo_labels` separately
treats an already-present `labels/<split>/` directory as done, skipping regeneration
unless `force=True`. Both checks were existence-only: "does an output path already
exist" — not "do the current inputs still match what produced it."

That's exactly the failure mode that bit: a `labels/` directory built *before* a
conversion-logic fix landed (RLE→polygon fidelity, ADR 0013's own fix) kept being served
on every subsequent `train`/`tune`/`sweep` run, silently, because nothing about the
existence check compared it against the fixed logic or the current source annotations.
Deleting `labels/` by hand was the only way to force a rebuild, and there was no signal
that it had gone stale short of noticing the trained model's masks looked wrong.

## Decision

Key both caches by a fingerprint of their *inputs* instead of by output existence alone:

- `_file_signature(path)`: a cheap per-file fingerprint — name + size + mtime, not a
  content hash (a content hash would mean reading every multi-hundred-MB ChickenDet
  annotation file just to decide whether to re-read it, defeating the point of caching).
- A module-level `LABEL_FORMAT_VERSION` constant folded into every signature, bumped
  whenever the conversion logic itself changes — so a pure logic fix invalidates the
  cache even when the source annotations didn't change (the exact scenario that caused
  the original incident).
- A small JSON manifest recorded alongside each cache, mapping source filename →
  signature. A split is only skipped when its cached output exists *and* the manifest's
  recorded signature for it still matches the current source file.

Applied in two places using the same signature scheme: `cache_polygon_annotations` (per
split JSON) and `convert_coco_to_yolo_labels` (the derived `labels/` directory as a
whole, via `_directory_fingerprint` over the annotations dir).

### Rejected alternatives

- **Content hash instead of name+size+mtime.** More precise, but reading a
  multi-hundred-MB annotations file in full just to fingerprint it costs roughly what
  re-converting it would — no actual caching win. name+size+mtime is the same
  good-enough-for-a-local-cache signal `make` and most build caches use.
- **Store the manifest inside the cache directory itself.** Tried first, since it's the
  obvious place — broke immediately: `convert_coco_to_yolo_labels` passes
  `cache_dir` straight to `convert_coco` as its `labels_dir` when `use_segments=True`,
  which globs every `*.json` file there (dotfiles included) expecting each one to be a
  COCO instances file. A manifest living inside got picked up as one and crashed with
  `KeyError: 'images'`. Fixed by writing the manifest as a sibling of `cache_dir`
  (`<cache_dir_name>.cache_manifest.json`) instead of a child.

## Consequences

- Regeneration is now automatic on a real source or logic change — no more manually
  deleting `labels/` to force a rebuild after a fix.
- mtime resolution is filesystem-dependent and coarser than a fast rewrite-in-place can
  land on; a source file rewritten within the same mtime tick as its previous version
  could theoretically go undetected as "changed." Not observed in practice, but the test
  suite works around it explicitly (`_bump_mtime_forward`) rather than relying on real
  wall-clock delay between writes.
- Two separate manifests now exist (polygon cache, `labels/`) rather than one shared
  one — deliberate, since the two caches can go stale independently (e.g. a
  `labels/`-only regeneration triggered by a class-remap change wouldn't need to touch
  the polygon cache at all).
