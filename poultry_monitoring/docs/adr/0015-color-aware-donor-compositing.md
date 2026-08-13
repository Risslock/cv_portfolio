# ADR 0015: LAB-space statistical color transfer for donor/scene color matching

## Status

Accepted

## Context

[ADR 0014](0014-copy-paste-donor-bank-design.md) flagged, but didn't fix, a visible gap: a
donor pulled from one ChickenVerse facility's lighting can carry a noticeably different color
cast than the target scene it gets pasted into — first spotted by eyeballing a raw (non-mask-
overlaid) composite while building a README example, where several pasted birds read as
obviously bluish/gray against an otherwise warm-lit scene.

Same shape of problem the domain-aware resize ([ADR 0014](0014-copy-paste-donor-bank-design.md))
already solved for size: a donor's native pixel statistics reflect *where it came from*, not
*where it's going*.

## Decision

`masked_pixel_stats(image, masks)` pools pixels from every real instance mask in the target
scene (not the background — litter is a completely different color from a bird) into a single
per-channel **LAB** mean + std. `match_color_to_target` then applies a Reinhard-style linear
transfer: recenter and rescale the donor's own LAB pixels from its own mean/std onto the
target's, with a `strength` blend factor and a floor on the donor's std (guards a near-solid-
color donor crop from an exploding scale factor).

Verified end-to-end against the real donor bank and a real target scene: 5 sampled donors with
LAB means ranging `[142,127,132]`–`[212,130,127]` before correction all converged to within
noise of the target's actual `[190.5, 121, 156.4]` after. Visually, texture and shading survive
the transfer untouched — only the color cast shifts.

### Rejected alternatives

- **RGB-space matching instead of LAB.** RGB's three channels are perceptually correlated —
  independently matching mean/std per RGB channel can fight itself, correcting brightness and
  color-cast as one entangled operation instead of two clean, separable ones. LAB's L channel
  isolates lightness from the a/b chrominance pair, so both the "too dark/bright" and the
  "wrong hue" parts of the mismatch get corrected together without interference.
- **Full histogram matching/specification** (matching the donor's empirical CDF to the target's,
  not just its mean/std). More precise in principle, but needs either a new dependency
  (`scikit-image`) or a hand-rolled CDF-mapping implementation, and is far more sensitive to a
  small reference sample — a scene with only a handful of real birds gives a noisy, spiky
  histogram to match against, where a mean/std summary stays stable. Same reasoning that already
  favored `Normal(mean, std)` over resampling raw per-instance values for size matching in ADR
  0014 — consistent design language, not a one-off choice.
- **Averaging per-instance color means instead of pooling all real-instance pixels together.**
  Would weight a 20-pixel sliver of a bird the same as a 2000-pixel full bird in the final
  estimate. Pooling raw pixels implicitly weights by how much of each instance is actually
  visible — a more reliable color sample dominates a noisier one, rather than being diluted by it.
- **Matching against the whole image (including background) instead of masked instance pixels
  only.** The litter background is a completely different color population from a bird's
  feathers; matching a donor toward "average scene color" would just tint it yellowish-green
  toward the ground, not toward what a real bird looks like under that scene's lighting.

## Consequences

- Resolves the color/lighting gap [ADR 0014](0014-copy-paste-donor-bank-design.md) documented as
  known-but-unfixed. Geometric (orientation, scale) and photometric (color) realism are now both
  handled for pasted donors.
- `masked_pixel_stats` pools *every* real instance in the scene into one mean/std, implicitly
  assuming roughly uniform lighting across the frame. A scene with strong local lighting
  variation (hard shadow on one side, bright spot on the other) would pull a donor toward the
  scene's *average* rather than wherever it specifically lands. Not an observed problem in
  ChickenVerse's fairly even overhead lighting so far — worth re-checking if a future scene shows
  otherwise, not something to pre-solve now.
- `match_color_to_target` is a plain function over arrays (no COCO/Albumentations dependency),
  same atomic-function shape as `resize_donor`/`sample_domain_scale_factor` — intended to move
  into `augmentation/segmentation.py` largely unchanged, per constitution Principle II.
