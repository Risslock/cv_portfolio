"""Task-agnostic augmentation: lighting/color jitter.

Bbox-aware augmentation (mosaic, occlusion-aware crops) belongs in `detection.py`;
mask-aware augmentation (copy-paste) in `segmentation.py` — per `CLAUDE.md` § Package
Layout. Everything here is image-only (no coordinates to update), so it plugs directly
into Ultralytics' own `Albumentations` wrapper via the `augmentations=` kwarg documented
in `ultralytics.data.augment.v8_transforms` — see `detection/yolo.py` for the call site.

Chosen over relying solely on Ultralytics' built-in augmentations (constitution §
Technology Stack: Albumentations over DALI for augmentation) and scoped to complement,
not duplicate, what `model.tune()`'s own search space already covers (its `hsv_v`
hyperparameter already jitters brightness in HSV space) — this module targets what that
space doesn't: color-channel invariance and the "washed-out white chicken" contrast
problem noted in notebook 02's Notes section.
"""

import albumentations as A

# Every keyword `build_domain_transforms` accepts, and the range `detection/yolo.py`'s
# `tune_augmentation_parameters` searches each one over. Kept alongside the function
# they parameterize so the two can't drift out of sync. `autocontrast_cutoff` isn't
# here: unlike `RandomBrightnessContrast` (whose `brightness_limit`/`contrast_limit`
# Albumentations auto-symmetrizes into a fresh ±range each call), `AutoContrast.cutoff`
# is a single fixed float applied identically every time it fires — tuning it to one
# scalar (e.g. the winning search's 1.2, only ~1% of pixels clipped) locks in that exact
# clip amount forever instead of giving the augmentation any real sample-to-sample
# variety. `_RandomCutoffAutoContrast` below fixes that by sampling a fresh cutoff each
# application instead, so there's nothing left to tune here as a single scalar.
PARAM_RANGES = {
    "p_color_invariance": (0.0, 0.3),
    "p_lighting": (0.1, 0.6),
    "brightness_limit": (0.1, 0.4),
    "contrast_limit": (0.1, 0.4),
}

# Fixed sampling range for _RandomCutoffAutoContrast, not searched — see the comment on
# PARAM_RANGES above for why a single tuned cutoff value undersold what this transform
# is for. 10% is a meaningfully stronger clip than the 1.2% the search converged on,
# without being so aggressive it starts discarding real signal.
AUTOCONTRAST_CUTOFF_RANGE = (0.0, 10.0)


class _RandomCutoffAutoContrast(A.AutoContrast):
    """`AutoContrast` with a fresh, randomly sampled `cutoff` each time it's applied.

    Albumentations' own `AutoContrast.apply()` reads `self.cutoff` directly rather than
    threading it through `get_params()`'s params dict, so `get_params()` mutates
    `self.cutoff` as a side effect instead of returning it — the standard `apply()`
    inherited from `AutoContrast` then picks up the fresh value on the next call.

    Draws from `self.py_random` (Albumentations' own per-transform seeded RNG, reset by
    `set_random_seed`/`Compose`'s pipeline seed), not the bare `random` module — using
    the latter silently broke seeded reproducibility (`generate_augmented_samples`'
    `seed=` argument had no effect on this transform's draws), caught by
    `test_seed_gives_reproducible_draws` once the full suite finally ran again.
    """

    def __init__(
        self,
        cutoff_range: tuple[float, float] = AUTOCONTRAST_CUTOFF_RANGE,
        p: float = 0.5,
    ) -> None:
        super().__init__(cutoff=cutoff_range[0], p=p)
        self.cutoff_range = cutoff_range

    def get_params(self) -> dict:
        """Sample a fresh `cutoff` for this application; nothing to thread via params."""
        self.cutoff = self.py_random.uniform(*self.cutoff_range)
        return {}


def build_domain_transforms(
    p_color_invariance: float = 0.1,
    p_lighting: float = 0.4,
    brightness_limit: float = 0.3,
    contrast_limit: float = 0.3,
) -> list:
    """Build this project's custom Albumentations transform list.

    Two `OneOf` groups, kept separate so the effects don't compound:

    - Color invariance (`ToGray`/`ChannelDropout`, low probability): chickens are
      mostly white/cream against brown litter — discourages over-relying on one hue.
    - Lighting/contrast (`RandomBrightnessContrast`/`_RandomCutoffAutoContrast`):
      cross-facility lighting variation, and the "washed-out white chicken"
      observation from notebook 02. The autocontrast variant's `cutoff` (percent of
      pixels clipped from each end before stretching) is resampled every application
      from `AUTOCONTRAST_CUTOFF_RANGE` rather than fixed — see that constant's comment.

    Not part of `model.tune()`'s search space (fixed scalar hyperparameters, not a
    transform list) — see `detection/yolo.py`'s `tune_augmentation_parameters`, which
    searches these four values instead, over `PARAM_RANGES` above.

    Args:
        p_color_invariance: Probability of applying the ToGray/ChannelDropout group.
        p_lighting: Probability of applying the brightness/contrast/autocontrast group.
        brightness_limit: `RandomBrightnessContrast` brightness jitter magnitude
            (applied as `±brightness_limit`).
        contrast_limit: `RandomBrightnessContrast` contrast jitter magnitude
            (applied as `±contrast_limit`).

    Returns:
        A list of Albumentations transforms, for `model.train(augmentations=...)`.
    """
    return [
        A.OneOf([A.ToGray(p=1.0), A.ChannelDropout(p=1.0)], p=p_color_invariance),
        A.OneOf(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=brightness_limit, contrast_limit=contrast_limit, p=1.0
                ),
                _RandomCutoffAutoContrast(p=1.0),
            ],
            p=p_lighting,
        ),
    ]
