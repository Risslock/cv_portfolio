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
# they parameterize so the two can't drift out of sync.
PARAM_RANGES = {
    "p_color_invariance": (0.0, 0.3),
    "p_lighting": (0.1, 0.6),
    "brightness_limit": (0.1, 0.4),
    "contrast_limit": (0.1, 0.4),
    "autocontrast_cutoff": (0.0, 15.0),
}


def build_domain_transforms(
    p_color_invariance: float = 0.1,
    p_lighting: float = 0.4,
    brightness_limit: float = 0.3,
    contrast_limit: float = 0.3,
    autocontrast_cutoff: float = 0.0,
) -> list:
    """Build this project's custom Albumentations transform list.

    Two `OneOf` groups, kept separate so the effects don't compound:

    - Color invariance (`ToGray`/`ChannelDropout`, low probability): chickens are
      mostly white/cream against brown litter — discourages over-relying on one hue.
    - Lighting/contrast (`RandomBrightnessContrast`/`AutoContrast`): cross-facility
      lighting variation, and the "washed-out white chicken" observation from
      notebook 02. `AutoContrast`'s `cutoff` clips extreme-value outliers (percent,
      each end) before stretching, so a few blown-out pixels don't mute the effect.

    Not part of `model.tune()`'s search space (fixed scalar hyperparameters, not a
    transform list) — see `detection/yolo.py`'s `tune_augmentation_parameters`, which
    searches these five values instead, over `PARAM_RANGES` above.

    Args:
        p_color_invariance: Probability of applying the ToGray/ChannelDropout group.
        p_lighting: Probability of applying the brightness/contrast/autocontrast group.
        brightness_limit: `RandomBrightnessContrast` brightness jitter magnitude
            (applied as `±brightness_limit`).
        contrast_limit: `RandomBrightnessContrast` contrast jitter magnitude
            (applied as `±contrast_limit`).
        autocontrast_cutoff: `AutoContrast`'s cutoff percentage — see above.

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
                A.AutoContrast(cutoff=autocontrast_cutoff, p=1.0),
            ],
            p=p_lighting,
        ),
    ]
