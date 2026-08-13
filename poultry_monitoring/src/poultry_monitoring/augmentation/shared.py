"""Task-agnostic augmentation: lighting/color jitter.

Image-only (no coordinates to update), so it plugs directly into Ultralytics' own
`Albumentations` wrapper via `model.train(augmentations=...)`. Bbox-aware augmentation
belongs in `detection.py`, mask-aware in `segmentation.py` — see CLAUDE.md § Package
Layout.
"""

import albumentations as A

# Searched by detection/yolo.py's tune_augmentation_parameters.
PARAM_RANGES = {
    "p_color_invariance": (0.0, 0.3),
    "p_lighting": (0.1, 0.6),
    "brightness_limit": (0.1, 0.4),
    "contrast_limit": (0.1, 0.4),
    "autocontrast_cutoff_max": (0.0, 20.0),
}


class _RandomCutoffAutoContrast(A.AutoContrast):
    """`AutoContrast` with a fresh `cutoff` sampled from `[0, cutoff_max]` each call."""

    def __init__(self, cutoff_max: float, p: float = 0.5) -> None:
        super().__init__(cutoff=0.0, p=p)
        self.cutoff_max = cutoff_max

    def get_params(self) -> dict:
        self.cutoff = self.py_random.uniform(0.0, self.cutoff_max)
        return {}


def build_domain_transforms(
    p_color_invariance: float = 0.1,
    p_lighting: float = 0.4,
    brightness_limit: float = 0.3,
    contrast_limit: float = 0.3,
    autocontrast_cutoff_max: float = 10.0,
) -> list:
    """Build this project's custom Albumentations transform list.

    Args:
        p_color_invariance: Probability of the ToGray/ChannelDropout group.
        p_lighting: Probability of the brightness/contrast/autocontrast group.
        brightness_limit: `RandomBrightnessContrast` brightness jitter (±).
        contrast_limit: `RandomBrightnessContrast` contrast jitter (±).
        autocontrast_cutoff_max: Upper bound for the autocontrast cutoff (percent of
            pixels clipped from each end), freshly resampled from `[0, this]` every
            time the transform fires.

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
                _RandomCutoffAutoContrast(cutoff_max=autocontrast_cutoff_max, p=1.0),
            ],
            p=p_lighting,
        ),
    ]
