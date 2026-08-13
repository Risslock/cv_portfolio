"""Bbox-aware augmentation: spatial transforms, need Albumentations' `bbox_params`.

Detection-specific per `CLAUDE.md` § Package Layout (segmentation's mask-aware
equivalent, copy-paste, lives in `segmentation.py` instead). Unlike `shared.py`'s
image-only transforms, `model.train(augmentations=...)`'s combined list only needs
`bbox_params` wired through once — Ultralytics' own `Albumentations` wrapper checks
`contains_spatial` across the whole list and threads boxes through automatically (see
`augmentation/visualize.py`'s `generate_augmented_samples` for the same pattern used
standalone).
"""

import albumentations as A

# See shared.py's PARAM_RANGES for the convention this follows.
PARAM_RANGES = {
    "p_occlusion": (0.0, 0.4),
}


def build_detection_transforms(p_occlusion: float = 0.3) -> list:
    """Build this project's custom bbox-aware Albumentations transform list.

    `CoarseDropout` blacks out random rectangular regions — verified empirically that
    it leaves bounding box coordinates untouched even when a box is fully covered
    (it's a pixel-level transform, not a geometric one), which is exactly what
    occlusion-robustness training wants: the label stays valid, forcing the model to
    use partial-visibility/context cues for birds that are now (partly) hidden. Not
    covered by `model.tune()`'s own search space or by `shared.py` (image-only) — a
    genuinely new axis, not a duplicate of Ultralytics' `degrees`/`scale`/`shear`.

    Args:
        p_occlusion: Probability of applying the dropout occlusion.

    Returns:
        A list of Albumentations transforms, for `model.train(augmentations=...)`.
    """
    return [
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(0.05, 0.15),
            hole_width_range=(0.05, 0.15),
            fill=0,
            p=p_occlusion,
        ),
    ]
