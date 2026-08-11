"""Smoke tests for `poultry_monitoring.detection.yolo`.

Constitution Principle VIII scope: deterministic, non-ML code only — the
hyperparameters-dict/augmentation-kwargs split, transform-list assembly, and the
progressive-unfreezing stage-chaining logic (with `train` mocked out — no actual
training happens here), not training/tuning itself.
"""

from pathlib import Path
from unittest.mock import patch

from poultry_monitoring.detection.yolo import (
    CUSTOM_AUGMENTATION_PARAM_RANGES,
    TrainOutcome,
    _augmentation_kwargs_from,
    _build_custom_augmentations,
    progressive_unfreeze_train,
)


class TestAugmentationKwargsFrom:
    def test_empty_dict_returns_empty(self):
        assert _augmentation_kwargs_from({}) == {}

    def test_pulls_only_custom_augmentation_keys(self):
        hyperparameters = {
            "lr0": 0.01,
            "p_color_invariance": 0.15,
            "p_lighting": 0.35,
            "brightness_limit": 0.2,
            "p_occlusion": 0.25,
        }

        result = _augmentation_kwargs_from(hyperparameters)

        assert result == {
            "p_color_invariance": 0.15,
            "p_lighting": 0.35,
            "brightness_limit": 0.2,
            "p_occlusion": 0.25,
        }
        assert "lr0" not in result

    def test_covers_every_param_ranges_key(self):
        hyperparameters = dict.fromkeys(CUSTOM_AUGMENTATION_PARAM_RANGES, 0.1)

        result = _augmentation_kwargs_from(hyperparameters)

        assert set(result) == set(CUSTOM_AUGMENTATION_PARAM_RANGES)


class TestBuildCustomAugmentations:
    def test_no_hyperparameters_uses_each_builder_defaults(self):
        transforms = _build_custom_augmentations({})

        # 2 from shared.build_domain_transforms (OneOf groups) + 1 from
        # detection.build_detection_transforms (CoarseDropout).
        assert len(transforms) == 3

    def test_routes_shared_and_detection_keys_to_their_own_builder(self):
        transforms = _build_custom_augmentations(
            {"p_color_invariance": 1.0, "p_occlusion": 1.0, "lr0": 0.01}
        )

        assert len(transforms) == 3
        probabilities = [t.p for t in transforms]
        assert 1.0 in probabilities


def _fake_outcome(stage: int) -> TrainOutcome:
    return TrainOutcome(
        weights_path=Path(f"stage{stage}/best.pt"),
        box_map50=0.9,
        box_map50_95=0.8,
        box_precision=0.9,
        box_recall=0.9,
    )


class TestProgressiveUnfreezeTrain:
    @patch("poultry_monitoring.detection.yolo.train")
    def test_each_stage_starts_from_the_previous_stages_weights(self, mock_train):
        mock_train.side_effect = [_fake_outcome(0), _fake_outcome(1)]
        stages = [
            {"freeze": 10, "lr0": 5e-4, "optimizer": "AdamW", "epochs": 5, "patience": 5},
            {"freeze": 0, "lr0": 2e-5, "optimizer": "AdamW", "epochs": 5, "patience": 5},
        ]

        outcomes = progressive_unfreeze_train(
            Path("data.yaml"),
            Path("project"),
            model_name="yolo26n",
            initial_weights=Path("initial/best.pt"),
            stages=stages,
        )

        assert outcomes == [_fake_outcome(0), _fake_outcome(1)]
        first_call, second_call = mock_train.call_args_list
        assert first_call.kwargs["weights_path"] == Path("initial/best.pt")
        assert first_call.kwargs["freeze"] == 10
        # Second stage must start from the first stage's *output*, not initial_weights again.
        assert second_call.kwargs["weights_path"] == Path("stage0/best.pt")
        assert second_call.kwargs["freeze"] == 0

    @patch("poultry_monitoring.detection.yolo.train")
    def test_stage_lr0_and_optimizer_override_fixed_hyperparameters(self, mock_train):
        mock_train.return_value = _fake_outcome(0)
        stages = [{"freeze": 5, "lr0": 1e-4, "optimizer": "AdamW", "epochs": 5, "patience": 5}]

        progressive_unfreeze_train(
            Path("data.yaml"),
            Path("project"),
            model_name="yolo26n",
            initial_weights=Path("initial/best.pt"),
            hyperparameters={"lr0": 0.01, "p_color_invariance": 0.2},
            stages=stages,
        )

        used_hyperparameters = mock_train.call_args.kwargs["hyperparameters"]
        assert used_hyperparameters["lr0"] == 1e-4  # stage override wins, not the fixed 0.01
        assert used_hyperparameters["p_color_invariance"] == 0.2  # fixed value passed through

    @patch("poultry_monitoring.detection.yolo.train")
    def test_no_stages_argument_uses_default_schedule(self, mock_train):
        mock_train.side_effect = [_fake_outcome(i) for i in range(3)]

        progressive_unfreeze_train(
            Path("data.yaml"), Path("project"), model_name="yolo26n", initial_weights=Path("w.pt")
        )

        assert mock_train.call_count == 3
        freezes = [c.kwargs["freeze"] for c in mock_train.call_args_list]
        assert freezes == sorted(freezes, reverse=True)  # freeze decreases each stage
