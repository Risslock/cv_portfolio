"""Smoke tests for `poultry_monitoring.detection.tuning`.

Constitution Principle VIII scope: deterministic, non-ML code only — the
progressive-unfreezing stage-chaining logic (with `tuning.train` mocked out — no actual
training happens here), not the search/tuning itself.
"""

from pathlib import Path
from unittest.mock import patch

from poultry_monitoring.detection.tuning import progressive_unfreeze_train
from poultry_monitoring.detection.yolo import TrainOutcome


def _fake_outcome(stage: int) -> TrainOutcome:
    return TrainOutcome(
        weights_path=Path(f"stage{stage}/best.pt"),
        box_map50=0.9,
        box_map50_95=0.8,
        box_precision=0.9,
        box_recall=0.9,
    )


class TestProgressiveUnfreezeTrain:
    @patch("poultry_monitoring.detection.tuning.train")
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

    @patch("poultry_monitoring.detection.tuning.train")
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

    @patch("poultry_monitoring.detection.tuning.train")
    def test_no_stages_argument_uses_default_schedule(self, mock_train):
        mock_train.side_effect = [_fake_outcome(i) for i in range(3)]

        progressive_unfreeze_train(
            Path("data.yaml"), Path("project"), model_name="yolo26n", initial_weights=Path("w.pt")
        )

        assert mock_train.call_count == 3
        freezes = [c.kwargs["freeze"] for c in mock_train.call_args_list]
        assert freezes == sorted(freezes, reverse=True)  # freeze decreases each stage
