"""Smoke tests for `poultry_monitoring.mlflow_utils`.

Constitution Principle VIII scope: deterministic, non-ML code — MLflow
bookkeeping, not model behavior.
"""

import mlflow
import pytest

from poultry_monitoring.mlflow_utils import finish_run, make_run_name


@pytest.fixture
def local_mlflow(tmp_path):
    """Point MLflow at a throwaway local SQLite store, isolated from the project's own mlflow.db."""
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("test_experiment")
    yield
    if mlflow.active_run() is not None:
        mlflow.end_run()


class TestMakeRunName:
    def test_requires_active_run(self, local_mlflow):
        with pytest.raises(RuntimeError):
            make_run_name("yolo26", "n-baseline")

    def test_sets_expected_name_and_returns_it(self, local_mlflow):
        with mlflow.start_run() as run:
            new_name = make_run_name("yolo26", "n-baseline")

        expected = f"yolo26-n-baseline-{run.info.run_id[:8]}"
        assert new_name == expected
        finished = mlflow.get_run(run.info.run_id)
        assert finished.data.tags["mlflow.runName"] == expected


class TestFinishRun:
    def test_requires_active_run(self, local_mlflow):
        with pytest.raises(RuntimeError):
            finish_run()

    def test_logs_extras_and_ends_run(self, local_mlflow):
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            finish_run(
                extra_params={"model_family": "yolo26"},
                extra_tags={"stage": "sweep"},
                extra_metrics={"box_map50": 0.953},
            )

        assert mlflow.active_run() is None
        finished = mlflow.get_run(run_id)
        assert finished.data.params["model_family"] == "yolo26"
        assert finished.data.tags["stage"] == "sweep"
        assert finished.data.metrics["box_map50"] == pytest.approx(0.953)
        assert finished.info.status == "FINISHED"
