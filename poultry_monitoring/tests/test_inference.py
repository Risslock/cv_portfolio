"""Smoke tests for the shared model-load path.

Constitution Principle VIII scope: deterministic, non-ML code only — artifact-to-backend
resolution, runtime configuration reporting, and the runtime-description shape. Actually
loading a model or running inference needs real artifacts and a GPU, so it is outside the
gate; `describe_runtime` is exercised against hand-built stand-ins rather than a session.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from poultry_monitoring.inference import (
    autobackend,
    configure_runtime,
    describe_runtime,
    load_model,
    resolve_backend,
)


class TestResolveBackend:
    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("best.pt", "pytorch"),
            ("model.onnx", "onnx"),
            ("model_int8.onnx", "onnx"),
            ("model.engine", "tensorrt"),
            ("model.tflite", "litert"),
        ],
    )
    def test_maps_known_suffixes(self, filename, expected):
        assert resolve_backend(Path(filename)) == expected

    def test_suffix_match_is_case_insensitive(self):
        assert resolve_backend(Path("MODEL.ONNX")) == "onnx"

    def test_openvino_recognized_by_directory_name_without_touching_disk(self):
        assert resolve_backend(Path("some/model_openvino_model")) == "openvino"

    def test_openvino_recognized_when_directory_exists(self, tmp_path):
        d = tmp_path / "whatever_openvino_model"
        d.mkdir()
        assert resolve_backend(d) == "openvino"

    @pytest.mark.parametrize("filename", ["model.bin", "model", "model.xml"])
    def test_rejects_unknown_artifacts(self, filename):
        with pytest.raises(ValueError, match="Unrecognized artifact"):
            resolve_backend(Path(filename))


class TestConfigureRuntime:
    def test_reports_the_fields_principle_v_disclosure_needs(self):
        info = configure_runtime()
        assert set(info) == {"torch_num_threads", "cpu_count", "autoinstall_disabled"}
        assert isinstance(info["torch_num_threads"], int)

    def test_autoinstall_is_disabled_by_importing_this_package(self):
        # inference.py sets YOLO_AUTOINSTALL at import time; if that regresses, Ultralytics
        # can pip-install backends outside uv.lock (constitution Principle VII).
        assert configure_runtime()["autoinstall_disabled"] is True

    def test_pins_torch_threads_when_asked(self):
        original = configure_runtime()["torch_num_threads"]
        try:
            assert configure_runtime(cpu_threads=2)["torch_num_threads"] == 2
        finally:
            configure_runtime(cpu_threads=original)

    def test_is_idempotent(self):
        assert configure_runtime() == configure_runtime()


class TestLoadModel:
    def test_missing_artifact_fails_before_any_backend_work(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No artifact at"):
            load_model(tmp_path / "absent.onnx")


def _fake_loaded(backend, device="cpu"):
    """Mimic a loaded exported artifact.

    The AutoBackend hangs off the predictor, and `.model` is the path string -- see
    `inference.autobackend`.
    """
    return SimpleNamespace(
        predictor=SimpleNamespace(model=backend), model="some/artifact.onnx", device=device
    )


class TestDescribeRuntime:
    def test_surfaces_real_providers_when_the_backend_has_a_session(self):
        # The point of the function: report what ONNX Runtime actually bound, since asking
        # for CUDA and silently getting CPU raises no exception.
        session = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])
        model = _fake_loaded(SimpleNamespace(session=session), device="cuda:0")
        assert describe_runtime(model)["providers"] == ["CPUExecutionProvider"]

    def test_omits_providers_for_backends_that_expose_none(self):
        assert "providers" not in describe_runtime(_fake_loaded(SimpleNamespace()))

    def test_always_reports_backend_and_device(self):
        info = describe_runtime(_fake_loaded(SimpleNamespace()))
        assert info["device"] == "cpu"
        assert info["backend"] == "SimpleNamespace"

    def test_reports_no_backend_when_the_predictor_was_never_built(self):
        # Regression: describe_runtime used to read YOLO.model directly, which for an
        # exported artifact is the path string -- so provider checks silently came back
        # empty for exactly the artifacts they exist to verify.
        info = describe_runtime(SimpleNamespace(predictor=None, model="x.onnx", device="cpu"))
        assert info["backend"] is None
        assert "providers" not in info


class TestAutobackend:
    def test_prefers_the_predictor_backend(self):
        # For anything but a .pt, YOLO.model is the path *string*; the real AutoBackend is
        # built lazily and only lives at model.predictor.model.
        backend = SimpleNamespace(session="live")
        model = SimpleNamespace(predictor=SimpleNamespace(model=backend), model="a/path.onnx")
        assert autobackend(model) is backend

    def test_ignores_a_string_model_when_no_predictor_exists(self):
        assert autobackend(SimpleNamespace(predictor=None, model="a/path.onnx")) is None

    def test_falls_back_to_a_pytorch_module(self):
        import torch

        module = torch.nn.Identity()
        assert autobackend(SimpleNamespace(predictor=None, model=module)) is module

    def test_handles_a_model_with_no_predictor_attribute(self):
        assert autobackend(SimpleNamespace(model="a/path.engine")) is None
