"""Smoke tests for the export matrix and its artifact manifest.

Constitution Principle VIII scope: deterministic, non-ML code only — the manifest
round-trip, the matrix's own invariants, and baseline selection. Building an artifact needs
a GPU and minutes per variant, so `export_variant`/`quantize_onnx_dynamic` are outside the
gate; Principle VIII's "export I/O round-trips" is covered here at the manifest layer.
"""

import json

import pytest

from poultry_monitoring.export import (
    EXPORT_MATRIX,
    UNMEASURED,
    _artifact_size_mb,
    baseline_for,
    default_manifest_path,
    entry_key,
    read_manifest,
    summarize_manifest,
    update_manifest_entry,
    write_manifest,
)


class TestExportMatrix:
    def test_variant_names_are_unique(self):
        names = [spec.variant for spec in EXPORT_MATRIX]
        assert len(names) == len(set(names))

    @pytest.mark.parametrize("spec", EXPORT_MATRIX, ids=lambda s: s.variant)
    def test_precision_and_quantize_agree(self, spec):
        # w8a32 carries no `quantize`: Ultralytics restricts that value to LiteRT, so the
        # scheme is applied afterwards by `post_quantize` instead.
        expected = {"fp32": None, "fp16": 16, "int8": 8, "w8a32": None}[spec.precision]
        assert spec.quantize == expected

    @pytest.mark.parametrize("spec", EXPORT_MATRIX, ids=lambda s: s.variant)
    def test_post_quantize_is_only_used_where_no_native_route_exists(self, spec):
        if spec.post_quantize is not None:
            assert spec.post_quantize == "onnx_dynamic"
            assert spec.quantize is None, "a native quantize= would make the post-pass moot"

    def test_weight_only_int8_keeps_symbolic_shapes(self):
        # w8a32 quantizes weights only and leaves the graph's dynamic axes intact, so it
        # compares against the rect baseline, not the square one.
        spec = next(s for s in EXPORT_MATRIX if s.precision == "w8a32")
        assert spec.dynamic is True

    @pytest.mark.parametrize("spec", EXPORT_MATRIX, ids=lambda s: s.variant)
    def test_variant_name_encodes_its_backend_and_precision(self, spec):
        assert spec.variant == f"{spec.fmt}-{spec.precision}"

    def test_tensorrt_engines_are_always_static(self):
        # A dynamic engine gives up the fixed-profile optimization that makes TensorRT the
        # fastest deployment target, which is the only reason it's in the matrix.
        assert all(not s.dynamic for s in EXPORT_MATRIX if s.fmt == "engine")

    def test_gpu_only_builds_do_not_request_a_cpu_export_device(self):
        # FP16 ONNX and every TensorRT build need a CUDA device at export time.
        for spec in EXPORT_MATRIX:
            if spec.fmt == "engine" or (spec.fmt == "onnx" and spec.precision == "fp16"):
                assert spec.device != "cpu", spec.variant


class TestManifestRoundTrip:
    def test_reading_a_missing_manifest_yields_no_entries(self, tmp_path):
        assert read_manifest(tmp_path / "absent.json") == {}

    def test_write_then_read_preserves_entries(self, tmp_path):
        path = tmp_path / "manifest.json"
        entries = {"m__onnx-fp32": {"backend": "onnx", "size_mb": 11.4}}
        write_manifest(path, entries)
        assert read_manifest(path) == entries

    def test_write_creates_missing_parent_directories(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "manifest.json"
        write_manifest(path, {})
        assert path.exists()

    def test_written_manifest_carries_schema_and_timestamp(self, tmp_path):
        path = tmp_path / "manifest.json"
        write_manifest(path, {})
        payload = json.loads(path.read_text())
        assert payload["schema"] == 1
        assert "updated" in payload


class TestUpdateManifestEntry:
    def test_merges_into_an_entry_without_dropping_its_other_fields(self, tmp_path):
        path = tmp_path / "manifest.json"
        write_manifest(path, {"k": {"backend": "onnx", "size_mb": 11.4}})
        entries = update_manifest_entry(path, "k", {"accuracy": {"mask_map50_95": 0.83}})
        assert entries["k"]["backend"] == "onnx"
        assert entries["k"]["accuracy"] == {"mask_map50_95": 0.83}

    def test_leaves_sibling_entries_untouched(self, tmp_path):
        path = tmp_path / "manifest.json"
        write_manifest(path, {"a": {"size_mb": 1.0}, "b": {"size_mb": 2.0}})
        entries = update_manifest_entry(path, "a", {"size_mb": 9.9})
        assert entries["b"] == {"size_mb": 2.0}

    def test_creates_the_entry_when_absent(self, tmp_path):
        path = tmp_path / "manifest.json"
        entries = update_manifest_entry(path, "new", {"backend": "tensorrt"})
        assert entries["new"] == {"backend": "tensorrt"}

    def test_persists_across_reads(self, tmp_path):
        path = tmp_path / "manifest.json"
        update_manifest_entry(path, "k", {"latency": {"latency_ms_mean": 7.4}})
        assert read_manifest(path)["k"]["latency"]["latency_ms_mean"] == 7.4


class TestEntryKeyAndBaseline:
    def test_entry_key_joins_model_and_variant(self):
        assert entry_key("yolo26n-seg-baseline-adamw", "onnx-int8") == (
            "yolo26n-seg-baseline-adamw__onnx-int8"
        )

    def test_dynamic_artifacts_compare_against_the_dynamic_baseline(self):
        assert baseline_for({"dynamic": True}) == "pytorch-fp32-dynamic"

    def test_static_artifacts_compare_against_the_static_baseline(self):
        # Static graphs are letterboxed square; comparing them to the rectangular baseline
        # would charge ~0.7 mask mAP50 of padding difference to quantization.
        assert baseline_for({"dynamic": False}) == "pytorch-fp32-static"

    def test_missing_dynamic_flag_is_treated_as_static(self):
        assert baseline_for({}) == "pytorch-fp32-static"


class TestSummarizeManifest:
    def test_empty_manifest_explains_what_to_run(self):
        assert "python -m poultry_monitoring.export" in summarize_manifest({})

    def test_lists_each_variant(self, tmp_path):
        entries = {
            "m__onnx-fp32": {
                "backend": "onnx",
                "precision": "fp32",
                "dynamic": True,
                "size_mb": 11.4,
            },
            "m__engine-int8": {
                "backend": "tensorrt",
                "precision": "int8",
                "dynamic": False,
                "size_mb": 4.2,
            },
        }
        table = summarize_manifest(entries)
        assert "m__onnx-fp32" in table
        assert "m__engine-int8" in table

    def test_shows_accuracy_once_measured_and_a_placeholder_before(self):
        scored = summarize_manifest({"k": {"accuracy": {"mask_map50_95": 0.8312}}})
        unscored = summarize_manifest({"k": {}})
        assert "0.8312" in scored
        assert UNMEASURED in unscored

    def test_renders_a_freshly_exported_manifest_that_has_no_metrics_yet(self):
        # Regression: a format spec applied to a `None`-or-float conditional raised
        # TypeError, so the table broke for every variant between export and scoring.
        assert summarize_manifest({"k": {"backend": "onnx", "size_mb": 11.4}})

    def test_surfaces_a_failed_export_rather_than_hiding_it(self):
        table = summarize_manifest({"k": {"error": "OutOfMemoryError: ..."}})
        assert "FAILED" in table and "OutOfMemoryError" in table


class TestArtifactSizeMb:
    def test_measures_a_single_file(self, tmp_path):
        f = tmp_path / "model.onnx"
        f.write_bytes(b"x" * 2_000_000)
        assert _artifact_size_mb(f) == pytest.approx(2.0, rel=1e-3)

    def test_sums_an_openvino_directory_recursively(self, tmp_path):
        d = tmp_path / "model_openvino_model"
        (d / "nested").mkdir(parents=True)
        (d / "model.bin").write_bytes(b"x" * 1_000_000)
        (d / "nested" / "model.xml").write_bytes(b"x" * 500_000)
        assert _artifact_size_mb(d) == pytest.approx(1.5, rel=1e-3)


class TestDefaultManifestPath:
    def test_lands_under_the_gitignored_export_tree(self, tmp_path):
        assert default_manifest_path(tmp_path).parts[-3:] == ("YOLO", "export", "manifest.json")
