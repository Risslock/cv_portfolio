"""Smoke tests for the latency harness.

Constitution Principle VIII scope: deterministic, non-ML code only — the timing statistics
and the hardware fingerprint's shape. `benchmark_variant` needs real artifacts and a GPU, so
it is outside the gate. `measure_latency` is exercised against plain callables, which is the
point: the timing logic should not care what it is timing.
"""

import pytest

from poultry_monitoring.benchmark import (
    DEFAULT_ITERATIONS_CPU,
    DEFAULT_ITERATIONS_GPU,
    DEFAULT_WARMUP,
    aggregate_results,
    build_summary,
    devices_for,
    hardware_fingerprint,
    letterbox_for,
    measure_latency,
)


class TestDevicesFor:
    def test_tensorrt_engines_are_gpu_only(self):
        assert devices_for({"backend": "tensorrt", "precision": "fp16"}) == [0]

    def test_openvino_is_cpu_only_here(self):
        assert devices_for({"backend": "openvino", "precision": "int8"}) == ["cpu"]

    @pytest.mark.parametrize("precision", ["int8", "w8a32"])
    def test_neither_int8_scheme_is_swept_on_gpu(self, precision):
        assert devices_for({"backend": "onnx", "precision": precision}) == ["cpu"]

    def test_onnx_int8_is_not_swept_on_gpu(self):
        # The CUDA provider has no INT8 kernels for this graph and would fall back to CPU,
        # yielding a duplicate CPU number mislabelled as GPU.
        assert devices_for({"backend": "onnx", "precision": "int8"}) == ["cpu"]

    @pytest.mark.parametrize("precision", ["fp32", "fp16"])
    def test_onnx_float_precisions_sweep_both_devices(self, precision):
        assert set(devices_for({"backend": "onnx", "precision": precision})) == {0, "cpu"}

    def test_pytorch_sweeps_both_devices(self):
        assert set(devices_for({"backend": "pytorch", "precision": "fp32"})) == {0, "cpu"}

    def test_unknown_backend_defers_to_ultralytics(self):
        assert devices_for({"backend": "something-new"}) == [None]


class TestLetterboxFor:
    def test_frozen_graphs_pad_square(self):
        assert letterbox_for({"backend": "tensorrt", "dynamic": False}) == "square"

    def test_dynamic_graphs_keep_aspect_ratio(self):
        assert letterbox_for({"backend": "onnx", "dynamic": True}) == "rect"

    @pytest.mark.parametrize("dynamic", [True, False])
    def test_pytorch_always_pads_rect_whatever_its_baseline_flag_says(self, dynamic):
        # `dynamic` on a PyTorch entry selects which *accuracy* baseline it represents
        # (val(rect=False) vs val(rect=True)); predict() has no rect argument and a .pt has
        # no frozen shape, so the runtime padding is rect either way. Reading the flag here
        # claimed a latency difference that measurement showed did not exist.
        assert letterbox_for({"backend": "pytorch", "dynamic": dynamic}) == "rect"


class TestMeasureLatency:
    def test_reports_the_requested_iteration_and_warmup_counts(self):
        stats = measure_latency(lambda: None, batch=1, warmup=2, iterations=5)
        assert stats.iterations == 5
        assert stats.warmup == 2

    def test_runs_the_callable_warmup_plus_iterations_times(self):
        calls = []
        measure_latency(lambda: calls.append(1), batch=1, warmup=3, iterations=7)
        assert len(calls) == 10

    def test_warmup_iterations_are_excluded_from_the_statistics(self):
        # A deliberately slow warmup must not show up in the timings, which is the whole
        # reason Principle V asks for warmup to be excluded.
        import time

        state = {"n": 0}

        def call():
            state["n"] += 1
            if state["n"] <= 2:
                time.sleep(0.05)

        stats = measure_latency(call, batch=1, warmup=2, iterations=5)
        assert stats.mean_ms < 25.0

    def test_throughput_accounts_for_batch_size(self):
        # Needs a call slow enough to survive rounding to 3 decimals, otherwise the
        # reported mean is 0.000 ms and there is nothing to divide by.
        import time

        stats = measure_latency(lambda: time.sleep(0.002), batch=8, warmup=0, iterations=3)
        assert stats.throughput_img_per_sec == pytest.approx(8 * 1000.0 / stats.mean_ms, rel=1e-2)

    def test_throughput_is_infinite_rather_than_dividing_by_zero(self):
        # A callable faster than the clock's resolution must still produce a record.
        assert measure_latency(lambda: None, batch=1, warmup=0, iterations=1).mean_ms >= 0.0

    def test_orders_min_median_and_p95_consistently(self):
        stats = measure_latency(lambda: None, batch=1, warmup=1, iterations=20)
        assert stats.min_ms <= stats.median_ms <= stats.p95_ms

    def test_single_iteration_reports_zero_deviation_rather_than_raising(self):
        # statistics.stdev needs two points; one timed iteration must still produce a record.
        assert measure_latency(lambda: None, batch=1, warmup=0, iterations=1).std_ms == 0.0

    @pytest.mark.parametrize("iterations", [0, -1])
    def test_rejects_a_nonpositive_iteration_count(self, iterations):
        with pytest.raises(ValueError, match="iterations must be >= 1"):
            measure_latency(lambda: None, batch=1, iterations=iterations)

    def test_zero_warmup_is_allowed(self):
        assert measure_latency(lambda: None, batch=1, warmup=0, iterations=2).warmup == 0


class TestDefaults:
    def test_cpu_gets_fewer_iterations_than_gpu(self):
        # CPU cells run ~10x slower; an equal count would make a sweep take hours.
        assert DEFAULT_ITERATIONS_CPU < DEFAULT_ITERATIONS_GPU

    def test_warmup_is_nonzero(self):
        assert DEFAULT_WARMUP > 0


class TestHardwareFingerprint:
    def test_reports_the_fields_principle_v_requires(self):
        info = hardware_fingerprint()
        for key in ["os", "cpu", "cpu_count_logical", "torch_num_threads", "torch", "python"]:
            assert key in info, key

    def test_records_every_backend_version_slot_even_when_absent(self):
        # A missing backend is recorded as None rather than omitted, so a results file
        # always states which backends existed on the measuring machine.
        info = hardware_fingerprint()
        for key in ["onnxruntime", "openvino", "tensorrt", "ultralytics"]:
            assert key in info, key

    def test_is_json_serializable(self):
        import json

        json.dumps(hardware_fingerprint())

    def test_gpu_details_accompany_each_other(self):
        info = hardware_fingerprint()
        if "gpu" in info:
            assert "gpu_compute_capability" in info
            assert "supports_bf16" in info


class TestBuildSummary:
    def _manifest(self, tmp_path):
        from poultry_monitoring.export import write_manifest

        p = tmp_path / "manifest.json"
        write_manifest(
            p,
            {
                "m__pytorch-fp32-static": {
                    "model": "m",
                    "variant": "pytorch-fp32-static",
                    "backend": "pytorch",
                    "precision": "fp32",
                    "dynamic": False,
                    "size_mb": 6.5,
                    "path": str(tmp_path / "a" / "b" / "best.pt"),
                    "accuracy_by_split": {"Test": {"box_map50_95": 0.75, "mask_map50_95": 0.70}},
                },
                "m__engine-int8": {
                    "model": "m",
                    "variant": "engine-int8",
                    "backend": "tensorrt",
                    "precision": "int8",
                    "dynamic": False,
                    "size_mb": 7.2,
                    "path": str(tmp_path / "a" / "b" / "model.engine"),
                    "accuracy_by_split": {"Test": {"box_map50_95": 0.74, "mask_map50_95": 0.69}},
                },
            },
        )
        return p

    def test_delta_is_computed_against_the_shape_matched_baseline(self, tmp_path):
        md = build_summary(self._manifest(tmp_path), tmp_path / "absent.json", tmp_path)
        assert "-0.0100" in md  # 0.74 - 0.75, engine-int8 vs the static baseline

    def test_baseline_row_shows_no_delta_against_itself(self, tmp_path):
        md = build_summary(self._manifest(tmp_path), tmp_path / "absent.json", tmp_path)
        row = next(ln for ln in md.splitlines() if "`pytorch-fp32-static`" in ln)
        assert row.rstrip().endswith("| — | — |")

    def test_tolerates_a_missing_benchmark_results_file(self, tmp_path):
        md = build_summary(self._manifest(tmp_path), tmp_path / "absent.json", tmp_path)
        assert "## Accuracy" in md and "## Latency" not in md

    def test_paths_are_repo_relative_not_absolute(self, tmp_path):
        md = build_summary(self._manifest(tmp_path), tmp_path / "absent.json", tmp_path)
        assert "a/b/best.pt" in md
        assert str(tmp_path) not in md

    def test_escapes_pipes_in_latency_cell_keys(self, tmp_path):
        import json

        results = tmp_path / "bench.json"
        results.write_text(
            json.dumps(
                {
                    "hardware": {"gpu": "Test GPU", "os": "Linux"},
                    "results": {
                        "m__engine-int8|cuda|b1": {
                            "letterbox": "square",
                            "forward": {"mean_ms": 3.1, "median_ms": 3.0, "p95_ms": 3.8},
                            "end_to_end": {"mean_ms": 9.3, "throughput_img_per_sec": 107.6},
                            "stage_ms": {"preprocess": 2.2, "inference": 3.1, "postprocess": 3.2},
                        }
                    },
                }
            )
        )
        md = build_summary(self._manifest(tmp_path), results, tmp_path)
        row = next(ln for ln in md.splitlines() if "engine-int8" in ln and "cuda" in ln)
        # A bare pipe would split the cell into extra columns and break the table.
        assert "engine-int8\|cuda\|b1" in row
        assert row.replace("\|", "\x00").count("|") == 9


class TestAggregateResults:
    def _write(self, tmp_path, name, e2e, fwd=None):
        import json

        cell = {"end_to_end": {"mean_ms": e2e, "throughput_img_per_sec": 1000 / e2e}}
        if fwd is not None:
            cell["forward"] = {"mean_ms": fwd}
        p = tmp_path / name
        p.write_text(json.dumps({"hardware": {"gpu": "x"}, "results": {"m__v|cpu|b1": cell}}))
        return p

    def test_takes_the_median_not_the_mean(self, tmp_path):
        # An outlier run must not drag the published number, which is the whole reason
        # repeats exist.
        paths = [
            self._write(tmp_path, f"r{i}.json", e2e) for i, e2e in enumerate([100.0, 104.0, 300.0])
        ]
        out = aggregate_results(paths)["results"]["m__v|cpu|b1"]
        assert out["end_to_end"]["mean_ms"] == 104.0

    def test_records_spread_so_a_noisy_cell_is_visible(self, tmp_path):
        paths = [
            self._write(tmp_path, f"r{i}.json", e2e) for i, e2e in enumerate([100.0, 200.0, 150.0])
        ]
        out = aggregate_results(paths)["results"]["m__v|cpu|b1"]
        assert out["spread_pct"] == pytest.approx(66.7, abs=0.1)
        assert out["runs"] == 3

    def test_medians_the_forward_block_too(self, tmp_path):
        paths = [
            self._write(tmp_path, f"r{i}.json", 100.0, fwd=f)
            for i, f in enumerate([10.0, 12.0, 50.0])
        ]
        assert aggregate_results(paths)["results"]["m__v|cpu|b1"]["forward"]["mean_ms"] == 12.0

    def test_a_single_run_still_aggregates(self, tmp_path):
        out = aggregate_results([self._write(tmp_path, "r.json", 42.0)])["results"]["m__v|cpu|b1"]
        assert out["end_to_end"]["mean_ms"] == 42.0
        assert "spread_pct" not in out  # nothing to spread across

    def test_failed_cells_survive_rather_than_crashing(self, tmp_path):
        import json

        p = tmp_path / "r.json"
        p.write_text(json.dumps({"hardware": {}, "results": {"m__v|cpu|b1": {"error": "OOM"}}}))
        assert "error" in aggregate_results([p])["results"]["m__v|cpu|b1"]

    def test_rejects_an_empty_file_list(self):
        with pytest.raises(ValueError, match="at least one"):
            aggregate_results([])
