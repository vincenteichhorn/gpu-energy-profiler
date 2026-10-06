import pandas as pd
import pytest
import torch

from gpu_energy_profiler.torch_profiler import TorchProfiler


def _profile() -> TorchProfiler:
    profiler = TorchProfiler(
        activities=[torch.profiler.ProfilerActivity.CPU],
        with_flops=False,
        profile_memory=True,
    )
    with profiler:
        with profiler.record_context("matmul"):
            torch.mm(torch.ones((2, 2)), torch.ones((2, 2)))
    return profiler


def test_torch_profiler_collects_dataframe_and_summaries() -> None:
    profiler = _profile()

    frame = profiler.to_pandas()
    summary = profiler.summary()
    totals = profiler.totals()

    assert not frame.empty
    assert {"name", "device", "record_step", "is_annotation"} <= set(frame.columns)
    assert isinstance(summary, pd.DataFrame)
    assert isinstance(totals, pd.Series)
    assert profiler.total_time("CPU") >= 0
    assert profiler.total_flops() == 0


def test_torch_profiler_reports_metrics_by_step() -> None:
    profiler = _profile()

    flops = profiler.flops_by_step()
    times = profiler.time_by_step()

    assert "matmul" in flops.index
    assert list(times.columns) == ["cpu_time", "gpu_time"]
    assert "matmul" in times.index


def test_torch_profiler_validates_device() -> None:
    profiler = _profile()

    with pytest.raises(AssertionError, match="CPU or CUDA"):
        profiler.total_time("TPU")
