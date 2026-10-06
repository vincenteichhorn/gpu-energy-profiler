from datetime import datetime
from multiprocessing import Array
from pathlib import Path

import pandas as pd
import pytest

from gpu_energy_profiler.nvidia_profiler import NvidiaProfiler
from gpu_energy_profiler.plotting_util import sample_colors


def _samples() -> list[tuple[int, datetime, float, float, str]]:
    return [
        (0, datetime(2026, 1, 1, 12, 0, 0), 100.0, 200.0, "step-a"),
        (0, datetime(2026, 1, 1, 12, 0, 1), 100.0, 300.0, "step-a"),
        (1, datetime(2026, 1, 1, 12, 0, 2), 50.0, 400.0, "step-b"),
    ]


def test_parse_nvidia_smi_row() -> None:
    current_step = Array("c", 1000)
    current_step.value = b"inference"  # type: ignore[attr-defined]

    row = NvidiaProfiler._parse_nvidia_smi_row(
        "0, 2026/01/01 12:00:00.123456, 100.5 W, 256 MiB",
        current_step,
    )

    assert row == (0, datetime(2026, 1, 1, 12, 0, 0, 123456), 100.5, 256.0, "inference")


def test_empty_profiler_returns_empty_metrics() -> None:
    profiler = NvidiaProfiler()

    assert profiler.total_energy() == 0.0
    assert profiler.total_time() == 0.0
    assert profiler.avg_memory_usage() == 0.0
    assert profiler.time_series_plot() is None


def test_sample_colors_always_returns_requested_palette() -> None:
    assert len(sample_colors("viridis", 3)) == 3
    assert len(sample_colors("viridis", 0)) == 2


def test_backend_is_validated() -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        NvidiaProfiler(backend="invalid")  # type: ignore[arg-type]


def test_pynvml_worker_collects_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    class Flag:
        def __init__(self) -> None:
            self.reads = 0

        @property
        def value(self) -> int:
            self.reads += 1
            return int(self.reads <= 2)

    class Event:
        def __init__(self) -> None:
            self.was_set = False

        def set(self) -> None:
            self.was_set = True

    class Memory:
        used = 256 * 1024 * 1024

    class FakePynvml:
        class NVMLError(Exception):
            pass

        initialized = False
        shutdown = False

        @classmethod
        def nvmlInit(cls) -> None:
            cls.initialized = True

        @classmethod
        def nvmlShutdown(cls) -> None:
            cls.shutdown = True

        @staticmethod
        def nvmlDeviceGetCount() -> int:
            return 1

        @staticmethod
        def nvmlDeviceGetHandleByIndex(index: int) -> int:
            return index

        @staticmethod
        def nvmlDeviceGetPowerUsage(_handle: int) -> int:
            del _handle
            return 125_000

        @staticmethod
        def nvmlDeviceGetMemoryInfo(_handle: int) -> Memory:
            del _handle
            return Memory()

    class Handler:
        def __init__(self) -> None:
            self.rows = []

        def __enter__(self) -> "Handler":
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def put(self, row: object) -> None:
            self.rows.append(row)

    monkeypatch.setattr(
        "gpu_energy_profiler.nvidia_profiler._load_pynvml",
        lambda: FakePynvml,
    )
    flag = Flag()
    started = Event()
    stopped = Event()
    handler = Handler()
    step = Array("c", 1000)
    step.value = b"inference"  # type: ignore[attr-defined]

    NvidiaProfiler._pynvml_profiling_process(
        flag,
        started,
        stopped,
        handler,  # type: ignore[arg-type]
        step,
        0,
    )

    assert FakePynvml.initialized is True
    assert FakePynvml.shutdown is True
    assert started.was_set is True
    assert stopped.was_set is True
    first_row = handler.rows[0]
    assert first_row[0] == 0
    assert first_row[2:4] == (125.0, 256.0)
    assert first_row[4] == "inference"
    assert handler.rows[-1] is None


def test_nvidia_metrics() -> None:
    profiler = NvidiaProfiler()
    profiler.data = _samples()

    assert profiler.profiled_gpus() == [0, 1]
    assert profiler.total_time() == 2.0
    assert profiler.avg_memory_usage(0) == 250.0
    assert profiler.total_energy(gpu_ids=[0]) == 100.0
    assert profiler.total_energy(gpu_ids=[0], return_data=True) == [100.0]


def test_nvidia_dataframe_and_plot() -> None:
    profiler = NvidiaProfiler()
    profiler.data = _samples()
    profiler.record_step("step-a")
    profiler.record_step("step-b")

    frame = profiler.to_pandas()
    figure = profiler.time_series_plot()

    assert list(frame.columns) == ["gpu_id", "timestamp", "power", "memory", "record_step"]
    assert pd.api.types.is_datetime64_any_dtype(frame["timestamp"])
    assert figure is not None
    assert sum(1 for _ in figure.data) == 4
    shapes = figure.layout.shapes
    assert isinstance(shapes, (list, tuple))
    assert sum(1 for _ in shapes) == 3


def test_from_cache_reads_samples(tmp_path: Path) -> None:
    cache_file = tmp_path / "samples.csv"
    cache_file.write_text(
        "gpu_id,timestamp,power,memory,record_step\n"
        "0,2026-01-01 12:00:00.000000,100.0,200.0,step-a\n",
        encoding="utf-8",
    )

    with pytest.warns(UserWarning, match="already exists"):
        profiler = NvidiaProfiler.from_cache(str(cache_file))

    assert profiler.data == [(0, "2026-01-01 12:00:00.000000", 100.0, 200.0, "step-a")]


def test_context_manager_requires_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("subprocess.getstatusoutput", lambda _: (1, "missing"))

    with pytest.raises(RuntimeError, match="nvidia-smi"):
        with NvidiaProfiler():
            pass


def test_context_manager_reports_empty_nvidia_smi_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("subprocess.getstatusoutput", lambda _: (0, ""))

    with pytest.raises(RuntimeError, match="could not query"):
        with NvidiaProfiler(backend="nvidia-smi"):
            pass


def test_context_manager_reports_missing_pynvml_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePynvml:
        class NVMLError(Exception):
            pass

        @staticmethod
        def nvmlInit() -> None:
            pass

        @staticmethod
        def nvmlDeviceGetCount() -> int:
            return 0

        @staticmethod
        def nvmlShutdown() -> None:
            pass

    monkeypatch.setattr(
        "gpu_energy_profiler.nvidia_profiler._load_pynvml",
        lambda: FakePynvml,
    )

    with pytest.raises(RuntimeError, match="found no NVIDIA GPU"):
        with NvidiaProfiler(backend="pynvml"):
            pass
