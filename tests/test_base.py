from datetime import datetime

import pytest

from gpu_energy_profiler.base import Profiler


def test_profiler_records_initial_and_named_steps() -> None:
    profiler = Profiler()
    profiler.record_step("load")

    assert [name for _, name in profiler.record_steps] == ["__init__", "load"]
    assert all(isinstance(timestamp, datetime) for timestamp, _ in profiler.record_steps)


def test_record_context_records_end_step_even_when_body_fails() -> None:
    profiler = Profiler()

    with pytest.raises(RuntimeError):
        with profiler.record_context("inference"):
            raise RuntimeError("failure")

    assert [name for _, name in profiler.record_steps] == [
        "__init__",
        "inference",
        "__other__",
    ]
