"""PyTorch profiler integration and event analysis."""

from typing import Any

import pandas as pd
from torch.autograd.profiler_util import EventList, FunctionEvent
from torch.profiler import ProfilerActivity, profile

from .base import Profiler


class TorchProfiler(profile, Profiler):
    """Collect and analyse CPU and CUDA events from PyTorch."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize a PyTorch profiler with useful profiling defaults.

        Args:
            *args: Positional arguments forwarded to ``torch.profiler.profile``.
            **kwargs: Keyword arguments overriding the profiler defaults.
        """
        defaults = {
            "with_flops": True,
            "profile_memory": True,
            "activities": [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        }
        profile.__init__(self, *args, **{**defaults, **kwargs})
        Profiler.__init__(self)
        self.numeric_columns = [
            "flops",
            "count",
            "self_device_time_total",
            "self_cpu_time_total",
            "device_time_total",
            "cpu_time_total",
            "self_device_memory_usage",
            "self_cpu_memory_usage",
            "device_memory_usage",
            "cpu_memory_usage",
        ]

    def _get_profiler_events(self) -> EventList:
        """Return the collected PyTorch events.

        Returns:
            The finalized PyTorch event list.

        Raises:
            AssertionError: If profiling has not been stopped correctly.
        """
        profiler = self.profiler
        assert profiler is not None, "Profiling not stopped correctly"
        profiler._ensure_function_events()
        events = profiler._function_events
        assert events is not None, "Profiler events are unavailable"
        return events

    def _get_profiler_events_by_record_step(self) -> dict[str, list[FunctionEvent]]:
        """Group events by the most recent recorded step.

        Returns:
            A mapping from recorded step names to their events.
        """
        events = self._get_profiler_events()
        matched_events = {step: [] for _, step in self.record_steps}
        profiler = self.profiler
        assert profiler is not None, "Profiling not stopped correctly"
        kineto_results = profiler.kineto_results
        assert kineto_results is not None, "Profiler trace is unavailable"
        base_timestamp = kineto_results.trace_start_ns() * 1e-3

        for event in events:
            event_timestamp = base_timestamp + event.time_range.start
            previous_steps = [
                (event_timestamp - timestamp.timestamp() * 1e6, name)
                for timestamp, name in self.record_steps
                if event_timestamp >= timestamp.timestamp() * 1e6
            ]
            matched_events[min(previous_steps)[1]].append(event)
        return matched_events

    def _event_rows(self) -> list[dict[str, Any]]:
        """Convert grouped PyTorch events into dataframe rows.

        Returns:
            Rows containing event metrics and their recorded step.
        """
        rows = []
        for step, events in self._get_profiler_events_by_record_step().items():
            for event in events:
                row = {column: getattr(event, column, None) for column in self.numeric_columns}
                row.update(
                    name=getattr(event, "name", None),
                    is_annotation=getattr(event, "is_user_annotation", None),
                    device=getattr(event, "device_type", None).name,  # type: ignore[union-attr]
                    record_step=step,
                )
                rows.append(row)
        return rows

    def to_pandas(self) -> pd.DataFrame:
        """Return one row per PyTorch event with timing and memory metrics.

        Returns:
            A dataframe containing event metrics, devices, and percentages.
        """
        df = pd.DataFrame(self._event_rows())
        df.loc[df["device"] == "CPU", ["self_device_time_total", "device_time_total"]] = 0
        df.loc[df["device"] == "CUDA", ["self_cpu_time_total", "cpu_time_total"]] = 0
        df["self_cpu_time_total_percentage"] = (
            df["self_cpu_time_total"] / df["self_cpu_time_total"].sum() * 100
        )
        df["cpu_time_total_percentage"] = df["cpu_time_total"] / df["cpu_time_total"].sum() * 100
        return df

    def summary(self) -> pd.DataFrame:
        """Return metrics summed by event name, excluding annotations.

        Returns:
            A dataframe with metrics grouped and sorted by event name.
        """
        df = self.to_pandas()
        return (
            df[~df["is_annotation"]][["name"] + self.numeric_columns]
            .groupby("name")
            .sum()
            .sort_values(by=["flops", "count"])
        )

    def totals(self) -> pd.Series:
        """Return total metrics for non-annotation events.

        Returns:
            A series containing the sum of each numeric metric.
        """
        df = self.to_pandas()
        return df[~df["is_annotation"]][self.numeric_columns].sum(axis=0)

    def total_time(self, device: str = "CUDA") -> float:
        """Return total self time in microseconds for one device.

        Args:
            device: Device to measure, either ``"CPU"`` or ``"CUDA"``.

        Returns:
            Total self time in microseconds.

        Raises:
            AssertionError: If ``device`` is not ``"CPU"`` or ``"CUDA"``.
        """
        assert device in {"CPU", "CUDA"}, "device must be either CPU or CUDA"
        time_field = "self_cpu_time_total" if device == "CPU" else "self_device_time_total"
        return sum(
            getattr(event, time_field, 0.0)
            for event in self._get_profiler_events()
            if event.device_type.name == device and not event.is_user_annotation
        )

    def total_flops(self) -> int:
        """Return the total FLOPs recorded by PyTorch.

        Returns:
            The total number of floating-point operations.
        """
        return int(sum((getattr(event, "flops", 0.0) or 0.0) for event in self._get_profiler_events()))

    def flops_by_step(self) -> pd.DataFrame:
        """Return FLOPs summed for each recorded step.

        Returns:
            A dataframe indexed by step name with a ``flops`` column.
        """
        flops_by_step = {
            name: sum((event.flops or 0.0) for event in events)
            for name, events in self._get_profiler_events_by_record_step().items()
        }
        return pd.DataFrame.from_dict(flops_by_step, orient="index", columns=["flops"])

    def time_by_step(self) -> pd.DataFrame:
        """Return CPU and CUDA self time in microseconds for each step.

        Returns:
            A dataframe indexed by step with ``cpu_time`` and ``gpu_time``.
        """
        time_by_step = {}
        for step, events in self._get_profiler_events_by_record_step().items():
            cpu_time = sum(
                getattr(event, "self_cpu_time_total", 0.0)
                for event in events
                if event.device_type.name == "CPU" and not event.is_user_annotation
            )
            gpu_time = sum(
                getattr(event, "self_device_time_total", 0.0)
                for event in events
                if event.device_type.name == "CUDA" and not event.is_user_annotation
            )
            time_by_step[step] = (cpu_time, gpu_time)
        return pd.DataFrame.from_dict(
            time_by_step, orient="index", columns=["cpu_time", "gpu_time"]
        )

    # Compatibility aliases for versions before the API cleanup.
    get_total_time = total_time
    get_total_flops = total_flops
    get_flops_by_step = flops_by_step
    get_time_by_step = time_by_step
