"""GPU power and memory profiling through NVML or ``nvidia-smi``."""

from datetime import datetime
import subprocess
from multiprocessing import Array, Event, Process, Value
import time
from typing import Any, Literal

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .base import Profiler
from .multiprocessing_util import FileCacheResultHandler, MPQueueResultHandler, ResultHandler
from .plotting_util import sample_colors


def _load_pynvml() -> Any:
    """Import and return the optional ``pynvml`` module.

    Returns:
        The imported NVML Python bindings.

    Raises:
        RuntimeError: If the optional dependency is not installed.
    """
    try:
        import pynvml
    except ImportError as error:
        raise RuntimeError(
            "The pynvml backend requires the optional 'nvidia-ml-py' dependency."
        ) from error
    return pynvml


class NvidiaProfiler(Profiler):
    """Sample GPU power and memory usage with a selectable backend."""

    _COLUMNS = ("gpu_id", "timestamp", "power", "memory", "record_step")
    _BACKENDS = ("nvidia-smi", "pynvml")

    def __init__(
        self,
        interval: int = 1,
        cache_file: str | None = None,
        force_cache: bool = False,
        backend: Literal["nvidia-smi", "pynvml"] = "nvidia-smi",
    ) -> None:
        """Initialize an NVIDIA power and memory profiler.

        Args:
            interval: Sampling interval in milliseconds.
            cache_file: Optional CSV path for persisted samples.
            force_cache: Whether to replace an existing cache file.
            backend: Sampling implementation, either ``"nvidia-smi"`` or
                ``"pynvml"``.

        Raises:
            ValueError: If ``backend`` is not supported.
        """
        if backend not in self._BACKENDS:
            raise ValueError(f"Unknown backend {backend!r}; choose from {self._BACKENDS}.")
        self.current_record_step: Any = Array("c", 1000)
        self.interval = interval
        self.backend = backend
        self.data: list[tuple[Any, ...]] = []
        self.should_profiling_run = Value("i", 1)
        self.profiling_started = Event()
        self.profiling_stopped = Event()
        self.result_handler: ResultHandler = (
            MPQueueResultHandler()
            if cache_file is None
            else FileCacheResultHandler(cache_file, force_cache)
        )
        self.result_handler.set_columns(self._COLUMNS, (int, str, float, float, str))
        self.process = Process(
            target=self._profiling_process,
            args=(
                self.should_profiling_run,
                self.profiling_started,
                self.profiling_stopped,
                self.result_handler,
                self.current_record_step,
                self.interval,
                self.backend,
            ),
        )
        super().__init__()

    def record_step(self, name: str) -> None:
        """Record a step and share its name with the sampling process.

        Args:
            name: Label for the recorded step.
        """
        super().record_step(name)
        self.current_record_step.value = name.encode("utf-8")

    @staticmethod
    def _parse_nvidia_smi_row(line: str, current_record_step: Any) -> tuple[Any, ...]:
        """Parse one CSV row emitted by ``nvidia-smi``.

        Args:
            line: Raw CSV sample line.
            current_record_step: Shared character array containing the step name.

        Returns:
            A typed sample tuple containing GPU, timestamp, power, memory, and step.
        """
        values = line.strip().split(", ")
        return (
            int(values[0]),
            datetime.strptime(values[1], "%Y/%m/%d %H:%M:%S.%f"),
            float(values[2].split(" ")[0]),
            float(values[3].split(" ")[0]),
            current_record_step.value.decode("utf-8"),
        )

    @staticmethod
    def _profiling_process(
        should_run: Any,
        started: Any,
        stopped: Any,
        result_handler: ResultHandler,
        current_record_step: Any,
        interval: int,
        backend: Literal["nvidia-smi", "pynvml"],
    ) -> None:
        """Run the selected sampling backend in a worker process.

        Args:
            should_run: Shared flag controlling the worker lifecycle.
            started: Event set after the sampler starts.
            stopped: Event set after sampling ends.
            result_handler: Destination for sampled rows.
            current_record_step: Shared current step name.
            interval: Sampling interval in milliseconds.
            backend: Backend to run.
        """
        if backend == "nvidia-smi":
            NvidiaProfiler._nvidia_smi_profiling_process(
                should_run,
                started,
                stopped,
                result_handler,
                current_record_step,
                interval,
            )
            return
        NvidiaProfiler._pynvml_profiling_process(
            should_run,
            started,
            stopped,
            result_handler,
            current_record_step,
            interval,
        )

    @staticmethod
    def _nvidia_smi_profiling_process(
        should_run: Any,
        started: Any,
        stopped: Any,
        result_handler: ResultHandler,
        current_record_step: Any,
        interval: int,
    ) -> None:
        """Read samples until ``should_run`` is cleared.

        Args:
            should_run: Shared flag controlling the worker lifecycle.
            started: Event set after the sampler starts.
            stopped: Event set after sampling ends.
            result_handler: Destination for sampled rows.
            current_record_step: Shared current step name.
            interval: Sampling interval in milliseconds.
        """
        command = (
            "nvidia-smi --query-gpu=index,timestamp,power.draw,memory.used "
            f"--format=csv -lms {interval}"
        )
        with (
            subprocess.Popen(
                command,
                shell=True,
                text=True,
                stdout=subprocess.PIPE,
            ) as nvidia_smi_process,
            result_handler as result,
        ):
            assert nvidia_smi_process.stdout is not None
            with nvidia_smi_process.stdout as output:
                output.readline()
                if should_run.value:
                    started.set()
                while should_run.value:
                    result.put(
                        NvidiaProfiler._parse_nvidia_smi_row(output.readline(), current_record_step)
                    )
        result.put(None)  # type: ignore[arg-type]
        stopped.set()

    @staticmethod
    def _pynvml_profiling_process(
        should_run: Any,
        started: Any,
        stopped: Any,
        result_handler: ResultHandler,
        current_record_step: Any,
        interval: int,
    ) -> None:
        """Read power and memory samples through the optional ``pynvml`` package.

        Args:
            should_run: Shared flag controlling the worker lifecycle.
            started: Event set after the sampler starts.
            stopped: Event set after sampling ends.
            result_handler: Destination for sampled rows.
            current_record_step: Shared current step name.
            interval: Sampling interval in milliseconds.
        """
        pynvml = _load_pynvml()
        pynvml.nvmlInit()
        try:
            with result_handler as result:
                if should_run.value:
                    started.set()
                while should_run.value:
                    for gpu_id in range(pynvml.nvmlDeviceGetCount()):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        result.put(
                            (
                                gpu_id,
                                datetime.now(),
                                pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
                                memory.used / (1024 * 1024),
                                current_record_step.value.decode("utf-8"),
                            )
                        )
                    time.sleep(interval / 1000.0)
                result.put(None)  # type: ignore[arg-type]
        finally:
            pynvml.nvmlShutdown()
            stopped.set()

    def __enter__(self) -> "NvidiaProfiler":
        """Start sampling and return this profiler.

        Returns:
            This active NVIDIA profiler.

        Raises:
            RuntimeError: If the selected backend is unavailable.
        """
        if self.backend == "nvidia-smi":
            status, output = subprocess.getstatusoutput(
                "nvidia-smi --query-gpu=index --format=csv,noheader"
            )
            if status != 0 or not output.strip():
                raise RuntimeError(
                    "nvidia-smi is unavailable or could not query an NVIDIA GPU. "
                    f"Command output: {output.strip() or '<empty>'}"
                )
        else:
            pynvml = _load_pynvml()
            try:
                pynvml.nvmlInit()
                if pynvml.nvmlDeviceGetCount() == 0:
                    raise RuntimeError("pynvml found no NVIDIA GPU devices.")
            except pynvml.NVMLError as error:
                raise RuntimeError(
                    "pynvml could not initialize or query an NVIDIA GPU."
                ) from error
            finally:
                pynvml.nvmlShutdown()
        self.process.start()
        if not self.profiling_started.wait(timeout=5):
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5)
            raise RuntimeError(
                f"The {self.backend} profiling worker did not start. "
                "Check the NVIDIA driver and backend installation."
            )
        return self

    def __exit__(self, *context_info: Any) -> None:
        """Stop sampling and collect all recorded rows.

        Args:
            *context_info: Context-manager exception information.
        """
        del context_info
        self.should_profiling_run.value = 0
        self.profiling_stopped.wait()
        self.data = self.result_handler.all()
        self.process.join()
        self.process.terminate()

    @staticmethod
    def from_cache(cache_file: str) -> "NvidiaProfiler":
        """Create a profiler populated from an existing cache file.

        Args:
            cache_file: Path to a previously written sample cache.

        Returns:
            A profiler containing the cached samples.
        """
        profiler = NvidiaProfiler(cache_file=cache_file)
        profiler.data = profiler.result_handler.all()
        return profiler

    def to_pandas(self) -> pd.DataFrame:
        """Return samples with parsed timestamps and standard column names.

        Returns:
            A dataframe containing GPU sample data.
        """
        df = pd.DataFrame(self.data, columns=self._COLUMNS)
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce"
        )
        return df

    def profiled_gpus(self) -> list[int]:
        """Return the IDs of GPUs present in the samples.

        Returns:
            Unique GPU IDs found in the samples.
        """
        return [int(gpu_id) for gpu_id in self.to_pandas()["gpu_id"].unique()]

    def total_energy(
        self,
        gpu_ids: list[int] | None = None,
        record_steps: list[str] | None = None,
        return_data: bool = False,
    ) -> float | list[float]:
        """Return sampled energy in watt-seconds.

        Args:
            gpu_ids: GPUs to include. Defaults to the first sampled GPU.
            record_steps: Step labels to include. Defaults to all steps.
            return_data: Return one energy value per record-step group.

        Returns:
            Total energy, or grouped energy values when ``return_data`` is true.
        """
        if not self.data:
            return 0.0
        df = self.to_pandas()
        df["record_step_id"] = df["record_step"].ne(df["record_step"].shift()).cumsum()
        gpu_ids = gpu_ids or [df["gpu_id"].unique()[0]]
        df = df[df["gpu_id"].isin(gpu_ids)].copy()
        df["time_interval"] = df["timestamp"].diff().dt.total_seconds().fillna(0)
        df["energy_interval"] = df["power"] * df["time_interval"]
        record_steps = record_steps or list(df["record_step"].unique())
        df = df[df["record_step"].isin(record_steps)]
        if return_data:
            return list(df.groupby("record_step_id")["energy_interval"].sum())
        return df["energy_interval"].sum()

    def total_time(self) -> float:
        """Return the time between the first and last sample in seconds.

        Returns:
            Elapsed profiling time in seconds.
        """
        if not self.data:
            return 0.0
        df = self.to_pandas()
        return (df["timestamp"].max() - df["timestamp"].min()).total_seconds()

    def avg_memory_usage(self, gpu_id: int | None = None) -> float:
        """Return average memory usage in MiB for a GPU.

        Args:
            gpu_id: GPU to measure. Defaults to the first sampled GPU.

        Returns:
            Average memory usage in MiB.
        """
        if not self.data:
            return 0.0
        df = self.to_pandas()
        gpu_id = gpu_id or df["gpu_id"].unique()[0]
        return df.loc[df["gpu_id"] == gpu_id, "memory"].mean()

    def time_series_plot(self) -> go.Figure | None:
        """Return a Plotly chart showing power, memory, and recorded steps.

        Returns:
            A Plotly figure, or ``None`` when no samples are available.
        """
        if not self.data:
            return None
        df = self.to_pandas()
        figure = make_subplots(specs=[[{"secondary_y": True}]])
        gpu_ids = self.profiled_gpus()
        gpu_colors = sample_colors("Rainbow", len(gpu_ids))
        for index, gpu_id in enumerate(gpu_ids):
            gpu_df = df[df["gpu_id"] == gpu_id]
            for metric, unit in (("power", "W"), ("memory", "MiB")):
                figure.add_trace(
                    go.Scatter(
                        x=gpu_df["timestamp"],
                        y=gpu_df[metric],
                        name=f"{metric.capitalize()} ({unit})",
                        mode="lines+markers",
                        legendgroup=str(gpu_id),
                        legendgrouptitle_text=f"GPU #{gpu_id}",
                        line=dict(
                            color=gpu_colors[index],
                            width=4,
                            dash="dot" if metric == "memory" else "solid",
                        ),
                    ),
                    secondary_y=metric == "memory",
                )
        _add_record_step_regions(figure, df, self.record_steps)
        figure.update_layout(
            title="GPU Memory and Power Usage", legend=dict(groupclick="toggleitem")
        )
        figure.update_xaxes(title_text="Time")
        figure.update_yaxes(title_text="Power (W)", secondary_y=False)
        figure.update_yaxes(title_text="Memory (MiB)", secondary_y=True)
        return figure

def _add_record_step_regions(
    figure: go.Figure,
    samples: pd.DataFrame,
    record_steps: list[tuple[datetime, str]],
) -> None:
    """Add shaded regions for recorded steps to a Plotly figure.

    Args:
        figure: Figure to modify.
        samples: Sample dataframe containing timestamps.
        record_steps: Timestamp and label pairs defining the regions.
    """
    max_timestamp: datetime = samples["timestamp"].max()
    steps = record_steps + [(max_timestamp, ".")]
    step_names = list(dict.fromkeys(name for _, name in steps))
    colors = dict(zip(step_names, sample_colors("viridis", len(step_names))))
    previous_timestamp, previous_name = steps[0]
    for timestamp, name in steps[1:]:
        figure.add_vrect(
            x0=previous_timestamp,
            x1=timestamp,
            annotation_text=previous_name,
            annotation_position="top left",
            line_width=0,
            opacity=0.25,
            fillcolor=colors[previous_name],
        )
        previous_timestamp, previous_name = timestamp, name
