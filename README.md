# gpu-energy-profiler

[![PyPI version](https://img.shields.io/pypi/v/gpu-energy-profiler.svg)](https://pypi.org/project/gpu-energy-profiler/)
[![Python versions](https://img.shields.io/pypi/pyversions/gpu-energy-profiler.svg)](https://pypi.org/project/gpu-energy-profiler/)
[![License: not specified](https://img.shields.io/badge/license-not%20specified-lightgrey.svg)](#license)
[![CI: not configured](https://img.shields.io/badge/CI-not%20configured-lightgrey.svg)](#development)

Profile PyTorch work and measure NVIDIA GPU power and memory use.

## Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Usage](#usage)
- [Configuration](#configuration)
- [API reference](#api-reference)
- [Examples](#examples)
- [Full documentation](#full-documentation)
- [Development and testing](#development-and-testing)
- [Contributing](#contributing)
- [License](#license)

## Overview

`gpu-energy-profiler` provides a small Python API for two profiling tasks:

1. Collect CPU and CUDA events from `torch.profiler`.
2. Sample NVIDIA GPU power and memory use with `nvidia-smi` or NVML.

The package groups measurements by named application steps. It returns
Pandas data frames for analysis and Plotly figures for interactive review.
NVIDIA samples can also be written to a CSV cache.

The package requires Python 3.14 or newer.

## Features

- Profile CPU and CUDA PyTorch events.
- Measure FLOPS and memory use reported by PyTorch.
- Group PyTorch metrics by named record steps.
- Sample NVIDIA power and memory use with `nvidia-smi`.
- Sample NVIDIA power and memory use with `pynvml`.
- Store NVIDIA samples in CSV files.
- Load cached NVIDIA samples for later analysis.
- Return Pandas data frames for downstream analysis.
- Create interactive Plotly GPU power and memory plots.
- Keep backend failures visible with clear runtime errors.

## Installation

Install the released package from PyPI:

```bash
python -m pip install gpu-energy-profiler
```

The package installs PyTorch, Pandas, Plotly, and `nvidia-ml-py`.
The `pynvml` backend uses `nvidia-ml-py`. The `nvidia-smi` backend also
requires the NVIDIA driver and the `nvidia-smi` executable on `PATH`.

There are no optional extras in version `0.1.0`. For a source checkout,
install the development dependencies with Poetry:

```bash
poetry install
```

## Quickstart

This example profiles a CPU matrix multiplication and prints the total FLOPS:

```python
import torch

from gpu_energy_profiler import TorchProfiler


with TorchProfiler(
    activities=[torch.profiler.ProfilerActivity.CPU],
    with_flops=True,
) as profiler:
    with profiler.record_context("matmul"):
        matrix = torch.ones((512, 512))
        torch.mm(matrix, matrix)

print(f"Total FLOPS: {profiler.total_flops()}")
print(profiler.summary())
```

## Usage

### Profile PyTorch events

`TorchProfiler` wraps `torch.profiler.profile`. It enables FLOPS and memory
collection by default. Pass PyTorch profiler options when you need different
activities or limits.

```python
import torch

from gpu_energy_profiler import TorchProfiler


profiler = TorchProfiler(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    with_flops=True,
    profile_memory=True,
)

with profiler:
    with profiler.record_context("inference"):
        inputs = torch.ones((1024, 1024), device="cuda")
        model_output = torch.mm(inputs, inputs)
        torch.cuda.synchronize()

print("CPU time (microseconds):", profiler.total_time("CPU"))
print("CUDA time (microseconds):", profiler.total_time("CUDA"))
print("Total FLOPS:", profiler.total_flops())
print(profiler.flops_by_step())
print(profiler.time_by_step())
print(profiler.summary())
```

Use a CPU-only activity list on systems without CUDA:

```python
import torch

from gpu_energy_profiler import TorchProfiler


with TorchProfiler(
    activities=[torch.profiler.ProfilerActivity.CPU],
) as profiler:
    with profiler.record_context("cpu-work"):
        tensor = torch.ones((256, 256))
        torch.mm(tensor, tensor)

print(profiler.total_time("CPU"))
```

### Profile NVIDIA power and memory

`NvidiaProfiler` samples all visible NVIDIA GPUs in a worker process.
Select the backend explicitly. Use a short interval for detailed time series
data and a longer interval for lower overhead.

```python
from gpu_energy_profiler import NvidiaProfiler


with NvidiaProfiler(
    backend="pynvml",
    interval=100,
    cache_file="profile-data/nvidia.csv",
    force_cache=True,
) as profiler:
    # Run the GPU workload here.
    pass

print("GPUs:", profiler.profiled_gpus())
print("Energy (watt-seconds):", profiler.total_energy())
print("Average memory (MiB):", profiler.avg_memory_usage())

figure = profiler.time_series_plot()
if figure is not None:
    figure.write_html("profile-data/nvidia.html")
```

Use `backend="nvidia-smi"` when the command-line backend is preferred:

```python
from gpu_energy_profiler import NvidiaProfiler


with NvidiaProfiler(backend="nvidia-smi", interval=100) as profiler:
    # Run the GPU workload here.
    pass
```

Name workload phases with `record_context`. The name is stored with every
sample collected during that phase:

```python
import time

from gpu_energy_profiler import NvidiaProfiler


with NvidiaProfiler(backend="pynvml", interval=50) as profiler:
    for _ in range(3):
        with profiler.record_context("matmul"):
            # Run the workload here.
            time.sleep(0.2)
        with profiler.record_context("idle"):
            time.sleep(0.2)
```

### Load cached samples

Use `from_cache` to analyse a CSV without starting a sampling process:

```python
from gpu_energy_profiler import NvidiaProfiler


profiler = NvidiaProfiler.from_cache("profile-data/nvidia.csv")
data = profiler.to_pandas()
print(data)
```

## Configuration

### `TorchProfiler`

`TorchProfiler` accepts the options supported by
`torch.profiler.profile`, including:

- `activities`: Select CPU, CUDA, or both.
- `with_flops`: Collect FLOPS when PyTorch supports the operation.
- `profile_memory`: Collect CPU and device memory metrics.
- `record_shapes`: Include tensor shape information.
- `with_stack`: Include Python stack information.

The package defaults are `with_flops=True`, `profile_memory=True`, and both
CPU and CUDA activities. Select explicit activities on CPU-only systems.

### `NvidiaProfiler`

The constructor accepts:

- `backend`: `"nvidia-smi"` or `"pynvml"`.
- `interval`: Sampling interval in milliseconds.
- `cache_file`: Optional CSV output path.
- `force_cache`: Remove an existing cache before sampling.

The `pynvml` backend needs a working NVIDIA driver and at least one visible
GPU. The `nvidia-smi` backend needs a working `nvidia-smi` command and at
least one queryable GPU. A missing dependency, command, driver, or GPU raises
a `RuntimeError` instead of returning incomplete results.

## API reference

This section documents every public symbol exported by `gpu_energy_profiler`.

### `Profiler`

Import with:

```python
from gpu_energy_profiler import Profiler
```

`Profiler` is the shared step-timeline class. It gives profilers a common
way to label workload phases. The initial `__init__` marker is added
automatically.

#### `record_step(name)`

Record a named timestamp. Use this when the application controls the phase
boundaries directly.

#### `record_context(name)`

Return a context manager that records `name` on entry and `__other__` on exit.
The `finally` behavior records the end marker even when the workload raises.

### `TorchProfiler`

Import with:

```python
from gpu_energy_profiler import TorchProfiler
```

`TorchProfiler` combines PyTorch event collection with the shared step
timeline. It exists so one profiler object can collect events and provide
ready-to-use aggregate views.

#### `to_pandas()`

Return one row per PyTorch event. The result includes event names, devices,
FLOPS, counts, CPU and device time, memory values, record steps, and time
percentages.

#### `summary()`

Group non-annotation events by event name. Use this for a compact operation
summary.

#### `totals()`

Return the sums of the numeric event columns for non-annotation events.
Use this when a Series is easier to consume than a grouped data frame.

#### `total_time(device="CUDA")`

Return total self time in microseconds for `CPU` or `CUDA`.
Use the device argument to compare host and device work.

#### `total_flops()`

Return the total FLOPS reported by PyTorch. Operations without FLOPS metadata
contribute zero.

#### `flops_by_step()`

Return a data frame with one `flops` value per recorded step. Use this to
compare named workload phases.

#### `time_by_step()`

Return CPU and CUDA self time per recorded step. The result has `cpu_time`
and `gpu_time` columns, both in microseconds.

### `NvidiaProfiler`

Import with:

```python
from gpu_energy_profiler import NvidiaProfiler
```

`NvidiaProfiler` samples GPU power and memory in a separate process. This
keeps sampling independent from the workload and supports both NVIDIA
backends.

#### Constructor

```python
NvidiaProfiler(
    interval=1,
    cache_file=None,
    force_cache=False,
    backend="nvidia-smi",
)
```

- `interval`: Sampling interval in milliseconds.
- `cache_file`: CSV file for samples. If omitted, samples stay in memory.
- `force_cache`: Replace an existing cache file.
- `backend`: Select `"nvidia-smi"` or `"pynvml"`.

#### `to_pandas()`

Return samples with these columns:
`gpu_id`, `timestamp`, `power`, `memory`, and `record_step`.
Power is in watts. Memory is in MiB.

#### `from_cache(path)`

Create a profiler from an existing CSV cache. Use this for offline analysis
without a GPU sampling process.

#### `profiled_gpus()`

Return the unique GPU IDs found in the samples.

#### `total_energy(gpu_ids=None, record_steps=None, return_data=False)`

Estimate sampled energy in watt-seconds. Filter by GPU or record step when
needed. Set `return_data=True` to receive one value per sampled step group.

#### `total_time()`

Return the time between the first and last sample in seconds.

#### `avg_memory_usage(gpu_id=None)`

Return average memory use in MiB. Without a GPU ID, use the first sampled GPU.

#### `time_series_plot()`

Return a Plotly figure with power and memory traces and shaded record-step
regions. Return `None` when there are no samples.

### `sample_colors`

`sample_colors` is an internal plotting utility in
`gpu_energy_profiler.plotting_util`. It returns evenly spaced colors from a Plotly
color scale so GPU traces and step regions remain distinct.

### Result handlers

The result handlers are internal building blocks in
`gpu_energy_profiler.multiprocessing_util`. They transfer worker results or write
typed rows to CSV. They are not exported from the package root.

#### `ResultHandler`

Define the queue-like result contract used by profiler workers. It provides
`set_columns`, `put`, `latest`, and `all`. Use a subclass when a new result
destination is needed.

#### `MPQueueResultHandler`

Store results in a multiprocessing queue. Use it when samples must stay in
memory and another process consumes them.

#### `FileCacheResultHandler`

Write typed result rows to a CSV file. Use it to keep samples after the
profiling process ends. `force=True` replaces an existing file.

#### `start_separate_process(target, arguments)`

Run a worker with a managed queue and return its first result. Use it for
small multiprocessing helpers outside the profiler classes.

## Examples

Run the CPU and NVIDIA examples from a source checkout:

```bash
poetry run python examples/torch_profile.py
poetry run python examples/nvidia_profile_pynvml.py
poetry run python examples/nvidia_profile_smi.py
```

The NVIDIA examples require a working NVIDIA GPU and backend. They write CSV
samples and interactive HTML plots to `examples/profile-data/`. The Torch
example writes `torch_flops.html` to the same directory.

## Full documentation

The complete API documentation is in this README. The source docstrings in
[`src/gpu_energy_profiler`](src/gpu_energy_profiler) provide implementation-level details.

## Development and testing

Clone the source tree and install the project with Poetry:

```bash
git clone <repository-url>
cd gpu-energy-profiler
poetry install
```

Run the test suite:

```bash
poetry run pytest -q
```

Compile the package and examples:

```bash
poetry run python -m compileall -q src tests examples
```

The test suite covers the shared profiler, PyTorch metrics, NVIDIA parsers,
both NVIDIA backends, cache handling, multiprocessing helpers, and plotting.

## Contributing

1. Create a focused branch.
2. Make a small, tested change.
3. Add or update tests for changed behavior.
4. Run `poetry run pytest -q`.
5. Update this README when the public API changes.
6. Open a pull request with a clear summary.

Keep public APIs typed, keep error messages actionable, and keep examples
small enough to copy and understand.

## License

No license file or license metadata is present in version `0.1.0`.
Add a license before publishing a release that grants reuse rights.
