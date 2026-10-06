"""Profile a small CPU-only PyTorch workload."""

import time
from pathlib import Path

import plotly.express as px
import torch

from gpu_energy_profiler import TorchProfiler


def main() -> None:
    """Run the example and print the collected summaries.

    Returns:
        ``None``. Results are printed to standard output.
    """
    profiler = TorchProfiler(
        activities=[torch.profiler.ProfilerActivity.CPU],
        with_flops=True,
    )
    with profiler:
        for _ in range(5):
            with profiler.record_context("matmul"):
                matrix = torch.ones((512, 512))
                torch.mm(matrix, matrix)
        with profiler.record_context("sleep"):
            torch.cuda.synchronize()
            time.sleep(0.1)

    print("Total CPU time (microseconds):", profiler.total_time("CPU"))
    print("\nSummary:")
    print(profiler.summary())
    print("\nTime by step:")
    print(profiler.time_by_step())
    print("\nTotal FLOPS:", profiler.total_flops())
    print("\nFLOPS by step:")
    flops_by_step = profiler.flops_by_step()
    print(flops_by_step)

    output_dir = Path("examples/profile-data")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure = px.bar(
        flops_by_step,
        y="flops",
        labels={"index": "Step", "flops": "FLOPS"},
        title="FLOPS by Profiling Step",
    )
    plot_path = output_dir / "torch_flops.html"
    figure.write_html(plot_path)
    print(f"\nPlot saved to {plot_path}")


if __name__ == "__main__":
    main()
