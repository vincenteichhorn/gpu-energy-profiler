"""Profile a CUDA matrix multiplication with nvidia-smi."""

from pathlib import Path
import time

import torch

from gpu_energy_profiler import NvidiaProfiler


def run_matmul_phase(workload: torch.Tensor, duration: float = 0.25) -> None:
    """Run synchronized matrix multiplications for a fixed duration.

    Args:
        workload: CUDA matrix used for multiplication.
        duration: Minimum phase duration in seconds.
    """
    deadline = time.perf_counter() + duration
    while time.perf_counter() < deadline:
        torch.mm(workload, workload)
        torch.cuda.synchronize()


def main() -> None:
    """Run the nvidia-smi example."""
    if not torch.cuda.is_available():
        print("No CUDA GPU is available. Install an NVIDIA driver and CUDA PyTorch.")
        return

    output_dir = Path("examples/profile-data")
    output_dir.mkdir(exist_ok=True)
    workload = torch.rand((2048, 2048), device="cuda")

    try:
        with NvidiaProfiler(
            interval=10,
            backend="nvidia-smi",
            cache_file=str(output_dir / "nvidia-smi.csv"),
            force_cache=True,
        ) as profiler:
            for _ in range(5):
                with profiler.record_context("matmul"):
                    run_matmul_phase(workload)
                with profiler.record_context("sleep"):
                    torch.cuda.synchronize()
                    time.sleep(0.25)

        print("Backend: nvidia-smi")
        print("GPUs:", profiler.profiled_gpus())
        print("Energy (Ws):", profiler.total_energy())
        print("Average memory (MiB):", profiler.avg_memory_usage())
        figure = profiler.time_series_plot()
        if figure is not None:
            plot_path = output_dir / "nvidia_smi.html"
            figure.write_html(plot_path)
            print(f"Plot saved to {plot_path}")
    except RuntimeError as error:
        print(f"nvidia-smi is unavailable: {error}")


if __name__ == "__main__":
    main()
