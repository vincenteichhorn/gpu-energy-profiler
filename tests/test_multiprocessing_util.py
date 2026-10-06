from pathlib import Path

from gpu_energy_profiler.multiprocessing_util import (
    FileCacheResultHandler,
    MPQueueResultHandler,
    start_separate_process,
)


def test_queue_result_handler_reads_until_sentinel() -> None:
    handler = MPQueueResultHandler()
    handler.put((1, "sample"))
    handler.put(None)  # type: ignore[arg-type]

    assert handler.all() == [(1, "sample")]


def test_file_cache_result_handler_writes_and_converts_rows(tmp_path: Path) -> None:
    cache_file = tmp_path / "nested" / "samples.csv"
    handler = FileCacheResultHandler(str(cache_file))
    handler.set_columns(("gpu_id", "power"), (int, float))
    handler.put((2, 75.5))

    assert handler.latest() == (2, 75.5)
    assert handler.all() == [(2, 75.5)]
    assert cache_file.read_text(encoding="utf-8") == "gpu_id,power\n2,75.5\n"


def test_file_cache_force_replaces_existing_file(tmp_path: Path) -> None:
    cache_file = tmp_path / "samples.csv"
    cache_file.write_text("old\n", encoding="utf-8")

    handler = FileCacheResultHandler(str(cache_file), force=True)
    handler.set_columns(("value",), (str,))

    assert cache_file.read_text(encoding="utf-8") == "value\n"


def _worker(queue, value: int) -> None:
    queue.put(value * 2)


def test_start_separate_process_returns_worker_result() -> None:
    assert start_separate_process(_worker, [21]) == 42
