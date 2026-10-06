"""Result handlers and helpers for profiler worker processes."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from itertools import zip_longest
from multiprocessing import Manager, Process, Queue
from pathlib import Path
from typing import Any
import warnings

Result = tuple[Any, ...]
ResultOrSentinel = Result | None


class ResultHandler(ABC):
    """Store worker results behind a queue-like interface."""

    def __init__(self) -> None:
        """Initialize an unconfigured result handler."""
        self.column_names: tuple[str, ...] = ()
        self.dtypes: tuple[type, ...] = ()

    def __enter__(self) -> "ResultHandler":
        """Enter the handler context.

        Returns:
            This result handler.
        """
        return self

    def __exit__(self, *context_info: Any) -> None:
        """Exit the handler context.

        Args:
            *context_info: Context-manager exception information.
        """
        del context_info

    def set_columns(
        self,
        names: tuple[str, ...],
        dtypes: tuple[type, ...] | None = None,
    ) -> None:
        """Set column names and conversion types used by file-backed results.

        Args:
            names: Column names in the serialized result rows.
            dtypes: Conversion types for serialized values. Defaults to ``str``.

        Raises:
            ValueError: If the number of names and types differs.
        """
        dtypes = dtypes or (str,) * len(names)
        if len(names) != len(dtypes):
            raise ValueError("names and dtypes must have the same length")
        self.column_names = names
        self.dtypes = dtypes

    @abstractmethod
    def put(self, data: ResultOrSentinel) -> None:
        """Store one result or ``None`` as the end-of-results sentinel.

        Args:
            data: Result tuple or ``None`` to mark the end of the stream.
        """

    @abstractmethod
    def latest(self) -> ResultOrSentinel:
        """Return the latest result, or ``None`` when no result is available."""

    @abstractmethod
    def all(self) -> list[Result]:
        """Return all results up to the end-of-results sentinel."""

    def get(self) -> ResultOrSentinel:
        """Return the next result using the legacy method name.

        Returns:
            The next result or ``None``.
        """
        return self.latest()

    def get_all(self) -> list[Result]:
        """Return all results using the legacy method name.

        Returns:
            All results up to the end-of-results sentinel.
        """
        return self.all()


class MPQueueResultHandler(ResultHandler):
    """Store results in a multiprocessing queue."""

    def __init__(self) -> None:
        """Initialize an in-memory queue result handler."""
        super().__init__()
        self.queue: Queue[ResultOrSentinel] = Queue()

    def put(self, data: ResultOrSentinel) -> None:
        """Put a result or sentinel into the queue.

        Args:
            data: Result tuple or ``None`` as the stream sentinel.
        """
        self.queue.put(data)

    def latest(self) -> ResultOrSentinel:
        """Block until and return the next queued result.

        Returns:
            The next queued result or ``None``.
        """
        return self.queue.get()

    def all(self) -> list[Result]:
        """Read results until the sentinel is encountered.

        Returns:
            All queued result tuples.
        """
        return [result for result in iter(self.queue.get, None)]


class FileCacheResultHandler(ResultHandler):
    """Append typed result rows to a CSV cache file."""

    def __init__(self, file_path: str, force: bool = False) -> None:
        """Initialize a CSV-backed result handler.

        Args:
            file_path: Path to the CSV cache.
            force: Whether to replace an existing cache.
        """
        super().__init__()
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if force and self.file_path.is_file():
            self.file_path.unlink()
        if self.file_path.is_file() and self.file_path.stat().st_size > 0:
            warnings.warn(
                f"File {self.file_path} already exists and is not empty",
                UserWarning,
                stacklevel=2,
            )

    def __enter__(self) -> "FileCacheResultHandler":
        """Enter the file-cache context.

        Returns:
            This file-cache result handler.
        """
        return self

    def set_columns(
        self,
        names: tuple[str, ...],
        dtypes: tuple[type, ...] | None = None,
    ) -> None:
        """Write the CSV header when the cache is empty."""
        super().set_columns(names, dtypes)
        if self.file_path.exists() and self.file_path.stat().st_size > 0:
            return
        self.file_path.write_text(",".join(names) + "\n", encoding="utf-8")

    def put(self, data: ResultOrSentinel) -> None:
        """Append a tuple result and ignore the worker sentinel.

        Args:
            data: Tuple to append, or ``None`` to end a worker stream.
        """
        if not isinstance(data, tuple):
            return
        with self.file_path.open("a", encoding="utf-8") as file:
            file.write(",".join(str(value) for value in data) + "\n")

    def _parse_line(self, line: str) -> Result:
        """Convert one serialized CSV row to a typed result tuple.

        Args:
            line: Comma-separated result row.

        Returns:
            The converted result tuple.
        """
        values = zip_longest(
            line.strip().split(","),
            self.dtypes,
            fillvalue=str,
        )
        return tuple(converter(value) for value, converter in values)

    def latest(self) -> ResultOrSentinel:
        """Return the last cached row, or ``None`` for an empty cache."""
        rows = self._rows()
        return self._parse_line(rows[-1]) if rows else None

    def all(self) -> list[Result]:
        """Return every cached row in file order.

        Returns:
            Typed result tuples in cache order.
        """
        return [self._parse_line(row) for row in self._rows()]

    def _rows(self) -> list[str]:
        """Read data rows from the cache file.

        Returns:
            Raw CSV data rows without the optional header.
        """
        with self.file_path.open("r", encoding="utf-8") as file:
            rows = file.readlines()
        return rows[1:] if self.column_names else rows


def start_separate_process(target: Callable[..., Any], arguments: list[Any]) -> Any:
    """Run ``target`` in a spawned process and return its first result.

    Args:
        target: Worker function receiving a managed queue first.
        arguments: Additional arguments passed to the worker.

    Returns:
        The first result placed in the managed queue.
    """
    with Manager() as manager:
        queue = manager.Queue()
        process = Process(target=target, args=[queue, *arguments])
        process.start()
        process.join()
        return queue.get()


# Compatibility alias for the original misspelled function name.
start_seprate_process = start_separate_process
