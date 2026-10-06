"""Shared functionality for profiler implementations."""

from contextlib import contextmanager
from datetime import datetime
from typing import Generator


class Profiler:
    """Record named timestamps while a profiling session is running."""

    def __init__(self) -> None:
        """Initialize an empty step timeline."""
        self.record_steps: list[tuple[datetime, str]] = []
        self.record_step("__init__")

    def record_step(self, name: str) -> None:
        """Record a named timestamp.

        Args:
            name: Label for the recorded step.
        """
        self.record_steps.append((datetime.now(), name))

    @contextmanager
    def record_context(self, name: str) -> Generator[None, None, None]:
        """Record a context's start and end markers.

        Args:
            name: Label recorded when entering the context.
        """
        try:
            self.record_step(name)
            yield
        finally:
            self.record_step("__other__")
