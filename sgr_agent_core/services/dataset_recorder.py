"""Dataset recording service for distillation.

Captures LLM interactions (raw calls and full agent trajectories) as JSONL
records, suitable for synthesizing instruction-tuning datasets via
distillation from a teacher model.
"""

import asyncio
import json
import logging
import os
from typing import Any

from sgr_agent_core.agent_definition import DatasetRecordingConfig

logger = logging.getLogger(__name__)

CALLS_FILENAME = "llm_calls.jsonl"
TRAJECTORIES_FILENAME = "trajectories.jsonl"


class DatasetRecorder:
    """Writes LLM interaction records to JSONL files.

    Two record types are produced (each independently gated by ``modes``):
    - ``raw`` records (``llm_calls.jsonl``): one per LLM call.
    - ``trajectory`` records (``trajectories.jsonl``): one per full agent run.

    Writes are serialized with an :class:`asyncio.Lock` so the recorder is safe
    to share across concurrently running agents that write to the same files.
    """

    def __init__(self, config: DatasetRecordingConfig, teacher_model: str | None = None) -> None:
        """Initialize the recorder and create the output directory.

        Args:
            config: Dataset recording configuration (output dir, modes, ...).
            teacher_model: Tag of the teacher model used for this recording.
        """
        self._config = config
        self._teacher_model = teacher_model
        self._lock = asyncio.Lock()
        self._output_dir = config.output_dir
        self._calls_path = os.path.join(self._output_dir, CALLS_FILENAME)
        self._trajectories_path = os.path.join(self._output_dir, TRAJECTORIES_FILENAME)
        os.makedirs(self._output_dir, exist_ok=True)

    @property
    def config(self) -> DatasetRecordingConfig:
        """The recording configuration."""
        return self._config

    @property
    def teacher_model(self) -> str | None:
        """The teacher model tag."""
        return self._teacher_model

    async def record_call(self, record: dict[str, Any]) -> None:
        """Append a raw LLM call record to ``llm_calls.jsonl``.

        No-op when ``raw`` is not among the configured modes.

        Args:
            record: A serialized raw-call record (Level A).
        """
        if "raw" not in self._config.modes:
            return
        await self._append(self._calls_path, record)

    async def record_trajectory(self, record: dict[str, Any]) -> None:
        """Append a full agent trajectory record to ``trajectories.jsonl``.

        No-op when ``trajectory`` is not among the configured modes.

        Args:
            record: A serialized trajectory record (Level B, sharegpt-style).
        """
        if "trajectory" not in self._config.modes:
            return
        await self._append(self._trajectories_path, record)

    async def _append(self, path: str, record: dict[str, Any]) -> None:
        """Serialize one record to a JSONL line and append it under the lock."""
        line = json.dumps(record, ensure_ascii=False, default=str)
        async with self._lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


# Module-level singleton so agents and the factory share a single recorder.
_recorder: DatasetRecorder | None = None


def get_recorder() -> DatasetRecorder | None:
    """Return the active shared :class:`DatasetRecorder`, or ``None``."""
    return _recorder


def set_recorder(recorder: DatasetRecorder | None) -> None:
    """Install ``recorder`` as the active shared recorder."""
    global _recorder
    _recorder = recorder


def reset_recorder() -> None:
    """Clear the active shared recorder (useful for tests)."""
    global _recorder
    _recorder = None
