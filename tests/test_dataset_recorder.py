"""Tests for the DatasetRecorder service."""

import asyncio
import json
import os

import pytest

from sgr_agent_core.agent_definition import DatasetRecordingConfig
from sgr_agent_core.services.dataset_recorder import DatasetRecorder, get_recorder, reset_recorder, set_recorder


def _read_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class TestDatasetRecorderFiles:
    """Tests for JSONL file creation and writing."""

    @pytest.mark.asyncio
    async def test_record_call_writes_llm_calls_jsonl(self, tmp_path):
        """record_call appends a record to llm_calls.jsonl when 'raw' mode is on."""
        reset_recorder()
        cfg = DatasetRecordingConfig(enabled=True, output_dir=str(tmp_path), modes=["raw"])
        recorder = DatasetRecorder(cfg, teacher_model="glm-5.2")
        await recorder.record_call({"record_type": "llm_call", "call_id": "c1", "phase": "action"})

        path = os.path.join(str(tmp_path), "llm_calls.jsonl")
        assert os.path.exists(path)
        records = _read_jsonl(path)
        assert len(records) == 1
        assert records[0]["call_id"] == "c1"
        assert records[0]["phase"] == "action"

    @pytest.mark.asyncio
    async def test_record_trajectory_writes_trajectories_jsonl(self, tmp_path):
        """record_trajectory appends a record to trajectories.jsonl when 'trajectory' mode is on."""
        reset_recorder()
        cfg = DatasetRecordingConfig(enabled=True, output_dir=str(tmp_path), modes=["trajectory"])
        recorder = DatasetRecorder(cfg, teacher_model="glm-5.2")
        await recorder.record_trajectory({"record_type": "trajectory", "messages": []})

        path = os.path.join(str(tmp_path), "trajectories.jsonl")
        assert os.path.exists(path)
        assert len(_read_jsonl(path)) == 1

    @pytest.mark.asyncio
    async def test_mode_filtering_skips_disabled_modes(self, tmp_path):
        """A recorder with only 'trajectory' mode does not write llm_calls.jsonl."""
        reset_recorder()
        cfg = DatasetRecordingConfig(enabled=True, output_dir=str(tmp_path), modes=["trajectory"])
        recorder = DatasetRecorder(cfg, teacher_model="glm-5.2")
        await recorder.record_call({"call_id": "c1"})
        assert not os.path.exists(os.path.join(str(tmp_path), "llm_calls.jsonl"))

    @pytest.mark.asyncio
    async def test_unicode_content_preserved(self, tmp_path):
        """Non-ASCII (Cyrillic) content is written with ensure_ascii=False."""
        reset_recorder()
        cfg = DatasetRecordingConfig(enabled=True, output_dir=str(tmp_path), modes=["trajectory"])
        recorder = DatasetRecorder(cfg, teacher_model="glm-5.2")
        await recorder.record_trajectory({"messages": [{"content": "Безопасность детей"}]})

        raw = open(os.path.join(str(tmp_path), "trajectories.jsonl"), encoding="utf-8").read()
        assert "Безопасность детей" in raw

    @pytest.mark.asyncio
    async def test_concurrent_writes_are_serialized(self, tmp_path):
        """Many concurrent record_call writes produce that many valid lines."""
        reset_recorder()
        cfg = DatasetRecordingConfig(enabled=True, output_dir=str(tmp_path), modes=["raw"])
        recorder = DatasetRecorder(cfg, teacher_model="glm-5.2")
        await asyncio.gather(*[recorder.record_call({"call_id": f"c{i}"}) for i in range(50)])

        path = os.path.join(str(tmp_path), "llm_calls.jsonl")
        records = _read_jsonl(path)
        assert len(records) == 50
        assert {r["call_id"] for r in records} == {f"c{i}" for i in range(50)}

    def test_teacher_model_and_config_accessors(self, tmp_path):
        cfg = DatasetRecordingConfig(enabled=True, output_dir=str(tmp_path), modes=["raw", "trajectory"])
        recorder = DatasetRecorder(cfg, teacher_model="glm-5.2")
        assert recorder.teacher_model == "glm-5.2"
        assert recorder.config is cfg


class TestDatasetRecorderSingleton:
    """Tests for the module-level singleton accessors."""

    def teardown_method(self):
        reset_recorder()

    def test_get_recorder_returns_none_by_default(self):
        reset_recorder()
        assert get_recorder() is None

    def test_set_and_get_recorder(self, tmp_path):
        cfg = DatasetRecordingConfig(enabled=True, output_dir=str(tmp_path))
        recorder = DatasetRecorder(cfg, teacher_model="glm-5.2")
        set_recorder(recorder)
        assert get_recorder() is recorder

    def test_reset_recorder_clears_singleton(self, tmp_path):
        cfg = DatasetRecordingConfig(enabled=True, output_dir=str(tmp_path))
        set_recorder(DatasetRecorder(cfg, teacher_model="glm-5.2"))
        reset_recorder()
        assert get_recorder() is None
