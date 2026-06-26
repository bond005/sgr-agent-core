"""Tests for the dataset export utility (trajectories -> sharegpt)."""

import json

from sgr_agent_core.cli.dataset_export import (
    collapse_final_answer,
    convert,
    substitute_cot_from_raw,
    to_sharegpt,
)


def _read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _traj(messages, tools=None):
    return {"record_type": "trajectory", "agent_id": "a1", "messages": messages, "tools": tools or []}


def _assistant_tool_call(name, arguments="{}", call_id="1"):
    """Build an assistant message that calls a tool (compact test helper)."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}],
    }


def _tool_result(call_id="1", content="results"):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


class TestCollapseFinalAnswer:
    def test_collapse_replaces_tool_roundtrip_with_text(self):
        msgs = [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "1",
                        "type": "function",
                        "function": {"name": "finalanswertool", "arguments": '{"answer": "FINAL"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "1", "content": "FINAL"},
        ]
        out = collapse_final_answer(msgs)
        assert out[-1] == {"role": "assistant", "content": "FINAL"}
        assert len(out) == 2

    def test_no_collapse_when_not_finishing(self):
        msgs = [
            _assistant_tool_call("web_search_tool"),
            _tool_result(),
        ]
        out = collapse_final_answer(msgs)
        assert out == msgs


class TestToSharegpt:
    def test_tool_mode_keeps_tool_calls(self):
        msgs = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "q"},
            _assistant_tool_call("web_search_tool"),
            _tool_result(),
        ]
        records = to_sharegpt([_traj(msgs, tools=[{"type": "function", "function": {"name": "web_search_tool"}}])])
        assert len(records) == 1
        assert "tools" in records[0]
        assert records[0]["messages"] == msgs
        assert records[0]["tools"][0]["function"]["name"] == "web_search_tool"


class TestCotSubstitution:
    def test_substitute_replaces_sgr_cot_with_reasoning_content(self):
        trajs = [_traj([{"role": "assistant", "content": "Reasoning steps:\n- x"}])]
        calls = [
            {"agent_id": "a1", "phase": "reasoning", "iteration": 1, "response": {"reasoning_content": "DEEP THOUGHT"}}
        ]
        out = substitute_cot_from_raw(trajs, calls)
        assert out[0]["messages"][0]["content"] == "DEEP THOUGHT"

    def test_no_calls_keeps_original(self):
        trajs = [_traj([{"role": "assistant", "content": "Reasoning steps:\n- x"}])]
        out = substitute_cot_from_raw(trajs, [])
        assert out[0]["messages"][0]["content"].startswith("Reasoning steps:")


class TestConvertEndToEnd:
    def test_convert_writes_train_and_val(self, tmp_path):
        # Write a trajectories.jsonl
        traj = _traj(
            [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "Reasoning steps:\n- a"},
                _assistant_tool_call("finalanswertool", '{"answer": "A"}'),
                _tool_result(content="A"),
            ],
            tools=[{"type": "function", "function": {"name": "finalanswertool"}}],
        )
        with open(tmp_path / "trajectories.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")
        # Also a non-trajectory line that must be ignored
        f = open(tmp_path / "trajectories.jsonl", "a", encoding="utf-8")
        f.write(json.dumps({"record_type": "llm_call"}, ensure_ascii=False) + "\n")
        f.close()

        out_dir = tmp_path / "out"
        counts = convert(
            str(tmp_path),
            str(out_dir),
            final_answer_as="text",
            cot_source="sgr_reasoning",
            val_ratio=0.0,
        )
        assert counts == {"train": 1, "val": 0}
        records = _read(out_dir / "train.jsonl")
        assert len(records) == 1
        # final answer collapsed to text
        assert records[0]["messages"][-1] == {"role": "assistant", "content": "A"}
        # reasoning kept as text
        assert any(m["content"].startswith("Reasoning steps:") for m in records[0]["messages"])
        assert records[0]["tools"][0]["function"]["name"] == "finalanswertool"

    def test_convert_with_val_split(self, tmp_path):
        with open(tmp_path / "trajectories.jsonl", "w", encoding="utf-8") as f:
            for i in range(5):
                f.write(json.dumps(_traj([{"role": "user", "content": str(i)}]), ensure_ascii=False) + "\n")
        out_dir = tmp_path / "out"
        counts = convert(str(tmp_path), str(out_dir), val_ratio=0.2, seed=0)
        assert counts["train"] == 4
        assert counts["val"] == 1
        assert (out_dir / "val.jsonl").exists()

    def test_convert_role_filter(self, tmp_path):
        with open(tmp_path / "trajectories.jsonl", "w", encoding="utf-8") as f:
            for role in ("factual_qa", "coder", "creative_writer"):
                traj = _traj([{"role": "user", "content": "q"}])
                traj["role"] = role
                f.write(json.dumps(traj, ensure_ascii=False) + "\n")
        out_dir = tmp_path / "out"
        counts = convert(str(tmp_path), str(out_dir), roles=["coder"])
        assert counts == {"train": 1, "val": 0}

    def test_convert_split_by_role(self, tmp_path):
        with open(tmp_path / "trajectories.jsonl", "w", encoding="utf-8") as f:
            for role in ("factual_qa", "factual_qa", "coder"):
                traj = _traj([{"role": "user", "content": "q"}])
                traj["role"] = role
                f.write(json.dumps(traj, ensure_ascii=False) + "\n")
        out_dir = tmp_path / "out"
        counts = convert(str(tmp_path), str(out_dir), split_by_role=True)
        assert counts == {"coder": {"train": 1, "val": 0}, "factual_qa": {"train": 2, "val": 0}}
        assert (out_dir / "train_factual_qa.jsonl").exists()
        assert (out_dir / "train_coder.jsonl").exists()

    def test_convert_language_filter(self, tmp_path):
        with open(tmp_path / "trajectories.jsonl", "w", encoding="utf-8") as f:
            for lang in ("en", "ru", "en"):
                traj = _traj([{"role": "user", "content": "q"}])
                traj["language"] = lang
                f.write(json.dumps(traj, ensure_ascii=False) + "\n")
        out_dir = tmp_path / "out"
        counts = convert(str(tmp_path), str(out_dir), languages=["ru"])
        assert counts == {"train": 1, "val": 0}
