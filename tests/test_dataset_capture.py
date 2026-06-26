"""Tests for BaseAgent._llm_call dataset capture."""

import json
from typing import Any
from unittest.mock import Mock

import pytest
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageFunctionToolCall
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import Function

from sgr_agent_core.agent_definition import (
    AgentConfig,
    DatasetRecordingConfig,
    ExecutionConfig,
    LLMConfig,
    PromptsConfig,
)
from sgr_agent_core.base_agent import BaseAgent
from sgr_agent_core.services.dataset_recorder import reset_recorder
from sgr_agent_core.tools import ReasoningTool


class _Stream:
    """Minimal mock of the OpenAI async stream context manager."""

    def __init__(self, completion: ChatCompletion):
        self._completion = completion

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def get_final_completion(self) -> ChatCompletion:
        return self._completion


def _make_tool_call(
    name: str = "web_search_tool", arguments: str = '{"query": "kids safety"}', call_id: str = "call-1"
) -> ChatCompletionMessageFunctionToolCall:
    return ChatCompletionMessageFunctionToolCall(
        id=call_id, type="function", function=Function(name=name, arguments=arguments)
    )


def _make_completion(
    *,
    content: str | None = "answer",
    reasoning_content: str | None = None,
    tool_calls: list[Any] | None = None,
    finish_reason: str = "stop",
    model: str = "glm-5.2",
) -> ChatCompletion:
    msg = ChatCompletionMessage(role="assistant", content=content, tool_calls=tool_calls)
    if reasoning_content is not None:
        setattr(msg, "reasoning_content", reasoning_content)
    completion = ChatCompletion(
        id="comp-1",
        choices=[Choice(index=0, message=msg, finish_reason=finish_reason)],
        created=1,
        model=model,
        object="chat.completion",
    )
    # Attach a simple usage object via attribute (defensive serializer uses getattr).
    usage = Mock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    usage.total_tokens = 15
    completion.usage = usage
    return completion


def _make_client(completion: ChatCompletion) -> AsyncOpenAI:
    client = Mock(spec=AsyncOpenAI)
    client.chat.completions.stream = Mock(return_value=_Stream(completion))
    return client


def _make_agent(client: AsyncOpenAI, dataset_cfg: DatasetRecordingConfig | None = None) -> BaseAgent:
    agent_config = AgentConfig(
        llm=LLMConfig(api_key="k", model="glm-5.2"),
        prompts=PromptsConfig(system_prompt_str="s", initial_user_request_str="i", clarification_response_str="c"),
        execution=ExecutionConfig(),
        dataset=dataset_cfg or DatasetRecordingConfig(),
        role="pediatric_advisor",
    )
    return BaseAgent(
        task_messages=[{"role": "user", "content": "hi"}],
        openai_client=client,
        agent_config=agent_config,
        toolkit=[ReasoningTool],
        def_name="my_agent",
    )


def _read(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class TestLLMCallCapture:
    """Tests that _llm_call records raw request/response when recording is on."""

    def teardown_method(self):
        reset_recorder()

    @pytest.mark.asyncio
    async def test_no_recording_when_disabled(self, tmp_path):
        """No recorder is created and no file is written when disabled."""
        reset_recorder()
        agent = _make_agent(_make_client(_make_completion(content="hi")))
        assert agent.recorder is None
        await agent._llm_call("action", messages=[{"role": "user", "content": "x"}], model="glm-5.2")
        assert not tmp_path.exists() or not list(tmp_path.glob("*.jsonl"))

    @pytest.mark.asyncio
    async def test_records_call_with_messages_and_response(self, tmp_path):
        """A raw record is written capturing messages, content, and teacher model."""
        cfg = DatasetRecordingConfig(enabled=True, output_dir=str(tmp_path), modes=["raw"])
        agent = _make_agent(_make_client(_make_completion(content="final answer")), dataset_cfg=cfg)
        # Recorder is created lazily in __init__ when enabled.
        assert agent.recorder is not None
        await agent._llm_call(
            "action",
            messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "q"}],
            model="glm-5.2",
            temperature=0.4,
        )

        records = _read(str(tmp_path / "llm_calls.jsonl"))
        assert len(records) == 1
        rec = records[0]
        assert rec["record_type"] == "llm_call"
        assert rec["phase"] == "action"
        assert rec["role"] == "pediatric_advisor"
        assert rec["teacher_model"] == "glm-5.2"
        assert rec["request"]["messages"][1]["content"] == "q"
        assert rec["request"]["params"]["model"] == "glm-5.2"
        assert rec["response"]["content"] == "final answer"
        assert rec["response"]["usage"]["total_tokens"] == 15
        assert rec["latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_records_tool_calls_and_reasoning_content(self, tmp_path):
        """tool_calls and reasoning_content are captured from the response."""
        cfg = DatasetRecordingConfig(enabled=True, output_dir=str(tmp_path), modes=["raw"])
        tc = _make_tool_call()
        agent = _make_agent(
            _make_client(
                _make_completion(
                    content=None, tool_calls=[tc], reasoning_content="thinking...", finish_reason="tool_calls"
                )
            ),
            dataset_cfg=cfg,
        )
        await agent._llm_call(
            "action",
            messages=[{"role": "user", "content": "q"}],
            tools=[{"type": "function", "function": {"name": "web_search_tool"}}],
            tool_choice="required",
        )

        rec = _read(str(tmp_path / "llm_calls.jsonl"))[0]
        assert rec["response"]["reasoning_content"] == "thinking..."
        assert rec["response"]["finish_reason"] == "tool_calls"
        assert rec["response"]["tool_calls"][0]["function"]["name"] == "web_search_tool"
        assert rec["response"]["tool_calls"][0]["function"]["arguments"] == '{"query": "kids safety"}'
        assert rec["request"]["tool_choice"] == "required"
        assert rec["request"]["tools"][0]["function"]["name"] == "web_search_tool"

    @pytest.mark.asyncio
    async def test_records_response_format_schema(self, tmp_path):
        """response_format pydantic class is serialized as json_schema."""
        cfg = DatasetRecordingConfig(enabled=True, output_dir=str(tmp_path), modes=["raw"])
        agent = _make_agent(_make_client(_make_completion()), dataset_cfg=cfg)
        await agent._llm_call("reasoning", messages=[{"role": "user", "content": "q"}], response_format=ReasoningTool)
        rec = _read(str(tmp_path / "llm_calls.jsonl"))[0]
        rf = rec["request"]["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "ReasoningTool"
        assert "properties" in rf["json_schema"]["schema"]
