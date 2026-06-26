"""Tests for full agent trajectory recording (Level B)."""

import json
from unittest.mock import Mock

import pytest
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion import Choice

from sgr_agent_core.agent_definition import (
    AgentConfig,
    DatasetRecordingConfig,
    ExecutionConfig,
    LLMConfig,
    PromptsConfig,
)
from sgr_agent_core.agents.sgr_tool_calling_agent import SGRToolCallingAgent
from sgr_agent_core.base_tool import BaseTool
from sgr_agent_core.models import AgentStatesEnum
from sgr_agent_core.services.dataset_recorder import reset_recorder
from sgr_agent_core.tools import FinalAnswerTool, ReasoningTool


class _Stream:
    def __init__(self, completion):
        self._completion = completion

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def get_final_completion(self):
        return self._completion


def _tool_call(tool: BaseTool, call_id: str) -> ChatCompletionMessageToolCall:
    tc = Mock(spec=ChatCompletionMessageToolCall)
    tc.id = call_id
    tc.type = "function"
    tc.function = Mock()
    tc.function.name = tool.tool_name
    tc.function.parsed_arguments = tool
    return tc


def _completion(tool_call) -> ChatCompletion:
    message = ChatCompletionMessage(role="assistant", content=None, tool_calls=[tool_call])
    return ChatCompletion(
        id="c1",
        choices=[Choice(index=0, message=message, finish_reason="tool_calls")],
        created=1,
        model="glm-5.2",
        object="chat.completion",
    )


class NonFinishingTool(BaseTool):
    """An action tool that returns a result without finishing the agent."""

    note: str = "working"

    async def __call__(self, context, config, **kwargs) -> str:  # type: ignore[override]
        return "intermediate result"


def _mock_client():
    """Returns a mock client that yields reasoning then action tools across turns."""
    client = Mock(spec=AsyncOpenAI)
    reasoning = ReasoningTool(
        reasoning_steps=["Analyze task", "Decide next step"],
        current_situation="Starting research",
        plan_status="On track",
        enough_data=False,
        remaining_steps=["Search", "Finalize"],
        task_completed=False,
    )
    final = FinalAnswerTool(
        reasoning="done",
        completed_steps=["s1"],
        answer="The final answer for the child safety question.",
        status=AgentStatesEnum.COMPLETED,
    )
    action_counter = {"n": 0}

    def mock_stream(**kwargs):
        tools_param = kwargs.get("tools", [])
        tool_name = None
        if tools_param and isinstance(tools_param[0], dict):
            tool_name = tools_param[0].get("function", {}).get("name")
        if tool_name == ReasoningTool.tool_name:
            return _Stream(_completion(_tool_call(reasoning, "r-call")))
        action_counter["n"] += 1
        tool = NonFinishingTool(note="wip") if action_counter["n"] == 1 else final
        return _Stream(_completion(_tool_call(tool, "a-call")))

    client.chat.completions.stream = Mock(side_effect=mock_stream)
    return client


def _read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture(autouse=True)
def _reset():
    reset_recorder()
    yield
    reset_recorder()


class TestTrajectoryRecording:
    @pytest.mark.asyncio
    async def test_trajectory_recorded_after_execution(self, tmp_path):
        cfg = DatasetRecordingConfig(
            enabled=True, output_dir=str(tmp_path), modes=["trajectory", "raw"], include_reasoning=True
        )
        agent_config = AgentConfig(
            llm=LLMConfig(api_key="k", model="glm-5.2"),
            prompts=PromptsConfig(
                system_prompt_str="You are a pediatric advisor.",
                initial_user_request_str="Current Date: 2026-01-01",
                clarification_response_str="c",
            ),
            execution=ExecutionConfig(max_iterations=10, max_clarifications=3),
            dataset=cfg,
            role="pediatric_advisor",
        )
        agent = SGRToolCallingAgent(
            task_messages=[{"role": "user", "content": "How to keep kids safe during sports?"}],
            openai_client=_mock_client(),
            agent_config=agent_config,
            toolkit=[FinalAnswerTool, NonFinishingTool],
            def_name="advisor",
        )
        agent.language = "ru"

        # Consume the SSE stream in the background so the agent doesn't block.
        async def consume():
            async for _ in agent.streaming_generator.stream():
                pass

        import asyncio

        consumer = asyncio.create_task(consume())
        await agent.execute()
        consumer.cancel()
        try:
            await consumer
        except asyncio.CancelledError:
            pass

        traj_path = str(tmp_path / "trajectories.jsonl")
        trajectories = _read(traj_path)
        assert len(trajectories) == 1
        traj = trajectories[0]
        assert traj["record_type"] == "trajectory"
        assert traj["role"] == "pediatric_advisor"
        assert traj["language"] == "ru"
        assert traj["teacher_model"] == "glm-5.2"
        assert traj["metadata"]["finish_state"] == "completed"
        assert traj["metadata"]["iterations"] >= 2

        # Tools are the toolkit function schemas
        tool_names = {t["function"]["name"] for t in traj["tools"]}
        assert "finalanswertool" in tool_names
        assert "nonfinishingtool" in tool_names

        # Messages start with system + user query
        msgs = traj["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are a pediatric advisor."
        assert any(m["role"] == "user" and "sports" in m["content"] for m in msgs)

        # Reasoning tool calls were transformed into assistant text (no reasoningtool tool_calls)
        assert not any(
            m.get("role") == "assistant"
            and m.get("tool_calls")
            and m["tool_calls"][0]["function"]["name"] == "reasoningtool"
            for m in msgs
        ), "Reasoning tool calls should have been transformed into text"
        # ...and the transformed CoT text is present
        assert any(
            m.get("role") == "assistant" and isinstance(m.get("content"), str) and "Reasoning steps" in m["content"]
            for m in msgs
        )

        # The final answer is present (either as a tool_call answer or tool result content)
        flat = json.dumps(msgs, ensure_ascii=False)
        assert "The final answer for the child safety question." in flat

    @pytest.mark.asyncio
    async def test_trajectory_skipped_when_mode_disabled(self, tmp_path):
        cfg = DatasetRecordingConfig(enabled=True, output_dir=str(tmp_path), modes=["raw"])
        agent_config = AgentConfig(
            llm=LLMConfig(api_key="k", model="glm-5.2"),
            prompts=PromptsConfig(system_prompt_str="s", initial_user_request_str="i", clarification_response_str="c"),
            execution=ExecutionConfig(max_iterations=10, max_clarifications=3),
            dataset=cfg,
        )
        agent = SGRToolCallingAgent(
            task_messages=[{"role": "user", "content": "q"}],
            openai_client=_mock_client(),
            agent_config=agent_config,
            toolkit=[FinalAnswerTool, NonFinishingTool],
        )
        import asyncio

        async def consume():
            async for _ in agent.streaming_generator.stream():
                pass

        consumer = asyncio.create_task(consume())
        await agent.execute()
        consumer.cancel()
        try:
            await consumer
        except asyncio.CancelledError:
            pass

        import os

        assert not os.path.exists(str(tmp_path / "trajectories.jsonl"))

    @pytest.mark.asyncio
    async def test_trajectory_skipped_when_not_completed(self, tmp_path):
        """Failed runs (e.g. a teacher API error) must not pollute the dataset."""
        import os

        from sgr_agent_core.models import AgentStatesEnum

        cfg = DatasetRecordingConfig(enabled=True, output_dir=str(tmp_path), modes=["trajectory"])
        agent_config = AgentConfig(
            llm=LLMConfig(api_key="k", model="glm-5.2"),
            prompts=PromptsConfig(system_prompt_str="s", initial_user_request_str="i", clarification_response_str="c"),
            execution=ExecutionConfig(max_iterations=10, max_clarifications=3),
            dataset=cfg,
        )
        agent = SGRToolCallingAgent(
            task_messages=[{"role": "user", "content": "q"}],
            openai_client=_mock_client(),
            agent_config=agent_config,
            toolkit=[FinalAnswerTool, NonFinishingTool],
        )
        assert agent.recorder is not None  # recording is active

        # Simulate a failed run (e.g. insufficient-balance API error).
        agent._context.state = AgentStatesEnum.FAILED
        await agent._record_trajectory()

        assert not os.path.exists(str(tmp_path / "trajectories.jsonl"))
