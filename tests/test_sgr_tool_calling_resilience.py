"""Resilience tests for ``SGRToolCallingAgent`` recovery from malformed /
missing streaming tool-call responses.

Reproduces two real production crashes observed during dataset generation with
the GLM-5.2 teacher, where ``temperature=0.4`` makes a retry effective but the
model occasionally:

  1. returns no tool-call (``tool_calls is None``) despite
     ``tool_choice="required"`` -> ``TypeError: 'NoneType' object is not
     subscriptable`` in ``_reasoning_phase``;
  2. after the retry budget is exhausted on a stuck input, a synthetic
     reasoning fallback must keep the run alive instead of crashing.

The streaming OpenAI client is mocked; no network access is performed.
"""

from typing import Any
from unittest.mock import Mock

import pytest
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion import Choice
from pydantic import ValidationError

from sgr_agent_core.agent_definition import AgentConfig, ExecutionConfig, LLMConfig, PromptsConfig
from sgr_agent_core.agents.sgr_tool_calling_agent import SGRToolCallingAgent
from sgr_agent_core.models import AgentStatesEnum
from sgr_agent_core.tools import FinalAnswerTool, ReasoningTool

pytestmark = pytest.mark.unit


class _Stream:
    """Minimal mock of the OpenAI streaming async context manager.

    Emulates ``chat.completions.stream(...)``: async context manager + async
    iterator (optionally raising ``error`` on first iteration to simulate a
    streamed tool-argument validation failure) + a final completion carrying
    either a parsed tool-call or ``tool_calls=None``.
    """

    def __init__(self, tool: Any | None = None, error: BaseException | None = None) -> None:
        self._tool = tool
        self._error = error

    async def __aenter__(self) -> "_Stream":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    def __aiter__(self) -> "_Stream":
        return self

    async def __anext__(self):
        if self._error is not None:
            err = self._error
            self._error = None
            raise err
        raise StopAsyncIteration

    async def get_final_completion(self) -> ChatCompletion:
        tool_calls = None
        if self._tool is not None:
            tc = Mock(spec=ChatCompletionMessageToolCall)
            tc.id = "call-1"
            tc.type = "function"
            tc.function = Mock()
            tc.function.name = self._tool.tool_name
            tc.function.parsed_arguments = self._tool
            tool_calls = [tc]
        message = ChatCompletionMessage(role="assistant", content=None, tool_calls=tool_calls)
        return ChatCompletion(
            id="c1",
            choices=[Choice(index=0, message=message, finish_reason="stop")],
            created=1,
            model="glm-5.2",
            object="chat.completion",
        )


def _make_client(reasoning_responses: list, action_responses: list) -> AsyncOpenAI:
    """Build a mock ``AsyncOpenAI`` dispatching by the first tool name.

    Each element of ``reasoning_responses`` / ``action_responses`` is either a
    tool instance (valid tool-call), ``None`` (model returned no tool-call), or
    an ``Exception`` instance (raised while streaming, e.g. a validation error).
    """

    client = Mock(spec=AsyncOpenAI)
    state = {"r": 0, "a": 0}

    def mock_stream(**kwargs):
        tools = kwargs.get("tools") or []
        first = tools[0] if tools else {}
        name = first.get("function", {}).get("name") if isinstance(first, dict) else None
        if name == ReasoningTool.tool_name:
            seq, key = reasoning_responses, "r"
        else:
            seq, key = action_responses, "a"
        idx = state[key]
        entry = seq[idx] if idx < len(seq) else None
        state[key] += 1
        if isinstance(entry, BaseException):
            return _Stream(error=entry)
        return _Stream(tool=entry)

    client.chat.completions.stream = Mock(side_effect=mock_stream)
    return client


def _reasoning(**overrides) -> ReasoningTool:
    base = dict(
        reasoning_steps=["Analyze", "Proceed"],
        current_situation="Proceeding",
        plan_status="Finalize",
        enough_data=True,
        remaining_steps=["Finalize"],
        task_completed=True,
    )
    base.update(overrides)
    return ReasoningTool(**base)


def _final_answer() -> FinalAnswerTool:
    return FinalAnswerTool(
        reasoning="done",
        completed_steps=["step1"],
        answer="Final answer",
        status=AgentStatesEnum.COMPLETED,
    )


def _config() -> AgentConfig:
    return AgentConfig(
        llm=LLMConfig(api_key="k", base_url="https://x", model="glm-5.2"),
        prompts=PromptsConfig(
            system_prompt_str="sys",
            initial_user_request_str="init",
            clarification_response_str="clr",
        ),
        execution=ExecutionConfig(max_iterations=5, max_clarifications=3, max_searches=3),
    )


def _build_agent(client: AsyncOpenAI) -> SGRToolCallingAgent:
    return SGRToolCallingAgent(
        task_messages=[{"role": "user", "content": "do it"}],
        openai_client=client,
        agent_config=_config(),
        toolkit=[FinalAnswerTool],
    )


@pytest.mark.asyncio
async def test_reasoning_recovers_on_retry_when_no_tool_call():
    """tool_calls=None once, then a valid reasoning tool on retry: the agent
    must complete (currently raises TypeError and fails)."""
    client = _make_client(reasoning_responses=[None, _reasoning()], action_responses=[_final_answer()])
    agent = _build_agent(client)

    await agent.execute()

    assert agent._context.state == AgentStatesEnum.COMPLETED
    # First reasoning attempt failed (None), second succeeded -> >= 2 stream calls.
    assert client.chat.completions.stream.call_count >= 2
    assert not getattr(agent, "_used_fallback", False)


@pytest.mark.asyncio
async def test_reasoning_fallback_after_retries_exhausted():
    """Every reasoning attempt returns no tool-call: after the retry budget is
    exhausted a synthetic reasoning fallback must keep the run alive (currently
    raises TypeError and fails), and the trajectory must be flagged."""
    client = _make_client(reasoning_responses=[None] * 10, action_responses=[_final_answer()])
    agent = _build_agent(client)

    await agent.execute()

    assert agent._context.state == AgentStatesEnum.COMPLETED
    assert getattr(agent, "_used_fallback", False) is True


def _validation_error() -> ValidationError:
    """Build a real pydantic ValidationError (the kind the OpenAI SDK raises
    while aggregating a streamed tool-call)."""
    try:
        ReasoningTool.model_validate({"reasoning_steps": "not-a-list"})
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a ValidationError")


@pytest.mark.asyncio
async def test_reasoning_recovers_when_stream_raises_validation_error():
    """A streamed tool-argument ValidationError (the bug #1 scenario) must be
    retried and recovered from, not crash the whole agent run."""
    client = _make_client(
        reasoning_responses=[_validation_error(), _reasoning()],
        action_responses=[_final_answer()],
    )
    agent = _build_agent(client)

    await agent.execute()

    assert agent._context.state == AgentStatesEnum.COMPLETED
    assert client.chat.completions.stream.call_count >= 2
    assert not getattr(agent, "_used_fallback", False)


@pytest.mark.asyncio
async def test_action_fallback_when_no_tool_call():
    """Reasoning succeeds but the action phase never returns a tool-call: the
    FinalAnswerTool fallback must complete the run (not crash) and flag the
    trajectory as degraded."""
    client = _make_client(reasoning_responses=[_reasoning()], action_responses=[None] * 10)
    agent = _build_agent(client)

    await agent.execute()

    assert agent._context.state == AgentStatesEnum.COMPLETED
    assert getattr(agent, "_used_fallback", False) is True
