import asyncio
import json
import logging
import os
import time
import traceback
import uuid
from datetime import datetime
from typing import Any, Type

from openai import AsyncOpenAI, pydantic_function_tool
from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessageParam
from pydantic import BaseModel

from sgr_agent_core.agent_definition import AgentConfig, ToolDefinition
from sgr_agent_core.models import AgentContext, AgentStatesEnum
from sgr_agent_core.services.dataset_recorder import DatasetRecorder, get_recorder, set_recorder
from sgr_agent_core.services.prompt_loader import PromptLoader
from sgr_agent_core.services.registry import AgentRegistry
from sgr_agent_core.stream import BaseStreamingGenerator, OpenAIStreamingGenerator
from sgr_agent_core.tools import (
    BaseTool,
    ClarificationTool,
    ReasoningTool,
)


class AgentRegistryMixin:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__name__ not in ("BaseAgent",):
            AgentRegistry.register(cls, name=cls.name)


def _json_safe(obj: Any) -> Any:
    """Best-effort conversion of ``obj`` into a JSON-serializable structure."""
    try:
        return json.loads(json.dumps(obj, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return str(obj)


def _serialize_response_format(response_format: Any) -> Any:
    """Serialize a ``response_format`` argument into a JSON-safe description.

    Pydantic model classes (used by SGRAgent structured output) are described via
    their JSON schema; everything else is best-effort serialized.
    """
    if response_format is None:
        return None
    if isinstance(response_format, type) and hasattr(response_format, "model_json_schema"):
        return {
            "type": "json_schema",
            "json_schema": {"name": response_format.__name__, "schema": response_format.model_json_schema()},
        }
    return _json_safe(response_format)


def _serialize_llm_request(openai_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Serialize the request kwargs of an LLM call into a JSON-safe record."""
    special = {"messages", "tools", "tool_choice", "response_format"}
    return {
        "messages": _json_safe(openai_kwargs["messages"]) if openai_kwargs.get("messages") is not None else [],
        "tools": _json_safe(openai_kwargs["tools"]) if openai_kwargs.get("tools") is not None else None,
        "tool_choice": openai_kwargs.get("tool_choice"),
        "response_format": _serialize_response_format(openai_kwargs.get("response_format")),
        "params": {k: _json_safe(v) for k, v in openai_kwargs.items() if k not in special},
    }


def _serialize_tool_call(tool_call: Any) -> dict[str, Any]:
    """Serialize a single tool call into a JSON-safe dict (defensive)."""
    function = getattr(tool_call, "function", None)
    return {
        "id": getattr(tool_call, "id", None),
        "type": getattr(tool_call, "type", "function"),
        "function": {
            "name": getattr(function, "name", None),
            "arguments": getattr(function, "arguments", None),
        },
    }


def _serialize_llm_response(completion: Any) -> dict[str, Any]:
    """Serialize a ChatCompletion into a JSON-safe response record."""
    try:
        choice = completion.choices[0]
        message = choice.message
    except (AttributeError, IndexError, TypeError):
        return {"raw": _json_safe(completion)}
    tool_calls = getattr(message, "tool_calls", None) or []
    usage = getattr(completion, "usage", None)
    usage_dict = None
    if usage is not None:
        usage_dict = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
    return {
        "content": getattr(message, "content", None),
        "reasoning_content": getattr(message, "reasoning_content", None),
        "tool_calls": [_serialize_tool_call(tc) for tc in tool_calls],
        "finish_reason": getattr(choice, "finish_reason", None),
        "model": getattr(completion, "model", None),
        "usage": usage_dict,
    }


def _render_reasoning_cot(arguments: str) -> str:
    """Render a ReasoningTool arguments JSON string as a readable CoT block."""
    try:
        data = json.loads(arguments) if arguments else {}
    except (TypeError, ValueError):
        return arguments or ""
    lines: list[str] = []
    steps = data.get("reasoning_steps")
    if steps:
        lines.append("Reasoning steps:")
        lines.extend(f"- {s}" for s in steps)
    for key in ("current_situation", "plan_status"):
        value = data.get(key)
        if value:
            label = key.replace("_", " ").capitalize()
            lines.append(f"{label}: {value}")
    remaining = data.get("remaining_steps")
    if remaining:
        lines.append("Remaining steps:")
        lines.extend(f"- {s}" for s in remaining)
    return "\n".join(lines) if lines else (arguments or "")


def _transform_trajectory_messages(
    messages: list[dict[str, Any]],
    reasoning_tool_name: str | None,
    include_reasoning: bool,
) -> list[dict[str, Any]]:
    """Clean and transform the conversation into a sharegpt-style message list.

    - When ``include_reasoning`` is True, the SGR reasoning tool call and its
      (empty) tool result are replaced by a single assistant message containing
      the rendered chain-of-thought text.
    """
    if not include_reasoning or not reasoning_tool_name:
        return messages
    result: list[dict[str, Any]] = []
    skip_tool_ids: set[str] = set()
    for message in messages:
        role = message.get("role")
        tool_calls = message.get("tool_calls")
        if role == "assistant" and tool_calls and len(tool_calls) == 1:
            function = tool_calls[0].get("function") or {}
            if function.get("name") == reasoning_tool_name:
                cot = _render_reasoning_cot(function.get("arguments", ""))
                result.append({"role": "assistant", "content": cot})
                call_id = tool_calls[0].get("id")
                if call_id is not None:
                    skip_tool_ids.add(call_id)
                continue
        if role == "tool" and message.get("tool_call_id") in skip_tool_ids:
            continue
        result.append(message)
    return result


class BaseAgent(AgentRegistryMixin):
    """Base class for agents."""

    name: str = "base_agent"

    def __init__(
        self,
        task_messages: list[ChatCompletionMessageParam],
        openai_client: AsyncOpenAI,
        agent_config: AgentConfig,
        toolkit: list[Type[BaseTool]],
        def_name: str | None = None,
        streaming_generator: type[BaseStreamingGenerator] = OpenAIStreamingGenerator,
        tool_configs: dict[str, ToolDefinition] | None = None,
        **kwargs: dict,
    ):
        self.id = f"{def_name or self.name}_{uuid.uuid4()}"
        self.streaming_generator = streaming_generator(agent_id=self.id)

        self.openai_client = openai_client
        self.config = agent_config
        self.creation_time = datetime.now()
        self.task_messages = task_messages
        self.toolkit = toolkit
        self.tool_configs = tool_configs or {}

        self._context = AgentContext()
        self.conversation = []
        self.logger = logging.getLogger(f"sgr_agent_core.agents.{self.id}")
        self.log = []

        self._execute_task: asyncio.Task | None = None

        self.def_name = def_name
        self.role = self._resolve_role(agent_config, def_name)
        self.language: str | None = None
        self.recorder = self._get_or_create_recorder()

    def _resolve_role(self, agent_config: AgentConfig, def_name: str | None) -> str:
        """Resolve the role tag: explicit config role, else def name, else class name."""
        role = getattr(agent_config, "role", None)
        if role:
            return role
        return def_name or self.name

    def _get_or_create_recorder(self) -> DatasetRecorder | None:
        """Return the shared dataset recorder, creating it if recording is enabled.

        Recording is enabled when the agent's ``dataset`` config has ``enabled=True``.
        A single shared recorder is reused across agents (module-level singleton).
        """
        cfg = getattr(self.config, "dataset", None)
        if cfg is None or not getattr(cfg, "enabled", False):
            return None
        existing = get_recorder()
        if existing is not None:
            return existing
        teacher = getattr(cfg, "teacher_model", None) or self.config.llm.model
        recorder = DatasetRecorder(cfg, teacher_model=teacher)
        set_recorder(recorder)
        return recorder

    async def _llm_call(self, phase: str, **openai_kwargs: Any) -> Any:
        """Execute a streaming chat completion and (optionally) record it.

        Centralizes the streaming consumption pattern shared by all agent phases
        (forwarding chunks to the streaming generator and aggregating the final
        completion) and captures the raw request/response for the dataset recorder
        when one is active.

        Args:
            phase: Phase label used for the stream phase id and dataset tagging
                (e.g. ``"reasoning"``, ``"action"``, ``"generate"``).
            **openai_kwargs: Keyword arguments forwarded verbatim to
                ``openai_client.chat.completions.stream`` (messages, tools,
                response_format, tool_choice, model params, ...).

        Returns:
            The aggregated :class:`~openai.types.chat.ChatCompletion`.
        """
        phase_id = f"{self._context.iteration}-{phase}"
        request_serialized = _serialize_llm_request(openai_kwargs) if self.recorder is not None else None
        started = time.monotonic()
        async with self.openai_client.chat.completions.stream(**openai_kwargs) as stream:
            async for event in stream:
                if event.type == "chunk":
                    self.streaming_generator.add_chunk(event.chunk, phase_id)
            completion = await stream.get_final_completion()
        if self.recorder is not None and request_serialized is not None:
            try:
                await self.recorder.record_call(
                    {
                        "record_type": "llm_call",
                        "call_id": str(uuid.uuid4()),
                        "timestamp": datetime.now().isoformat(),
                        "agent_id": self.id,
                        "agent_class": type(self).__name__,
                        "role": self.role,
                        "language": self.language,
                        "phase": phase,
                        "iteration": self._context.iteration,
                        "request": request_serialized,
                        "response": _serialize_llm_response(completion),
                        "teacher_model": self.recorder.teacher_model,
                        "latency_ms": int((time.monotonic() - started) * 1000),
                    }
                )
            except Exception as e:  # noqa: BLE001 - recording must never break execution
                self.logger.warning(f"Failed to record LLM call for dataset: {e}")
        return completion

    def get_tool_config(self, tool_class: Type[BaseTool]) -> BaseModel | dict[str, Any]:
        """Return resolved config for a tool as a Pydantic model or raw dict.

        If the tool defines config_model, builds and returns a validated
        instance from tool_configs. Otherwise returns the raw dict.
        """
        raw = self.tool_configs.get(tool_class.tool_name, {})
        config_model = getattr(tool_class, "config_model", None)
        if config_model is None:
            return raw
        return config_model(**raw)

    async def provide_clarification(
        self,
        messages: list[ChatCompletionMessageParam],
        replace_conversation: bool = False,
    ) -> None:
        """Receive clarification from an external source in OpenAI messages
        format.

        Args:
            messages: Clarification messages in OpenAI ChatCompletionMessageParam format.
            replace_conversation: When True, clear the conversation
                before applying messages (continuing stateful conversation / stateless mode).
                Use this for stateless clients that re-send the full history on every turn.
        """
        if replace_conversation:
            self.conversation = []
        self.conversation.extend(messages)
        self.conversation.append(
            {"role": "user", "content": PromptLoader.get_clarification_template(messages, self.config.prompts)}
        )

        self._context.clarifications_used += 1
        self._context.clarification_received.set()
        self._context.state = AgentStatesEnum.RESEARCHING
        self.logger.info(f"✅ Clarification received: {len(messages)} messages")

    def _log_reasoning(self, result: ReasoningTool) -> None:
        next_step = result.remaining_steps[0] if result.remaining_steps else "Completing"
        self.logger.info(
            f"""
    ###############################################
    🤖 LLM RESPONSE DEBUG:
       🧠 Reasoning Steps: {result.reasoning_steps}
       📊 Current Situation: '{result.current_situation[:400]}...'
       📋 Plan Status: '{result.plan_status[:400]}...'
       🔍 Searches Done: {self._context.searches_used}
       🔍 Clarifications Done: {self._context.clarifications_used}
       ✅ Enough Data: {result.enough_data}
       📝 Remaining Steps: {result.remaining_steps}
       🏁 Task Completed: {result.task_completed}
       ➡️ Next Step: {next_step}
    ###############################################"""
        )
        self.log.append(
            {
                "step_number": self._context.iteration,
                "timestamp": datetime.now().isoformat(),
                "step_type": "reasoning",
                "agent_reasoning": result.model_dump(mode="json"),
            }
        )

    def _log_tool_execution(self, tool: BaseTool, result: str):
        self.logger.info(
            f"""
###############################################
🛠️ TOOL EXECUTION DEBUG:
    🔧 Tool Name: {tool.tool_name}
    📋 Tool Model: {tool.model_dump_json(indent=2)}
    🔍 Result: '{result[:400]}...'
###############################################"""
        )
        self.log.append(
            {
                "step_number": self._context.iteration,
                "timestamp": datetime.now().isoformat(),
                "step_type": "tool_execution",
                "tool_name": tool.tool_name,
                "agent_tool_context": tool.model_dump(mode="json"),
                "agent_tool_execution_result": result,
            }
        )

    def _save_agent_log(self):
        from sgr_agent_core.agent_config import GlobalConfig

        logs_dir = GlobalConfig().execution.logs_dir
        # Skip saving if logs_dir is None or empty string
        if not logs_dir:
            self.logger.debug("Skipping agent log save: logs_dir is not configured")
            return

        os.makedirs(logs_dir, exist_ok=True)
        filepath = os.path.join(logs_dir, f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{self.id}-log.json")
        agent_log = {
            "id": self.id,
            "model_config": self.config.llm.model_dump(
                exclude={"api_key", "proxy"}, mode="json"
            ),  # Sensitive data excluded by default
            "task_messages": self.task_messages,
            "toolkit": [tool.tool_name for tool in self.toolkit],
            "log": self.log,
        }

        json.dump(agent_log, open(filepath, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    async def _record_trajectory(self) -> None:
        """Assemble and record a full agent trajectory (Level B record).

        Builds a sharegpt-style message list from the final conversation plus the
        toolkit function schemas, applies the reasoning transformation, and writes
        one ``trajectory`` record to ``trajectories.jsonl``. No-op unless recording
        is enabled with the ``trajectory`` mode.
        """
        if self.recorder is None or "trajectory" not in self.recorder.config.modes:
            return
        # Only keep successfully completed runs; FAILED/ERROR/CANCELLED (e.g. a
        # teacher API error such as insufficient balance) would produce a
        # degenerate trajectory without a final answer and pollute the dataset.
        if self._context.state != AgentStatesEnum.COMPLETED:
            self.logger.debug(f"Skipping trajectory record: agent state is {self._context.state} (not COMPLETED)")
            return
        try:
            messages = await self._prepare_context()
            # Drop synthetic system messages (e.g. "Agent {id} started"); the real
            # system prompt is always the first message.
            cleaned = [m for i, m in enumerate(messages) if not (m.get("role") == "system" and i != 0)]
            reasoning_tool_name: str | None = None
            reasoning_tool_cls = getattr(self, "ReasoningTool", None)
            if reasoning_tool_cls is not None:
                reasoning_tool_name = getattr(reasoning_tool_cls, "tool_name", None)
            cleaned = _transform_trajectory_messages(
                cleaned, reasoning_tool_name, self.recorder.config.include_reasoning
            )
            tools = [pydantic_function_tool(tool, name=tool.tool_name) for tool in self.toolkit]
            state = self._context.state
            finish_state = state.value if hasattr(state, "value") else str(state)
            await self.recorder.record_trajectory(
                {
                    "record_type": "trajectory",
                    "timestamp": datetime.now().isoformat(),
                    "agent_id": self.id,
                    "agent_class": type(self).__name__,
                    "role": self.role,
                    "language": self.language,
                    "teacher_model": self.recorder.teacher_model,
                    "tools": _json_safe(tools),
                    "messages": _json_safe(cleaned),
                    "metadata": {
                        "iterations": self._context.iteration,
                        "finish_state": finish_state,
                    },
                }
            )
        except Exception as e:  # noqa: BLE001 - recording must never break execution
            self.logger.warning(f"Failed to record trajectory for dataset: {e}")

    async def _prepare_context(self) -> list[dict]:
        """Prepare a conversation context with system prompt, task data and any
        other context.

        Note: Override this method to change the context setup for the agent.

        Returns a list of dictionaries OpenAI like format, each
        containing a role and content key by default.
        """

        return [
            {"role": "system", "content": PromptLoader.get_system_prompt(self.toolkit, self.config.prompts)},
            *self.task_messages,
            {"role": "user", "content": PromptLoader.get_initial_user_request(self.task_messages, self.config.prompts)},
            *self.conversation,
        ]

    async def _prepare_tools(self) -> list[ChatCompletionFunctionToolParam]:
        """Prepare available tools for the current agent state and progress.

        Note: Override this method to change the tool setup or conditions for tool
        usage.

        Returns a list of ChatCompletionFunctionToolParam based
        available tools.
        """
        tools = set(self.toolkit)
        if self._context.iteration >= self.config.execution.max_iterations:
            raise RuntimeError("Max iterations reached")
        return [pydantic_function_tool(tool, name=tool.tool_name) for tool in tools]

    async def _reasoning_phase(self) -> ReasoningTool:
        """Call LLM to decide next action based on current context."""
        raise NotImplementedError("_reasoning_phase must be implemented by subclass")

    async def _select_action_phase(self, reasoning: ReasoningTool) -> BaseTool:
        """Select the most suitable tool for the action decided in the
        reasoning phase.

        Returns the tool suitable for the action.
        """
        raise NotImplementedError("_select_action_phase must be implemented by subclass")

    async def _action_phase(self, tool: BaseTool) -> str:
        """Call Tool for the action decided in the select_action phase.

        Returns string or dumped JSON result of the tool execution.
        """
        raise NotImplementedError("_action_phase must be implemented by subclass")

    async def _execution_step(self):
        """Execute a single step of the agent workflow.

        Note: Override this method to change the agent workflow for each step.
        """
        reasoning = await self._reasoning_phase()
        self._context.current_step_reasoning = reasoning
        action_tool = await self._select_action_phase(reasoning)
        await self._action_phase(action_tool)

        if isinstance(action_tool, ClarificationTool):
            self.logger.info("\n⏸️  Research paused - please answer questions")
            self.streaming_generator.finish(
                phase_id="{self._context.iteration}-final", content=self._context.execution_result
            )
            self._context.clarification_received.clear()
            await self._context.clarification_received.wait()

    async def cancel(self) -> None:
        """Cancel the agent execution.

        Cancels the running execute task if it exists and sets the agent
        state to CANCELLED.
        """
        if self._execute_task and not self._execute_task.done():
            self._execute_task.cancel()
            try:
                await self._execute_task
            except asyncio.CancelledError:
                pass

    async def execute(self) -> str | None:
        """Start agent execution and return the result.

        Creates an asyncio task for the agent execution, stores it
        in _execute_task for later cancellation, and awaits completion.

        Returns:
            The execution result (final answer) or None.
        """
        self._execute_task = asyncio.create_task(self._execute())
        return await self._execute_task

    async def _execute(self):
        """Internal execution loop for the agent.

        This method contains the main agent execution logic. It is
        called by execute() which wraps it in an asyncio task.
        """
        self.logger.info(f"🚀 User provided {len(self.task_messages)} messages.")
        init_message = f"Agent {self.id} started\n"
        self.conversation.append({"role": "system", "content": init_message})
        self.streaming_generator.add_content_delta(init_message, "0-start")
        try:
            while self._context.state not in AgentStatesEnum.FINISH_STATES.value:
                self._context.iteration += 1
                self.logger.info(f"Step {self._context.iteration} started")
                await self._execution_step()
            return self._context.execution_result

        except asyncio.CancelledError:
            self.logger.info("⏹️ Agent execution cancelled")
            self._context.state = AgentStatesEnum.CANCELLED
            raise

        except Exception as e:
            self.logger.error(f"❌ Agent execution error: {str(e)}")
            self._context.state = AgentStatesEnum.FAILED
            traceback.print_exc()
        finally:
            if self.streaming_generator is not None:
                self.streaming_generator.finish(
                    phase_id=f"{self._context.iteration}-final", content=self._context.execution_result
                )
            self._save_agent_log()
            if self.recorder is not None:
                await self._record_trajectory()
