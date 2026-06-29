from typing import Any, Literal, Type

from openai import AsyncOpenAI, pydantic_function_tool
from pydantic import ValidationError

from sgr_agent_core.agent_config import AgentConfig
from sgr_agent_core.base_agent import BaseAgent
from sgr_agent_core.models import AgentStatesEnum
from sgr_agent_core.tools import (
    BaseTool,
    FinalAnswerTool,
    ReasoningTool,
    SystemBaseTool,
)


class SGRToolCallingAgent(BaseAgent):
    """Agent that uses OpenAI native function calling to select and execute
    tools based on SGR like a reasoning scheme."""

    name: str = "sgr_tool_calling_agent"

    # Maximum number of attempts (including the first) for a single tool-call
    # LLM request in both the reasoning and the action phases. Each retry is a
    # fresh streaming request; at temperature>0 (and for the Z.ai prefix cache,
    # which caches input computation but never replayed outputs) a retry yields a
    # freshly sampled response, so transient "no tool-call" / parse failures
    # commonly recover. After this budget is exhausted a synthetic fallback keeps
    # the run alive instead of crashing. See AGENTS.md / the truncation notes.
    tool_call_max_attempts: int = 3

    def __init__(
        self,
        task_messages: list,
        openai_client: AsyncOpenAI,
        agent_config: AgentConfig,
        toolkit: list[Type[BaseTool]],
        *,
        def_name: str | None = None,
        reasoning_tool_cls: type[SystemBaseTool] = ReasoningTool,
        **kwargs: dict,
    ):
        super().__init__(
            task_messages=task_messages,
            openai_client=openai_client,
            agent_config=agent_config,
            toolkit=toolkit,
            def_name=def_name,
            **kwargs,
        )
        self.tool_choice: Literal["required"] = "required"
        self.ReasoningTool: type[SystemBaseTool] = reasoning_tool_cls

    @staticmethod
    def _parse_tool_call(completion: Any) -> BaseTool | None:
        """Extract the parsed tool from a chat completion.

        Returns ``None`` if the model returned no tool-call (``tool_calls`` is
        ``None``/empty) or the structure is otherwise unusable. A ``ValidationError``
        raised by lazily parsing ``parsed_arguments`` is allowed to propagate so
        the retry loop in :meth:`_llm_call_with_retry` can catch it.
        """
        try:
            return completion.choices[0].message.tool_calls[0].function.parsed_arguments
        except (IndexError, AttributeError, TypeError):
            return None

    async def _llm_call_with_retry(
        self,
        phase: str,
        max_attempts: int,
        **openai_kwargs: Any,
    ) -> tuple[Any, BaseTool | None]:
        """Call :meth:`_llm_call` up to ``max_attempts`` times and parse a
        tool.

        Catches transient failures of two kinds observed with GLM-5.2: (1) the
        model returns no tool-call despite ``tool_choice="required"`` (parsed as
        ``None``); (2) streamed tool-argument validation raises (``ValidationError``)
        during the streaming aggregation. Each retry is a fresh streaming request
        with the same context (no conversation mutation between attempts), so the
        recorded trajectory stays clean. Returns ``(last_completion, parsed_tool)``
        where ``parsed_tool`` is ``None`` when every attempt failed - the caller
        is then responsible for producing a synthetic fallback.
        """
        last_completion = None
        for attempt in range(1, max_attempts + 1):
            try:
                completion = await self._llm_call(phase, **openai_kwargs)
                last_completion = completion
                parsed = self._parse_tool_call(completion)
            except (ValidationError, IndexError, AttributeError, TypeError, ValueError) as e:
                self.logger.warning(
                    f"{phase} attempt {attempt}/{max_attempts} failed " f"({type(e).__name__}: {e}); retrying"
                )
                continue
            if parsed is not None:
                return last_completion, parsed
            self.logger.warning(f"{phase} attempt {attempt}/{max_attempts} returned no usable tool-call; retrying")
        return last_completion, None

    def _fallback_reasoning(self) -> ReasoningTool:
        """Build a synthetic reasoning when the model produced no usable tool-
        call.

        Keeps the run alive instead of crashing. The trajectory is flagged via
        ``_used_fallback`` so it can be filtered out of the distillation dataset
        during export (a fabricated reasoning step is lower-quality CoT).
        """
        self._used_fallback = True
        self.logger.warning(
            f"reasoning: no usable tool-call after {self.tool_call_max_attempts} attempts; "
            "using synthetic reasoning fallback to keep the run alive"
        )
        return ReasoningTool(
            reasoning_steps=[
                "Structured reasoning was unavailable from the model; proceeding cautiously",
                "Continuing with the next action",
            ],
            current_situation="Reasoning phase did not return a structured tool call; recovering.",
            plan_status="Recovering from a missing reasoning tool call.",
            enough_data=False,
            remaining_steps=["Select the next action"],
            task_completed=False,
        )

    async def _reasoning_phase(self) -> ReasoningTool:
        phase_id = f"{self._context.iteration}-reasoning"
        _completion, reasoning = await self._llm_call_with_retry(
            "reasoning",
            max_attempts=self.tool_call_max_attempts,
            messages=await self._prepare_context(),
            tools=[pydantic_function_tool(self.ReasoningTool, name=self.ReasoningTool.tool_name)],
            tool_choice=self.tool_choice,
            **self.config.llm.to_openai_client_kwargs(),
        )
        if reasoning is None:
            reasoning = self._fallback_reasoning()
        self.streaming_generator.add_tool_call(phase_id, reasoning)
        self.conversation.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "id": phase_id,
                        "function": {
                            "name": reasoning.tool_name,
                            "arguments": reasoning.model_dump_json(),
                        },
                    }
                ],
            }
        )
        tool_call_result = await reasoning(self._context, self.config)
        self.streaming_generator.add_tool_result(phase_id, tool_call_result, reasoning.tool_name)
        self.conversation.append({"role": "tool", "content": tool_call_result, "tool_call_id": phase_id})
        self._log_reasoning(reasoning)
        return reasoning

    async def _select_action_phase(self, reasoning: ReasoningTool) -> BaseTool:
        phase_id = f"{self._context.iteration}-action"
        completion, tool = await self._llm_call_with_retry(
            "action",
            max_attempts=self.tool_call_max_attempts,
            messages=await self._prepare_context(),
            tools=await self._prepare_tools(),
            tool_choice=self.tool_choice,
            **self.config.llm.to_openai_client_kwargs(),
        )
        if tool is None or not isinstance(tool, BaseTool):
            final_content = "Task completed successfully"
            try:
                final_content = completion.choices[0].message.content or final_content
            except (IndexError, AttributeError, TypeError):
                pass
            tool = FinalAnswerTool(
                reasoning="Agent decided to complete the task",
                completed_steps=["Task finalized via fallback (no tool-call produced)"],
                answer=final_content,
                status=AgentStatesEnum.COMPLETED,
            )
            self._used_fallback = True
            self.logger.warning("action: no usable tool-call after retries; using FinalAnswerTool fallback")

        self.conversation.append(
            {
                "role": "assistant",
                "content": reasoning.remaining_steps[0] if reasoning.remaining_steps else "Completing",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": phase_id,
                        "function": {
                            "name": tool.tool_name,
                            "arguments": tool.model_dump_json(),
                        },
                    }
                ],
            }
        )
        self.streaming_generator.add_tool_call(phase_id, tool)
        return tool

    async def _action_phase(self, tool: BaseTool) -> str:
        phase_id = f"{self._context.iteration}-action"
        result = await tool(self._context, self.config, **self.tool_configs.get(tool.tool_name, {}))
        self.conversation.append({"role": "tool", "content": result, "tool_call_id": phase_id})
        self.streaming_generator.add_tool_result(phase_id, result, tool.tool_name)
        self._log_tool_execution(tool, result)
        return result
