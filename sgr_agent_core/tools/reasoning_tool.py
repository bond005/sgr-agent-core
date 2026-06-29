from __future__ import annotations

from pydantic import Field, field_validator

from sgr_agent_core.base_tool import SystemBaseTool

# Length budget for the free-text reasoning fields. These limits are kept in the
# JSON schema (as ``maxLength``) so the model sees a soft hint to stay concise,
# while the ``mode="before"`` validators below truncate any over-long value
# instead of raising. This is required because OpenAI structured-outputs parsing
# validates streamed tool arguments against this Pydantic model on the client
# side: without truncation a single over-long ``current_situation`` (common with
# verbose/thinking-enabled models such as GLM-5.2) crashes the whole agent run.
CURRENT_SITUATION_MAX_LENGTH = 1200
PLAN_STATUS_MAX_LENGTH = 600


def _truncate(value: object, limit: int) -> object:
    """Truncate a string to ``limit`` chars with an ellipsis; pass through non-
    string input unchanged (let Pydantic handle type coercion/errors)."""
    if isinstance(value, str) and len(value) > limit:
        return value[: limit - 1].rstrip() + "…"
    return value


class ReasoningTool(SystemBaseTool):
    """Agent core logic determines the next reasoning step with adaptive
    planning by schema-guided-reasoning capabilities. Keep all text fields
    concise and focused.

    Usage: Required tool. Use this tool before any other tool execution
    """

    # Reasoning chain - step-by-step thinking process (helps stabilize model)
    reasoning_steps: list[str] = Field(
        description="Step-by-step reasoning (brief, 1 sentence each)",
        min_length=2,
        max_length=3,
    )

    # Reasoning and state assessment
    current_situation: str = Field(
        description="Current research situation (2-3 sentences MAX)",
        max_length=CURRENT_SITUATION_MAX_LENGTH,
    )
    plan_status: str = Field(
        description="Status of current plan (1 sentence)",
        max_length=PLAN_STATUS_MAX_LENGTH,
    )
    enough_data: bool = Field(
        default=False,
        description="Sufficient data collected for comprehensive report?",
    )

    # Next step planning
    remaining_steps: list[str] = Field(
        description="0-3 remaining steps (brief, action-oriented)",
        max_length=3,
    )
    task_completed: bool = Field(description="Is the research task finished?")

    @field_validator("current_situation", mode="before")
    @classmethod
    def _truncate_current_situation(cls, v: object) -> object:
        return _truncate(v, CURRENT_SITUATION_MAX_LENGTH)

    @field_validator("plan_status", mode="before")
    @classmethod
    def _truncate_plan_status(cls, v: object) -> object:
        return _truncate(v, PLAN_STATUS_MAX_LENGTH)

    async def __call__(self, *args, **kwargs):
        return ""
