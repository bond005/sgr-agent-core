from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field, field_validator

from sgr_agent_core.base_tool import SystemBaseTool, truncate_list

if TYPE_CHECKING:
    from sgr_agent_core.agent_config import AgentConfig
    from sgr_agent_core.models import AgentContext

# Mirror the Field(max_length=...) values below; referenced by the truncating
# validators so they never drift out of sync.
PLAN_CHANGES_MAX_LENGTH = 3
NEXT_STEPS_MAX_LENGTH = 4


class AdaptPlanTool(SystemBaseTool):
    """Adapt a research plan based on new findings."""

    reasoning: str = Field(description="Why plan needs adaptation based on new data")
    original_goal: str = Field(description="Original research goal")
    new_goal: str = Field(description="Updated research goal")
    plan_changes: list[str] = Field(
        description="Specific changes made to plan", min_length=1, max_length=PLAN_CHANGES_MAX_LENGTH
    )
    next_steps: list[str] = Field(description="Updated remaining steps", min_length=2, max_length=NEXT_STEPS_MAX_LENGTH)

    @field_validator("plan_changes", mode="before")
    @classmethod
    def _truncate_plan_changes(cls, v: object) -> object:
        return truncate_list(v, PLAN_CHANGES_MAX_LENGTH)

    @field_validator("next_steps", mode="before")
    @classmethod
    def _truncate_next_steps(cls, v: object) -> object:
        return truncate_list(v, NEXT_STEPS_MAX_LENGTH)

    async def __call__(self, context: AgentContext, config: AgentConfig, **_) -> str:
        return ""
