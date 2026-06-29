from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field, field_validator

from sgr_agent_core.base_tool import SystemBaseTool, truncate_list

if TYPE_CHECKING:
    from sgr_agent_core.agent_definition import AgentConfig
    from sgr_agent_core.models import AgentContext

# Mirror the Field(max_length=...) values below; referenced by the truncating
# validators so they never drift out of sync.
PLANNED_STEPS_MAX_LENGTH = 4
SEARCH_STRATEGIES_MAX_LENGTH = 3


class GeneratePlanTool(SystemBaseTool):
    """Generate a research plan.

    Useful to split complex request into manageable steps.
    """

    reasoning: str = Field(description="Justification for research approach")
    research_goal: str = Field(description="Primary research objective")
    planned_steps: list[str] = Field(
        description="List of 3-4 planned steps", min_length=3, max_length=PLANNED_STEPS_MAX_LENGTH
    )
    search_strategies: list[str] = Field(
        description="Information search strategies", min_length=2, max_length=SEARCH_STRATEGIES_MAX_LENGTH
    )

    @field_validator("planned_steps", mode="before")
    @classmethod
    def _truncate_planned_steps(cls, v: object) -> object:
        return truncate_list(v, PLANNED_STEPS_MAX_LENGTH)

    @field_validator("search_strategies", mode="before")
    @classmethod
    def _truncate_search_strategies(cls, v: object) -> object:
        return truncate_list(v, SEARCH_STRATEGIES_MAX_LENGTH)

    async def __call__(self, context: AgentContext, config: AgentConfig, **_) -> str:
        return ""
