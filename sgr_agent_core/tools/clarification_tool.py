from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field, field_validator

from sgr_agent_core.base_tool import SystemBaseTool, truncate_list
from sgr_agent_core.models import AgentContext, AgentStatesEnum

if TYPE_CHECKING:
    from sgr_agent_core.agent_config import AgentConfig

# All list fields share the same cap; mirror the Field(max_length=...) values
# below and referenced by the truncating validators.
CLARIFICATION_LIST_MAX_LENGTH = 3


class ClarificationTool(SystemBaseTool):
    """Ask clarifying questions when facing an ambiguous request.

    Keep all fields concise - brief reasoning, short terms, and clear questions.
    """

    reasoning: str = Field(description="Why clarification is needed (1-2 sentences MAX)", max_length=200)
    unclear_terms: list[str] = Field(
        description="List of unclear terms (brief, 1-3 words each)",
        min_length=1,
        max_length=CLARIFICATION_LIST_MAX_LENGTH,
    )
    assumptions: list[str] = Field(
        description="Possible interpretations (short, 1 sentence each)",
        min_length=2,
        max_length=CLARIFICATION_LIST_MAX_LENGTH,
    )
    questions: list[str] = Field(
        description="3 specific clarifying questions (short and direct)",
        min_length=1,
        max_length=CLARIFICATION_LIST_MAX_LENGTH,
    )

    @field_validator("unclear_terms", "assumptions", "questions", mode="before")
    @classmethod
    def _truncate_lists(cls, v: object) -> object:
        return truncate_list(v, CLARIFICATION_LIST_MAX_LENGTH)

    async def __call__(self, context: AgentContext, config: AgentConfig, **_) -> str:
        context.state = AgentStatesEnum.WAITING_FOR_CLARIFICATION
        context.execution_result = "\n".join(self.questions)
        return ""
