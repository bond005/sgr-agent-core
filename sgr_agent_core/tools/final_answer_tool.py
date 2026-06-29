from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from pydantic import Field, field_validator

from sgr_agent_core.base_tool import SystemBaseTool, truncate_list
from sgr_agent_core.models import AgentStatesEnum

if TYPE_CHECKING:
    from sgr_agent_core.agent_definition import AgentConfig
    from sgr_agent_core.models import AgentContext

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Mirrors the Field(max_length=...) below; referenced by the truncating
# validator so the two never drift out of sync.
COMPLETED_STEPS_MAX_LENGTH = 5


class FinalAnswerTool(SystemBaseTool):
    """Finalize a task and complete agent execution after all steps are
    completed.

    Usage: Call after you are ready to finalize your work and provide the final answer to the user.
    """

    reasoning: str = Field(description="Why task is now complete and how answer was verified")
    completed_steps: list[str] = Field(
        description="Summary of completed steps including verification",
        min_length=1,
        max_length=COMPLETED_STEPS_MAX_LENGTH,
    )
    answer: str = Field(description="Comprehensive final answer with EXACT factual details (dates, numbers, names)")
    status: Literal[AgentStatesEnum.COMPLETED, AgentStatesEnum.FAILED] = Field(description="Task completion status")

    @field_validator("completed_steps", mode="before")
    @classmethod
    def _truncate_completed_steps(cls, v: object) -> object:
        return truncate_list(v, COMPLETED_STEPS_MAX_LENGTH)

    async def __call__(self, context: AgentContext, config: AgentConfig, **_) -> str:
        context.state = self.status
        context.execution_result = self.answer
        return self.answer
