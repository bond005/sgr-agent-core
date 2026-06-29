from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, ClassVar, TypeVar

from fastmcp import Client
from pydantic import BaseModel

from sgr_agent_core._compat import Self
from sgr_agent_core.agent_config import GlobalConfig
from sgr_agent_core.services.registry import ToolRegistry

if TYPE_CHECKING:
    from sgr_agent_core.agent_definition import AgentConfig
    from sgr_agent_core.models import AgentContext

logger = logging.getLogger(__name__)


def truncate_list(value: object, limit: int) -> object:
    """Truncate a list to ``limit`` items; pass non-list input through
    unchanged.

    The OpenAI SDK validates streamed tool-call JSON arguments against the
    Pydantic schema on the client side, so an over-long list (common with
    verbose models such as GLM-5.2) raises ``ValidationError`` and crashes the
    whole agent run. Used in ``field_validator(..., mode="before")`` so the
    schema's ``max_length`` stays as a model-facing hint while runtime
    over-long values are silently capped instead of rejecting the call.
    Non-list input is returned unchanged so Pydantic still reports genuine type
    errors.
    """
    if isinstance(value, list) and len(value) > limit:
        return value[:limit]
    return value


class ToolRegistryMixin:
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if cls.__name__ not in ("BaseTool", "MCPBaseTool", "SystemBaseTool"):
            ToolRegistry.register(cls, name=cls.tool_name)


ToolConfig = TypeVar("ToolConfig", bound=BaseModel | None)


class BaseTool(BaseModel, ToolRegistryMixin):
    """Class to provide tool handling capabilities."""

    tool_name: ClassVar[str] = None
    description: ClassVar[str] = None
    isSystemTool: ClassVar[bool] = False
    # Optional: Pydantic model for tool config; agent.get_tool_config(tool_class) returns an instance of it
    config_model: ClassVar[ToolConfig] = None

    async def __call__(self, context: AgentContext, config: AgentConfig, **kwargs) -> str:
        """The result should be a string or dumped JSON."""
        raise NotImplementedError("Execute method must be implemented by subclass")

    def __init_subclass__(cls, **kwargs) -> None:
        if "tool_name" not in cls.__dict__:
            cls.tool_name = cls.__name__.lower()
        if "description" not in cls.__dict__:
            cls.description = cls.__doc__ or ""
        super().__init_subclass__(**kwargs)


class SystemBaseTool(BaseTool):
    """Base class for system tools that are always available and never
    filtered."""

    isSystemTool: ClassVar[bool] = True


ReasoningToolStubType = TypeVar("ReasoningToolStubType", bound=SystemBaseTool)


class MCPBaseTool(BaseTool):
    """Base model for MCP Tool schema."""

    _client: ClassVar[Client | None] = None

    async def __call__(self, context: AgentContext, config: AgentConfig, **kwargs) -> str:
        config = GlobalConfig()
        payload = self.model_dump(mode="json")
        try:
            async with self._client:
                result = await self._client.call_tool(self.tool_name, payload)
                return json.dumps([m.model_dump_json() for m in result.content], ensure_ascii=False)[
                    : config.execution.mcp_context_limit
                ]
        except Exception as e:
            logger.error(f"Error processing MCP tool {self.tool_name}: {e}")
            return f"Error: {e}"

    @classmethod
    def model_validate_json(cls, json_data: str | bytes | bytearray, **kwargs) -> Self:
        return super().model_validate_json(json_data=json_data or "{}", **kwargs)
