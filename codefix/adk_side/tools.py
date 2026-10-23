"""MCP tools as native Google ADK tools.

Each tool the CodeFix MCP server advertises becomes an ADK `BaseTool` whose declaration is
the server's own JSON schema and whose execution is an MCP `call_tool` through the shared
Toolbox (so every call is still logged to the Blackboard under the calling agent).
"""

from __future__ import annotations

import asyncio
from typing import Any

from google.adk.tools import BaseTool
from google.genai import types

from codefix.mcp_client import Toolbox


class McpTool(BaseTool):
    def __init__(self, toolbox: Toolbox, name: str, spec: dict, task_id: str, agent: str):
        super().__init__(name=name, description=(spec.get("description") or name).strip())
        self._toolbox = toolbox
        self._schema = spec.get("schema") or {"type": "object", "properties": {}}
        self._task_id = task_id
        self._agent = agent

    def _get_declaration(self) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(name=self.name, description=self.description,
                                         parameters_json_schema=self._schema)

    async def run_async(self, *, args: dict[str, Any], tool_context) -> Any:
        # the MCP client is sync (its session lives on its own loop thread)
        result = await asyncio.to_thread(self._toolbox.call, self.name, dict(args or {}),
                                         task_id=self._task_id, agent=self._agent)
        return result if isinstance(result, dict) else {"result": result}


def mcp_tools(toolbox: Toolbox, names: list[str], task_id: str, agent: str) -> list[McpTool]:
    return [McpTool(toolbox, n, spec, task_id, agent) for n, spec in toolbox.tool_specs(names).items()]
