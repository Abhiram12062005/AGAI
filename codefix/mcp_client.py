"""Thin MCP client: binds the CodeFix MCP server's tools as LangChain tools.

The MCP Python SDK is async; LangGraph nodes here are sync. A dedicated background event
loop thread owns the stdio session for the lifetime of the toolbox, and sync callers submit
coroutines to it. Every tool call is logged (args, latency, ok/error) to the Blackboard and
the JSONL trace under the calling agent's name.

Toolboxes are cached per repo so a benchmark of 40+ runs reuses one server process.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from collections.abc import Callable
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from codefix import config

_JSON_TO_PY = {"string": str, "integer": int, "number": float, "boolean": bool, "array": list, "object": dict}


class ToolError(RuntimeError):
    pass


def _args_model(name: str, schema: dict) -> type[BaseModel]:
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields: dict[str, Any] = {}
    for pname, p in props.items():
        t = p.get("type")
        if t is None and "anyOf" in p:
            t = next((x.get("type") for x in p["anyOf"] if x.get("type") != "null"), "string")
        py = _JSON_TO_PY.get(t, str)
        if pname in required:
            fields[pname] = (py, Field(..., description=p.get("description", "")))
        else:
            fields[pname] = (py | None, Field(p.get("default"), description=p.get("description", "")))
    return create_model(f"{name}_args", **fields)


class Toolbox:
    """Holds an MCP session and exposes tools as LangChain StructuredTools."""

    def __init__(self, repo_root: Path, transport: str = "stdio"):
        self.repo = Path(repo_root).resolve()
        self.transport = transport
        self.specs: dict[str, dict] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._session = None
        self._stack: AsyncExitStack | None = None
        self._local = None
        self._lock = threading.Lock()
        # (task_id, agent) -> telemetry sink; set per call via bind()
        self.listener: Callable[[dict], None] | None = None
        if transport == "stdio":
            self._start_stdio()
        else:  # in-process: same tool implementations, no transport (unit tests only)
            from mcp_server.tools import ToolContext

            self._local = ToolContext(self.repo, config.SANDBOX_DIR, config.INDEX_DIR)
            self.specs = _local_specs()

    # ------------------------------------------------------------------ stdio session
    def _start_stdio(self) -> None:
        ready = threading.Event()
        err: list[BaseException] = []

        def runner() -> None:
            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._connect())
            except BaseException as e:  # surface startup errors to the caller
                err.append(e)
                ready.set()
                return
            ready.set()
            loop.run_forever()

        threading.Thread(target=runner, name=f"mcp-{self.repo.name}", daemon=True).start()
        ready.wait(timeout=120)
        if err:
            raise ToolError(f"failed to start MCP server: {err[0]!r}")
        if self._session is None:
            raise ToolError("MCP server did not start within 120s")

    async def _connect(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        root = Path(__file__).resolve().parent.parent
        env = {**os.environ, "CODEFIX_REPO_ROOT": str(self.repo),
               "CODEFIX_SANDBOX_DIR": str(config.SANDBOX_DIR), "CODEFIX_INDEX_DIR": str(config.INDEX_DIR),
               "PYTHONPATH": str(root), "PYTHONIOENCODING": "utf-8"}
        params = StdioServerParameters(command=sys.executable, args=["-m", "mcp_server.server"],
                                       env=env, cwd=str(root))
        self._stack = AsyncExitStack()
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        listed = await self._session.list_tools()
        self.specs = {t.name: {"description": t.description or "", "schema": _schema_of(t)}
                      for t in listed.tools}

    async def _call_async(self, name: str, args: dict) -> Any:
        res = await self._session.call_tool(name, args)
        text = "\n".join(getattr(c, "text", "") for c in res.content)
        if getattr(res, "is_error", None) or getattr(res, "isError", None):
            raise ToolError(text or f"{name} failed")
        structured = getattr(res, "structured_content", None) or getattr(res, "structuredContent", None)
        if structured is not None:
            # FastMCP wraps non-object returns as {"result": ...}
            return structured.get("result", structured) if set(structured) == {"result"} else structured
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    # ------------------------------------------------------------------ public API
    def call(self, name: str, args: dict | None = None, *, task_id: str | None = None,
             agent: str = "system", timeout: float = 900) -> Any:
        args = {k: v for k, v in (args or {}).items() if v is not None}
        started = time.time()
        ok, result, error = True, None, None
        try:
            if self._local is not None:
                result = getattr(self._local, name)(**args)
            else:
                fut = asyncio.run_coroutine_threadsafe(self._call_async(name, args), self._loop)
                result = fut.result(timeout=timeout)
            return result
        except Exception as e:
            ok, error = False, f"{type(e).__name__}: {e}"
            raise ToolError(error) from e
        finally:
            if self.listener and task_id:
                self.listener({"task_id": task_id, "agent": agent, "tool": name, "args": args, "ok": ok,
                               "latency_ms": int((time.time() - started) * 1000),
                               "result": result if ok else error})

    def langchain_tools(self, names: list[str] | None = None, *, task_id: str | None = None,
                        agent: str = "system") -> list[StructuredTool]:
        """Bind MCP tools as LangChain tools (sync)."""
        tools = []
        for name, spec in self.specs.items():
            if names and name not in names:
                continue

            def _fn(_name=name, **kwargs):
                return self.call(_name, kwargs, task_id=task_id, agent=agent)

            tools.append(StructuredTool.from_function(
                func=_fn, name=name, description=spec["description"],
                args_schema=_args_model(name, spec["schema"])))
        return tools

    def close(self) -> None:
        if self._loop and self._stack:
            fut = asyncio.run_coroutine_threadsafe(self._stack.aclose(), self._loop)
            try:
                fut.result(timeout=10)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)


def _local_specs() -> dict[str, dict]:
    """Tool specs for the in-process transport, derived from the FastMCP registrations."""
    from mcp_server import server

    specs = {}
    for t in asyncio.run(server.mcp.list_tools()):
        specs[t.name] = {"description": t.description or "", "schema": _schema_of(t)}
    return specs


def _schema_of(tool) -> dict:
    return getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {}


def close_toolbox(repo_root: str | Path) -> None:
    """Stop the MCP server process(es) serving this repo (they hold an embedder in memory)."""
    key_repo = str(Path(repo_root).resolve())
    with _CACHE_LOCK:
        doomed = [_CACHE.pop(k) for k in [k for k in _CACHE if k[0] == key_repo]]
    for tb in doomed:
        tb.close()


_CACHE: dict[tuple[str, str], Toolbox] = {}
_CACHE_LOCK = threading.Lock()


def get_toolbox(repo_root: str | Path, transport: str | None = None) -> Toolbox:
    transport = transport or os.environ.get("CODEFIX_MCP_TRANSPORT", "stdio")
    key = (str(Path(repo_root).resolve()), transport)
    with _CACHE_LOCK:
        if key not in _CACHE:
            _CACHE[key] = Toolbox(Path(repo_root), transport)
        return _CACHE[key]
