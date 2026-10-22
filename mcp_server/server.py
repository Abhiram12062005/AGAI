"""CodeFix MCP server (MCPServer, the mcp 2.x name for FastMCP; stdio transport).

Every agent reaches the repository, the test runner, git and RAG through this server, so
tool access is uniform regardless of which model backs an agent.

Run standalone:
    python -m mcp_server.server --repo <path>          # stdio
    mcp dev mcp_server/server.py                        # MCP inspector
Configuration via args or env: CODEFIX_REPO_ROOT, CODEFIX_SANDBOX_DIR, CODEFIX_INDEX_DIR.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.mcpserver import MCPServer  # noqa: E402

from mcp_server.tools import ToolContext  # noqa: E402

mcp = MCPServer("codefix", instructions="Repository, test, git and RAG tools for CodeFix AI agents.",
                log_level="WARNING")
_ctx: ToolContext | None = None


def ctx() -> ToolContext:
    global _ctx
    if _ctx is None:
        root = Path(__file__).resolve().parent.parent
        _ctx = ToolContext(
            os.environ["CODEFIX_REPO_ROOT"],
            os.environ.get("CODEFIX_SANDBOX_DIR", root / ".codefix" / "sandbox"),
            os.environ.get("CODEFIX_INDEX_DIR", root / ".codefix" / "index"),
        )
    return _ctx


@mcp.tool()
def code_search(query: str, k: int = 8) -> dict[str, Any]:
    """Hybrid code search over the repo: ripgrep literal matches fused (reciprocal-rank) with
    vector search over AST-chunked code. Returns file, line, snippet and match source."""
    return ctx().code_search(query, k)


@mcp.tool()
def read_file(path: str, start: int = 1, end: int | None = None, workspace: str | None = None,
              raw: bool = False) -> dict[str, Any]:
    """Read a repo-relative file with line numbers (max 400 lines per call). Paths outside the
    repo root and secret files (.env, .git, CI secrets, keys) are refused. raw=true returns the
    exact unnumbered content."""
    return ctx().read_file(path, start, end, workspace, raw)


@mcp.tool()
def list_tree(path: str = ".", max_depth: int = 3, workspace: str | None = None) -> dict[str, Any]:
    """List files under a repo-relative directory (ignores .git, node_modules, venvs)."""
    return ctx().list_tree(path, max_depth, workspace)


@mcp.tool()
def parse_stacktrace(text: str) -> dict[str, Any]:
    """Parse a Python/pytest/JavaScript stack trace into structured frames mapped to repo files."""
    return ctx().parse_stacktrace(text)


@mcp.tool()
def static_analysis(paths: list[str], workspace: str | None = None) -> dict[str, Any]:
    """Run ruff (Python) or eslint / node --check (JavaScript) on the given repo-relative paths."""
    return ctx().static_analysis(paths, workspace)


@mcp.tool()
def rag_retrieve(query: str, k: int = 5, collection: str = "code") -> dict[str, Any]:
    """Retrieve chunks from a RAG collection: 'code' (function-level chunks with file+line),
    'bugs' (past bugs and their accepted fixes from run history) or 'docs' (README/docs)."""
    return ctx().rag_retrieve(query, k, collection)


@mcp.tool()
def index_repo(force: bool = False) -> dict[str, Any]:
    """(Re)build the RAG index for the repo. Incremental on file mtime unless force=true."""
    return ctx().index_repo(force)


@mcp.tool()
def record_fix(task_id: str, bug_report: str, root_cause: str, diff: str, files: list[str]) -> dict[str, Any]:
    """Add an accepted fix to the 'bugs' RAG collection (past bugs + fixes)."""
    return ctx().record_fix(task_id, bug_report, root_cause, diff, files)


@mcp.tool()
def create_workspace(name: str) -> dict[str, Any]:
    """Create an isolated clone of the target repo (the repo itself is never mutated)."""
    return ctx().create_workspace(name)


@mcp.tool()
def discard_workspace(workspace: str) -> dict[str, Any]:
    """Delete an isolated workspace."""
    return ctx().discard_workspace(workspace)


@mcp.tool()
def apply_patch(diff: str, workspace: str) -> dict[str, Any]:
    """Apply a unified diff inside an isolated workspace (git apply --check first)."""
    return ctx().apply_patch(diff, workspace)


@mcp.tool()
def run_tests(workspace: str, selector: str | None = None, timeout_s: int = 180) -> dict[str, Any]:
    """Run the project's test suite (pytest / jest / node:test auto-detected) in a workspace,
    in a subprocess with a hard timeout. Optional selector narrows the run."""
    return ctx().run_tests(workspace, selector, timeout_s)


@mcp.tool()
def git_branch(workspace: str, branch: str) -> dict[str, Any]:
    """Create/switch to a codefix/<task_id> branch inside a workspace."""
    return ctx().git_branch(workspace, branch)


@mcp.tool()
def git_commit(workspace: str, message: str) -> dict[str, Any]:
    """Commit all changes in a workspace; returns the commit sha."""
    return ctx().git_commit(workspace, message)


@mcp.tool()
def diff_workspace(workspace: str) -> dict[str, Any]:
    """Show the uncommitted diff of a workspace against its HEAD."""
    return ctx().diff_workspace(workspace)


@mcp.tool()
def open_pull_request(workspace: str, branch: str, title: str, body: str, confirmed: bool = False) -> dict[str, Any]:
    """Push a codefix/* branch and open a GitHub PR. Refuses unless confirmed=true was set by an
    explicit human confirmation, and never pushes to the default branch."""
    return ctx().open_pull_request(workspace, branch, title, body, confirmed)


def _quiet() -> None:
    import logging

    # stdout is the MCP channel; keep library chatter (HF downloads, httpx) off stderr too
    for name in ("httpx", "huggingface_hub", "sentence_transformers", "chromadb", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    if (hub / "models--sentence-transformers--all-MiniLM-L6-v2").is_dir():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # model cached: no network calls per server start
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main() -> None:
    _quiet()
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.environ.get("CODEFIX_REPO_ROOT"))
    ap.add_argument("--sandbox", default=os.environ.get("CODEFIX_SANDBOX_DIR"))
    ap.add_argument("--index", default=os.environ.get("CODEFIX_INDEX_DIR"))
    a = ap.parse_args()
    if not a.repo:
        ap.error("--repo or CODEFIX_REPO_ROOT is required")
    os.environ["CODEFIX_REPO_ROOT"] = a.repo
    if a.sandbox:
        os.environ["CODEFIX_SANDBOX_DIR"] = a.sandbox
    if a.index:
        os.environ["CODEFIX_INDEX_DIR"] = a.index
    import logging

    logging.getLogger().setLevel(logging.WARNING)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
