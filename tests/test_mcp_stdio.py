"""End-to-end over the real MCP stdio transport (spawns `python -m mcp_server.server`)."""

import pytest

from codefix.mcp_client import Toolbox, ToolError


@pytest.fixture()
def toolbox(sample_repo, tmp_path, monkeypatch):
    from codefix import config

    monkeypatch.setattr(config, "SANDBOX_DIR", tmp_path / "sandbox")
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "index")
    tb = Toolbox(sample_repo, transport="stdio")
    yield tb
    tb.close()


def test_stdio_lists_all_tools(toolbox):
    expected = {"code_search", "read_file", "list_tree", "parse_stacktrace", "static_analysis", "run_tests",
                "apply_patch", "git_branch", "git_commit", "open_pull_request", "rag_retrieve", "create_workspace"}
    assert expected <= set(toolbox.specs)


def test_stdio_calls_and_telemetry(toolbox):
    seen = []
    toolbox.listener = seen.append
    r = toolbox.call("read_file", {"path": "pkg/mathx.py", "raw": True}, task_id="t1", agent="diagnoser")
    assert r["content"].replace("\r\n", "\n").startswith("def add(a, b):\n")
    assert seen[-1]["tool"] == "read_file" and seen[-1]["ok"] and seen[-1]["agent"] == "diagnoser"
    with pytest.raises(ToolError):
        toolbox.call("read_file", {"path": ".env"}, task_id="t1", agent="diagnoser")
    assert seen[-1]["ok"] is False


def test_langchain_binding(toolbox):
    tools = {t.name: t for t in toolbox.langchain_tools(["read_file", "list_tree"])}
    assert set(tools) == {"read_file", "list_tree"}
    out = tools["list_tree"].invoke({"path": "."})
    assert "pkg/mathx.py" in out["files"]
