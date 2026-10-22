"""MCP tool layer: sandboxing, workspaces, patch application, tests, stack traces, search."""

import pytest

from codefix.agents.patching import EditError, build_diff, is_test_path
from mcp_server.sandbox import SandboxError, is_blocked, safe_path
from mcp_server.tools import ToolContext


@pytest.fixture()
def tc(sample_repo, tmp_path):
    return ToolContext(sample_repo, tmp_path / "sandbox", tmp_path / "index")


# ----------------------------------------------------------------------------- sandbox
@pytest.mark.parametrize("p", [".env", "sub/.env", ".env.local", ".git/config", ".github/workflows/ci.yml",
                               "keys/id_rsa", "server.pem", "config/secrets.json"])
def test_blocklist(p):
    assert is_blocked(p)


@pytest.mark.parametrize("p", ["pkg/mathx.py", ".env.example", "README.md", "src/environment.py"])
def test_not_blocked(p):
    assert not is_blocked(p)


def test_safe_path_refuses_escape(sample_repo):
    with pytest.raises(SandboxError):
        safe_path(sample_repo, "../outside.txt")
    with pytest.raises(SandboxError):
        safe_path(sample_repo, str(sample_repo.parent / "x.py"))
    assert safe_path(sample_repo, "pkg/mathx.py").name == "mathx.py"


def test_read_file_numbered_and_raw(tc):
    r = tc.read_file("pkg/mathx.py", 1, 2)
    assert "    1| def add(a, b):" in r["content"] and r["total_lines"] == 6
    assert tc.read_file("pkg/mathx.py", raw=True)["content"].startswith("def add")
    with pytest.raises(SandboxError):
        tc.read_file(".env")


def test_list_tree_hides_secrets(tc):
    files = tc.list_tree(".")["files"]
    assert "pkg/mathx.py" in files and ".env" not in files and not any(f.startswith(".git/") for f in files)


# ----------------------------------------------------------------------------- patching
def _diff(tc):
    original = tc.read_file("pkg/mathx.py", raw=True)["content"].replace("\r\n", "\n")
    return build_diff({"pkg/mathx.py": original},
                      [{"path": "pkg/mathx.py", "find": "return a - b", "replace": "return a + b"}]).diff


def test_build_diff_errors():
    with pytest.raises(EditError):
        build_diff({"f.py": "x = 1\nx = 1\n"}, [{"path": "f.py", "find": "x = 1", "replace": "x = 2"}])
    with pytest.raises(EditError):
        build_diff({"f.py": "x = 1\n"}, [{"path": "f.py", "find": "y", "replace": "z"}])
    with pytest.raises(EditError):
        build_diff({"f.py": "x = 1\n"}, [{"path": "g.py", "find": "x", "replace": "z"}])


def test_diff_handles_missing_trailing_newline():
    r = build_diff({"f.py": "a = 1\nb = 2"}, [{"path": "f.py", "find": "b = 2", "replace": "b = 3"}])
    assert "\\ No newline at end of file" in r.diff


def test_workspace_isolation_apply_and_run_tests(tc, sample_repo):
    tc.create_workspace("t1-a1")
    before = tc.run_tests("t1-a1")
    assert before["passed"] is False and before["failed"] == 1 and before["total"] == 2
    assert any("test_add" in a for a in before["failing_assertions"])

    res = tc.apply_patch(_diff(tc), "t1-a1")
    assert res["ok"], res
    after = tc.run_tests("t1-a1")
    assert after["passed"] is True and after["total"] == 2
    # target repo untouched
    assert "return a - b" in (sample_repo / "pkg" / "mathx.py").read_text()
    tc.discard_workspace("t1-a1")
    with pytest.raises(SandboxError):
        tc.run_tests("t1-a1")


def test_apply_patch_rejects_bad_diff_and_blocked_paths(tc):
    tc.create_workspace("t2-a1")
    assert tc.apply_patch("--- a/pkg/mathx.py\n+++ b/pkg/mathx.py\n@@ -1 +1 @@\n-nope\n+x\n", "t2-a1")["ok"] is False
    evil = "--- a/.env\n+++ b/.env\n@@ -1 +1 @@\n-SECRET=1\n+SECRET=2\n"
    with pytest.raises(SandboxError):
        tc.apply_patch(evil, "t2-a1")
    with pytest.raises(SandboxError):
        tc.apply_patch(evil.replace(".env", "../x.py"), "t2-a1")


def test_invalid_workspace_names(tc):
    for bad in ("../x", "a/b", "", "x" * 200):
        with pytest.raises(SandboxError):
            tc.create_workspace(bad)


def test_selector_runs_subset(tc):
    tc.create_workspace("t3-a1")
    r = tc.run_tests("t3-a1", selector="tests/test_mathx.py::test_mul")
    assert r["passed"] and r["total"] == 1


def test_static_analysis_flags_undefined_name(tc):
    tc.create_workspace("t4-a1")
    diff = build_diff({"pkg/mathx.py": tc.read_file("pkg/mathx.py", raw=True)["content"].replace("\r\n", "\n")},
                      [{"path": "pkg/mathx.py", "find": "return a - b", "replace": "return a + c"}]).diff
    assert tc.apply_patch(diff, "t4-a1")["ok"]
    f = tc.static_analysis(["pkg/mathx.py"], workspace="t4-a1")["findings"]
    assert any(x["code"] == "F821" and x["gating"] for x in f)


def test_git_branch_commit_and_pr_guards(tc):
    tc.create_workspace("t5-a1")
    tc.apply_patch(_diff(tc), "t5-a1")
    with pytest.raises(SandboxError):
        tc.git_branch("t5-a1", "main")
    tc.git_branch("t5-a1", "codefix/t5")
    assert len(tc.git_commit("t5-a1", "fix add")["sha"]) == 40
    with pytest.raises(SandboxError):
        tc.open_pull_request("t5-a1", "codefix/t5", "t", "b", confirmed=False)
    with pytest.raises(SandboxError):
        tc.open_pull_request("t5-a1", "main", "t", "b", confirmed=True)
    r = tc.open_pull_request("t5-a1", "codefix/t5", "t", "b", confirmed=True)
    assert r["pushed"] is False and "no 'origin' remote" in r["note"]


# ----------------------------------------------------------------------------- analysis
def test_parse_stacktrace_python_maps_to_repo(tc, sample_repo):
    trace = (
        "Traceback (most recent call last):\n"
        '  File "/usr/lib/python3.11/site-packages/foo.py", line 3, in run\n'
        f'  File "{sample_repo / "pkg" / "mathx.py"}", line 2, in add\n'
        "    return a - b\n"
        "TypeError: unsupported operand\n"
    )
    r = tc.parse_stacktrace(trace)
    assert r["exception"].startswith("TypeError")
    assert r["innermost_repo_frame"]["repo_path"] == "pkg/mathx.py"
    assert r["frames"][0]["in_repo"] is False


def test_parse_stacktrace_pytest_and_js(tc):
    r = tc.parse_stacktrace("pkg/mathx.py:2: in add\nE   assert -1 == 5\n")
    assert r["innermost_repo_frame"]["repo_path"] == "pkg/mathx.py"
    js = tc.parse_stacktrace("TypeError: x is undefined\n    at slugify (/home/u/proj/src/slug.js:2:5)\n")
    assert js["frames"] and js["exception"].startswith("TypeError")


def test_code_search_literal(tc):
    r = tc.code_search("return a - b", k=3)
    assert r["results"][0]["file"] == "pkg/mathx.py" and r["results"][0]["line"] == 2


def test_is_test_path():
    assert is_test_path("tests/test_x.py") and is_test_path("src/a.test.js") and is_test_path("conftest.py")
    assert not is_test_path("pkg/mathx.py")


def test_rag_index_incremental_and_query(tc, sample_repo):
    stats = tc.index_repo()
    assert stats["code_chunks"] >= 3 and stats["doc_chunks"] >= 1
    assert tc.index_repo()["files_changed"] == 0  # incremental: nothing changed
    (sample_repo / "pkg" / "extra.py").write_text("def subtract(a, b):\n    return a - b\n")
    assert tc.index_repo()["files_changed"] == 1
    chunks = tc.rag_retrieve("add numbers", k=3, collection="code")["chunks"]
    assert chunks and all(c["id"].startswith("code:") for c in chunks)
    tc.record_fix("t9", "add broken", "subtracts", "diff", ["pkg/mathx.py"])
    assert tc.rag_retrieve("add broken", 1, "bugs")["chunks"][0]["id"] == "bugs:t9"
