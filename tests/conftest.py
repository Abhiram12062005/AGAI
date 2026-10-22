import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The suite must pass with no API key and no network model downloads.
os.environ.pop("ANTHROPIC_API_KEY", None)
os.environ["CODEFIX_EMBEDDER"] = "hash"


@pytest.fixture()
def bb(tmp_path):
    from codefix.blackboard import Blackboard

    return Blackboard(tmp_path / "runs.db", tmp_path / "traces")


@pytest.fixture()
def sample_repo(tmp_path):
    """A tiny git repo with one buggy function and a failing pytest test."""
    import subprocess

    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "pkg" / "__init__.py").write_text("")
    (repo / "pkg" / "mathx.py").write_text(
        "def add(a, b):\n    return a - b\n\n\ndef mul(a, b):\n    return a * b\n"
    )
    (repo / "tests" / "test_mathx.py").write_text(
        "from pkg.mathx import add, mul\n\n\ndef test_add():\n    assert add(2, 3) == 5\n\n\n"
        "def test_mul():\n    assert mul(2, 3) == 6\n"
    )
    (repo / "README.md").write_text("# mathx\nTiny math helpers. `add` adds two numbers.\n")
    (repo / ".env").write_text("SECRET=1\n")
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t"}
    for cmd in (["git", "init", "-q", "-b", "main"], ["git", "add", "-A"], ["git", "commit", "-qm", "init"]):
        subprocess.run(cmd, cwd=repo, check=True, env=env)
    return repo


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    """Point every persistent path (db, traces, checkpoints, index, sandbox) at tmp_path."""
    from codefix import config, mcp_client
    from codefix.graph import runner

    monkeypatch.setenv("CODEFIX_MCP_TRANSPORT", "inprocess")
    for name, val in (("DB_PATH", tmp_path / "runs.db"), ("TRACES_DIR", tmp_path / "traces"),
                      ("CHECKPOINT_DB", tmp_path / "ckpt.db"), ("INDEX_DIR", tmp_path / "index"),
                      ("SANDBOX_DIR", tmp_path / "sandbox")):
        monkeypatch.setattr(config, name, val)
    monkeypatch.setattr(runner, "_BB", None)
    monkeypatch.setattr(runner, "_CKPT", None)
    monkeypatch.setattr(mcp_client, "_CACHE", {})
    return tmp_path
