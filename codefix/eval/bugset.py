"""Seeded bug set: load, materialise (inject into a pristine repo copy), verify, known-good fix."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from codefix.agents.patching import build_diff
from codefix.config import ROOT
from mcp_server import testrunner
from mcp_server.sandbox import GIT_IDENTITY, rmtree, run

BUGSET_DIR = ROOT / "benchmarks" / "bugset"
REPOS_DIR = BUGSET_DIR / "repos"


@dataclass
class Bug:
    id: str
    repo: str
    difficulty: str
    failing_test: str
    inject: list[dict]
    report: str
    root_cause: str
    with_trace: bool = True
    extra: dict = field(default_factory=dict)

    @property
    def files(self) -> list[str]:
        return sorted({i["file"] for i in self.inject})

    @property
    def fix_edits(self) -> list[dict]:
        return [{"path": i["file"], "find": i["replace"], "replace": i["find"]} for i in reversed(self.inject)]


def load_bugs(ids: list[str] | None = None, path: Path | None = None) -> list[Bug]:
    data = yaml.safe_load((path or BUGSET_DIR / "bugs.yaml").read_text(encoding="utf-8"))
    bugs = [Bug(**{k: v for k, v in b.items() if k in Bug.__dataclass_fields__}) for b in data["bugs"]]
    for b in bugs:
        b.report = " ".join(b.report.split())
    if ids:
        wanted = set(ids)
        bugs = [b for b in bugs if b.id in wanted or any(b.id.startswith(w) for w in wanted)]
    return bugs


def _read(p: Path) -> str:
    with open(p, encoding="utf-8", newline="") as f:
        return f.read().replace("\r\n", "\n")


def materialize(bug: Bug, dest: Path) -> Path:
    """Copy the pristine repo to `dest`, inject the bug, commit it as the repo's HEAD."""
    rmtree(dest)
    shutil.copytree(REPOS_DIR / bug.repo, dest, ignore=shutil.ignore_patterns(
        "__pycache__", ".pytest_cache", "node_modules", ".codefix-junit.xml"))
    originals = {f: _read(dest / f) for f in bug.files}
    edits = [{"path": i["file"], "find": i["find"], "replace": i["replace"]} for i in bug.inject]
    from codefix.agents.patching import apply_edits_in_memory

    for path, text in apply_edits_in_memory(originals, edits).items():
        with open(dest / path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    run(["git", "init", "-q", "-b", "main"], cwd=dest, check=True)
    run(["git", "config", "core.autocrlf", "false"], cwd=dest, check=True)
    run(["git", "add", "-A"], cwd=dest, check=True)
    run(["git", *GIT_IDENTITY, "commit", "-qm", f"bench: {bug.id}"], cwd=dest, check=True)
    return dest


def known_fix_diff(bug: Bug, repo: Path) -> str:
    originals = {f: _read(repo / f) for f in bug.files}
    return build_diff(originals, bug.fix_edits).diff


def failing_output(bug: Bug, repo: Path, timeout_s: int = 120) -> dict:
    """Run the bug's failing test in the (injected) repo — read-only for the repo's sources."""
    r = testrunner.run_tests(repo, selector=bug.failing_test, timeout_s=timeout_s)
    junk = repo / ".codefix-junit.xml"
    if junk.exists():
        os.remove(junk)
    return r


def trace_text(r: dict) -> str:
    return (r.get("stack_trace") or r.get("output") or "")[-3500:]


def verify(bug: Bug, workdir: Path) -> dict:
    """Bug must fail its test when injected and pass with the known-good fix."""
    repo = materialize(bug, workdir / bug.id)
    before = failing_output(bug, repo)
    diff = known_fix_diff(bug, repo)
    pf = repo / ".git" / "fix.patch"
    pf.write_text(diff, encoding="utf-8", newline="\n")
    run(["git", "apply", str(pf)], cwd=repo, check=True)
    after_target = testrunner.run_tests(repo, selector=bug.failing_test)
    after_all = testrunner.run_tests(repo)
    ok = (not before["passed"]) and after_target["passed"] and after_all["passed"]
    rmtree(repo)
    return {"bug": bug.id, "ok": ok, "fails_when_injected": not before["passed"],
            "target_passes_after_fix": after_target["passed"], "suite_passes_after_fix": after_all["passed"]}
