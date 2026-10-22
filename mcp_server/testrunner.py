"""Test-suite + static-analysis execution (pytest / jest / node:test, ruff / eslint).

Everything runs in a subprocess with a hard timeout and captured output. Results are
normalised into the TestReport shape via JUnit XML (pytest, node:test) or jest --json.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from mcp_server.sandbox import iter_files, run

MAX_TRACE = 6000
# flake8's "real error" selection: syntax errors, undefined names, invalid comparisons...
GATING_RUFF_PREFIXES = ("E9", "F63", "F7", "F82")


def detect_project(root: Path) -> str:
    root = Path(root)
    pkg = root / "package.json"
    if pkg.exists():
        try:
            data = json.loads(pkg.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
        deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
        if "jest" in deps or "jest" in data.get("scripts", {}).get("test", ""):
            return "jest"
        return "node"
    if any((root / f).exists() for f in ("pyproject.toml", "setup.py", "setup.cfg", "pytest.ini", "tox.ini")):
        return "pytest"
    if any(rel.endswith(".py") for _, rel in iter_files(root)):
        return "pytest"
    return "unknown"


def _test_env(cwd: Path) -> dict:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(cwd), "CI": "1",
           "NO_COLOR": "1", "FORCE_COLOR": "0"}
    for secret in ("ANTHROPIC_API_KEY", "GITHUB_TOKEN", "GH_TOKEN"):
        env.pop(secret, None)  # the code under test never sees our credentials
    return env


def _parse_junit(path: Path) -> tuple[int, int, list[str], str]:
    tree = ET.parse(path)
    total, failed, assertions, traces = 0, 0, [], []
    for case in tree.iter("testcase"):
        total += 1
        bad = case.find("failure")
        if bad is None:
            bad = case.find("error")
        if bad is not None:
            failed += 1
            name = f"{case.get('classname', '')}::{case.get('name', '')}".strip(":")
            msg = (bad.get("message") or "").strip().splitlines()
            assertions.append(f"{name}: {msg[0] if msg else bad.tag}"[:400])
            traces.append(f"--- {name}\n{(bad.text or '').strip()}")
    return total, failed, assertions, "\n".join(traces)[-MAX_TRACE:]


def run_tests(cwd: Path, selector: str | None = None, timeout_s: int = 180) -> dict:
    cwd = Path(cwd)
    kind = detect_project(cwd)
    junit = cwd / ".codefix-junit.xml"
    junit.unlink(missing_ok=True)
    started = time.time()
    if kind == "pytest":
        argv = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={junit}",
                "-o", "junit_family=xunit1"]
        if selector:
            argv.append(selector)
    elif kind == "node":
        argv = ["node", "--test", "--test-reporter=spec", "--test-reporter-destination=stdout",
                "--test-reporter=junit", f"--test-reporter-destination={junit}"]
        if selector:
            argv.append(f"--test-name-pattern={selector}")
    elif kind == "jest":
        jest = cwd / "node_modules" / ".bin" / ("jest.cmd" if os.name == "nt" else "jest")
        argv = [str(jest) if jest.exists() else "npx", *([] if jest.exists() else ["jest"]),
                "--ci", "--json", f"--outputFile={junit.with_suffix('.json')}"]
        if selector:
            argv += ["-t", selector]
    else:
        return {"passed": False, "total": 0, "failed": 0, "failing_assertions": [],
                "stack_trace": "No supported test framework detected", "duration_ms": 0,
                "failure_stage": "infra", "framework": kind, "output": ""}
    try:
        cp = run(argv, cwd=cwd, timeout=timeout_s, env=_test_env(cwd))
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") if isinstance(e.stdout, str) else ""
        return {"passed": False, "total": 0, "failed": 0, "failing_assertions": [],
                "stack_trace": f"Test run exceeded {timeout_s}s timeout\n{out[-2000:]}",
                "duration_ms": int((time.time() - started) * 1000), "failure_stage": "timeout",
                "framework": kind, "output": out[-4000:]}
    duration = int((time.time() - started) * 1000)
    output = (cp.stdout + "\n" + cp.stderr)[-8000:]
    total = failed = 0
    assertions: list[str] = []
    trace = ""
    if kind == "jest":
        jf = junit.with_suffix(".json")
        if jf.exists():
            data = json.loads(jf.read_text(encoding="utf-8"))
            total, failed = data.get("numTotalTests", 0), data.get("numFailedTests", 0)
            for suite in data.get("testResults", []):
                for t in suite.get("assertionResults", []):
                    if t.get("status") == "failed":
                        assertions.append(f"{t.get('fullName')}: {(t.get('failureMessages') or [''])[0][:200]}")
                        trace += "\n".join(t.get("failureMessages") or []) + "\n"
            jf.unlink(missing_ok=True)
    elif junit.exists():
        try:
            total, failed, assertions, trace = _parse_junit(junit)
        except ET.ParseError:
            pass
        junit.unlink(missing_ok=True)
    passed = cp.returncode == 0 and failed == 0 and total > 0
    stage = None
    if not passed:
        stage = "tests" if total > 0 else "infra"  # infra: collection/import error, no tests ran
        if not trace:
            trace = output[-MAX_TRACE:]
    return {"passed": passed, "total": total, "failed": failed if total else 0,
            "failing_assertions": assertions[:30], "stack_trace": trace[-MAX_TRACE:],
            "duration_ms": duration, "failure_stage": stage, "framework": kind,
            "exit_code": cp.returncode, "output": output[-4000:]}


def static_analysis(cwd: Path, paths: list[str]) -> list[dict]:
    cwd = Path(cwd)
    py = [p for p in paths if p.endswith(".py") and (cwd / p).exists()]
    js = [p for p in paths if p.endswith((".js", ".mjs", ".cjs", ".ts")) and (cwd / p).exists()]
    findings: list[dict] = []
    if py:
        cp = run([sys.executable, "-m", "ruff", "check", "--no-cache", "--isolated", "--output-format",
                  "json", "--select", "E9,F", *py], cwd=cwd, timeout=60)
        try:
            for f in json.loads(cp.stdout or "[]"):
                code = f.get("code") or "E999"
                findings.append({
                    "path": Path(f["filename"]).resolve().relative_to(cwd.resolve()).as_posix(),
                    "line": (f.get("location") or {}).get("row", 0), "code": code,
                    "message": f.get("message", ""), "gating": code.startswith(GATING_RUFF_PREFIXES),
                })
        except (json.JSONDecodeError, ValueError):
            findings.append({"path": ",".join(py), "line": 0, "code": "RUFF", "message": cp.stderr[:300],
                             "gating": False})
    if js:
        eslint = cwd / "node_modules" / ".bin" / ("eslint.cmd" if os.name == "nt" else "eslint")
        if eslint.exists():
            cp = run([str(eslint), "-f", "json", *js], cwd=cwd, timeout=120)
            try:
                for file in json.loads(cp.stdout or "[]"):
                    for m in file.get("messages", []):
                        findings.append({
                            "path": Path(file["filePath"]).resolve().relative_to(cwd.resolve()).as_posix(),
                            "line": m.get("line", 0), "code": m.get("ruleId") or "eslint",
                            "message": m.get("message", ""), "gating": m.get("severity") == 2,
                        })
            except json.JSONDecodeError:
                pass
        elif shutil.which("node"):
            for p in js:  # no eslint installed: fall back to a syntax check
                if p.endswith(".ts"):
                    continue
                cp = run(["node", "--check", p], cwd=cwd, timeout=30)
                if cp.returncode != 0:
                    findings.append({"path": p, "line": 0, "code": "node-syntax",
                                     "message": cp.stderr.strip().splitlines()[-1][:300] if cp.stderr else "",
                                     "gating": True})
    return findings
