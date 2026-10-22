"""Retry / re-delegation / escalation logic for all three architectures (mock LLM, no API key)."""

import json

import pytest

from codefix import config
from codefix.agents.base import FixerOutput, ManagerDecision, ReActStep
from codefix.graph import runner

WRONG = {"rationale": "use multiplication", "edits": [{"path": "pkg/mathx.py", "find": "return a - b",
                                                        "replace": "return a * b"}], "citations": [], "confidence": 0.5}
RIGHT = {"rationale": "add should add", "edits": [{"path": "pkg/mathx.py", "find": "return a - b",
                                                    "replace": "return a + b"}], "citations": ["file:pkg/mathx.py"],
         "confidence": 0.9}


def final(files=("pkg/mathx.py",), conf=0.9):
    return {"thought": "add() uses subtraction", "action": "final_answer", "action_input": "{}",
            "final_answer": {"root_cause": "add() returns a - b instead of a + b", "implicated_files": list(files),
                             "implicated_symbols": ["add"], "evidence": ["pkg/mathx.py:2"], "confidence": conf,
                             "citations": ["code:pkg/mathx.py:1-2", "bogus:id"]}}


class Script:
    """Mock handler: diagnoser reads the file then answers; fixer follows `fixes` in order."""

    def __init__(self, fixes, manager=None, diag_conf=0.9, react_loop=False):
        self.fixes, self.manager, self.diag_conf, self.react_loop = list(fixes), manager, diag_conf, react_loop
        self.fixer_prompts, self.diag_calls, self.manager_calls = [], 0, 0
        self._diag_step = 0

    def __call__(self, role, system, messages, schema, n):
        if schema is ReActStep:
            self.diag_calls += 1
            first = len(messages) == 1
            if first or self.react_loop:
                return {"thought": "read the code", "action": "read_file",
                        "action_input": json.dumps({"path": "pkg/mathx.py"}), "final_answer": None}
            return final(conf=self.diag_conf)
        if schema is FixerOutput:
            self.fixer_prompts.append(messages[0]["content"])
            return self.fixes.pop(0) if len(self.fixes) > 1 else self.fixes[0]
        if schema is ManagerDecision:
            self.manager_calls += 1
            return self.manager or {"decision": "fixer", "reason": "retry", "confidence": 0.8}
        raise AssertionError(schema)


pytestmark = pytest.mark.usefixtures("isolated")


def _run(repo, arch, script, **kw):
    return runner.run_pipeline("add(2, 3) returns -1 instead of 5", repo, arch=arch, llm_mode="mock",
                               mock_handler=script, **kw)


def test_hierarchical_retries_fixer_with_feedback_then_passes(sample_repo):
    s = Script([WRONG, RIGHT])
    snap = _run(sample_repo, "hierarchical", s)
    run = snap["run"]
    assert run["status"] == "awaiting_approval" and run["outcome"] == "passed"
    assert run["attempts"] == 2 and s.diag_calls == 2  # diagnosed ONCE (1 tool step + final)
    assert "PREVIOUS DIFF" not in s.fixer_prompts[0] and "PREVIOUS DIFF" in s.fixer_prompts[1]
    assert "return a * b" in s.fixer_prompts[1]  # failed diff was attached to the retry
    dispatches = [m for m in snap["messages"] if m["status"] == "dispatched" and m["sender"] == "manager"]
    assert [m["target_agent"] for m in dispatches] == ["diagnoser", "fixer", "tester", "fixer", "tester", "pr"]
    retry = dispatches[3]["payload"]
    assert set(retry["artifact_refs"]) == {"diagnosis", "patch", "test_report"}
    assert "diff" not in json.dumps(retry)  # refs only, never content
    assert snap["pr"]["branch"] == f"codefix/{run['task_id']}" and snap["pr"]["pushed"] is False
    # citations validated against retrieved ids
    assert "bogus:id" not in snap["diagnosis"]["citations"]
    # completion latency recorded per agent
    assert all(m["latency_ms"] is not None for m in snap["messages"]
               if m["status"] in ("success", "failure") and m["sender"] != "manager")


def test_hierarchical_escalates_after_max_attempts(sample_repo):
    snap = _run(sample_repo, "hierarchical", Script([WRONG]), max_attempts=3)
    run = snap["run"]
    assert run["status"] == "escalated" and run["attempts"] == 3 and run["failure_stage"] == "tests"
    esc = snap["escalation"]
    assert esc["attempts"] == 3 and len(esc["tried"]) == 3
    assert sum(1 for m in snap["messages"] if m["sender"] == "fixer" and m["status"] != "in_progress") == 3


def test_configurable_attempt_limit(sample_repo):
    snap = _run(sample_repo, "hierarchical", Script([WRONG]), max_attempts=1)
    assert snap["run"]["attempts"] == 1 and snap["run"]["status"] == "escalated"


def test_sequential_restarts_from_scratch_without_feedback(sample_repo):
    s = Script([WRONG, WRONG, RIGHT])
    snap = _run(sample_repo, "sequential", s)
    assert snap["run"]["outcome"] == "passed" and snap["run"]["attempts"] == 3
    assert s.diag_calls == 6  # re-diagnosed on every restart
    assert all("PREVIOUS DIFF" not in p for p in s.fixer_prompts)
    assert not any(m["sender"] == "manager" for m in snap["messages"])


def test_p2p_hands_feedback_directly_no_manager(sample_repo):
    s = Script([WRONG, RIGHT])
    snap = _run(sample_repo, "p2p", s)
    assert snap["run"]["outcome"] == "passed" and s.diag_calls == 2
    assert "PREVIOUS DIFF" in s.fixer_prompts[1]
    senders = {(m["sender"], m["target_agent"]) for m in snap["messages"] if m["status"] in ("success", "failure")}
    assert ("tester", "fixer") in senders and ("diagnoser", "fixer") in senders
    assert not any(m["sender"] == "manager" for m in snap["messages"])


def test_p2p_and_sequential_bounded(sample_repo):
    for arch in ("p2p", "sequential"):
        snap = _run(sample_repo, arch, Script([WRONG]), max_attempts=2)
        assert snap["run"]["status"] == "escalated" and snap["run"]["attempts"] == 2


def test_manager_llm_invalid_decision_falls_back(sample_repo):
    s = Script([WRONG, RIGHT], manager={"decision": "launch_rockets", "reason": "?", "confidence": 1})
    snap = _run(sample_repo, "hierarchical", s, extra={"mock_manager_llm": True})
    assert s.manager_calls >= 1 and snap["run"]["outcome"] == "passed"


def test_manager_can_escalate_early(sample_repo):
    s = Script([WRONG], manager={"decision": "escalate", "reason": "infra", "confidence": 1})
    snap = _run(sample_repo, "hierarchical", s, extra={"mock_manager_llm": True})
    assert snap["run"]["status"] == "escalated" and snap["run"]["attempts"] == 1


def test_weak_diagnosis_triggers_one_rediagnosis(sample_repo):
    s = Script([RIGHT], diag_conf=0.1)
    snap = _run(sample_repo, "hierarchical", s)
    targets = [m["target_agent"] for m in snap["messages"] if m["status"] == "dispatched" and m["sender"] == "manager"]
    assert targets[:3] == ["diagnoser", "diagnoser", "fixer"]  # exactly one re-diagnosis, then proceed
    assert snap["run"]["outcome"] == "passed"


def test_react_cap_degrades_gracefully(sample_repo):
    s = Script([RIGHT], react_loop=True)
    snap = _run(sample_repo, "hierarchical", s)
    d = snap["diagnosis"]
    assert d["degraded"] is True and d["react_steps"] == config.settings().pipeline.react_max_iterations + 1
    assert "pkg/mathx.py" in d["implicated_files"]


def test_test_file_edits_rejected(sample_repo):
    cheat = {"rationale": "change test", "edits": [{"path": "tests/test_mathx.py", "find": "== 5",
                                                    "replace": "== -1"}], "citations": [], "confidence": 1}
    snap = _run(sample_repo, "hierarchical", Script([cheat]), max_attempts=1)
    assert snap["run"]["status"] == "escalated"
    assert snap["test_report"]["failure_stage"] == "patch"


def test_pr_gate_resume(sample_repo):
    snap = _run(sample_repo, "hierarchical", Script([RIGHT]))
    tid = snap["run"]["task_id"]
    with pytest.raises(KeyError):
        runner.resolve_pr(tid + "x", True)
    out = runner.resolve_pr(tid, confirmed=True)
    assert out["run"]["status"] == "done"
    assert "origin" in out["pr"]["note"]  # sample repo has no remote: nothing pushed
    with pytest.raises(ValueError):
        runner.resolve_pr(tid, confirmed=True)  # cannot resolve twice


def test_trace_has_all_event_types(sample_repo):
    snap = _run(sample_repo, "hierarchical", Script([WRONG, RIGHT]))
    tid = snap["run"]["task_id"]
    types = {json.loads(line)["type"] for line in (config.TRACES_DIR / f"{tid}.jsonl").read_text().splitlines()}
    assert {"run_started", "envelope", "tool_call", "llm_call", "react_step", "manager_decision",
            "run_finished"} <= types
    assert (config.TRACES_DIR / f"{tid}.react.md").exists()
