import json
import time

import pytest

from codefix.blackboard import Diagnosis, DispatchRef, Envelope, Patch, TestReport
from codefix.config import ConfigError, Settings, _validate


def _run(bb, tid="t1"):
    bb.create_run(tid, "add is broken", "/repo", "hierarchical", max_attempts=3)
    return tid


def test_create_and_snapshot(bb):
    tid = _run(bb)
    snap = bb.snapshot(tid)
    assert snap["run"]["status"] == "running"
    assert snap["diagnosis"] is None and snap["messages"] == []


def test_post_stores_artifact_and_updates_pointer(bb):
    tid = _run(bb)
    env = bb.post(Envelope(task_id=tid, sender="diagnoser", target_agent="manager", status="success",
                           payload=Diagnosis(root_cause="x", confidence=0.8), confidence=0.8))
    assert env.artifact_id and env.artifact_id.startswith("diag-")
    assert bb.get_run(tid)["diagnosis_id"] == env.artifact_id
    assert isinstance(bb.current(tid, "diagnosis"), Diagnosis)
    assert bb.snapshot(tid)["diagnosis"]["root_cause"] == "x"


def test_dispatch_does_not_create_artifact(bb):
    tid = _run(bb)
    env = bb.post(Envelope(task_id=tid, sender="manager", target_agent="fixer", status="dispatched",
                           payload=DispatchRef(instruction="fix", artifact_refs={"diagnosis": "diag-1"}),
                           confidence=1))
    assert env.artifact_id is None
    assert bb.artifacts(tid) == []


def test_latency_filled_from_matching_dispatch(bb):
    tid = _run(bb)
    bb.post(Envelope(task_id=tid, sender="manager", target_agent="fixer", status="dispatched",
                     payload=DispatchRef(instruction="fix"), confidence=1))
    time.sleep(0.05)
    done = bb.post(Envelope(task_id=tid, sender="fixer", target_agent="manager", status="success",
                            payload=Patch(diff="d", rationale="r", attempt_no=1), confidence=0.7))
    assert done.latency_ms is not None and done.latency_ms >= 40
    msgs = bb.messages(tid)
    assert [m["status"] for m in msgs] == ["dispatched", "success"]
    assert msgs[1]["delta_ms"] >= 40


def test_message_log_is_append_only_and_ordered(bb):
    tid = _run(bb)
    for i in range(3):
        bb.post(Envelope(task_id=tid, sender="tester", target_agent="manager", status="failure",
                         payload=TestReport(passed=False, failed=i + 1), confidence=1))
    msgs = bb.messages(tid)
    assert [m["payload"]["failed"] for m in msgs] == [1, 2, 3]
    assert len(bb.artifacts(tid, "test_report")) == 3
    assert bb.current(tid, "test_report").failed == 3  # latest pointer
    assert bb.messages(tid, since_id=msgs[0]["id"])[0]["payload"]["failed"] == 2


def test_attempt_counter_and_cost_accumulates(bb):
    tid = _run(bb)
    assert bb.increment_attempts(tid) == 1
    assert bb.increment_attempts(tid) == 2
    bb.record_llm_call(tid, agent="fixer", model="m", input_tokens=10, output_tokens=5, cost_usd=0.25)
    bb.record_llm_call(tid, agent="fixer", model="m", input_tokens=10, output_tokens=5, cost_usd=0.25)
    assert bb.get_run(tid)["cost_usd"] == pytest.approx(0.5)
    assert bb.snapshot(tid)["llm_usage"][0]["n"] == 2


def test_trace_jsonl_written(bb):
    tid = _run(bb)
    bb.post(Envelope(task_id=tid, sender="manager", target_agent="diagnoser", status="dispatched",
                     payload=DispatchRef(instruction="go"), confidence=1))
    lines = bb.trace.path(tid).read_text().strip().splitlines()
    types = [json.loads(line)["type"] for line in lines]
    assert types == ["run_started", "envelope"]


def test_update_run_rejects_unknown_fields(bb):
    tid = _run(bb)
    with pytest.raises(ValueError):
        bb.update_run(tid, drop_table=1)


def test_query_is_read_only(bb):
    with pytest.raises(ValueError):
        bb.query("DELETE FROM runs")


def test_judge_must_differ_from_generators():
    roles = {r: {"model": m} for r, m in
             [("manager", "h"), ("diagnoser", "o"), ("fixer", "s"), ("judge", "s")]}
    with pytest.raises(ConfigError):
        _validate(Settings.model_validate({"roles": roles}))
    roles["judge"] = {"model": "f"}
    assert _validate(Settings.model_validate({"roles": roles}))
