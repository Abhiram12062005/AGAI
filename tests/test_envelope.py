import pytest
from pydantic import ValidationError

from codefix.blackboard import Diagnosis, DispatchRef, Envelope, Patch, TestReport


def _diag(**kw):
    return Diagnosis(root_cause="add() subtracts", implicated_files=["pkg/mathx.py"], confidence=0.9, **kw)


def test_envelope_roundtrip_discriminates_payload():
    env = Envelope(task_id="t-1", sender="diagnoser", target_agent="manager", status="success",
                   payload=_diag(), confidence=0.9)
    again = Envelope.model_validate_json(env.model_dump_json())
    assert isinstance(again.payload, Diagnosis)
    assert again.payload.root_cause == "add() subtracts"


@pytest.mark.parametrize("payload", [
    Patch(diff="--- a\n+++ b\n", rationale="r", attempt_no=1),
    TestReport(passed=True, total=2),
    DispatchRef(instruction="fix it", artifact_refs={"diagnosis": "diag-1"}),
])
def test_all_payload_types_roundtrip(payload):
    env = Envelope(task_id="t1", sender="manager", target_agent="fixer", status="dispatched",
                   payload=payload, confidence=1.0)
    assert type(Envelope.model_validate_json(env.model_dump_json()).payload) is type(payload)


def test_confidence_bounds():
    with pytest.raises(ValidationError):
        Envelope(task_id="t1", sender="manager", target_agent="fixer", status="dispatched",
                 payload=DispatchRef(instruction="x"), confidence=1.5)
    with pytest.raises(ValidationError):
        Diagnosis(root_cause="x", confidence=-0.1)


def test_unknown_status_and_agent_rejected():
    with pytest.raises(ValidationError):
        Envelope(task_id="t1", sender="manager", target_agent="fixer", status="done",
                 payload=DispatchRef(instruction="x"), confidence=1)
    with pytest.raises(ValidationError):
        Envelope(task_id="t1", sender="intern", target_agent="fixer", status="dispatched",
                 payload=DispatchRef(instruction="x"), confidence=1)


def test_extra_fields_forbidden_and_task_id_sanitised():
    with pytest.raises(ValidationError):
        Diagnosis(root_cause="x", confidence=0.5, surprise=True)
    with pytest.raises(ValidationError):
        Envelope(task_id="../etc", sender="manager", target_agent="fixer", status="dispatched",
                 payload=DispatchRef(instruction="x"), confidence=1)


def test_payload_from_dict_uses_kind():
    env = Envelope.model_validate({
        "task_id": "t1", "sender": "tester", "target_agent": "manager", "status": "failure",
        "payload": {"kind": "test_report", "passed": False, "failed": 1}, "confidence": 1,
    })
    assert isinstance(env.payload, TestReport) and env.payload.failed == 1
