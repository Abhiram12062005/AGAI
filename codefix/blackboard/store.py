"""SQLite-backed Blackboard: the single source of truth for pipeline state.

Keyed by task_id. Holds the bug report, current Diagnosis, active Patch, latest TestReport,
attempt counter and the full append-only message log. In hierarchical mode agents read and
write *only* through this object. `snapshot(task_id)` dumps the complete pipeline state at
any instant — the observability audit point.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from codefix.blackboard.schemas import PAYLOAD_TYPES, Envelope
from codefix.observability.tracer import TraceWriter

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    task_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    bug_report TEXT NOT NULL,
    stack_trace TEXT DEFAULT '',
    repo TEXT NOT NULL,
    arch TEXT NOT NULL,
    llm_mode TEXT NOT NULL DEFAULT 'live',
    status TEXT NOT NULL DEFAULT 'pending',      -- pending|running|awaiting_approval|done|escalated|failed|error
    outcome TEXT,                                -- passed|escalated|failed|error
    failure_stage TEXT,                          -- diagnosis|patch|apply|lint|tests|timeout|infra|error
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    diagnosis_id TEXT,
    patch_id TEXT,
    test_report_id TEXT,
    escalation_id TEXT,
    pr_id TEXT,
    cost_usd REAL NOT NULL DEFAULT 0,
    duration_ms INTEGER,
    bench_id TEXT,
    bug_id TEXT,
    meta TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    body TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    sender TEXT NOT NULL,
    target_agent TEXT NOT NULL,
    status TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    artifact_id TEXT,
    confidence REAL NOT NULL,
    timestamp TEXT NOT NULL,
    latency_ms INTEGER,
    delta_ms INTEGER
);
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    agent TEXT NOT NULL,
    model TEXT NOT NULL,
    served_model TEXT,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0,
    latency_ms INTEGER,
    stop_reason TEXT,
    error TEXT,
    timestamp TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    agent TEXT NOT NULL,
    tool TEXT NOT NULL,
    args TEXT,
    ok INTEGER NOT NULL,
    latency_ms INTEGER,
    result_preview TEXT,
    timestamp TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    judge_model TEXT NOT NULL,
    verdict TEXT NOT NULL,
    score REAL,
    reasoning TEXT,
    objective_pass INTEGER,
    target_test_pass INTEGER,
    agreement INTEGER,
    cost_usd REAL DEFAULT 0,
    timestamp TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_messages_task ON messages(task_id, id);
CREATE INDEX IF NOT EXISTS ix_llm_task ON llm_calls(task_id);
CREATE INDEX IF NOT EXISTS ix_tool_task ON tool_calls(task_id);
CREATE INDEX IF NOT EXISTS ix_eval_task ON evaluations(task_id);
"""

_POINTER_FOR_KIND = {
    "diagnosis": "diagnosis_id",
    "patch": "patch_id",
    "test_report": "test_report_id",
    "escalation": "escalation_id",
    "pr_result": "pr_id",
}

RUN_FIELDS = {
    "status", "outcome", "failure_stage", "attempts", "max_attempts", "diagnosis_id",
    "patch_id", "test_report_id", "escalation_id", "pr_id", "cost_usd", "duration_ms",
    "bench_id", "bug_id", "meta", "llm_mode",
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def new_task_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


class Blackboard:
    def __init__(self, db_path: Path | str, traces_dir: Path | str):
        self.db_path = str(db_path)
        self.trace = TraceWriter(Path(traces_dir))
        self._wlock = threading.RLock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    # ------------------------------------------------------------------ plumbing
    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ runs
    def create_run(
        self,
        task_id: str,
        bug_report: str,
        repo: str,
        arch: str,
        max_attempts: int,
        stack_trace: str = "",
        llm_mode: str = "live",
        **extra: Any,
    ) -> None:
        now = _now()
        meta = extra.pop("meta", {})
        with self._wlock, self._conn() as c:
            c.execute(
                "INSERT INTO runs(task_id, created_at, updated_at, bug_report, stack_trace, repo, arch,"
                " llm_mode, status, max_attempts, bench_id, bug_id, meta)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (task_id, now, now, bug_report, stack_trace, repo, arch, llm_mode, "running",
                 max_attempts, extra.get("bench_id"), extra.get("bug_id"), json.dumps(meta)),
            )
        self.trace.emit(task_id, "run_started", repo=repo, arch=arch, llm_mode=llm_mode,
                        bug_report=bug_report, max_attempts=max_attempts)

    def update_run(self, task_id: str, **fields: Any) -> None:
        bad = set(fields) - RUN_FIELDS
        if bad:
            raise ValueError(f"unknown run fields {bad}")
        if "meta" in fields and not isinstance(fields["meta"], str):
            fields["meta"] = json.dumps(fields["meta"])
        cols = ", ".join(f"{k}=?" for k in fields) + ", updated_at=?"
        with self._wlock, self._conn() as c:
            c.execute(f"UPDATE runs SET {cols} WHERE task_id=?", (*fields.values(), _now(), task_id))

    def get_run(self, task_id: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM runs WHERE task_id=?", (task_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["meta"] = json.loads(d.get("meta") or "{}")
        return d

    def list_runs(self, outcome: str | None = None, arch: str | None = None,
                  bench_id: str | None = None, limit: int = 500) -> list[dict]:
        q, args = "SELECT * FROM runs WHERE 1=1", []
        for col, val in (("outcome", outcome), ("arch", arch), ("bench_id", bench_id)):
            if val:
                q += f" AND {col}=?"
                args.append(val)
        q += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        with self._conn() as c:
            return [dict(r) for r in c.execute(q, args).fetchall()]

    def increment_attempts(self, task_id: str) -> int:
        with self._wlock, self._conn() as c:
            c.execute("UPDATE runs SET attempts = attempts + 1, updated_at=? WHERE task_id=?",
                      (_now(), task_id))
            return c.execute("SELECT attempts FROM runs WHERE task_id=?", (task_id,)).fetchone()[0]

    # ------------------------------------------------------------------ artifacts
    def put_artifact(self, task_id: str, payload: BaseModel) -> str:
        kind = payload.kind  # type: ignore[attr-defined]
        art_id = f"{kind[:4]}-{uuid.uuid4().hex[:8]}"
        with self._wlock, self._conn() as c:
            c.execute(
                "INSERT INTO artifacts(id, task_id, kind, body, created_at) VALUES (?,?,?,?,?)",
                (art_id, task_id, kind, payload.model_dump_json(), _now()),
            )
            pointer = _POINTER_FOR_KIND.get(kind)
            if pointer:
                c.execute(f"UPDATE runs SET {pointer}=?, updated_at=? WHERE task_id=?",
                          (art_id, _now(), task_id))
        return art_id

    def get_artifact(self, artifact_id: str | None) -> BaseModel | None:
        if not artifact_id:
            return None
        with self._conn() as c:
            row = c.execute("SELECT kind, body FROM artifacts WHERE id=?", (artifact_id,)).fetchone()
        if not row:
            return None
        return PAYLOAD_TYPES[row["kind"]].model_validate_json(row["body"])

    def artifacts(self, task_id: str, kind: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM artifacts WHERE task_id=?", [task_id]
        if kind:
            q += " AND kind=?"
            args.append(kind)
        with self._conn() as c:
            rows = c.execute(q + " ORDER BY created_at", args).fetchall()
        return [{**dict(r), "body": json.loads(r["body"])} for r in rows]

    def current(self, task_id: str, kind: str) -> BaseModel | None:
        run = self.get_run(task_id) or {}
        return self.get_artifact(run.get(_POINTER_FOR_KIND[kind]))

    # ------------------------------------------------------------------ messages
    def post(self, env: Envelope) -> Envelope:
        """Append an envelope to the message log (and store its artifact payload).

        On a completion message (success/failure) latency_ms is filled in as the time
        since the matching dispatch to the sending agent.
        """
        Envelope.model_validate(env.model_dump())  # strict re-validation at the boundary
        kind = env.payload.kind
        with self._wlock:
            if kind in _POINTER_FOR_KIND and env.artifact_id is None:
                env.artifact_id = self.put_artifact(env.task_id, env.payload)
            with self._conn() as c:
                prev = c.execute(
                    "SELECT timestamp FROM messages WHERE task_id=? ORDER BY id DESC LIMIT 1",
                    (env.task_id,),
                ).fetchone()
                delta = None
                if prev:
                    delta = int((env.timestamp - datetime.fromisoformat(prev[0])).total_seconds() * 1000)
                if env.status in ("success", "failure") and env.latency_ms is None:
                    # The hand-off that gave this agent its work: a Manager dispatch (hierarchical)
                    # or a peer's completion message addressed to it (sequential / p2p).
                    disp = c.execute(
                        "SELECT timestamp FROM messages WHERE task_id=? AND target_agent=? AND sender<>?"
                        " AND status IN ('dispatched','success','failure') ORDER BY id DESC LIMIT 1",
                        (env.task_id, env.sender, env.sender),
                    ).fetchone()
                    if disp:
                        env.latency_ms = int(
                            (env.timestamp - datetime.fromisoformat(disp[0])).total_seconds() * 1000)
                c.execute(
                    "INSERT INTO messages(task_id, sender, target_agent, status, kind, payload, artifact_id,"
                    " confidence, timestamp, latency_ms, delta_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (env.task_id, env.sender, env.target_agent, env.status, kind,
                     env.payload.model_dump_json(), env.artifact_id, env.confidence,
                     env.timestamp.isoformat(), env.latency_ms, delta),
                )
        self.trace.emit(env.task_id, "envelope", envelope=env.model_dump(mode="json"), msg_delta_ms=delta)
        return env

    def messages(self, task_id: str, since_id: int = 0) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM messages WHERE task_id=? AND id>? ORDER BY id",
                             (task_id, since_id)).fetchall()
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]

    # ------------------------------------------------------------------ telemetry rows
    def record_llm_call(self, task_id: str, **row: Any) -> None:
        row.setdefault("timestamp", _now())
        cols = ", ".join(row)
        with self._wlock, self._conn() as c:
            c.execute(f"INSERT INTO llm_calls(task_id, {cols}) VALUES (?{',?' * len(row)})",
                      (task_id, *row.values()))
            c.execute("UPDATE runs SET cost_usd = cost_usd + ? WHERE task_id=?",
                      (row.get("cost_usd", 0.0) or 0.0, task_id))

    def record_tool_call(self, task_id: str, **row: Any) -> None:
        row.setdefault("timestamp", _now())
        cols = ", ".join(row)
        with self._wlock, self._conn() as c:
            c.execute(f"INSERT INTO tool_calls(task_id, {cols}) VALUES (?{',?' * len(row)})",
                      (task_id, *row.values()))

    def record_evaluation(self, task_id: str, **row: Any) -> None:
        row.setdefault("timestamp", _now())
        cols = ", ".join(row)
        with self._wlock, self._conn() as c:
            c.execute("DELETE FROM evaluations WHERE task_id=?", (task_id,))
            c.execute(f"INSERT INTO evaluations(task_id, {cols}) VALUES (?{',?' * len(row)})",
                      (task_id, *row.values()))

    def query(self, sql: str, args: tuple = ()) -> list[dict]:
        """Read-only helper for dashboards/reports."""
        if not sql.lstrip().upper().startswith("SELECT"):
            raise ValueError("query() is read-only")
        with self._conn() as c:
            return [dict(r) for r in c.execute(sql, args).fetchall()]

    # ------------------------------------------------------------------ audit point
    def snapshot(self, task_id: str) -> dict:
        """Complete pipeline state at this instant (observability audit point)."""
        run = self.get_run(task_id)
        if run is None:
            raise KeyError(task_id)

        def dump(art_id: str | None) -> dict | None:
            a = self.get_artifact(art_id)
            return a.model_dump(mode="json") if a else None

        with self._conn() as c:
            llm = c.execute(
                "SELECT agent, COUNT(*) n, SUM(input_tokens) inp, SUM(output_tokens) outp,"
                " SUM(cost_usd) cost, SUM(latency_ms) lat FROM llm_calls WHERE task_id=? GROUP BY agent",
                (task_id,)).fetchall()
            tools = c.execute("SELECT COUNT(*) FROM tool_calls WHERE task_id=?", (task_id,)).fetchone()[0]
            ev = c.execute("SELECT * FROM evaluations WHERE task_id=?", (task_id,)).fetchone()
        return {
            "run": run,
            "bug_report": run["bug_report"],
            "attempts": run["attempts"],
            "diagnosis": dump(run["diagnosis_id"]),
            "patch": dump(run["patch_id"]),
            "test_report": dump(run["test_report_id"]),
            "escalation": dump(run["escalation_id"]),
            "pr": dump(run["pr_id"]),
            "messages": self.messages(task_id),
            "llm_usage": [dict(r) for r in llm],
            "tool_call_count": tools,
            "evaluation": dict(ev) if ev else None,
        }
