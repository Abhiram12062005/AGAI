"""Structured JSONL tracing: one file per run at traces/<task_id>.jsonl.

Every envelope, tool call, LLM call (model/tokens/cost), ReAct step and lifecycle event is
appended as a single JSON line. The web UI tails these files for live updates (SSE).
"""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


def _default(o: Any) -> Any:
    if isinstance(o, datetime):
        return o.isoformat()
    if hasattr(o, "model_dump"):
        return o.model_dump(mode="json")
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


class TraceWriter:
    def __init__(self, traces_dir: Path):
        self.dir = Path(traces_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._last_ts: dict[str, float] = {}

    def path(self, task_id: str) -> Path:
        return self.dir / f"{task_id}.jsonl"

    def emit(self, task_id: str, event_type: str, **data: Any) -> dict:
        now = time.time()
        with _LOCK:
            prev = self._last_ts.get(task_id)
            self._last_ts[task_id] = now
            record = {
                "ts": datetime.fromtimestamp(now, UTC).isoformat(),
                "task_id": task_id,
                "type": event_type,
                # latency delta between consecutive trace events of this run
                "delta_ms": None if prev is None else int((now - prev) * 1000),
                **data,
            }
            with self.path(task_id).open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=_default) + "\n")
        return record

    def read(self, task_id: str, offset: int = 0) -> list[dict]:
        p = self.path(task_id)
        if not p.exists():
            return []
        out = []
        with p.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < offset or not line.strip():
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    break  # partially written last line; caller retries later
        return out
