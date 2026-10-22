"""Deterministic diff construction.

DESIGN POINT: the Fixer model emits anchored search/replace edits, and the unified diff is
derived here with difflib. LLM-written hunks with hand-computed line numbers are the #1
cause of "patch does not apply" failures; deriving the diff removes that failure class
while keeping a standard unified diff as the Patch artifact.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path


class EditError(ValueError):
    pass


@dataclass
class EditResult:
    diff: str
    files: list[str]


def apply_edits_in_memory(originals: dict[str, str], edits: list[dict]) -> dict[str, str]:
    """Apply find/replace edits to file contents; each `find` must match exactly once."""
    new = dict(originals)
    for i, e in enumerate(edits, 1):
        path, find, replace = e["path"], e["find"], e["replace"]
        if path not in new:
            raise EditError(f"edit {i}: file '{path}' was not provided / does not exist")
        text = new[path]
        if find == "":
            raise EditError(f"edit {i}: empty 'find' anchor for {path}")
        count = text.count(find)
        if count == 0:
            # tolerate CRLF vs LF and trailing-whitespace drift in the anchor
            norm = find.replace("\r\n", "\n")
            if text.replace("\r\n", "\n").count(norm) == 1:
                text = text.replace("\r\n", "\n")
                find = norm
                count = 1
        if count == 0:
            raise EditError(f"edit {i}: 'find' text not found in {path}")
        if count > 1:
            raise EditError(f"edit {i}: 'find' text is ambiguous in {path} ({count} matches); include more context")
        new[path] = text.replace(find, replace, 1)
    return new


def unified_diff(path: str, old: str, new: str) -> str:
    if old == new:
        return ""
    a, b = old.splitlines(keepends=True), new.splitlines(keepends=True)
    out = [f"diff --git a/{path} b/{path}\n"]
    for line in difflib.unified_diff(a, b, fromfile=f"a/{path}", tofile=f"b/{path}", n=3):
        if not line.endswith("\n"):
            line += "\n\\ No newline at end of file\n"
        out.append(line)
    return "".join(out)


def build_diff(originals: dict[str, str], edits: list[dict]) -> EditResult:
    new = apply_edits_in_memory(originals, edits)
    parts, files = [], []
    for path in sorted(new):
        d = unified_diff(path, originals[path], new[path])
        if d:
            parts.append(d)
            files.append(path)
    if not parts:
        raise EditError("edits produce no change")
    return EditResult("".join(parts), files)


def is_test_path(path: str) -> bool:
    p = Path(path)
    name = p.name
    return (any(part in ("tests", "test", "__tests__", "spec") for part in p.parts[:-1])
            or name.startswith("test_") or name.endswith(("_test.py", ".test.js", ".spec.js", ".test.ts",
                                                           ".spec.ts", ".test.mjs"))
            or name == "conftest.py")
