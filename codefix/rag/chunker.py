"""AST/function-level chunking with file + line metadata.

Python: real `ast` parse — one chunk per top-level function, per class (or per method for
large classes) plus a module-header chunk. JavaScript/TypeScript: brace-matched function /
class / method extraction. Anything else (or unparsable code): 50-line sliding windows.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

CODE_EXT = (".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx")
DOC_EXT = (".md", ".rst", ".txt")
MAX_CLASS_LINES = 80


@dataclass
class Chunk:
    file: str
    start_line: int
    end_line: int
    symbol: str
    text: str
    kind: str  # function|class|method|module|window|doc

    @property
    def id(self) -> str:
        prefix = "docs" if self.kind == "doc" else "code"
        return f"{prefix}:{self.file}:{self.start_line}-{self.end_line}"


def _window(file: str, lines: list[str], size: int = 50, overlap: int = 10) -> list[Chunk]:
    out, i = [], 0
    while i < len(lines):
        j = min(len(lines), i + size)
        out.append(Chunk(file, i + 1, j, f"{file}:{i + 1}", "\n".join(lines[i:j]), "window"))
        if j == len(lines):
            break
        i = j - overlap
    return out


def chunk_python(file: str, source: str) -> list[Chunk]:
    lines = source.splitlines()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _window(file, lines)
    chunks: list[Chunk] = []
    covered = set()

    def add(node, symbol, kind):
        start = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno])
        end = node.end_lineno or node.lineno
        chunks.append(Chunk(file, start, end, symbol, "\n".join(lines[start - 1:end]), kind))
        covered.update(range(start, end + 1))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add(node, node.name, "function")
        elif isinstance(node, ast.ClassDef):
            size = (node.end_lineno or node.lineno) - node.lineno
            methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if size > MAX_CLASS_LINES and methods:
                for m in methods:
                    add(m, f"{node.name}.{m.name}", "method")
                head_end = methods[0].lineno - 1
                chunks.append(Chunk(file, node.lineno, head_end, node.name,
                                    "\n".join(lines[node.lineno - 1:head_end]), "class"))
            else:
                add(node, node.name, "class")
    header = [i for i in range(1, len(lines) + 1) if i not in covered and lines[i - 1].strip()]
    if header:
        s, e = header[0], header[-1]
        text = "\n".join(lines[i - 1] for i in header)[:3000]
        chunks.append(Chunk(file, s, e, f"{file}:<module>", text, "module"))
    return chunks or _window(file, lines)


_JS_START = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:"
    r"function\s*\*?\s*(?P<f>[A-Za-z_$][\w$]*)\s*\("
    r"|class\s+(?P<c>[A-Za-z_$][\w$]*)"
    r"|(?:const|let|var)\s+(?P<v>[A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:function\b|\([^)]*\)\s*=>|[A-Za-z_$][\w$]*\s*=>)"
    r")")


def _match_brace(lines: list[str], start: int) -> int:
    depth, seen = 0, False
    for i in range(start, len(lines)):
        code = re.sub(r"(['\"`]).*?\1|//.*$", "", lines[i])
        depth += code.count("{") - code.count("}")
        seen = seen or "{" in code
        if seen and depth <= 0:
            return i
        if not seen and i > start and code.rstrip().endswith(";"):
            return i
    return len(lines) - 1


def chunk_js(file: str, source: str) -> list[Chunk]:
    lines = source.splitlines()
    chunks, i = [], 0
    covered = set()
    while i < len(lines):
        m = _JS_START.match(lines[i])
        if m:
            end = _match_brace(lines, i)
            sym = m.group("f") or m.group("c") or m.group("v")
            chunks.append(Chunk(file, i + 1, end + 1, sym, "\n".join(lines[i:end + 1]),
                                "class" if m.group("c") else "function"))
            covered.update(range(i + 1, end + 2))
            i = end + 1
        else:
            i += 1
    rest = [n for n in range(1, len(lines) + 1) if n not in covered and lines[n - 1].strip()]
    if rest:
        chunks.append(Chunk(file, rest[0], rest[-1], f"{file}:<module>",
                            "\n".join(lines[n - 1] for n in rest)[:3000], "module"))
    return chunks or _window(file, lines)


def chunk_doc(file: str, source: str, max_chars: int = 1500) -> list[Chunk]:
    lines = source.splitlines()
    chunks: list[Chunk] = []
    start, size = 0, 0

    def flush(end: int) -> None:
        text = "\n".join(lines[start:end])
        if text.strip():
            title = next((ln.lstrip("# ") for ln in lines[start:end] if ln.startswith("#")), file)
            chunks.append(Chunk(file, start + 1, end, title, text, "doc"))

    for i, line in enumerate(lines):
        if i > start and (re.match(r"^#{1,3} ", line) or size > max_chars):
            flush(i)
            start, size = i, 0
        size += len(line) + 1
    flush(len(lines))
    return chunks


def chunk_file(file: str, source: str) -> list[Chunk]:
    if file.endswith(".py"):
        return chunk_python(file, source)
    if file.endswith(CODE_EXT):
        return chunk_js(file, source)
    if file.endswith(DOC_EXT):
        return chunk_doc(file, source)
    return []
