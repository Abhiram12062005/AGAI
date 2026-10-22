"""Plain-Python implementations of every MCP tool.

Kept separate from the FastMCP wiring (server.py) so they are unit-testable without a
transport. All paths are validated through the sandbox; nothing here ever writes to the
target repository itself — writes only happen inside workspaces created by
`create_workspace`.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from mcp_server import testrunner
from mcp_server.sandbox import (
    GIT_IDENTITY,
    SandboxError,
    WorkspaceManager,
    is_blocked,
    iter_files,
    norm_rel,
    rel_of,
    run,
    safe_path,
)

DIFF_FILE_RE = re.compile(r"^(?:---|\+\+\+) (?:[ab]/)?(\S+)", re.M)


class ToolContext:
    def __init__(self, repo_root: str | Path, sandbox_dir: str | Path, index_dir: str | Path | None = None):
        self.repo = Path(repo_root).resolve()
        if not self.repo.is_dir():
            raise FileNotFoundError(f"repo not found: {self.repo}")
        self.workspaces = WorkspaceManager(self.repo, Path(sandbox_dir))
        self.index_dir = Path(index_dir) if index_dir else None
        self._rag = None

    # ----------------------------------------------------------------- helpers
    def root_for(self, workspace: str | None) -> Path:
        return self.workspaces.existing(workspace) if workspace else self.repo

    @property
    def rag(self):
        if self._rag is None and self.index_dir is not None:
            from codefix.rag.index import RagIndex

            self._rag = RagIndex(self.repo, self.index_dir)
        return self._rag

    # ----------------------------------------------------------------- read-only tools
    def read_file(self, path: str, start: int = 1, end: int | None = None, workspace: str | None = None,
                  raw: bool = False) -> dict:
        root = self.root_for(workspace)
        full = safe_path(root, path, must_exist=True)
        if full.is_dir():
            raise IsADirectoryError(path)
        with open(full, encoding="utf-8", errors="replace", newline="") as f:
            text = f.read()
        if raw:  # exact bytes-as-text, used to derive diffs deterministically
            return {"path": rel_of(root, full), "content": text}
        lines = text.splitlines()
        start = max(1, int(start or 1))
        end = min(len(lines), int(end) if end else min(len(lines), start + 399))
        body = "\n".join(f"{i:>5}| {lines[i - 1]}" for i in range(start, end + 1))
        return {"path": rel_of(root, full), "start": start, "end": end, "total_lines": len(lines),
                "content": body}

    def list_tree(self, path: str = ".", max_depth: int = 3, workspace: str | None = None) -> dict:
        root = self.root_for(workspace)
        base = safe_path(root, path or ".", must_exist=True)
        out: list[str] = []
        for full, rel in iter_files(base):
            depth = rel.count("/")
            if depth < max_depth:
                out.append(rel_of(root, full))
            if len(out) >= 400:
                out.append("... (truncated)")
                break
        return {"root": rel_of(root, base), "files": out}

    def literal_search(self, query: str, k: int = 20, root: Path | None = None) -> list[dict]:
        root = root or self.repo
        hits: list[dict] = []
        rg = shutil.which("rg")
        if rg:
            cp = subprocess.run(
                [rg, "--no-heading", "-n", "-F", "-i", "--max-count", "5", "--max-columns", "300",
                 "-g", "!.git", "-g", "!node_modules", "-g", "!.venv", "-g", "!*.lock", "--", query, "."],
                cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
            for line in cp.stdout.splitlines():
                m = re.match(r"^(?:\./)?(.+?):(\d+):(.*)$", line)
                if m and not is_blocked(m.group(1).replace("\\", "/")):
                    hits.append({"file": m.group(1).replace("\\", "/"), "line": int(m.group(2)),
                                 "snippet": m.group(3).strip()[:240]})
                if len(hits) >= k:
                    break
            return hits
        q = query.lower()
        for full, rel in iter_files(root):
            try:
                for i, text in enumerate(full.read_text(encoding="utf-8").splitlines(), 1):
                    if q in text.lower():
                        hits.append({"file": rel, "line": i, "snippet": text.strip()[:240]})
                        if len(hits) >= k:
                            return hits
            except (UnicodeDecodeError, OSError):
                continue
        return hits

    def code_search(self, query: str, k: int = 8) -> dict:
        """Hybrid search: ripgrep literal matches fused with vector search (RRF)."""
        k = max(1, min(int(k or 8), 30))
        terms = [query.strip()]
        terms += [t for t in re.findall(r"[A-Za-z_][A-Za-z0-9_\.]{3,}", query) if t != query.strip()][:4]
        scores: dict[tuple, float] = {}
        meta: dict[tuple, dict] = {}
        for ti, term in enumerate(terms):
            for rank, h in enumerate(self.literal_search(term, k=k * 2)):
                key = (h["file"], h["line"])
                scores[key] = scores.get(key, 0) + 1.0 / (60 + rank) * (1.0 if ti == 0 else 0.7)
                meta.setdefault(key, {**h, "source": "literal"})
        try:
            vec = self.rag.query(query, k=k, collection="code") if self.rag else []
        except Exception:  # index missing / embedder unavailable: literal-only degrade
            vec = []
        for rank, c in enumerate(vec):
            key = (c["file"], c["start_line"])
            scores[key] = scores.get(key, 0) + 1.0 / (60 + rank)
            if key in meta:
                meta[key]["source"] = "both"
            else:
                meta[key] = {"file": c["file"], "line": c["start_line"], "end_line": c["end_line"],
                             "snippet": c["text"][:400], "source": "vector", "chunk_id": c["id"]}
        ranked = sorted(scores, key=lambda key: -scores[key])[:k]
        return {"query": query, "results": [{**meta[key], "score": round(scores[key], 4)} for key in ranked]}

    def parse_stacktrace(self, text: str) -> dict:
        frames: list[dict] = []
        patterns = [
            (re.compile(r'File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<func>[^\s]+)'), "python"),
            (re.compile(r"^(?P<file>[^\s:]+\.py):(?P<line>\d+): (?:in (?P<func>\S+)|\w+)", re.M), "pytest"),
            (re.compile(r"at (?:(?P<func>[^\s(]+) )?\(?(?:file://)?(?P<file>[^\s()]+?\.(?:m?js|cjs|ts)):(?P<line>\d+):\d+\)?"),
             "javascript"),
        ]
        seen = set()
        for rx, lang in patterns:
            for m in rx.finditer(text or ""):
                key = (m.group("file"), m.group("line"))
                if key in seen:
                    continue
                seen.add(key)
                rp = self._map_to_repo(m.group("file"))
                frames.append({"file": m.group("file"), "line": int(m.group("line")),
                               "function": m.group("func") or "", "lang": lang,
                               "repo_path": rp, "in_repo": rp is not None})
        exc = None
        for line in reversed((text or "").splitlines()):
            s = re.sub(r"^E\s+", "", line.strip())  # pytest prefixes failure lines with "E   "
            if re.match(r"^[\w\.]*(Error|Exception|Failure|Warning)\b", s) or s.startswith("assert"):
                exc = s[:300]
                break
        repo_frames = [f for f in frames if f["in_repo"]]
        return {"exception": exc, "frames": frames, "repo_frames": repo_frames,
                "innermost_repo_frame": repo_frames[-1] if repo_frames else None}

    def _map_to_repo(self, file: str) -> str | None:
        f = file.replace("\\", "/")
        p = Path(f)
        try:
            if p.is_absolute():
                return p.resolve().relative_to(self.repo).as_posix()
        except ValueError:
            pass
        if not p.is_absolute() and (self.repo / f).is_file():
            return norm_rel(f)
        parts = Path(f).parts
        for n in range(min(4, len(parts)), 0, -1):  # longest repo-relative suffix match
            suffix = "/".join(parts[-n:])
            for _, rel in iter_files(self.repo):
                if rel == suffix or rel.endswith("/" + suffix):
                    if "site-packages" in f or "node_modules" in f:
                        return None
                    return rel
        return None

    def static_analysis(self, paths: list[str], workspace: str | None = None) -> dict:
        root = self.root_for(workspace)
        rels = [rel_of(root, safe_path(root, p, must_exist=True)) for p in paths]
        return {"findings": testrunner.static_analysis(root, rels)}

    def rag_retrieve(self, query: str, k: int = 5, collection: str = "code") -> dict:
        if collection not in ("code", "bugs", "docs"):
            raise ValueError("collection must be one of code|bugs|docs")
        if not self.rag:
            return {"chunks": []}
        return {"chunks": self.rag.query(query, k=int(k or 5), collection=collection)}

    def index_repo(self, force: bool = False) -> dict:
        # Indexing runs inside the server process so the same Chroma client serves queries.
        return self.rag.index(force=bool(force))

    def record_fix(self, task_id: str, bug_report: str, root_cause: str, diff: str, files: list[str]) -> dict:
        return {"chunk_id": self.rag.add_bug(task_id, bug_report, root_cause, diff, files)}

    # ----------------------------------------------------------------- workspace tools
    def create_workspace(self, name: str) -> dict:
        ws = self.workspaces.create(name)
        return {"workspace": name, "path": str(ws)}

    def discard_workspace(self, workspace: str) -> dict:
        self.workspaces.discard(workspace)
        return {"discarded": workspace}

    def apply_patch(self, diff: str, workspace: str) -> dict:
        ws = self.workspaces.existing(workspace)
        files = sorted({f for f in DIFF_FILE_RE.findall(diff or "") if f != "/dev/null"})
        if not files:
            return {"ok": False, "files": [], "error": "diff contains no file headers"}
        for f in files:  # sandbox + blocklist check before touching anything
            safe_path(ws, f)
        pf = ws / ".git" / "codefix-attempt.patch"
        pf.write_text(diff if diff.endswith("\n") else diff + "\n", encoding="utf-8", newline="\n")
        flags = ["--whitespace=nowarn", "--recount"]
        check = run(["git", "apply", "--check", *flags, str(pf)], cwd=ws)
        if check.returncode != 0:  # retry tolerant of CRLF/whitespace drift in context lines
            flags.append("--ignore-whitespace")
            check = run(["git", "apply", "--check", *flags, str(pf)], cwd=ws)
        if check.returncode != 0:
            return {"ok": False, "files": files, "error": (check.stderr or check.stdout).strip()[:2000]}
        cp = run(["git", "apply", *flags, str(pf)], cwd=ws)
        return {"ok": cp.returncode == 0, "files": files, "error": cp.stderr.strip()[:2000] or None}

    def run_tests(self, workspace: str, selector: str | None = None, timeout_s: int = 180) -> dict:
        ws = self.workspaces.existing(workspace)
        return testrunner.run_tests(ws, selector=selector, timeout_s=min(int(timeout_s or 180), 900))

    def git_branch(self, workspace: str, branch: str) -> dict:
        ws = self.workspaces.existing(workspace)
        if not re.fullmatch(r"codefix/[A-Za-z0-9_\-]+", branch or ""):
            raise SandboxError("branches must be named codefix/<task_id>")
        run(["git", "checkout", "-q", "-B", branch], cwd=ws, check=True)
        return {"branch": branch}

    def git_commit(self, workspace: str, message: str) -> dict:
        ws = self.workspaces.existing(workspace)
        run(["git", "add", "-A", "--", ".", ":!.codefix-junit.xml", ":!node_modules"], cwd=ws, check=True)
        cp = run(["git", *GIT_IDENTITY, "commit", "-q", "-m", message], cwd=ws)
        if cp.returncode != 0 and "nothing to commit" not in cp.stdout + cp.stderr:
            raise RuntimeError(cp.stderr.strip())
        sha = run(["git", "rev-parse", "HEAD"], cwd=ws, check=True).stdout.strip()
        return {"sha": sha}

    def diff_workspace(self, workspace: str) -> dict:
        ws = self.workspaces.existing(workspace)
        return {"diff": run(["git", "diff", "HEAD"], cwd=ws).stdout}

    def open_pull_request(self, workspace: str, branch: str, title: str, body: str,
                          confirmed: bool = False) -> dict:
        """Push the codefix branch and open a PR. Requires explicit human confirmation."""
        if confirmed is not True:
            raise SandboxError("open_pull_request requires explicit human confirmation (confirmed=true)")
        if not re.fullmatch(r"codefix/[A-Za-z0-9_\-]+", branch or ""):
            raise SandboxError("refusing to push: branch must be codefix/<task_id>")
        ws = self.workspaces.existing(workspace)
        origin = run(["git", "remote", "get-url", "origin"], cwd=self.repo).stdout.strip()
        if not origin:
            return {"pushed": False, "url": None, "branch": branch,
                    "note": f"Target repo has no 'origin' remote. Branch {branch} is committed in {ws}."}
        default = self._default_branch(origin)
        if branch in (default, "main", "master"):
            raise SandboxError("refusing to push to the default branch")
        push = run(["git", "push", origin, f"HEAD:refs/heads/{branch}"], cwd=ws, timeout=180)
        if push.returncode != 0:
            return {"pushed": False, "url": None, "branch": branch, "note": push.stderr.strip()[:1000]}
        url, note = self._create_pr(origin, branch, default, title, body)
        return {"pushed": True, "url": url, "branch": branch, "base": default, "note": note}

    def _default_branch(self, origin: str) -> str:
        cp = run(["git", "ls-remote", "--symref", origin, "HEAD"], cwd=self.repo, timeout=60)
        m = re.search(r"ref: refs/heads/(\S+)\s+HEAD", cp.stdout)
        return m.group(1) if m else "main"

    def _create_pr(self, origin: str, branch: str, base: str, title: str, body: str) -> tuple[str | None, str]:
        m = re.search(r"github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$", origin)
        if not m:
            return None, "Branch pushed; remote is not GitHub, open the PR manually."
        owner, name = m.groups()
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if token:
            from github import Auth, Github

            gh = Github(auth=Auth.Token(token))
            pr = gh.get_repo(f"{owner}/{name}").create_pull(title=title, body=body, head=branch, base=base)
            return pr.html_url, "opened via PyGithub"
        if shutil.which("gh"):
            cp = run(["gh", "pr", "create", "--repo", f"{owner}/{name}", "--head", branch, "--base", base,
                      "--title", title, "--body", body], cwd=self.repo, timeout=120)
            if cp.returncode == 0:
                return cp.stdout.strip().splitlines()[-1], "opened via gh CLI"
            return None, cp.stderr.strip()[:500]
        compare = f"https://github.com/{owner}/{name}/compare/{base}...{branch}?expand=1"
        return compare, "No GITHUB_TOKEN or gh CLI: branch pushed, open the PR from this compare URL."
