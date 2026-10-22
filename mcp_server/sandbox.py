"""Safety layer: path sandboxing, secret blocklist and isolated per-attempt workspaces.

Hard rules enforced here (independent of what any model asks for):
  * Never mutate the target repo in place — every patch/test happens in a throwaway clone.
  * Refuse any path that resolves outside the allowed root.
  * Blocklist secrets: .env*, .git internals, CI secrets, key material.
  * All subprocesses get argv lists (never a shell string), so model output is never
    shell-interpolated.
"""

from __future__ import annotations

import fnmatch
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path, PurePosixPath

BLOCKED_PATTERNS = [
    ".env", ".env.*", "*/.env", "*/.env.*",
    ".git", ".git/*", "*/.git/*",
    ".github/workflows/*", ".gitlab-ci.yml", ".circleci/*",
    "*.pem", "*.key", "*.p12", "*.pfx", "id_rsa*", "id_ed25519*",
    ".npmrc", ".pypirc", ".netrc", "*secrets*", "*.secret",
]
ALLOWED_EXCEPTIONS = [".env.example", "*/.env.example", ".env.sample"]

IGNORED_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".pytest_cache",
                ".ruff_cache", ".mypy_cache", "dist", "build", ".codefix", ".tox", ".idea", ".vscode"}

WORKSPACE_RE = re.compile(r"^[A-Za-z0-9_-]{1,100}$")

GIT_IDENTITY = ["-c", "user.name=CodeFix AI", "-c", "user.email=codefix-ai@users.noreply.github.com"]


class SandboxError(PermissionError):
    pass


def norm_rel(path: str) -> str:
    """Normalise a repo-relative path: forward slashes, no leading './' (keeps dotfiles intact)."""
    p = path.replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return "" if p == "." else p


def is_blocked(rel_posix: str) -> bool:
    rel = norm_rel(rel_posix)
    if any(fnmatch.fnmatch(rel, p) for p in ALLOWED_EXCEPTIONS):
        return False
    parts = PurePosixPath(rel).parts
    if ".git" in parts:
        return True
    return any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(PurePosixPath(rel).name, p)
               for p in BLOCKED_PATTERNS)


def safe_path(root: Path, rel: str, *, must_exist: bool = False) -> Path:
    """Resolve `rel` inside `root`, refusing escapes and blocklisted paths."""
    root = Path(root).resolve()
    if rel is None:
        raise SandboxError("path is required")
    rel = str(rel).replace("\\", "/")
    candidate = Path(rel)
    full = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        rel_posix = full.relative_to(root).as_posix()
    except ValueError:
        raise SandboxError(f"path '{rel}' is outside the repository root") from None
    if rel_posix != "." and is_blocked(rel_posix):
        raise SandboxError(f"path '{rel_posix}' is blocklisted (secrets / git internals / CI)")
    if must_exist and not full.exists():
        raise FileNotFoundError(rel_posix)
    return full


def rel_of(root: Path, full: Path) -> str:
    return Path(full).resolve().relative_to(Path(root).resolve()).as_posix()


def iter_files(root: Path, max_bytes: int = 400_000):
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS and not d.startswith(".codefix"))
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            rel = p.relative_to(root).as_posix()
            if is_blocked(rel):
                continue
            try:
                if p.stat().st_size > max_bytes:
                    continue
            except OSError:
                continue
            yield p, rel


def run(argv: list[str], cwd: Path | None = None, timeout: int = 120, env: dict | None = None,
        check: bool = False) -> subprocess.CompletedProcess:
    if isinstance(argv, str):  # defence in depth: never accept a shell string
        raise SandboxError("commands must be argv lists")
    exe = shutil.which(argv[0]) or argv[0]
    cp = subprocess.run([exe, *argv[1:]], cwd=cwd, capture_output=True, text=True, timeout=timeout,
                        env=env, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL)
    if check and cp.returncode != 0:
        raise RuntimeError(f"{' '.join(argv)} failed: {cp.stderr.strip() or cp.stdout.strip()}")
    return cp


def is_git_repo(path: Path) -> bool:
    return (Path(path) / ".git").exists()


def _rm_readonly(func, path, _exc):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def rmtree(path: Path) -> None:
    if Path(path).exists():
        shutil.rmtree(path, onerror=_rm_readonly)


class WorkspaceManager:
    """Creates isolated clones of the target repo under the sandbox dir."""

    def __init__(self, repo_root: Path, sandbox_dir: Path):
        self.repo = Path(repo_root).resolve()
        self.sandbox = Path(sandbox_dir).resolve()
        self.sandbox.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        if not WORKSPACE_RE.match(name or ""):
            raise SandboxError(f"invalid workspace id '{name}'")
        p = (self.sandbox / name).resolve()
        if p.parent != self.sandbox:
            raise SandboxError("workspace escapes sandbox")
        return p

    def existing(self, name: str) -> Path:
        p = self.path(name)
        if not p.exists():
            raise SandboxError(f"unknown workspace '{name}' (create_workspace first)")
        return p

    def create(self, name: str) -> Path:
        ws = self.path(name)
        rmtree(ws)
        if is_git_repo(self.repo):
            # autocrlf=false: workspace files are the exact committed blobs (LF), whatever the host config
            run(["git", "clone", "-q", "--no-hardlinks", "-c", "core.autocrlf=false", str(self.repo), str(ws)],
                timeout=300, check=True)
            # Overlay uncommitted tracked changes so the workspace matches what the user sees.
            diff = run(["git", "diff", "HEAD", "--binary"], cwd=self.repo, timeout=60).stdout
            if diff.strip():
                pf = ws / ".git" / "codefix-overlay.patch"
                pf.write_text(diff, encoding="utf-8")
                run(["git", "apply", "--whitespace=nowarn", str(pf)], cwd=ws, check=True)
                run(["git", *GIT_IDENTITY, "commit", "-qam", "codefix: uncommitted baseline"], cwd=ws)
        else:
            shutil.copytree(self.repo, ws, ignore=shutil.ignore_patterns(*IGNORED_DIRS))
            run(["git", "init", "-q"], cwd=ws, check=True)
            run(["git", "add", "-A"], cwd=ws, check=True)
            run(["git", *GIT_IDENTITY, "commit", "-qm", "codefix baseline"], cwd=ws, check=True)
        self._link_node_modules(ws)
        return ws

    def _link_node_modules(self, ws: Path) -> None:
        src = self.repo / "node_modules"
        dst = ws / "node_modules"
        if not src.is_dir() or dst.exists():
            return
        try:
            os.symlink(src, dst, target_is_directory=True)
        except OSError:
            if os.name == "nt":  # directory junctions need no admin rights on Windows
                subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(src)], capture_output=True)

    def discard(self, name: str) -> None:
        ws = self.path(name)
        nm = ws / "node_modules"  # only ever our link (real node_modules is never copied)
        if nm.is_symlink():
            nm.unlink()
        elif nm.exists():
            try:
                os.rmdir(nm)  # removes a junction without touching its target
            except OSError:
                pass
        rmtree(ws)
