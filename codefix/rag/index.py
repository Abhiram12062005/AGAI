"""ChromaDB-backed RAG over three collections per repo:

  code - AST/function-chunked source with file + line metadata
  bugs - past bugs and their accepted fixes from this project's own run history
  docs - README / docs

Incremental re-index on file mtime (manifest per repo). Embeddings are computed by our own
Embedder and passed explicitly, so Chroma never loads a model of its own.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

from codefix.rag.chunker import CODE_EXT, DOC_EXT, chunk_file
from codefix.rag.embedder import Embedder, get_embedder

_CLIENTS: dict[str, object] = {}
_LOCK = threading.Lock()


def _client(path: Path):
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    key = str(path)
    with _LOCK:
        if key not in _CLIENTS:
            path.mkdir(parents=True, exist_ok=True)
            _CLIENTS[key] = chromadb.PersistentClient(
                path=key, settings=ChromaSettings(anonymized_telemetry=False))
        return _CLIENTS[key]


def repo_key(repo: Path) -> str:
    return hashlib.sha1(str(Path(repo).resolve()).lower().encode()).hexdigest()[:10]


def _is_doc(rel: str) -> bool:
    return rel.endswith(DOC_EXT) and (rel.count("/") == 0 or rel.startswith(("docs/", "doc/")))


class RagIndex:
    def __init__(self, repo: Path, index_dir: Path, embedder: Embedder | None = None):
        self.repo = Path(repo).resolve()
        self.embedder = embedder or get_embedder()
        self.dir = Path(index_dir)
        self.key = repo_key(self.repo)
        self.client = _client(self.dir / "chroma")
        self.manifest_path = self.dir / f"manifest_{self.key}_{self.embedder.name}.json"

    def collection(self, name: str):
        return self.client.get_or_create_collection(
            name=f"{name}_{self.key}_{self.embedder.name}", metadata={"hnsw:space": "cosine"},
            embedding_function=None)

    # ------------------------------------------------------------------ indexing
    def index(self, force: bool = False) -> dict:
        from mcp_server.sandbox import iter_files

        manifest = {} if force or not self.manifest_path.exists() else json.loads(self.manifest_path.read_text())
        seen, changed = set(), []
        for full, rel in iter_files(self.repo):
            if not (rel.endswith(CODE_EXT) or _is_doc(rel)):
                continue
            seen.add(rel)
            mtime = full.stat().st_mtime
            if manifest.get(rel) != mtime:
                changed.append((full, rel, mtime))
        removed = [r for r in manifest if r not in seen]
        code, docs = self.collection("code"), self.collection("docs")
        n_chunks = 0
        for rel in removed:
            for col in (code, docs):
                col.delete(where={"file": rel})
            manifest.pop(rel, None)
        for full, rel, mtime in changed:
            col = docs if _is_doc(rel) else code
            col.delete(where={"file": rel})
            try:
                chunks = chunk_file(rel, full.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
            uniq = {c.id: c for c in chunks if c.text.strip()}
            if uniq:
                ids = list(uniq)
                texts = [f"{uniq[i].file} {uniq[i].symbol}\n{uniq[i].text}" for i in ids]
                col.add(ids=ids, embeddings=self.embedder.embed(texts), documents=[uniq[i].text for i in ids],
                        metadatas=[{"file": uniq[i].file, "start_line": uniq[i].start_line,
                                    "end_line": uniq[i].end_line, "symbol": uniq[i].symbol,
                                    "kind": uniq[i].kind} for i in ids])
                n_chunks += len(ids)
            manifest[rel] = mtime
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(manifest, indent=0))
        return {"repo": str(self.repo), "files_changed": len(changed), "files_removed": len(removed),
                "chunks_added": n_chunks, "files_indexed": len(manifest),
                "code_chunks": code.count(), "doc_chunks": docs.count(), "embedder": self.embedder.name}

    def ensure_indexed(self) -> dict:
        return self.index(force=False)

    def add_bug(self, task_id: str, bug_report: str, root_cause: str, diff: str, files: list[str]) -> str:
        cid = f"bugs:{task_id}"
        text = f"BUG: {bug_report}\nROOT CAUSE: {root_cause}\nFILES: {', '.join(files)}\nFIX:\n{diff}"
        col = self.collection("bugs")
        col.upsert(ids=[cid], embeddings=self.embedder.embed([f"{bug_report}\n{root_cause}"]),
                   documents=[text[:6000]],
                   metadatas=[{"file": ",".join(files)[:500], "start_line": 0, "end_line": 0,
                               "symbol": task_id, "kind": "bug"}])
        return cid

    # ------------------------------------------------------------------ retrieval
    def query(self, text: str, k: int = 5, collection: str = "code") -> list[dict]:
        col = self.collection(collection)
        n = col.count()
        if n == 0:
            return []
        res = col.query(query_embeddings=self.embedder.embed([text]), n_results=min(k, n))
        out = []
        for i, cid in enumerate(res["ids"][0]):
            md = res["metadatas"][0][i]
            out.append({"id": cid, "collection": collection, "file": md.get("file"),
                        "start_line": md.get("start_line"), "end_line": md.get("end_line"),
                        "symbol": md.get("symbol"), "text": res["documents"][0][i],
                        "distance": round(res["distances"][0][i], 4)})
        return out
