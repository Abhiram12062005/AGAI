"""Embedders behind one interface.

Default: sentence-transformers all-MiniLM-L6-v2 (local, no second API key).
Fallback/tests: a deterministic feature-hashing embedder (no model download), selected with
CODEFIX_EMBEDDER=hash or automatically if sentence-transformers is unavailable.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import threading
from typing import Protocol


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def split_identifiers(text: str) -> list[str]:
    toks = []
    for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", text):
        toks.append(raw.lower())
        parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", raw).replace("_", " ").split()
        if len(parts) > 1:
            toks.extend(p.lower() for p in parts)
    return toks


class HashingEmbedder:
    name = "hash384"
    dim = 384

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for tok in split_identifiers(t):
                h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
                v[h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


class SentenceTransformerEmbedder:
    name = "minilm"
    dim = 384
    _model = None
    _lock = threading.Lock()

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name

    def _load(self):
        with self._lock:
            if SentenceTransformerEmbedder._model is None:
                from sentence_transformers import SentenceTransformer

                SentenceTransformerEmbedder._model = SentenceTransformer(self.model_name, device="cpu")
        return SentenceTransformerEmbedder._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        return model.encode(texts, batch_size=32, normalize_embeddings=True,
                            show_progress_bar=False).tolist()


def get_embedder() -> Embedder:
    choice = os.environ.get("CODEFIX_EMBEDDER", "minilm").lower()
    if choice == "hash":
        return HashingEmbedder()
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return HashingEmbedder()
    return SentenceTransformerEmbedder()
