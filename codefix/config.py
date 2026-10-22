"""Configuration loading: config/models.yaml + .env."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.environ.get("CODEFIX_CONFIG", ROOT / "config" / "models.yaml"))
DATA_DIR = Path(os.environ.get("CODEFIX_DATA_DIR", ROOT))
TRACES_DIR = DATA_DIR / "traces"
DB_PATH = DATA_DIR / "runs.db"
CHECKPOINT_DB = DATA_DIR / "checkpoints.db"
INDEX_DIR = DATA_DIR / ".codefix" / "index"
SANDBOX_DIR = DATA_DIR / ".codefix" / "sandbox"


class RoleConfig(BaseModel):
    model: str
    effort: str | None = None
    max_tokens: int = 8000
    thinking_display: str | None = None
    fallbacks: str | None = None


class Price(BaseModel):
    input: float
    output: float
    cache_read: float = 0.0
    cache_write: float = 0.0


class PipelineConfig(BaseModel):
    max_attempts: int = 3
    react_max_iterations: int = 8
    max_rediagnoses: int = 1
    rediagnose_below_confidence: float = 0.35
    test_timeout_s: int = 180
    allow_test_edits: bool = False
    rag_k: int = 6


class Settings(BaseModel):
    provider: Literal["anthropic", "google"] = "anthropic"
    roles: dict[str, RoleConfig]
    pricing: dict[str, Price] = Field(default_factory=dict)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)

    def role(self, name: str) -> RoleConfig:
        return self.roles[name]


class ConfigError(ValueError):
    pass


def _validate(s: Settings) -> Settings:
    for required in ("manager", "diagnoser", "fixer", "judge"):
        if required not in s.roles:
            raise ConfigError(f"models.yaml is missing role '{required}'")
    judge = s.roles["judge"].model
    generators = {s.roles["diagnoser"].model, s.roles["fixer"].model}
    if judge in generators and judge != "mock":
        raise ConfigError(
            f"Judge model '{judge}' must differ from the generating agents {sorted(generators)} "
            "(self-preference bias)."
        )
    return s


def load_settings(path: Path | None = None) -> Settings:
    p = Path(path or CONFIG_PATH)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return _validate(Settings.model_validate(data))


@lru_cache(maxsize=1)
def settings() -> Settings:
    load_dotenv()
    return load_settings()


def load_dotenv() -> None:
    try:
        from dotenv import load_dotenv as _ld

        _ld(ROOT / ".env", override=False)
    except ImportError:  # pragma: no cover
        pass


def ensure_dirs() -> None:
    for d in (TRACES_DIR, INDEX_DIR, SANDBOX_DIR):
        d.mkdir(parents=True, exist_ok=True)
