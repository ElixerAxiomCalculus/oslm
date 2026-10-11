from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))


class Config(dict[str, Any]):
    """Nested dict with attribute access and project-root-relative path helpers."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Config(value) if isinstance(value, dict) else value

    def path(self, key: str) -> Path:
        return ROOT / self["paths"][key]


@lru_cache(maxsize=1)
def load_config(path: Path | None = None) -> Config:
    load_dotenv(ROOT / ".env")
    cfg_path = path or ROOT / "config.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        return Config(yaml.safe_load(f))
