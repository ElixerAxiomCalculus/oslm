from __future__ import annotations

import shutil
from pathlib import Path

from oslm.config import ROOT

# Regenerable or intermediate artefacts (§11). Never: data/course, models/adapter, the final model, the
# index, SFT data or evaluation results.
TARGETS = [
    ".pytest_cache",
    ".pytest_tmp",
    ".ruff_cache",
    "web/test-results",
    "web/playwright-report",
    "web/blob-report",
    "web/dist",
    "data/cache",
    "models/checkpoints",
    "models/merged",
    "models/oslm-ft-f16.gguf",
    "logs",
]
PROTECTED = ["data/course", "models/adapter", "models/oslm-ft", "data/chroma", "data/sft", "data/eval"]


def _candidates() -> list[Path]:
    paths = [ROOT / t for t in TARGETS if (ROOT / t).exists()]
    paths += [p for p in ROOT.rglob("__pycache__") if ".venv" not in p.parts and "node_modules" not in p.parts]
    return paths


def run_clean(dry_run: bool = True) -> int:
    paths = _candidates()
    for p in paths:
        rel = p.relative_to(ROOT).as_posix()
        if any(rel == q or rel.startswith(q + "/") for q in PROTECTED):
            continue
        print(("would delete " if dry_run else "deleting ") + rel)
        if not dry_run:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
    if dry_run:
        print("\n(dry run — pass --yes to delete)")
    return 0
