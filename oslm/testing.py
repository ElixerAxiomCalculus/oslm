from __future__ import annotations

import subprocess
import sys

from oslm.config import ROOT


def _run(name: str, cmd: list[str], cwd=ROOT) -> bool:  # noqa: ANN001
    print(f"\n=== {name}: {' '.join(cmd)}", flush=True)
    ok = subprocess.call(cmd, cwd=cwd) == 0
    print(f"=== {name}: {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def run_all_tests(integration: bool = False, frontend: bool = True, e2e: bool = False) -> int:
    """The §9.5 gate: ruff, pytest (optionally integration), tsc, vitest, Playwright."""
    npx = "npx.cmd" if sys.platform == "win32" else "npx"
    results = {
        "ruff": _run("ruff", [sys.executable, "-m", "ruff", "check", "oslm", "tests"]),
        "pytest": _run("pytest", [sys.executable, "-m", "pytest", "-q"]),
    }
    if integration:
        results["pytest-integration"] = _run(
            "pytest integration", [sys.executable, "-m", "pytest", "-q", "-m", "integration"]
        )
    if frontend:
        web = ROOT / "web"
        results["tsc"] = _run("tsc", [npx, "tsc", "--noEmit"], cwd=web)
        results["vitest"] = _run("vitest", [npx, "vitest", "run"], cwd=web)
        if e2e:
            results["playwright"] = _run("playwright", [npx, "playwright", "test"], cwd=web)
    print("\nSummary: " + ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in results.items()))
    return 0 if all(results.values()) else 1
