from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from pathlib import Path, PureWindowsPath

from oslm.checker.report import CheckReport, build_report
from oslm.config import load_config

log = logging.getLogger(__name__)

SCRIPT = Path(__file__).resolve().parent / "wsl" / "run_checks.sh"


def to_wsl_path(path: Path) -> str:
    win = PureWindowsPath(path)
    if not win.drive:
        return path.as_posix()
    drive = win.drive.rstrip(":").lower()
    return "/mnt/" + drive + "/" + "/".join(win.parts[1:])


def _unavailable(reason: str) -> CheckReport:
    return CheckReport("checker_unavailable", f"Code checker offline: {reason}")


class CheckerBridge:
    """Runs C programs through run_checks.sh inside WSL, one wsl.exe process per program."""

    def __init__(
        self,
        distro: str | None = None,
        timeout_s: float | None = None,
        max_bytes: int | None = None,
        concurrency: int = 3,
        wsl_exe: str = "wsl.exe",
    ) -> None:
        cfg = load_config().wsl
        self.distro = distro or cfg["distro"]
        self.timeout_s = timeout_s or cfg["checker_timeout_s"]
        self.max_bytes = max_bytes or cfg["max_source_bytes"]
        self.wsl_exe = wsl_exe
        self._limit = asyncio.Semaphore(concurrency)
        self.available: bool | None = None

    def _command(self) -> list[str]:
        # --exec bypasses the login shell, which would otherwise re-expand `$` in arguments.
        return [self.wsl_exe, "-d", self.distro, "--exec", "bash", to_wsl_path(SCRIPT)]

    async def warmup(self) -> bool:
        """Boots the WSL VM so the first real check is not slowed by a cold start."""
        try:
            proc = await asyncio.create_subprocess_exec(
                self.wsl_exe,
                "-d",
                self.distro,
                "--exec",
                "true",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=60)
            self.available = proc.returncode == 0
        except (FileNotFoundError, OSError, TimeoutError):
            self.available = False
        log.info("WSL checker warmup: available=%s", self.available)
        return self.available

    async def check(self, code: str) -> CheckReport:
        source = code.replace("\r\n", "\n").encode("utf-8")
        if not source.strip():
            return CheckReport("rejected", "No source code was provided.")
        if len(source) > self.max_bytes:
            return CheckReport("rejected", f"Source is larger than {self.max_bytes} bytes.")
        async with self._limit:
            return await self._run(source)

    async def check_many(self, codes: Sequence[str]) -> list[CheckReport]:
        return list(await asyncio.gather(*(self.check(c) for c in codes)))

    async def _run(self, source: bytes) -> CheckReport:
        try:
            proc = await asyncio.create_subprocess_exec(
                *self._command(),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (FileNotFoundError, OSError) as exc:
            self.available = False
            return _unavailable(f"cannot start wsl.exe ({exc})")

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(source), timeout=self.timeout_s)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return CheckReport(
                "checker_error",
                f"Checker did not respond within {self.timeout_s:.0f} s.",
                timed_out=True,
            )

        line = next((ln for ln in reversed(stdout.splitlines()) if ln.startswith(b"{")), None)
        if line is None:
            err = stderr.replace(b"\x00", b"").decode("utf-8", "replace").strip()
            self.available = False
            return _unavailable(err[-300:] or f"wsl.exe exited with code {proc.returncode}")

        self.available = True
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            return CheckReport("checker_error", f"Malformed checker output: {exc}")
        return build_report(raw)


_bridge: CheckerBridge | None = None


def get_bridge() -> CheckerBridge:
    global _bridge
    if _bridge is None:
        _bridge = CheckerBridge()
    return _bridge
