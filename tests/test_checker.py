from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from oslm.checker.report import build_report, parse_compiler, parse_helgrind, parse_tsan
from oslm.checker.wsl_bridge import CheckerBridge, to_wsl_path

FIXTURES = Path(__file__).parent / "fixtures"
C_DIR = FIXTURES / "c"


def load_raw(name: str) -> dict:
    return json.loads((FIXTURES / "checker" / f"{name}.json").read_text(encoding="utf-8"))


def step(rc: int = 0, ms: int = 100, output: str = "") -> dict:
    return {"exit_code": rc, "elapsed_ms": ms, "output": output, "truncated": False}


def completed(**steps: dict) -> dict:
    base = {"compile_plain": step(), "compile_tsan": step(), "run": step(), "tsan": step(), "helgrind": step()}
    base.update(steps)
    return {"status": "completed", "steps": base, "memory_scoped": True}


class TestParsers:
    def test_tsan_data_race_from_real_output(self) -> None:
        findings = parse_tsan(load_raw("racy_counter")["steps"]["tsan"]["output"])
        assert [f.kind for f in findings] == ["data race"]
        assert findings[0].lines == [8]
        assert "thread T1" in findings[0].threads

    def test_helgrind_data_race_excludes_thread_creation_lines(self) -> None:
        findings = parse_helgrind(load_raw("racy_counter")["steps"]["helgrind"]["output"])
        races = [f for f in findings if f.kind == "data race"]
        assert races and all(15 not in f.lines and 16 not in f.lines for f in races)

    def test_lock_order_inversion_detected_by_both_tools(self) -> None:
        raw = load_raw("lock_order")
        tsan = parse_tsan(raw["steps"]["tsan"]["output"])
        helgrind = parse_helgrind(raw["steps"]["helgrind"]["output"])
        assert any(f.kind == "lock-order inversion" for f in tsan)
        assert any(f.kind == "lock-order inversion" for f in helgrind)
        assert {8, 9, 17, 18} <= set(tsan[0].lines)

    def test_compiler_diagnostics(self) -> None:
        out = "prog.c:5:5: error: expected ';' before 'x'\nprog.c:2:1: warning: unused variable 'y'\n"
        diags = parse_compiler(out)
        assert [(d.kind, d.lines) for d in diags] == [("error", [5]), ("warning", [2])]


class TestBuildReport:
    def test_racy_program_has_sync_bug(self) -> None:
        report = build_report(load_raw("racy_counter"))
        assert report.status == "issues"
        assert report.has_sync_bug
        assert report.tsan_clean is False and report.helgrind_clean is False
        assert "data race" in report.summary

    def test_clean_program(self) -> None:
        report = build_report(completed())
        assert report.status == "clean"
        assert report.tsan_clean and report.helgrind_clean
        assert not report.has_sync_bug

    def test_wall_timeout_reported_as_possible_deadlock(self) -> None:
        report = build_report(completed(run=step(124, 10_100), tsan=step(124, 10_100), helgrind=step(124, 10_100)))
        assert report.timed_out and report.has_sync_bug
        assert "possible deadlock" in report.summary
        assert report.tsan_clean is False

    def test_cpu_limit_reported_as_busy_wait(self) -> None:
        report = build_report(completed(run=step(152, 5_100)))
        assert report.timed_out
        assert "busy waiting" in report.summary

    def test_oom_kill_is_resource_limit_not_deadlock(self) -> None:
        report = build_report(completed(run=step(137, 800)))
        assert any(f.kind == "resource_limit" for f in report.findings)
        assert not report.timed_out

    def test_output_limit(self) -> None:
        report = build_report(completed(run=step(153, 300)))
        assert "1 MB" in report.summary

    def test_compile_error(self) -> None:
        raw = {"status": "compile_error", "steps": {"compile_plain": step(1, output="prog.c:3:1: error: boom\n")}}
        report = build_report(raw)
        assert report.status == "compile_error"
        assert "line 3" in report.summary

    @pytest.mark.parametrize(("status", "text"), [("too_large", "larger"), ("empty", "No source")])
    def test_rejections(self, status: str, text: str) -> None:
        report = build_report({"status": status, "steps": {}, "max_bytes": 20480})
        assert report.status == "rejected" and text in report.summary

    def test_to_dict_hides_raw(self) -> None:
        data = build_report(completed()).to_dict()
        assert "raw" not in data and data["has_sync_bug"] is False


class TestBridge:
    def test_to_wsl_path(self) -> None:
        assert to_wsl_path(Path(r"C:\dev\OSlm\x.sh")) == "/mnt/c/dev/OSlm/x.sh"

    async def test_missing_wsl_returns_unavailable(self) -> None:
        bridge = CheckerBridge(wsl_exe="definitely-not-wsl-oslm.exe")
        report = await bridge.check("int main(void){return 0;}")
        assert report.status == "checker_unavailable"
        assert bridge.available is False

    async def test_missing_wsl_warmup_is_false(self) -> None:
        assert await CheckerBridge(wsl_exe="definitely-not-wsl-oslm.exe").warmup() is False

    async def test_empty_and_oversized_rejected_without_calling_wsl(self) -> None:
        bridge = CheckerBridge(wsl_exe="definitely-not-wsl-oslm.exe", max_bytes=100)
        assert (await bridge.check("   \n")).status == "rejected"
        assert (await bridge.check("x" * 101)).status == "rejected"
        assert bridge.available is None

    async def test_parses_json_from_fake_wsl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = json.dumps(completed())
        bridge = CheckerBridge()
        monkeypatch.setattr(
            bridge, "_command", lambda: [sys.executable, "-c", f"import sys; sys.stdin.read(); print({payload!r})"]
        )
        report = await bridge.check("int main(void){return 0;}")
        assert report.status == "clean" and bridge.available is True

    async def test_crlf_normalised_before_sending(self, monkeypatch: pytest.MonkeyPatch) -> None:
        echo = (
            "import sys, json; d = sys.stdin.buffer.read();"
            "print(json.dumps({'status': 'completed', 'steps': {'run': {'exit_code': 0, 'elapsed_ms': 1,"
            "'output': str(b'\\r' in d), 'truncated': False}}}))"
        )
        bridge = CheckerBridge()
        monkeypatch.setattr(bridge, "_command", lambda: [sys.executable, "-c", echo])
        report = await bridge.check("int main(void)\r\n{return 0;}\r\n")
        assert report.program_output == "False"

    async def test_bridge_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        bridge = CheckerBridge(timeout_s=1)
        monkeypatch.setattr(bridge, "_command", lambda: [sys.executable, "-c", "import time; time.sleep(10)"])
        report = await bridge.check("int main(void){return 0;}")
        assert report.status == "checker_error" and report.timed_out

    async def test_garbage_output_is_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        bridge = CheckerBridge()
        monkeypatch.setattr(
            bridge,
            "_command",
            lambda: [
                sys.executable,
                "-c",
                "import sys; sys.stderr.write('There is no distribution with the supplied name.')",
            ],
        )
        report = await bridge.check("int main(void){return 0;}")
        assert report.status == "checker_unavailable"
        assert "no distribution" in report.summary


@pytest.mark.integration
class TestSandboxLive:
    """§9.2 edge cases against the real WSL sandbox."""

    @pytest.fixture(scope="class")
    def bridge(self) -> CheckerBridge:
        return CheckerBridge()

    async def run(self, bridge: CheckerBridge, name: str):
        return await bridge.check((C_DIR / f"{name}.c").read_bytes().decode("utf-8"))

    async def test_cold_call_within_timeout(self, bridge: CheckerBridge) -> None:
        assert await bridge.warmup()

    async def test_known_race_detected(self, bridge: CheckerBridge) -> None:
        report = await self.run(bridge, "racy_counter")
        assert report.has_sync_bug and any(f.kind == "data race" for f in report.findings)

    async def test_lock_order_inversion_detected_by_helgrind(self, bridge: CheckerBridge) -> None:
        report = await self.run(bridge, "lock_order")
        assert any(f.tool == "helgrind" and f.kind == "lock-order inversion" for f in report.findings)

    async def test_correct_program_clean(self, bridge: CheckerBridge) -> None:
        report = await self.run(bridge, "correct_counter")
        assert report.status == "clean"

    async def test_crlf_and_unicode(self, bridge: CheckerBridge) -> None:
        report = await self.run(bridge, "crlf_unicode")
        assert report.status == "clean" and "fertig" in report.program_output

    async def test_deadlock_reported_as_timeout(self, bridge: CheckerBridge) -> None:
        report = await self.run(bridge, "deadlock")
        assert report.timed_out and "possible deadlock" in report.summary

    async def test_infinite_loop(self, bridge: CheckerBridge) -> None:
        assert (await self.run(bridge, "infinite_loop")).timed_out

    async def test_fork_bomb_contained(self, bridge: CheckerBridge) -> None:
        report = await self.run(bridge, "fork_bomb")
        assert report.status == "issues"

    async def test_memory_bomb_contained(self, bridge: CheckerBridge) -> None:
        report = await self.run(bridge, "memory_bomb")
        assert any(f.kind in ("resource_limit", "cpu_limit", "timeout") for f in report.findings)

    async def test_writes_outside_tempdir_blocked(self, bridge: CheckerBridge) -> None:
        report = await self.run(bridge, "write_outside")
        assert "WROTE" not in report.program_output and report.program_output.count("blocked") == 4

    async def test_network_blocked(self, bridge: CheckerBridge) -> None:
        report = await self.run(bridge, "network")
        assert "network blocked" in report.program_output

    async def test_huge_output_truncated(self, bridge: CheckerBridge) -> None:
        report = await self.run(bridge, "huge_output")
        assert report.raw["steps"]["run"]["truncated"]
        assert len(report.program_output) <= 4000

    async def test_compile_error(self, bridge: CheckerBridge) -> None:
        assert (await self.run(bridge, "compile_error")).status == "compile_error"

    async def test_non_c_text(self, bridge: CheckerBridge) -> None:
        report = await bridge.check("Hello! Please explain what a semaphore is.")
        assert report.status == "compile_error"

    async def test_empty_input(self, bridge: CheckerBridge) -> None:
        assert (await bridge.check("")).status == "rejected"

    async def test_input_over_limit_rejected_inside_wsl(self, bridge: CheckerBridge) -> None:
        bridge_no_guard = CheckerBridge(max_bytes=10**6)
        report = await bridge_no_guard.check("/*" + "a" * 30_000 + "*/ int main(void){return 0;}")
        assert report.status == "rejected"

    async def test_unlock_unheld(self, bridge: CheckerBridge) -> None:
        report = await self.run(bridge, "unlock_unheld")
        assert any(f.kind == "unlock of unheld mutex" for f in report.findings)


def test_summary_prefers_helgrinds_exact_line() -> None:
    # TSan (-O1) blames the loop header (line 8); Helgrind (-O0) names counter++ (line 9).
    report = build_report(load_raw("racy_counter"))
    assert report.summary == "data race (line 9)"
