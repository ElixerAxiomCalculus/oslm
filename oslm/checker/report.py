from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

RUN_WALL_MS = 10_000
SIGKILL_RC = 137
TIMEOUT_RC = 124
SIGXCPU_RC = 128 + 24
SIGXFSZ_RC = 128 + 25

Status = Literal["clean", "issues", "compile_error", "rejected", "checker_unavailable", "checker_error"]


@dataclass
class Finding:
    tool: Literal["tsan", "helgrind", "runtime", "compiler"]
    kind: str
    message: str
    lines: list[int] = field(default_factory=list)
    threads: list[str] = field(default_factory=list)


@dataclass
class CheckReport:
    status: Status
    summary: str
    tsan_clean: bool | None = None
    helgrind_clean: bool | None = None
    timed_out: bool = False
    findings: list[Finding] = field(default_factory=list)
    compiler: list[Finding] = field(default_factory=list)
    program_output: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def has_sync_bug(self) -> bool:
        return any(f.tool in ("tsan", "helgrind") for f in self.findings) or self.timed_out

    def to_dict(self, include_raw: bool = False) -> dict[str, Any]:
        data = asdict(self)
        data["has_sync_bug"] = self.has_sync_bug
        if not include_raw:
            data.pop("raw")
        return data


_SRC_LINE = re.compile(r"\bprog\.c:(\d+)")
_TSAN_HEADER = re.compile(r"WARNING: ThreadSanitizer: (.+?)(?: \(pid=\d+\))?$", re.M)
_TSAN_THREAD = re.compile(r"\b(?:by |in )?(main thread|thread T\d+)")
_HG_PREFIX = re.compile(r"^==\d+== ?", re.M)
_HG_THREAD = re.compile(r"[Tt]hread #(\d+)")
_GCC_DIAG = re.compile(r"^prog\.c:(\d+):\d+: (warning|error): (.+)$", re.M)

_HELGRIND_KINDS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"Possible data race"), "data race"),
    (re.compile(r"lock order .* violated"), "lock-order inversion"),
    (re.compile(r"unlocked a not-locked lock|unlock.*not.*locked|with no corresponding"), "unlock of unheld mutex"),
    (re.compile(r"unlocked lock .* currently held by thread"), "unlock by non-owner thread"),
    (re.compile(r"re-lock a non-recursive lock"), "double lock (self-deadlock)"),
    (re.compile(r"Exiting thread still holds"), "thread exits holding a lock"),
    (re.compile(r"pthread_cond_\w+.*(?:not held|no.* associated|bound)"), "condition variable misuse"),
    (re.compile(r"pthread_mutex_destroy of a locked mutex"), "destroying a locked mutex"),
    (re.compile(r"sem_\w+"), "semaphore misuse"),
]


def _unique_ints(values: list[str]) -> list[int]:
    return sorted({int(v) for v in values})


# Paragraphs describing where threads were created or where memory/locks live point at
# setup code, not at the faulty access, so they are excluded from a finding's lines.
_CONTEXT_PARAGRAPH = re.compile(r"^\s*(Thread T\d+|Thread #\d+ was created|Location is|Address 0x|Lock at|Block 0x)")


def _primary_lines(block: str) -> list[int]:
    lines: set[int] = set()
    for para in re.split(r"\n\s*\n", block):
        if _CONTEXT_PARAGRAPH.search(para):
            continue
        first = _SRC_LINE.search(para)
        if first:
            lines.add(int(first.group(1)))
    return sorted(lines) or _unique_ints(_SRC_LINE.findall(block))


_TSAN_KINDS = {
    "data race": "data race",
    "lock-order-inversion": "lock-order inversion",
    "unlock of an unlocked mutex": "unlock of unheld mutex",
    "double lock": "double lock (self-deadlock)",
    "destroy of a locked mutex": "destroying a locked mutex",
    "use of an invalid mutex": "invalid mutex",
    "signal-unsafe call": "signal-unsafe call",
    "thread leak": "thread leak (missing pthread_join)",
    "heap-use-after-free": "use after free",
}


def _tsan_kind(raw: str) -> str:
    return next((name for key, name in _TSAN_KINDS.items() if raw.startswith(key)), raw)


def parse_tsan(output: str) -> list[Finding]:
    findings: list[Finding] = []
    for block in output.split("==================")[1:]:
        header = _TSAN_HEADER.search(block)
        if not header:
            continue
        kind = _tsan_kind(header.group(1).strip())
        threads = sorted({m.group(1) for m in _TSAN_THREAD.finditer(block)})
        summary = re.search(r"SUMMARY: ThreadSanitizer: (.+)", block)
        findings.append(
            Finding(
                tool="tsan",
                kind=kind,
                message=summary.group(1).strip() if summary else kind,
                lines=_primary_lines(block.split("SUMMARY:")[0]),
                threads=threads,
            )
        )
    return findings


def parse_helgrind(output: str) -> list[Finding]:
    text = _HG_PREFIX.sub("", output)
    findings: list[Finding] = []
    for block in re.split(r"^-{20,}$", text, flags=re.M):
        if "Thread-Announcement" in block:
            continue
        first = next((ln.strip() for ln in block.splitlines() if ln.strip()), "")
        kind = next((name for pat, name in _HELGRIND_KINDS if pat.search(first)), None)
        if kind is None:
            continue
        findings.append(
            Finding(
                tool="helgrind",
                kind=kind,
                message=first,
                lines=_primary_lines(block),
                threads=sorted({f"thread #{t}" for t in _HG_THREAD.findall(block)}),
            )
        )
    return findings


def parse_compiler(output: str) -> list[Finding]:
    return [
        Finding(tool="compiler", kind=sev, message=msg.strip(), lines=[int(line)])
        for line, sev, msg in _GCC_DIAG.findall(output)
    ]


def _runtime_finding(label: str, step: dict[str, Any]) -> Finding | None:
    rc, elapsed = step.get("exit_code", 0), step.get("elapsed_ms", 0)
    if rc == TIMEOUT_RC or (rc == SIGKILL_RC and elapsed >= RUN_WALL_MS):
        return Finding("runtime", "timeout", f"{label}: no exit within 10 s — possible deadlock or infinite loop")
    if rc == SIGXCPU_RC:
        return Finding("runtime", "cpu_limit", f"{label}: CPU limit hit — possible infinite loop or busy waiting")
    if rc == SIGXFSZ_RC:
        return Finding("runtime", "output_limit", f"{label}: output exceeded the 1 MB limit (truncated)")
    # A cgroup OOM kill surfaces as SIGKILL, or as SIGTERM (bare 15 from bwrap) when systemd tears down the scope.
    if rc in (SIGKILL_RC, 15, 128 + 15):
        return Finding("runtime", "resource_limit", f"{label}: killed by the sandbox (memory or process limit)")
    if rc > 128 and label == "run":
        return Finding("runtime", "crash", f"{label}: terminated by signal {rc - 128}")
    return None


def _dedupe(findings: list[Finding]) -> list[Finding]:
    seen: set[tuple[str, str, tuple[int, ...]]] = set()
    unique = []
    for f in findings:
        key = (f.tool, f.kind, tuple(f.lines))
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


_RUNTIME_SUMMARY = {
    "timeout": "program did not finish — possible deadlock or infinite loop",
    "cpu_limit": "CPU limit hit — possible infinite loop or busy waiting",
    "output_limit": "output exceeded the 1 MB limit",
    "resource_limit": "killed by the sandbox's memory or process limit",
    "crash": "program crashed",
}


def precise_lines(findings: list[Finding], kind: str) -> list[int]:
    """Lines for one bug kind; Helgrind's -O0 build names the exact statement where TSan's -O1 may name the loop."""
    helgrind = sorted({ln for f in findings if f.kind == kind and f.tool == "helgrind" for ln in f.lines})
    return helgrind or sorted({ln for f in findings if f.kind == kind and f.tool == "tsan" for ln in f.lines})


def _summarise(findings: list[Finding]) -> str:
    sync = [f for f in findings if f.tool in ("tsan", "helgrind")]
    parts = []
    for kind in dict.fromkeys(f.kind for f in sync):
        lines = precise_lines(sync, kind)
        where = f" (line{'s' if len(lines) > 1 else ''} {', '.join(map(str, lines))})" if lines else ""
        parts.append(f"{kind}{where}")
    for kind in dict.fromkeys(f.kind for f in findings if f.tool == "runtime"):
        parts.append(_RUNTIME_SUMMARY[kind])
    if not parts:
        return "No synchronization problems detected by ThreadSanitizer or Helgrind."
    return "; ".join(parts)


def build_report(raw: dict[str, Any]) -> CheckReport:
    status = raw.get("status")
    steps: dict[str, dict[str, Any]] = raw.get("steps", {})
    if status == "too_large":
        return CheckReport("rejected", f"Source is larger than {raw.get('max_bytes', 20480)} bytes.", raw=raw)
    if status == "empty":
        return CheckReport("rejected", "No source code was provided.", raw=raw)

    compiler = []
    for label in ("compile_plain", "compile_tsan"):
        compiler += parse_compiler(steps.get(label, {}).get("output", ""))
    compiler = _dedupe(compiler)

    if status == "compile_error":
        errors = [f for f in compiler if f.kind == "error"]
        detail = "; ".join(f"line {f.lines[0]}: {f.message}" for f in errors[:5])
        if not errors:
            failing = steps.get("compile_plain", {}) or steps.get("compile_tsan", {})
            detail = failing.get("output", "")[-400:].strip() or "compiler failed or timed out"
        return CheckReport("compile_error", f"Compilation failed: {detail}", compiler=compiler, raw=raw)

    if status != "completed":
        return CheckReport("checker_error", f"Unexpected checker status: {status!r}", raw=raw)

    tsan = parse_tsan(steps.get("tsan", {}).get("output", ""))
    helgrind = parse_helgrind(steps.get("helgrind", {}).get("output", ""))
    runtime = [
        f
        for label in ("run", "tsan", "helgrind")
        if label in steps and (f := _runtime_finding(label, steps[label])) is not None
    ]
    timed_out = any(f.kind in ("timeout", "cpu_limit") for f in runtime)
    findings = _dedupe(tsan + helgrind + runtime)

    tsan_step = steps.get("tsan", {})
    hg_step = steps.get("helgrind", {})
    tsan_clean = not tsan and tsan_step.get("exit_code") not in (66, TIMEOUT_RC)
    helgrind_clean = not helgrind and hg_step.get("exit_code") not in (67, TIMEOUT_RC)

    return CheckReport(
        status="issues" if findings else "clean",
        summary=_summarise(findings),
        tsan_clean=tsan_clean,
        helgrind_clean=helgrind_clean,
        timed_out=timed_out,
        findings=findings,
        compiler=compiler,
        program_output=steps.get("run", {}).get("output", "")[:4000],
        raw=raw,
    )
