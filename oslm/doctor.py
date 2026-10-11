from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass

from oslm.config import ROOT, load_config

TSAN_PROBE = b"""#include <pthread.h>
static int x;
static void *f(void *a) { (void)a; x++; return 0; }
int main(void) {
    pthread_t a, b;
    pthread_create(&a, 0, f, 0); pthread_create(&b, 0, f, 0);
    pthread_join(a, 0); pthread_join(b, 0);
    return 0;
}
"""


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    fatal: bool = True


def _run(args: list[str], stdin: bytes | None = None, timeout: int = 60) -> tuple[int, str]:
    try:
        proc = subprocess.run(args, input=stdin, capture_output=True, timeout=timeout)
    except FileNotFoundError:
        return 127, f"{args[0]} not found"
    except subprocess.TimeoutExpired:
        return 124, "timed out"
    # wsl.exe -l prints UTF-16LE; everything else is UTF-8.
    raw = proc.stdout + proc.stderr
    text = raw.decode("utf-16-le", "ignore") if b"\x00" in raw else raw.decode("utf-8", "ignore")
    return proc.returncode, text.strip()


def check_gpu() -> CheckResult:
    code, out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    return CheckResult("nvidia-smi", code == 0 and "4070" in out, out)


def check_torch() -> CheckResult:
    try:
        import torch

        ok = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if ok else "CUDA unavailable"
        return CheckResult("torch.cuda", ok, f"torch {torch.__version__}, cuda {torch.version.cuda}, {name}")
    except Exception as exc:
        return CheckResult("torch.cuda", False, repr(exc))


def check_bitsandbytes() -> CheckResult:
    try:
        import bitsandbytes

        return CheckResult("bitsandbytes", True, bitsandbytes.__version__)
    except Exception as exc:
        return CheckResult("bitsandbytes", False, repr(exc))


def check_ollama() -> CheckResult:
    cfg = load_config()
    code, out = _run(["ollama", "list"])
    if code != 0:
        return CheckResult("ollama", False, out)
    tags = [line.split()[0] for line in out.splitlines()[1:] if line.strip()]
    base = cfg.ollama.base_tag
    ok = base in tags or f"{base}:latest" in tags
    return CheckResult("ollama", ok, f"tags={tags}; base '{base}' {'present' if ok else 'MISSING'}")


def check_ollama_ft() -> CheckResult:
    cfg = load_config()
    code, out = _run(["ollama", "list"])
    tags = [line.split()[0] for line in out.splitlines()[1:] if line.strip()] if code == 0 else []
    ft = cfg.ollama.ft_tag
    ok = any(t.split(":")[0] == ft for t in tags)
    return CheckResult("ollama fine-tuned", ok, f"'{ft}' {'present' if ok else 'not built yet'}", fatal=False)


def check_wsl() -> CheckResult:
    distro = load_config().wsl.distro
    code, out = _run(["wsl.exe", "-l", "-v"])
    lines = [ln.replace("*", "").split() for ln in out.splitlines()[1:]]
    found = [ln for ln in lines if ln and ln[0] == distro]
    ok = code == 0 and bool(found) and found[0][-1] == "2"
    hint = "" if ok else " — run `wsl --install -d Ubuntu` as admin, then reboot"
    return CheckResult("wsl2 distro", ok, f"{distro}: {found[0] if found else 'missing'}{hint}")


def check_tsan() -> CheckResult:
    distro = load_config().wsl.distro
    script = (
        'd=$(mktemp -d); trap "rm -rf $d" EXIT; cat > "$d/t.c"; '
        'gcc -fsanitize=thread -g -O1 -pthread "$d/t.c" -o "$d/t" && "$d/t"; echo "rc=$?"'
    )
    code, out = _run(["wsl.exe", "-d", distro, "--exec", "bash", "-c", script], stdin=TSAN_PROBE)
    ok = "ThreadSanitizer: data race" in out
    return CheckResult("gcc -fsanitize=thread", ok, "race detected" if ok else out[-400:])


def check_wsl_tool(name: str, args: list[str]) -> Callable[[], CheckResult]:
    def check() -> CheckResult:
        distro = load_config().wsl.distro
        code, out = _run(["wsl.exe", "-d", distro, "--exec", *args])
        return CheckResult(name, code == 0, out.splitlines()[0] if out else "")

    return check


def check_node() -> CheckResult:
    code, out = _run(["node", "-v"]) if shutil.which("node") else (127, "node not found")
    major = int(out.lstrip("v").split(".")[0]) if code == 0 else 0
    # Brief targets Node 20 LTS; any LTS >= 20 works with Vite 6.
    return CheckResult("node", major >= 20, out)


def check_long_paths() -> CheckResult:
    if sys.platform != "win32":
        return CheckResult("long paths", True, "n/a", fatal=False)
    import winreg

    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem")
        value, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
    except OSError:
        value = 0
    hint = (
        ""
        if value == 1
        else " — run as admin: reg add HKLM\\SYSTEM\\CurrentControlSet\\Control\\FileSystem "
        "/v LongPathsEnabled /t REG_DWORD /d 1 /f"
    )
    return CheckResult("long paths", value == 1, f"LongPathsEnabled={value}{hint}", fatal=False)


def check_project_path() -> CheckResult:
    synced = "onedrive" in str(ROOT).lower()
    return CheckResult("project path", not synced, str(ROOT))


CHECKS: list[Callable[[], CheckResult]] = [
    check_project_path,
    check_gpu,
    check_torch,
    check_bitsandbytes,
    check_ollama,
    check_ollama_ft,
    check_wsl,
    check_tsan,
    check_wsl_tool("helgrind", ["valgrind", "--tool=helgrind", "--version"]),
    check_wsl_tool("bwrap", ["bwrap", "--version"]),
    check_wsl_tool("prlimit", ["prlimit", "--version"]),
    check_node,
    check_long_paths,
]


def run_doctor(as_json: bool = False) -> int:
    results = [check() for check in CHECKS]
    if as_json:
        print(json.dumps([r.__dict__ for r in results], indent=2))
    else:
        for r in results:
            mark = "OK  " if r.ok else ("FAIL" if r.fatal else "WARN")
            print(f"[{mark}] {r.name:<24} {r.detail}")
    failed = [r for r in results if r.fatal and not r.ok]
    if failed and not as_json:
        print(f"\n{len(failed)} required check(s) failed.")
    return 1 if failed else 0
