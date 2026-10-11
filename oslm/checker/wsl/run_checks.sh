#!/usr/bin/env bash
# Compiles C source read from stdin and runs it plain, under ThreadSanitizer and under
# Helgrind, each inside a bubblewrap sandbox. Prints exactly one JSON object on stdout.
set -uo pipefail

MAX_BYTES=${OSLM_MAX_BYTES:-20480}
RUN_WALL_S=10
COMPILE_WALL_S=12
CPU_S=5
FSIZE=1048576
OUTPUT_LIMIT=65536
MEM_MAX=768M
PLAIN_AS=$((1024 * 1024 * 1024))
# second_deadlock_stack gives both acquisition sites, which the tutor needs to explain lock-order bugs.
TSAN_OPTS="halt_on_error=0 exitcode=66 second_deadlock_stack=1 allocator_may_return_null=1 max_allocation_size_mb=512 report_signal_unsafe=0"

work=$(mktemp -d /tmp/oslm-check.XXXXXX)
trap 'rm -rf "$work"' EXIT
mkdir "$work/box" "$work/res"

head -c $((MAX_BYTES + 1)) > "$work/raw.c"
tr -d '\r' < "$work/raw.c" > "$work/box/prog.c"

use_scope=0
# The systemd user session sometimes is still starting right after wsl.exe boots the distro.
for _ in 1 2 3 4; do
    if systemd-run --user --scope -q true >/dev/null 2>&1; then
        use_scope=1
        break
    fi
    sleep 0.5
done

now_ms() { echo $(($(date +%s%N) / 1000000)); }

sandbox() {
    local label=$1 as_limit=$2
    shift 2
    local scope=()
    if ((use_scope)); then
        scope=(systemd-run --user --scope -q -p "MemoryMax=$MEM_MAX" -p MemorySwapMax=0 -p TasksMax=64)
    fi
    # Soft CPU limit delivers SIGXCPU first so runaway loops are distinguishable from OOM kills.
    local limits=(--cpu="$CPU_S:$((CPU_S + 1))" --fsize="$FSIZE" --core=0 --nofile=256)
    [[ $as_limit != 0 ]] && limits+=(--as="$as_limit")
    local wall=$RUN_WALL_S
    [[ $label == compile_* ]] && wall=$COMPILE_WALL_S
    local start
    start=$(now_ms)
    timeout --kill-after=2 "$wall" "${scope[@]}" \
        bwrap --unshare-all --die-with-parent --new-session \
        --ro-bind /usr /usr \
        --symlink usr/bin /bin --symlink usr/sbin /sbin \
        --symlink usr/lib /lib --symlink usr/lib64 /lib64 \
        --ro-bind /etc/ld.so.cache /etc/ld.so.cache \
        --proc /proc --dev /dev \
        --bind "$work/box" /work --remount-ro / --chdir /work \
        --clearenv --setenv PATH /usr/bin --setenv TMPDIR /work --setenv TSAN_OPTIONS "$TSAN_OPTS" --setenv LANG C.UTF-8 \
        prlimit "${limits[@]}" -- "$@" \
        > "$work/res/$label.out" 2>&1 < /dev/null
    echo $? > "$work/res/$label.rc"
    echo $(($(now_ms) - start)) > "$work/res/$label.ms"
}

emit() {
    python3 - "$work" "$1" "$OUTPUT_LIMIT" "$use_scope" "$MAX_BYTES" <<'PY'
import json, os, sys
work, status, limit, scoped, max_bytes = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] == "1", int(sys.argv[5])
res = os.path.join(work, "res")
steps = {}
for label in ("compile_plain", "compile_tsan", "run", "tsan", "helgrind"):
    out_path = os.path.join(res, label + ".out")
    if not os.path.exists(out_path):
        continue
    with open(out_path, "rb") as f:
        data = f.read(limit + 1)
    def num(ext):
        with open(os.path.join(res, label + ext)) as f:
            return int(f.read().strip() or -1)
    steps[label] = {
        "exit_code": num(".rc"),
        "elapsed_ms": num(".ms"),
        "output": data[:limit].decode("utf-8", "replace"),
        "truncated": len(data) > limit,
    }
helgrind_log = os.path.join(work, "box", "helgrind.log")
if "helgrind" in steps and os.path.exists(helgrind_log):
    with open(helgrind_log, "rb") as f:
        data = f.read(limit + 1)
    steps["helgrind"]["output"] = data[:limit].decode("utf-8", "replace")
    steps["helgrind"]["truncated"] = len(data) > limit
print(json.dumps({
    "status": status,
    "source_bytes": os.path.getsize(os.path.join(work, "raw.c")),
    "max_bytes": max_bytes,
    "memory_scoped": scoped,
    "steps": steps,
}))
PY
}

size=$(stat -c %s "$work/raw.c")
if ((size > MAX_BYTES)); then
    emit too_large
    exit 0
fi
if ! grep -q '[^[:space:]]' "$work/box/prog.c"; then
    emit empty
    exit 0
fi

sandbox compile_plain 0 gcc -Wall -Wextra -pthread -g -fdiagnostics-color=never -o prog prog.c &
sandbox compile_tsan 0 gcc -fsanitize=thread -g -O1 -pthread -fdiagnostics-color=never -o prog_tsan prog.c &
wait

if [[ ! -x $work/box/prog || ! -x $work/box/prog_tsan ]]; then
    emit compile_error
    exit 0
fi

sandbox run "$PLAIN_AS" ./prog &
sandbox tsan 0 ./prog_tsan &
sandbox helgrind 0 valgrind --tool=helgrind --vgdb=no --error-exitcode=67 --log-file=helgrind.log ./prog &
wait

emit completed
