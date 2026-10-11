#!/usr/bin/env bash
# Installs the code-checker toolchain inside WSL. Run as root; safe to re-run.
set -euo pipefail

packages=(gcc libc6-dev valgrind bubblewrap util-linux coreutils)
missing=()
for p in "${packages[@]}"; do
    dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p")
done

if ((${#missing[@]})); then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq "${missing[@]}"
fi

gcc --version | head -1
valgrind --version
bwrap --version
echo "oslm-checker-setup: ok"
