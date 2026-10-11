from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

from oslm.config import ROOT, load_config
from oslm.train.merge import merge_adapter

log = logging.getLogger(__name__)


def base_modelfile_parts(base_tag: str) -> tuple[str, list[str]]:
    """TEMPLATE block and PARAMETER lines from the base model, so the fine-tune uses the same chat format."""
    out = subprocess.run(
        ["ollama", "show", "--modelfile", base_tag], capture_output=True, text=True, encoding="utf-8", check=True
    ).stdout
    template = re.search(r'^TEMPLATE """(.*?)"""', out, re.S | re.M)
    if not template:
        raise RuntimeError(f"no TEMPLATE in `ollama show --modelfile {base_tag}`")
    params = [ln for ln in out.splitlines() if ln.startswith("PARAMETER ")]
    return template.group(1), params


def write_modelfile(source: Path, modelfile: Path, base_tag: str) -> None:
    import os

    template, params = base_modelfile_parts(base_tag)
    # Ollama resolves FROM relative to the Modelfile, so the tracked Modelfile works on any checkout path.
    rel = Path(os.path.relpath(source, modelfile.parent)).as_posix()
    lines = [f"FROM {rel}", f'TEMPLATE """{template}"""', *params]
    if not any("num_ctx" in p for p in params):
        lines.append(f"PARAMETER num_ctx {load_config().ollama['num_ctx']}")
    modelfile.parent.mkdir(parents=True, exist_ok=True)
    modelfile.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ollama_create(tag: str, modelfile: Path, quantize: str | None) -> bool:
    cmd = ["ollama", "create", tag, "-f", str(modelfile)]
    if quantize:
        cmd += ["--quantize", quantize]
    log.info("running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    log.info(proc.stdout[-2000:] + proc.stderr[-2000:])
    return proc.returncode == 0


def _llama_cpp_fallback(merged: Path, models: Path) -> tuple[Path, str | None]:
    """F16 GGUF via llama.cpp's converter when Ollama cannot import the safetensors architecture.

    Returns (gguf, quantize): Ollama quantizes an F16 GGUF to Q4_K_M itself, so llama-quantize.exe is only used
    when it is present in tools/.
    """
    import os
    import sys

    tools = ROOT / "tools" / "llama.cpp"
    if not (tools / "convert_hf_to_gguf.py").exists():
        raise RuntimeError(
            "Ollama import failed and tools/llama.cpp is missing: "
            "git clone --depth 1 https://github.com/ggml-org/llama.cpp tools/llama.cpp"
        )
    f16 = models / "oslm-ft-f16.gguf"
    if not f16.exists():
        env = {**os.environ, "PYTHONPATH": str(tools / "gguf-py") + os.pathsep + os.environ.get("PYTHONPATH", "")}
        subprocess.run(
            [
                sys.executable,
                str(tools / "convert_hf_to_gguf.py"),
                str(merged),
                "--outfile",
                str(f16),
                "--outtype",
                "f16",
            ],
            check=True,
            env=env,
        )
    quantizer = next(iter(tools.rglob("llama-quantize.exe")), None)
    if quantizer is None:
        return f16, "q4_K_M"
    q4 = models / "oslm-ft-q4_k_m.gguf"
    subprocess.run([str(quantizer), str(f16), str(q4), "Q4_K_M"], check=True)
    return q4, None


def run_export(force: bool = False) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    models = cfg.path("models")
    adapter = models / "adapter"
    if not (adapter / "adapter_config.json").exists():
        raise SystemExit("models/adapter is missing — run `oslm train` first")
    merged = merge_adapter(adapter, models / "merged", force=force)
    tag, base = cfg.ollama["ft_tag"], cfg.ollama["base_tag"]
    modelfile = models / "oslm-ft" / "Modelfile"
    write_modelfile(merged, modelfile, base)
    if not _ollama_create(tag, modelfile, "q4_K_M"):
        log.warning("Ollama could not import the safetensors; falling back to llama.cpp")
        gguf, quantize = _llama_cpp_fallback(merged, models)
        write_modelfile(gguf, modelfile, base)
        if not _ollama_create(tag, modelfile, quantize):
            raise SystemExit("ollama create failed — see log above")
    smoke = subprocess.run(
        ["ollama", "run", tag, "In one sentence, what is a critical section?"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    print(f"{tag} registered. Smoke answer: {smoke.stdout.strip()[:300]}")
    return 0
