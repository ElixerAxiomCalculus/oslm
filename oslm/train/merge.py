from __future__ import annotations

import gc
import json
import logging
import shutil
from pathlib import Path

from oslm.config import load_config

log = logging.getLogger(__name__)

OUT_SHARD_BYTES = 1 << 30
COPY_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
)


def _base_dir(name: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(name, local_files_only=True))


def _lora_deltas(adapter_dir: Path) -> tuple[dict[str, tuple[object, object]], float]:
    """Maps base weight names ('model.layers.0.self_attn.q_proj.weight') to their (A, B) LoRA factors."""
    from safetensors.torch import load_file

    cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    scale = cfg["lora_alpha"] / cfg["r"]
    tensors = load_file(str(adapter_dir / "adapter_model.safetensors"))
    pairs: dict[str, dict[str, object]] = {}
    for key, value in tensors.items():
        if ".lora_A." not in key and ".lora_B." not in key:
            continue
        which = "A" if ".lora_A." in key else "B"
        module = key.split(".lora_")[0].removeprefix("base_model.model.")
        pairs.setdefault(module + ".weight", {})[which] = value
    return {k: (v["A"], v["B"]) for k, v in pairs.items()}, scale


def merge_adapter(adapter_dir: Path, out_dir: Path, force: bool = False, base_dir: Path | None = None) -> Path:
    """Folds the LoRA update W' = W + (alpha/r)·B·A into the bf16 base weights, one tensor at a time.

    Streaming keeps peak memory near one output shard (~1 GB) instead of the whole 8 GB model, so the merge
    runs on a 16 GB laptop with other applications open, without the GPU.
    """
    if (out_dir / "config.json").exists() and not force:
        log.info("merged weights already exist at %s", out_dir)
        return out_dir
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    base_dir = base_dir or _base_dir(load_config().models["hf_base"])
    deltas, scale = _lora_deltas(adapter_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.safetensors"):
        old.unlink()

    weight_map: dict[str, str] = {}
    shard: dict[str, torch.Tensor] = {}
    shard_bytes, shard_idx, merged = 0, 0, 0

    def flush() -> None:
        nonlocal shard, shard_bytes, shard_idx
        if not shard:
            return
        shard_idx += 1
        name = f"model-{shard_idx:05d}.safetensors"
        save_file(shard, str(out_dir / name), metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(shard, name))
        shard, shard_bytes = {}, 0
        gc.collect()

    for src in sorted(base_dir.glob("*.safetensors")):
        with safe_open(str(src), framework="pt") as f:
            for key in f.keys():  # noqa: SIM118 - safe_open handles are not dicts
                tensor = f.get_tensor(key)
                if key in deltas:
                    a, b = deltas.pop(key)
                    update = (b.float() @ a.float()) * scale
                    tensor = (tensor.float() + update).to(tensor.dtype)
                    merged += 1
                shard[key] = tensor.contiguous()
                shard_bytes += tensor.numel() * tensor.element_size()
                if shard_bytes >= OUT_SHARD_BYTES:
                    flush()
    flush()
    if deltas:
        raise RuntimeError(f"{len(deltas)} LoRA modules had no matching base weight, e.g. {next(iter(deltas))}")

    total = sum((out_dir / n).stat().st_size for n in set(weight_map.values()))
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2), encoding="utf-8"
    )
    for name in COPY_FILES:
        for src_dir in (adapter_dir, base_dir):
            if (src_dir / name).exists():
                shutil.copy2(src_dir / name, out_dir / name)
                break
    log.info("merged %d LoRA modules (scale %.2f) into %s (%d shards)", merged, scale, out_dir, shard_idx)
    return out_dir
