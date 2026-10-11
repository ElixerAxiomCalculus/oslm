# models/

Only the Modelfile is tracked; everything else here is rebuilt locally and gitignored.

| Path | What | How to rebuild |
|---|---|---|
| `adapter/` | LoRA adapter (r=16, alpha=32) + `train_summary.json` (loss curve, best step) | `oslm train` (needs `data/sft/train.jsonl`; quit Ollama first) |
| `checkpoints/` | Trainer checkpoints (every 7 steps for the ~44-step run) | created by `oslm train`; resume with `oslm train --resume latest` |
| `merged/` | bf16 weights with the adapter folded in (intermediate, ~8 GB) | `oslm export` |
| `oslm-ft-f16.gguf` | F16 GGUF converted from `merged/` (intermediate, ~8 GB) | `oslm export` |
| `oslm-ft/Modelfile` | Ollama Modelfile: `FROM ../oslm-ft-f16.gguf`, the base model's chat TEMPLATE and stop PARAMETERs | `oslm export` |

`oslm export`:

1. merges the adapter with a streaming pass over the base safetensors (W' = W + (alpha/r)·B·A, one tensor at a time, ~1 GB peak RAM; no GPU needed);
2. tries `ollama create oslm-ft --quantize q4_K_M` straight from the merged safetensors;
3. Ollama 0.17 cannot import `Qwen3ForCausalLM` safetensors, so it falls back to llama.cpp's converter
   (`git clone --depth 1 https://github.com/ggml-org/llama.cpp tools/llama.cpp`) to write an F16 GGUF, and then
   `ollama create ... --quantize q4_K_M` quantizes that GGUF.

The result uses the same Q4_K_M quantization as the base `qwen3:4b-instruct-2507-q4_K_M`, which keeps the four-setup
comparison fair. After `oslm-ft` is registered, `merged/` and `oslm-ft-f16.gguf` can be deleted (`oslm clean --yes`),
but keep the GGUF if you may need to re-register the model in Ollama.
