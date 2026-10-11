from __future__ import annotations

import gc
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

from oslm.config import ROOT, load_config
from oslm.sft.store import read_jsonl

log = logging.getLogger(__name__)

VRAM_PROBE_STEPS = 20
FALLBACK_MAX_LENGTH = 2048
LOSS_CHUNK = 512


def _paths() -> dict[str, Path]:
    cfg = load_config()
    models = cfg.path("models")
    return {
        "checkpoints": models / "checkpoints",
        "adapter": models / "adapter",
        "decisions": cfg.path("logs") / "train_decisions.jsonl",
        "vram": cfg.path("logs") / "vram.json",
    }


def _log_decision(text: str, **extra: Any) -> None:
    path = _paths()["decisions"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "decision": text, **extra}) + "\n")
    log.info("DECISION: %s %s", text, extra)


def _latest_checkpoint(directory: Path) -> str | None:
    ckpts = sorted(directory.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    return str(ckpts[-1]) if ckpts else None


def build_model(cfg: Any):  # noqa: ANN201
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    name = cfg.models["hf_base"]
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        name, quantization_config=bnb, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa"
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    # prepare_model_for_kbit_training upcasts the 152k x 2560 embedding (tied to lm_head) to fp32 (~1.6 GB);
    # it is frozen, so bf16 is enough and keeps training inside 8 GB instead of spilling to shared memory.
    model.get_input_embeddings().to(torch.bfloat16)
    model.get_output_embeddings().to(torch.bfloat16)
    t = cfg.train
    lora = LoraConfig(
        r=t["lora_r"],
        lora_alpha=t["lora_alpha"],
        lora_dropout=t["lora_dropout"],
        target_modules=list(t["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, tokenizer


def encode_example(messages: list[dict[str, str]], tokenizer: Any) -> dict[str, list[int]]:
    """Token ids for one chat; labels mask everything except the final assistant answer and its end-of-turn token."""
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False)
    if not full.startswith(prompt):
        raise ValueError("chat template renders the prompt differently inside the full conversation")
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(full[len(prompt) :], add_special_tokens=False)["input_ids"]
    return {
        "input_ids": prompt_ids + answer_ids,
        "attention_mask": [1] * (len(prompt_ids) + len(answer_ids)),
        "labels": [-100] * len(prompt_ids) + answer_ids,
    }


class PadCollator:
    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        width = max(len(b["input_ids"]) for b in batch)

        def pad(key: str, value: int) -> Any:
            return torch.tensor([b[key] + [value] * (width - len(b[key])) for b in batch])

        return {
            "input_ids": pad("input_ids", self.pad_id),
            "attention_mask": pad("attention_mask", 0),
            "labels": pad("labels", -100),
        }


def answer_only_loss(model: Any, inputs: dict[str, Any], num_items_in_batch: Any = None) -> Any:
    """Causal-LM loss computed only at answer positions.

    Full logits over Qwen's ~152k vocabulary for a 3k-token sequence need several GB, more than an 8 GB GPU
    has left after the 4-bit model; projecting just the labelled positions gives the identical loss.
    """
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    causal_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    labels = inputs["labels"]
    hidden = causal_lm.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).last_hidden_state
    targets = labels[:, 1:]
    mask = targets != -100
    h, y = hidden[:, :-1][mask], targets[mask]
    weight = causal_lm.lm_head.weight

    def chunk_loss(hc: Any, yc: Any) -> Any:
        return F.cross_entropy(F.linear(hc.to(weight.dtype), weight).float(), yc, reduction="sum")

    # Recomputing each chunk's logits in backward caps logit memory at LOSS_CHUNK rows at a time.
    total = sum(
        checkpoint(chunk_loss, h[i : i + LOSS_CHUNK], y[i : i + LOSS_CHUNK], use_reentrant=False)
        for i in range(0, h.shape[0], LOSS_CHUNK)
    )
    denom = num_items_in_batch if num_items_in_batch is not None else mask.sum()
    return total / denom


def _train_once(max_length: int, resume: str | None, max_steps: int | None) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from transformers import Trainer, TrainerCallback, TrainingArguments

    class AnswerOnlyTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # noqa: ANN001, ANN201
            loss = answer_only_loss(model, inputs, num_items_in_batch)
            return (loss, {"loss": loss}) if return_outputs else loss

    cfg = load_config()
    t = cfg.train
    paths = _paths()
    sft_dir = cfg.path("sft")
    # Truncation would cut answers mid-sentence, so over-length samples are dropped instead.
    train_rows = [r for r in read_jsonl(sft_dir / "train.jsonl") if r.get("n_tokens", 0) <= max_length]
    val_rows = [r for r in read_jsonl(sft_dir / "val.jsonl") if r.get("n_tokens", 0) <= max_length]
    if not train_rows:
        raise SystemExit("data/sft/train.jsonl is empty — run `oslm sft` first")
    model, tokenizer = build_model(cfg)
    train_ds = Dataset.from_list([encode_example(r["messages"], tokenizer) for r in train_rows])
    val_ds = Dataset.from_list([encode_example(r["messages"], tokenizer) for r in val_rows])
    probe = {"exceeded": False, "peak_gb": 0.0}

    class VramProbe(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001, ANN201
            if state.global_step <= VRAM_PROBE_STEPS:
                peak = torch.cuda.max_memory_reserved() / 1024**3
                probe["peak_gb"] = max(probe["peak_gb"], peak)
                if state.global_step == VRAM_PROBE_STEPS:
                    paths["vram"].write_text(
                        json.dumps({"max_length": max_length, "peak_reserved_gb": round(peak, 2)}), encoding="utf-8"
                    )
                    if peak > t["vram_limit_gb"] and max_length > FALLBACK_MAX_LENGTH:
                        probe["exceeded"] = True
                        control.should_training_stop = True
            return control

    # transformers 5 dropped warmup_ratio; derive the same 3% as a step count.
    total_steps = max_steps or math.ceil(len(train_rows) / (t["batch_size"] * t["grad_accum"])) * t["epochs"]
    warmup_steps = max(1, round(total_steps * t["warmup_ratio"]))
    # A few hundred examples give only ~44 optimizer steps, so evaluate/save about six times per run instead of
    # the configured every-50/100 steps (which would never fire and leave no best checkpoint to pick).
    eval_steps = min(t["eval_steps"], max(2, total_steps // 6))
    if eval_steps != t["eval_steps"]:
        _log_decision(f"eval/save every {eval_steps} steps (run has only {total_steps} optimizer steps)")
    args = TrainingArguments(
        output_dir=str(paths["checkpoints"]),
        per_device_train_batch_size=t["batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=t["grad_accum"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        learning_rate=t["lr"],
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        num_train_epochs=t["epochs"],
        max_steps=max_steps or -1,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps if eval_steps != t["eval_steps"] else t["save_steps"],
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        logging_steps=5,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        prediction_loss_only=True,
        seed=cfg["seed"],
        data_seed=cfg["seed"],
        report_to="none",
    )
    # Plain Trainer: TRL 1.15's SFTTrainer always routes the loss through a Triton kernel, which is Linux-only.
    trainer = AnswerOnlyTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=PadCollator(tokenizer.pad_token_id),
        callbacks=[VramProbe()],
    )
    resume_from = _latest_checkpoint(paths["checkpoints"]) if resume == "latest" else resume
    trainer.train(resume_from_checkpoint=resume_from)
    result = {"exceeded": probe["exceeded"], "peak_gb": probe["peak_gb"], "log_history": trainer.state.log_history}
    if not probe["exceeded"]:
        paths["adapter"].mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(str(paths["adapter"]))
        tokenizer.save_pretrained(str(paths["adapter"]))
        evals = [h for h in trainer.state.log_history if "eval_loss" in h]
        summary = {
            "max_length": max_length,
            "peak_reserved_gb_first_steps": round(probe["peak_gb"], 2),
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_eval_loss": trainer.state.best_metric,
            "eval_losses": [(h["step"], round(h["eval_loss"], 4)) for h in evals],
            "train_losses": [(h["step"], round(h["loss"], 4)) for h in trainer.state.log_history if "loss" in h],
            "train_examples": len(train_rows),
            "val_examples": len(val_rows),
        }
        (paths["adapter"] / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_training(resume: str | None = None, max_steps: int | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    free, total = torch.cuda.mem_get_info()
    if free / 1024**3 < 6.5:
        raise SystemExit(
            f"only {free / 1024**3:.1f} GB of {total / 1024**3:.1f} GB VRAM free — quit Ollama completely first"
        )
    cfg = load_config()
    # Spilling past dedicated VRAM into Windows shared memory preceded a driver crash (bugcheck 0x1E) on
    # 2026-10-10; capping the allocator keeps every tensor on the card (profiled peak ~6.5 GB at 2048 tokens).
    torch.cuda.set_per_process_memory_fraction(min(1.0, cfg.train["vram_limit_gb"] / (total / 1024**3)))
    max_length = cfg.train["max_seq_length"]
    _log_decision("start training", max_length=max_length, resume=resume, max_steps=max_steps, pid=os.getpid())
    result = _train_once(max_length, resume, max_steps)
    if result["exceeded"]:
        _log_decision(
            f"peak VRAM {result['peak_gb']:.2f} GB > {cfg.train['vram_limit_gb']} GB in first "
            f"{VRAM_PROBE_STEPS} steps; restarting with max_length={FALLBACK_MAX_LENGTH}"
        )
        for ckpt in _paths()["checkpoints"].glob("checkpoint-*"):
            import shutil

            shutil.rmtree(ckpt, ignore_errors=True)
        result = _train_once(FALLBACK_MAX_LENGTH, None, max_steps)
    _log_decision("training finished", peak_gb=round(result["peak_gb"], 2))
    print(f"adapter saved to {_paths()['adapter'].relative_to(ROOT)}")
    return 0
