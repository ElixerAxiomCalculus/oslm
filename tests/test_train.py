from __future__ import annotations

import pytest
import torch

from oslm.train import qlora


def _tiny_qwen():  # noqa: ANN202
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=600,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        tie_word_embeddings=True,
    )
    return Qwen3ForCausalLM(cfg).eval()


def test_answer_only_loss_matches_full_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qlora, "LOSS_CHUNK", 3)  # force several chunks
    model = _tiny_qwen()
    ids = torch.randint(5, 600, (2, 12))
    labels = ids.clone()
    labels[0, :7] = -100
    labels[1, :4] = -100
    labels[1, 10:] = -100
    attn = torch.ones_like(ids)
    inputs = {"input_ids": ids, "attention_mask": attn, "labels": labels}
    reference = model(input_ids=ids, attention_mask=attn, labels=labels).loss
    ours = qlora.answer_only_loss(model, inputs)
    assert torch.allclose(ours, reference, atol=1e-5)


def test_answer_only_loss_backpropagates() -> None:
    model = _tiny_qwen().train()
    ids = torch.randint(5, 600, (1, 8))
    labels = ids.clone()
    labels[0, :5] = -100
    loss = qlora.answer_only_loss(model, {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels})
    loss.backward()
    assert model.model.layers[0].self_attn.q_proj.weight.grad is not None


def test_pad_collator() -> None:
    batch = qlora.PadCollator(pad_id=0)(
        [
            {"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1], "labels": [-100, 6, 7]},
            {"input_ids": [8], "attention_mask": [1], "labels": [8]},
        ]
    )
    assert batch["input_ids"].tolist() == [[5, 6, 7], [8, 0, 0]]
    assert batch["labels"].tolist() == [[-100, 6, 7], [8, -100, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


@pytest.mark.integration
def test_encode_example_masks_everything_but_the_answer() -> None:
    from oslm.index.embed import get_tokenizer

    tok = get_tokenizer()
    msgs = [
        {"role": "system", "content": "You are a tutor."},
        {"role": "user", "content": "What is a mutex?"},
        {"role": "assistant", "content": "A mutex is a lock [1]."},
    ]
    enc = qlora.encode_example(msgs, tok)
    trained = [t for t, lab in zip(enc["input_ids"], enc["labels"], strict=True) if lab != -100]
    text = tok.decode(trained)
    assert text.startswith("A mutex is a lock [1].") and "<|im_end|>" in text
    assert "What is a mutex?" not in text and len(enc["input_ids"]) == len(enc["labels"])


def test_streaming_merge_matches_peft(tmp_path) -> None:  # noqa: ANN001
    from peft import LoraConfig, get_peft_model

    from oslm.train.merge import merge_adapter

    base = _tiny_qwen()
    base_dir = tmp_path / "base"
    base.save_pretrained(str(base_dir), safe_serialization=True)
    peft_model = get_peft_model(_tiny_qwen(), LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "down_proj"]))
    with torch.no_grad():
        for name, p in peft_model.named_parameters():
            if "lora_B" in name:
                p.normal_(0, 0.02)  # lora_B starts at zero; give it values so the merge actually changes weights
    adapter_dir = tmp_path / "adapter"
    peft_model.save_pretrained(str(adapter_dir))
    reference = peft_model.merge_and_unload().state_dict()

    out = merge_adapter(adapter_dir, tmp_path / "merged", base_dir=base_dir)
    from safetensors.torch import load_file

    merged = {}
    for f in out.glob("*.safetensors"):
        merged.update(load_file(str(f)))
    for key in ("model.layers.0.self_attn.q_proj.weight", "model.layers.1.mlp.down_proj.weight"):
        assert torch.allclose(merged[key].float(), reference[key].float(), atol=1e-5)
        assert not torch.allclose(merged[key].float(), base.state_dict()[key].float())
    assert (out / "config.json").exists() and (out / "model.safetensors.index.json").exists()
