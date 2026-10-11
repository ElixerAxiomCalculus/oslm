from __future__ import annotations

import logging
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from oslm.config import load_config
from oslm.rag.citations import cited_numbers
from oslm.rag.prompts import SYSTEM_PROMPT
from oslm.sft.refresh import refresh_item
from oslm.sft.store import read_jsonl, write_jsonl

log = logging.getLogger(__name__)


def to_messages(item: dict[str, Any]) -> list[dict[str, str]]:
    """Exactly the message layout the pipeline sends to Ollama (system, history, user turn, then the answer)."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": h["role"], "content": h["content"]} for h in item.get("history", [])]
    messages.append({"role": "user", "content": item["user_message"]})
    messages.append({"role": "assistant", "content": item["answer"]})
    return messages


def dedup_key(item: dict[str, Any]) -> str:
    text = item["question"]
    if item.get("history"):
        text = item["history"][0]["content"][:300] + " || " + text
    return re.sub(r"\s+", " ", text).strip()


def _normalise(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def near_duplicates(keys: list[str], refs: list[str], threshold: float) -> tuple[set[int], set[int]]:
    """Returns (indices duplicated within `keys`, indices too close to any of `refs`)."""
    from oslm.index.embed import embed_passages

    vecs = embed_passages(keys)
    dup: set[int] = set()
    sims = vecs @ vecs.T
    for i in range(len(keys)):
        if i in dup:
            continue
        for j in np.nonzero(sims[i, i + 1 :] > threshold)[0]:
            dup.add(i + 1 + int(j))
    leak: set[int] = set()
    if refs:
        ref_vecs = embed_passages(refs)
        leak = {int(i) for i in np.nonzero((vecs @ ref_vecs.T).max(axis=1) > threshold)[0]}
        exact = {_normalise(r) for r in refs}
        leak |= {i for i, k in enumerate(keys) if _normalise(k) in exact}
    return dup, leak


def eval_questions() -> list[str]:
    eval_dir = load_config().path("eval")
    refs = []
    for name in ("testset.jsonl", "probes.jsonl"):
        for row in read_jsonl(eval_dir / name):
            q = row.get("question") or row.get("message") or ""
            if q.strip():
                refs.append(q)
    return refs


def token_length(messages: list[dict[str, str]]) -> int:
    from oslm.index.embed import get_tokenizer

    tokenizer = get_tokenizer()
    # transformers 5 returns a dict from apply_chat_template(tokenize=True), so render text and encode it.
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return len(tokenizer.encode(text, add_special_tokens=False))


def format_dataset(verified: list[dict[str, Any]], out_dir: Path, seed: int) -> dict[str, Any]:
    cfg = load_config()
    max_len = cfg.train["max_seq_length"]
    ok = [it for it in verified if it.get("ok")]
    keys = [dedup_key(it) for it in ok]
    dup, leak = near_duplicates(keys, eval_questions(), cfg.sft["dedup_cosine"])
    kept, dropped = [], Counter()
    for i, it in enumerate(ok):
        if i in leak:
            dropped["eval leakage"] += 1
            continue
        if i in dup:
            dropped["near duplicate"] += 1
            continue
        messages = to_messages(it)
        n_tok = token_length(messages)
        # Long items (mostly bug reviews with code) shed trailing context chunks the answer never cites,
        # instead of being dropped; the question and answer stay intact.
        while n_tok > max_len and it.get("chunk_ids") and len(it["chunk_ids"]) > 1:
            last = len(it["chunk_ids"])
            if last in cited_numbers(it["answer"], last):
                break
            it["chunk_ids"] = it["chunk_ids"][:-1]
            refresh_item(it)
            messages = to_messages(it)
            n_tok = token_length(messages)
            dropped["context chunk trimmed"] += 1
        if n_tok > max_len:
            dropped["too long"] += 1
            continue
        kept.append(
            {
                "id": it["id"],
                "category": it["category"],
                "variant": it.get("variant", ""),
                "mode": it["mode"],
                "n_tokens": n_tok,
                "messages": messages,
            }
        )

    rng = random.Random(seed)
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in kept:
        by_cat[row["category"]].append(row)
    train, val = [], []
    for rows in by_cat.values():
        rng.shuffle(rows)
        n_val = max(1, round(len(rows) * cfg.sft["val_frac"]))
        val += rows[:n_val]
        train += rows[n_val:]
    rng.shuffle(train)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)
    stats = {
        "verified_ok": len(ok),
        "kept": len(kept),
        "train": len(train),
        "val": len(val),
        "dropped": dict(dropped),
        "by_category": dict(Counter(r["category"] for r in kept)),
        "by_variant": dict(Counter(r["variant"] for r in kept)),
        "tokens_p95": int(np.percentile([r["n_tokens"] for r in kept], 95)) if kept else 0,
    }
    log.info("formatted: %s", stats)
    return stats
