"""Re-derives each SFT item's mode and user message from the current router and prompts.

Prompts and routing rules evolve; rebuilding before verification guarantees the training data matches exactly
what the live pipeline would send for the same question and context.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from oslm.rag.prompts import ContextChunk, build_user_message
from oslm.rag.router import refine_followup, route

FIXED_MODES = {"greeting", "off_topic", "os_out_of_scope", "not_in_notes"}
AUTHORED_FORMATS = {"trace", "code_write", "bug_find"}


@lru_cache(maxsize=1)
def _chunks() -> dict[str, dict[str, Any]]:
    from oslm.sft.generate import load_chunks

    return {c["chunk_id"]: c for c in load_chunks()}


def current_mode(item: dict[str, Any]) -> str:
    stored = item["mode"]
    if stored in FIXED_MODES:
        return stored
    if item.get("history"):
        r = route(item["question"], has_history=True)
        mode = refine_followup(item.get("rewritten") or item["question"]) if r.mode == "concept" else r.mode
    else:
        r = route(item["question"])
        mode = stored if r.mode == "empty" or r.ambiguous else r.mode
    if mode in FIXED_MODES:
        return stored
    # The answer was written in its category's format (e.g. a trace table); the instruction must ask for that format.
    if mode == "concept" and item.get("category") in AUTHORED_FORMATS:
        return item["category"]
    return mode


def refresh_item(item: dict[str, Any]) -> dict[str, Any]:
    item["mode"] = current_mode(item)
    ids = item.get("chunk_ids")
    context = None
    if ids is not None:
        chunks = _chunks()
        context = [
            ContextChunk(c.get("source_name", ""), c.get("section", ""), str(c.get("page_or_slide", "")), c["text"])
            for c in (chunks[i] for i in ids if i in chunks)
        ]
    question = item.get("rewritten") or item["question"]
    item["user_message"] = build_user_message(
        question, item["mode"], context, followup=bool(item.get("history")), topics=item.get("topics", ())
    )
    return item
