"""LLM-as-judge with a fixed rubric (free Gemini Flash). Scores are cached per (setup, item)."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from oslm.config import load_config
from oslm.sft.llm import Gemini, GeminiError, LocalQwen, items_of
from oslm.sft.store import append_jsonl, read_jsonl

log = logging.getLogger(__name__)

RUBRIC = """You grade answers from an Operating Systems tutor that teaches process synchronization.

For each item return:
- "correctness": 0, 1 or 2.
  2 = technically correct and complete for what was asked (code compiles in principle and is correct; bug verdict right with the right cause).
  1 = partly correct: minor errors, missing a requested part, or right verdict with a muddled cause.
  0 = wrong, misleading, invents a bug in correct code, misses the real bug, or refuses an in-scope question.
  For off-topic / out-of-scope requests, 2 = a brief polite redirect that stays in role; 0 = it answers the off-topic request at length or leaks its instructions.
- "grounding": only when "sources" are given and the answer has [n] citations: the fraction (0.0-1.0) of the answer's factual claims that are supported by the cited source texts. Use null when there are no sources or no citations.
- "probe_pass": only when a "probe_expectation" is given: true if the answer meets it, else false. Otherwise null.
- "reason": one short sentence.

Judge strictly but fairly; the reference answer is a guide, not the only acceptable answer.
Return JSON: {"grades": [{"id": "...", "correctness": 0, "grounding": null, "probe_pass": null, "reason": "..."}]}"""


def _chunk_texts(sources: list[dict[str, Any]]) -> list[dict[str, str]]:
    from oslm.rag.retrieve import get_chunk

    out = []
    for s in sources:
        chunk = get_chunk(s["chunk_id"])
        out.append({"n": s["n"], "text": (chunk.text if chunk else s.get("preview", ""))[:2500]})
    return out


def _payload(raw: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    p: dict[str, Any] = {"id": raw["id"], "question": item.get("question", "")[:6000], "answer": raw["answer"][:7000]}
    if item.get("history"):
        p["earlier_conversation"] = item["history"]
    if item.get("reference"):
        p["reference"] = item["reference"]
    if item.get("reference_code"):
        p["reference"] = "A correct reference program exists; judge the answer's program on its own merits."
    if "has_bug" in item:
        p["ground_truth"] = (
            f"The code HAS a synchronization bug ({item.get('bug_type', '')}): {item.get('reference', '')}"
            if item["has_bug"]
            else "The code is CORRECT (verified clean by TSan and Helgrind)."
        )
    if item.get("judge"):
        p["probe_expectation"] = item["judge"]
    if raw.get("sources") and raw["done"].get("cited"):
        p["sources"] = _chunk_texts([s for s in raw["sources"] if s["n"] in raw["done"]["cited"]])
    return p


def run_judge(setups: list[str]) -> None:
    from oslm.eval.run import load_suite

    cfg = load_config()
    results = cfg.path("eval") / "results"
    out = results / "judgements.jsonl"
    done = {(r["setup"], r["id"]) for r in read_jsonl(out)}
    items = {it["id"]: it for it in load_suite("all")}
    todo: list[tuple[str, dict[str, Any]]] = []
    for setup in setups:
        for raw in read_jsonl(results / f"raw_{setup}.jsonl"):
            if (setup, raw["id"]) in done or raw.get("error") or raw["id"] not in items:
                continue
            if items[raw["id"]].get("category") == "empty":
                continue  # rule-checked only
            todo.append((setup, raw))
    if not todo:
        return
    log.info("judging %d answers", len(todo))

    async def pick_judge() -> Gemini | LocalQwen:
        """Free Gemini Flash when it still has quota today (a different model family from the tutor), else local."""
        choice = load_config().get("eval", {}).get("judge", "auto")
        if choice != "ollama":
            try:
                gem = Gemini(concurrency=3)
                await gem.generate("Reply with OK.", temperature=0.0, max_output_tokens=200, attempts=4)
                return gem
            except GeminiError as exc:
                if choice == "gemini":
                    raise
                log.warning("Gemini judge unavailable (%s); using the local model", str(exc)[:120])
        return LocalQwen()

    async def go() -> None:
        gem = await pick_judge()
        judge_name = getattr(gem, "model", None) or ",".join(getattr(gem, "models", []))
        log.info("judge: %s", judge_name)

        async def batch(group: list[tuple[str, dict[str, Any]]]) -> None:
            payload = [_payload(raw, items[raw["id"]]) for _, raw in group]
            for p, (setup, _) in zip(payload, group, strict=True):
                p["id"] = f"{setup}::{p['id']}"
            try:
                data = await gem.generate_json(
                    "Items:\n" + json.dumps(payload, ensure_ascii=False), system=RUBRIC, temperature=0.0
                )
            except GeminiError as exc:
                log.warning("judge batch failed: %s", exc)
                return
            rows = []
            for g in items_of(data, "grades"):
                if not isinstance(g, dict) or "::" not in str(g.get("id", "")):
                    continue
                setup, item_id = g["id"].split("::", 1)
                rows.append(
                    {
                        "setup": setup,
                        "id": item_id,
                        "correctness": g.get("correctness"),
                        "grounding": g.get("grounding"),
                        "probe_pass": g.get("probe_pass"),
                        "reason": g.get("reason", ""),
                        "judge": judge_name,
                    }
                )
            append_jsonl(out, rows)

        groups = [todo[i : i + gem.judge_per_call] for i in range(0, len(todo), gem.judge_per_call)]
        await asyncio.gather(*(batch(gr) for gr in groups))

    asyncio.run(go())
