from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter
from typing import Any

from oslm.rag.citations import cited_numbers
from oslm.rag.pipeline import complete_c_programs

log = logging.getLogger(__name__)

DECLINE = re.compile(
    r"(do(es)?n['’]?t|do(es)? not|don['’]t|aren['’]?t|are not|isn['’]?t|is not|not)\s+(cover|covered|include|contain|discuss|address|in (the|my|your|our) notes)"
    r"|outside (what|the scope)|only (help|cover|teach)|beyond (the|what)|not part of",
    re.I,
)
VERDICT_BUG = re.compile(r"^\W*verdict\W*\s*bug found", re.I | re.M)
VERDICT_OK = re.compile(r"^\W*verdict\W*\s*no (synchronization )?bug", re.I | re.M)
FILLER = re.compile(r"^(great|good|excellent) question|^as an ai|^sure[,!]", re.I)


def n_context(item: dict[str, Any]) -> int:
    ids = item.get("chunk_ids")
    return len(ids) if ids else 0


def rule_problems(item: dict[str, Any]) -> list[str]:
    a = item.get("answer", "")
    problems: list[str] = []
    if len(a) < 20:
        return ["empty answer"]
    if FILLER.search(a.strip()):
        problems.append("filler opening")
    n = n_context(item)
    raw_cites = [int(x) for x in re.findall(r"\[(\d{1,2})\]", re.sub(r"```.*?```", "", a, flags=re.S))]
    if any(c < 1 or c > n for c in raw_cites):
        problems.append(f"citation outside 1..{n}")
    if item.get("chunk_ids") is None and raw_cites:
        problems.append("citations in a no-notes answer")
    variant, mode, cat = item.get("variant", ""), item.get("mode"), item.get("category")

    if variant == "distractor":
        bad = re.search(r"chunk \[(\d)\]", item.get("note", ""))
        if bad and int(bad.group(1)) in cited_numbers(a, n):
            problems.append("cited the distractor chunk")
    if variant in ("irrelevant_context", "not_in_notes_mode") or mode == "not_in_notes":
        if not DECLINE.search(a):
            problems.append("should decline but does not")
        if raw_cites:
            problems.append("declining answer has citations")
        if len(a.split()) > 160:
            problems.append("decline too long")
    if mode in ("greeting", "off_topic") and len(a.split()) > 90:
        problems.append("redirect too long")
    if mode == "off_topic" and "How you teach" in a:
        problems.append("leaked system prompt")
    if cat == "bug_find" and mode == "bug_find":
        if item.get("has_bug") and not VERDICT_BUG.search(a):
            problems.append("missing 'Verdict: bug found'")
        if item.get("has_bug") is False and not VERDICT_OK.search(a):
            problems.append("missing 'Verdict: no synchronization bug'")
    if mode == "trace" and len(re.findall(r"^\s*\|", a, re.M)) < 4:
        problems.append("trace without an interleaving table")
    if mode == "code_write" and not complete_c_programs(a):
        problems.append("code answer without a complete C program")
    return problems


async def verify_items(items: list[dict[str, Any]], bridge: Any) -> list[dict[str, Any]]:
    """Adds `problems`, `checks` and `ok` to each item. Programs in answers must be TSan/Helgrind clean."""
    programs: list[tuple[int, str]] = []
    for idx, it in enumerate(items):
        it["problems"] = rule_problems(it)
        for code in complete_c_programs(it.get("answer", "")):
            programs.append((idx, code))
    log.info("checking %d programs from answers", len(programs))
    reports = await bridge.check_many([code for _, code in programs])
    for (idx, _), report in zip(programs, reports, strict=True):
        it = items[idx]
        it.setdefault("checks", []).append({"status": report.status, "summary": report.summary})
        if report.status == "checker_unavailable":
            it["problems"].append("checker unavailable")
        elif report.status != "clean":
            it["problems"].append(f"program not clean: {report.summary[:120]}")
    for it in items:
        it["ok"] = not it["problems"]
    counts = Counter(p.split(":")[0] for it in items for p in it["problems"])
    log.info("verified %d items: %d ok; problems: %s", len(items), sum(it["ok"] for it in items), dict(counts))
    return items


def run_verify(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from oslm.checker.wsl_bridge import CheckerBridge

    async def go() -> list[dict[str, Any]]:
        bridge = CheckerBridge(concurrency=3)
        await bridge.warmup()
        return await verify_items(items, bridge)

    return asyncio.run(go())
