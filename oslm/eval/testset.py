"""Builds data/eval/testset.jsonl: 40 concept, 40 code-write, 40 find-the-bug items, kept apart from SFT data."""

from __future__ import annotations

import asyncio
import logging
import random
from collections import Counter
from typing import Any

from oslm.config import load_config
from oslm.sft.generate import BUG_TYPES, _hid, flagged, load_chunks, verified_correct
from oslm.sft.llm import Gemini, GeminiError, items_of
from oslm.sft.store import append_jsonl, read_jsonl

log = logging.getLogger(__name__)

TARGET = {"concept": 40, "code_write": 40, "bug_find": 40}
SEED_OFFSET = 1000  # different sampling stream from SFT generation

CODE_TASKS = [
    "bounded buffer with semaphores",
    "bounded buffer with condition variables",
    "readers-writers (readers preference)",
    "readers-writers (writer preference)",
    "dining philosophers without deadlock",
    "sleeping barber",
    "cigarette smokers problem",
    "reusable barrier for N threads",
    "thread-safe bank transfer avoiding deadlock",
    "spinlock built on __atomic_test_and_set",
    "counting semaphore built from mutex and condvar",
    "ping-pong alternation of two threads",
    "parallel sum with per-thread partial sums",
    "thread-safe queue",
    "parking lot with K spaces",
    "print numbers in order with 3 threads",
    "rendezvous of two threads with semaphores",
    "monitor-style counter with condition variable wait-until-zero",
    "H2O molecule building problem",
    "unisex bathroom problem",
]


async def _concept(gem: Gemini, rng: random.Random, n: int) -> list[dict[str, Any]]:
    chunks = [c for c in load_chunks() if 150 <= c["n_tokens"] <= 600 and not c["has_code"]]
    seeds = rng.sample(chunks, k=n)
    rows: list[dict[str, Any]] = []
    for i in range(0, n, 5):
        batch = seeds[i : i + 5]
        prompt = "For each excerpt, write ONE exam-style conceptual question about process synchronization that it answers, and a concise reference answer (80-150 words) drawn from it.\n\n"
        prompt += "\n\n".join(f'Excerpt {j + 1}:\n"""\n{c["text"]}\n"""' for j, c in enumerate(batch))
        prompt += '\n\nReturn {"items": [{"excerpt": 1, "question": "...", "reference": "..."}]}.'
        try:
            data = await gem.generate_json(prompt, temperature=0.4)
        except GeminiError as exc:
            log.warning("concept batch failed: %s", exc)
            continue
        for it in items_of(data, "items"):
            try:
                c = batch[int(it["excerpt"]) - 1]
            except (KeyError, ValueError, IndexError, TypeError):
                continue
            rows.append(
                {
                    "id": _hid("t-concept", it["question"]),
                    "category": "concept",
                    "question": it["question"],
                    "reference": it["reference"],
                    "gold_urls": [c["url"]],
                    "gold_doc_ids": [c["doc_id"]],
                }
            )
    return rows


async def _code(gem: Gemini, bridge: Any, rng: random.Random, n: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tasks = rng.sample(CODE_TASKS, k=len(CODE_TASKS))
    for i in range(0, len(tasks), 4):
        batch = tasks[i : i + 4]
        prompt = (
            "For each topic write TWO different student requests to write a C program (pthreads) — one plain, one with "
            "an extra constraint — and a reference solution for each: complete C, compiles with gcc -Wall -Wextra "
            "-pthread, terminates within one second, TSan/Helgrind clean.\nTopics: "
            + "; ".join(batch)
            + '\nReturn {"items": [{"topic": "...", "request": "...", "reference_code": "..."}]}.'
        )
        try:
            data = await gem.generate_json(prompt, temperature=0.5)
        except GeminiError as exc:
            log.warning("code batch failed: %s", exc)
            continue
        for it in items_of(data, "items"):
            code = (it.get("reference_code") or "").strip() if isinstance(it, dict) else ""
            if not code:
                continue
            report = await bridge.check(code)
            rows.append(
                {
                    "id": _hid("t-code", it["request"]),
                    "category": "code_write",
                    "question": it["request"],
                    "topic": it.get("topic", ""),
                    "reference_code": code,
                    "reference_verified": verified_correct(report),
                }
            )
    return rows[:n] if len(rows) > n else rows


async def _bugs(gem: Gemini, bridge: Any, rng: random.Random, n: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    plan = [(bug, 3) for bug in BUG_TYPES] + [("none", 3)] * 4
    for bug, k in plan:
        desc = (
            "contain NO synchronization bugs (correct, TSan/Helgrind clean, finish < 1 s)"
            if bug == "none"
            else (
                f"each contain exactly this bug: {BUG_TYPES[bug]}. The bug must show when run (TSan/Helgrind report, failing assert, or hang); otherwise correct and fast"
            )
        )
        prompt = (
            f"Write {k} short C programs (25-60 lines, pthreads) that {desc}. Use unusual scenarios "
            f"(e.g. {', '.join(rng.sample(['airport gates', 'robot arms', 'library books', 'game lobby', 'sensor readings', 'pizza oven', 'traffic lights', 'file chunks'], 3))}). "
            'Return {"programs": [{"code": "...", "buggy_lines": [ints], "explanation": "..."}]}.'
        )
        try:
            data = await gem.generate_json(prompt, temperature=0.8)
        except GeminiError as exc:
            log.warning("bug batch failed: %s", exc)
            continue
        for p in items_of(data, "programs"):
            code = (p.get("code") or "").strip() if isinstance(p, dict) else ""
            if not code:
                continue
            report = await bridge.check(code)
            ok = (
                verified_correct(report)
                if bug == "none"
                else (report.status in ("clean", "issues") and flagged(report))
            )
            if ok:
                rows.append(
                    {
                        "id": _hid("t-bug", code),
                        "category": "bug_find",
                        "question": f"Is there a synchronization bug in this code?\n```c\n{code}\n```",
                        "code": code,
                        "has_bug": bug != "none",
                        "bug_type": bug,
                        "buggy_lines": p.get("buggy_lines", []),
                        "reference": p.get("explanation", ""),
                        "checker": report.summary,
                    }
                )
    buggy = [r for r in rows if r["has_bug"]]
    clean = [r for r in rows if not r["has_bug"]]
    n_clean = min(len(clean), n // 3)
    return clean[:n_clean] + buggy[: n - n_clean]


def build_testset(force: bool = False) -> int:
    from oslm.checker.wsl_bridge import CheckerBridge

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    out = cfg.path("eval") / "testset.jsonl"
    if force:
        out.unlink(missing_ok=True)
    have = Counter(r["category"] for r in read_jsonl(out))
    rng = random.Random(cfg["seed"] + SEED_OFFSET)

    async def go() -> None:
        gem = Gemini()
        bridge = CheckerBridge()
        await bridge.warmup()
        if have["concept"] < TARGET["concept"]:
            append_jsonl(out, await _concept(gem, rng, TARGET["concept"] - have["concept"]))
        if have["code_write"] < TARGET["code_write"]:
            append_jsonl(out, await _code(gem, bridge, rng, TARGET["code_write"] - have["code_write"]))
        if have["bug_find"] < TARGET["bug_find"]:
            append_jsonl(out, await _bugs(gem, bridge, rng, TARGET["bug_find"] - have["bug_find"]))

    asyncio.run(go())
    counts = Counter(r["category"] for r in read_jsonl(out))
    print(f"testset: {dict(counts)} -> {out}")
    return 0
