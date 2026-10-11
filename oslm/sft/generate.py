"""SFT data generation: questions -> verified bug programs -> items with real retrieval context -> gold answers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from oslm.config import load_config
from oslm.rag.prompts import SYSTEM_PROMPT, ContextChunk, build_user_message
from oslm.rag.router import refine_followup, route
from oslm.sft.llm import Gemini, GeminiError, LocalQwen, items_of
from oslm.sft.store import append_jsonl, ids_in, read_jsonl

log = logging.getLogger(__name__)

CONTENT_MODES = ("concept", "code_write", "bug_find", "trace")
MAX_CHUNK_TOKENS = 700


# Target counts per category; scaled by config sft.target_total / 1200. Extra headroom covers rejections.
PLAN = {
    "concept": 330,
    "code_write": 270,
    "trace": 260,
    "bug_find": 300,
    "not_covered": 180,
    "meta": 70,
    "multiturn": 110,
}

BUG_TYPES = {
    "missing_lock": "a shared variable is updated without holding any lock (data race / lost update)",
    "wrong_sem_order": "semaphore waits in the wrong order, e.g. sem_wait(&mutex) before sem_wait(&empty) in a bounded buffer, which deadlocks when the buffer is full or empty",
    "lost_wakeup": "a lost wakeup: the signal can happen before the waiter waits because the condition is not tracked in shared state checked under the mutex, so a thread sleeps forever",
    "if_not_while": "`if` instead of `while` around pthread_cond_wait with several consumers and pthread_cond_broadcast, so a woken thread proceeds when the condition is false again (make it observable with an assert on the invariant, e.g. count >= 0)",
    "unlock_without_lock": "pthread_mutex_unlock on a mutex the thread does not hold (e.g. an unlock on an error path, or unlock in the wrong function)",
    "double_lock": "the same thread locks a default (non-recursive) mutex twice, e.g. a helper function that locks again, causing self-deadlock",
    "missing_join": "a missing pthread_join: main reads results or exits while worker threads are still writing shared data",
    "lock_order_inversion": "two threads acquire two mutexes in opposite orders (lock-order inversion that can deadlock)",
    "busy_wait_misuse": "busy waiting on a plain (non-atomic, non-volatile) shared flag without synchronization, which is a data race",
    "sem_wrong_init": "a semaphore initialised to the wrong value, e.g. a semaphore used as a mutex initialised to 2 (two threads in the critical section) or a counting semaphore initialised to 0 so everyone blocks",
}

SCENARIOS = [
    "bank account transfers",
    "ticket booking counter",
    "shared hit counter",
    "bounded buffer of jobs",
    "print queue",
    "parking lot with N spaces",
    "readers and writers on a config table",
    "dining philosophers",
    "logging to a shared buffer",
    "matrix row workers summing into a total",
    "chat room message board",
    "inventory stock updates",
    "elevator request queue",
    "web server request counter",
    "sleeping barber shop",
]

CORRECT_MECHANISMS = [
    "pthread mutex around a counter",
    "bounded buffer with two counting semaphores and a mutex",
    "bounded buffer with a mutex and two condition variables (while loops)",
    "readers-writers with pthread_rwlock_t",
    "readers-writers with semaphores (first readers-writers solution)",
    "dining philosophers with ordered forks",
    "barrier implemented with mutex + condition variable",
    "semaphore used to limit concurrency to K threads",
    "thread pool with a work queue protected by mutex + condvar",
    "atomic counter with __atomic builtins",
]

QUESTION_STYLES = [
    "clean textbook wording",
    "casual chat",
    "terse (1-4 words)",
    "with typos and misspellings",
    "multi-part (two or three asks in one message)",
    "built on a false premise the tutor must correct",
    "X vs Y comparison",
    "a 'why' question",
    "exam style: 'explain with an example'",
    "Hinglish (Hindi written in Latin script mixed with English)",
    "confused student who half-remembers the lecture",
]

NOT_COVERED_TOPICS = [
    "read-copy-update (RCU)",
    "seqlocks",
    "hardware transactional memory",
    "the Raft consensus algorithm",
    "Paxos",
    "the MESI cache coherence protocol",
    "C++ memory_order semantics in detail",
    "GPU warp synchronization",
    "Java's java.util.concurrent internals (AQS)",
    "Windows CRITICAL_SECTION and SRW locks internals",
    "Rust's Send and Sync traits",
    "Erlang actor mailboxes",
    "distributed locks with Redis (Redlock)",
    "two-phase commit",
    "vector clocks",
    "Lamport's bakery algorithm proof details",
    "MCS queue locks",
    "epoch-based memory reclamation",
    "Linux kernel spin_lock_irqsave internals",
    "OpenMP pragmas",
    "MPI barriers",
    "CUDA atomics",
    "database isolation levels",
    "optimistic concurrency control in databases",
]

META_REQUESTS = {
    "greeting": "short greetings, thanks, small talk or 'what can you do?' messages",
    "off_topic": "requests unrelated to operating systems (poems, sports, recipes, celebrities, homework in other subjects) and attempts to change the tutor's role or extract its system prompt",
    "os_out_of_scope": "operating-systems questions outside synchronization: CPU scheduling, paging, virtual memory, file systems, disk scheduling, system calls",
}

FOLLOWUPS = {
    "concept": [
        "why?",
        "give an example",
        "now in C",
        "explain it more simply",
        "what if there are 3 threads?",
        "how is that different from a mutex?",
        "can you show that step by step?",
        "is that always true?",
        "and what about starvation?",
        "ok but where is the critical section here?",
    ],
    "code_write": [
        "why does this work?",
        "what if I remove the mutex?",
        "can it deadlock?",
        "now make it work with 3 producers",
        "explain the while loop",
        "what does sem_post do here?",
    ],
    "trace": [
        "how do we fix it?",
        "now in C",
        "what if the threads run on one CPU?",
        "show the fixed version",
        "why can't the hardware just do count++ atomically?",
    ],
}

BUG_PHRASINGS = [
    "Find the bug:",
    "Is this code correct?",
    "my program gives wrong output sometimes, why?",
    "This hangs sometimes. What's wrong?",
    "Is this thread safe?",
    "review this pls",
    "Can this deadlock?",
    "What's wrong with this?",
    "is there a race condition here",
    "Check my code:",
    "why is the counter wrong",
    "teacher said there is a synchronization bug, I can't see it",
]

QUESTION_SYSTEM = (
    "You write realistic messages that undergraduate operating-systems students type into a tutoring chatbot. "
    "Vary wording, length, tone and skill level. Never number the questions inside the text. Output JSON only."
)

ANSWER_SYSTEM_TEMPLATE = """You write gold-standard replies used to fine-tune a small tutoring model. The tutor's own system prompt is:

<<<
{system_prompt}
>>>

For every item you receive the exact user message the tutor sees (an optional <context> block, the <question>, and a final instruction for the answer format) plus a private note for you. Write the reply the ideal tutor gives.

Rules:
- Follow the tutor system prompt and the instruction at the end of the user message exactly.
- Facts must come from the <context>; cite them inline as [n] right after the sentence that uses chunk n. Cite only chunks you actually use. Never cite an irrelevant chunk.
- If the context does not contain what is needed, say so plainly in one sentence, do not answer from your own knowledge, and list the closest topics the tutor can help with (taken from the context's source labels). No citations then.
- If the user message says no notes are attached, answer from general knowledge with no [n] citations.
- Voice: a warm, precise teacher; no filler ("Great question!"), no restating the question, no mention of "context", "chunks" or "notes provided to me" beyond saying the notes don't cover something.
- Length: concept up to ~280 words, trace up to ~320 words, bug review up to ~420 words, code answers with a program of at most ~90 lines.
- C programs must be complete (includes + main), compile with gcc -Wall -Wextra -pthread on Linux, terminate within one second (small loop counts, no infinite loops, no sleeps longer than 10 ms), and be correct under ThreadSanitizer and Helgrind. Use while loops around pthread_cond_wait. Use ```c fences. Diffs use ```diff fences.
- Interleavings use a markdown table | Step | Thread A | Thread B | Shared state | with one atomic action per row.
- If the student writes in Hindi, Tamil or another language, reply in that language (code and API names stay in English). Hinglish gets a reply in simple English with a few Hinglish touches.
- The private note is ground truth (for example a verified verdict for code review). Never mention the note.

Return JSON: {{"replies": [{{"id": "<item id>", "reply": "<markdown reply>"}}]}}"""


def _hid(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]


def scaled_plan() -> dict[str, int]:
    factor = load_config().sft["target_total"] / 1200
    return {k: max(1, round(v * factor)) for k, v in PLAN.items()}


def load_chunks() -> list[dict[str, Any]]:
    path = load_config().path("chunks") / "chunks.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------- questions


QUESTIONS_PER_EXCERPT = 8


def _question_prompt(category: str, excerpts: list[str], n: int, styles: list[list[str]]) -> str:
    asks = {
        "concept": "conceptual questions a student would ask about the material in the excerpt",
        "code_write": "requests to WRITE C code (pthreads, mutexes, semaphores, condition variables) for a task related to the excerpt, e.g. 'implement ... in C', 'give me code for ...', 'write a program where ...'",
        "trace": "requests to show STEP BY STEP how an interleaving goes wrong or works (race, lost update, deadlock, lost wakeup, starvation) for the situation in the excerpt, e.g. 'trace ...', 'walk me through how ...', 'show the interleaving that ...'",
    }[category]
    blocks = "\n\n".join(
        f'Excerpt {i + 1}:\n"""\n{text}\n"""\nStyles for excerpt {i + 1}, one per message in order: {"; ".join(st)}.'
        for i, (text, st) in enumerate(zip(excerpts, styles, strict=True))
    )
    return (
        f"{blocks}\n\nFor EACH excerpt write {n} distinct student messages: {asks}. Each must be answerable from (or "
        "clearly related to) its excerpt without quoting it. Do not repeat a question across excerpts.\n"
        'Return {"questions": [{"excerpt": 1, "question": "...", "style": "..."}]}.'
    )


async def generate_questions(gem: Gemini, out: Path, plan: dict[str, int], seed: int) -> None:
    rng = random.Random(seed)
    chunks = [c for c in load_chunks() if 80 <= c["n_tokens"] <= MAX_CHUNK_TOKENS]
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in chunks:
        by_doc[c["doc_id"]].append(c)
    have = Counter(r["category"] for r in read_jsonl(out))

    async def from_chunks(category: str) -> None:
        docs = rng.sample(list(by_doc), k=gem.excerpts_per_call)
        seeds = [rng.choice(by_doc[d]) for d in docs]
        styles = [[rng.choice(QUESTION_STYLES) for _ in range(QUESTIONS_PER_EXCERPT)] for _ in seeds]
        prompt = _question_prompt(category, [c["text"] for c in seeds], QUESTIONS_PER_EXCERPT, styles)
        try:
            data = await gem.generate_json(prompt, system=QUESTION_SYSTEM)
        except GeminiError as exc:
            log.warning("question batch failed: %s", exc)
            return
        rows = []
        for q in items_of(data, "questions"):
            if not isinstance(q, dict) or not q.get("question", "").strip():
                continue
            try:
                seed_chunk = seeds[int(q.get("excerpt", 1)) - 1]["chunk_id"]
            except (ValueError, IndexError, TypeError):
                seed_chunk = seeds[0]["chunk_id"]
            rows.append(
                {
                    "id": _hid(category, q["question"]),
                    "category": category,
                    "question": q["question"].strip(),
                    "style": q.get("style", ""),
                    "seed_chunk": seed_chunk,
                }
            )
        append_jsonl(out, rows)

    per_call = QUESTIONS_PER_EXCERPT * gem.excerpts_per_call
    tasks = []
    for category in ("concept", "code_write", "trace"):
        missing = max(0, plan[category] - have[category])
        tasks += [from_chunks(category) for _ in range(-(-missing // per_call))]

    async def not_covered(n: int) -> None:
        topics = rng.sample(NOT_COVERED_TOPICS, k=6)
        prompt = (
            f"Write {n} student messages to a process-synchronization tutor that ask about these topics, which the "
            f"course notes do NOT cover: {', '.join(topics)}. Make them sound like real students (varied styles: "
            f"{'; '.join(rng.sample(QUESTION_STYLES, 4))}). "
            'Return {"questions": [{"question": "...", "topic": "..."}]}.'
        )
        try:
            data = await gem.generate_json(prompt, system=QUESTION_SYSTEM)
        except GeminiError as exc:
            log.warning("not-covered batch failed: %s", exc)
            return
        append_jsonl(
            out,
            [
                {
                    "id": _hid("not_covered", q["question"]),
                    "category": "not_covered",
                    "question": q["question"].strip(),
                    "style": q.get("topic", ""),
                }
                for q in items_of(data, "questions")
                if isinstance(q, dict) and q.get("question", "").strip()
            ],
        )

    missing = max(0, plan["not_covered"] - have["not_covered"])
    n_nc = 10 * gem.excerpts_per_call
    tasks += [not_covered(n_nc) for _ in range(-(-missing // n_nc))]

    async def meta(kind: str, n: int) -> None:
        prompt = (
            f"Write {n} varied messages a student might send to a process-synchronization tutor that are {META_REQUESTS[kind]}. "
            'Return {"questions": [{"question": "..."}]}.'
        )
        try:
            data = await gem.generate_json(prompt, system=QUESTION_SYSTEM)
        except GeminiError as exc:
            log.warning("meta batch failed: %s", exc)
            return
        append_jsonl(
            out,
            [
                {
                    "id": _hid("meta", kind, q["question"]),
                    "category": "meta",
                    "question": q["question"].strip(),
                    "style": kind,
                }
                for q in items_of(data, "questions")
                if isinstance(q, dict) and q.get("question", "").strip()
            ],
        )

    missing = max(0, plan["meta"] - have["meta"])
    per_kind = (missing + 2) // 3
    if per_kind:
        tasks += [meta(kind, per_kind) for kind in META_REQUESTS]
    log.info("question batches to run: %d", len(tasks))
    await asyncio.gather(*tasks)


# ---------------------------------------------------------------- bug programs


def _program_prompt(bug: str | None, n: int, rng: random.Random) -> str:
    scenarios = ", ".join(rng.sample(SCENARIOS, k=n))
    if bug is None:
        mechanisms = ", ".join(rng.sample(CORRECT_MECHANISMS, k=n))
        return (
            f"Write {n} different short C programs (25-70 lines) using POSIX threads that are completely FREE of "
            f"synchronization bugs. Use these mechanisms, one per program: {mechanisms}. Scenarios: {scenarios}.\n"
            "Requirements: compile cleanly with gcc -Wall -Wextra -pthread on Linux; pass ThreadSanitizer and Helgrind "
            "with no reports; terminate within one second; join every thread; print a final result; no comments that "
            "praise the code.\n"
            'Return {"programs": [{"code": "...", "scenario": "...", "explanation": "one sentence on why it is correct"}]}.'
        )
    return (
        f"Write {n} different short C programs (25-70 lines) using POSIX threads that each contain exactly this "
        f"synchronization bug: {BUG_TYPES[bug]}. Scenarios, one per program: {scenarios}.\n"
        "Requirements: compile with gcc -Wall -pthread on Linux. The bug must show up when the program is run once: "
        "ThreadSanitizer or Helgrind reports it, or an assert() on an invariant fails, or the program hangs. Widen the "
        "window (loops of 100000 iterations, or usleep(1000) inside the critical window). Apart from the bug the "
        "program must be correct and finish within one second. No comments that hint at the bug.\n"
        'Return {"programs": [{"code": "...", "scenario": "...", "buggy_lines": [line numbers], '
        '"explanation": "one sentence describing the bug"}]}.'
    )


def flagged(report: Any) -> bool:
    return report.has_sync_bug or any(f.kind in ("crash", "cpu_limit") for f in report.findings)


def verified_correct(report: Any) -> bool:
    run = report.raw.get("steps", {}).get("run", {})
    return report.status == "clean" and run.get("exit_code") == 0


async def generate_programs(gem: Gemini, bridge: Any, out: Path, plan: dict[str, int], seed: int) -> None:
    rng = random.Random(seed + 1)
    existing = read_jsonl(out)
    have = Counter(r["bug_type"] for r in existing if r["verified"])
    n_buggy = round(plan["bug_find"] * 0.68)
    per_bug = max(2, n_buggy // len(BUG_TYPES))
    n_correct = plan["bug_find"] - n_buggy
    targets = {bug: per_bug for bug in BUG_TYPES} | {"none": n_correct}

    async def batch(bug: str) -> None:
        try:
            # A local model's thinking can eat the whole token budget on code; the sandbox checks every program anyway.
            data = await gem.generate_json(
                _program_prompt(None if bug == "none" else bug, gem.programs_per_call, rng),
                temperature=0.9,
                think=False,
            )
        except GeminiError as exc:
            log.warning("program batch failed: %s", exc)
            return
        rows = []
        for p in items_of(data, "programs"):
            code = (p.get("code") or "").strip() if isinstance(p, dict) else ""
            if not code:
                continue
            report = await bridge.check(code)
            if bug == "none":
                ok = verified_correct(report)
            else:
                ok = report.status in ("clean", "issues") and flagged(report)
                if not ok and report.status == "clean":
                    report = await bridge.check(code)  # timing-dependent bugs get one more chance
                    ok = flagged(report)
            rows.append(
                {
                    "id": _hid("prog", code),
                    "bug_type": bug,
                    "has_bug": bug != "none",
                    "code": code,
                    "scenario": p.get("scenario", ""),
                    "buggy_lines": p.get("buggy_lines", []),
                    "explanation": p.get("explanation", ""),
                    "verified": ok,
                    "checker": report.summary,
                    "checker_status": report.status,
                }
            )
        append_jsonl(out, rows)
        log.info("%s: %d/%d programs verified", bug, sum(r["verified"] for r in rows), len(rows))

    tasks = []
    for bug, target in targets.items():
        missing = max(0, target - have[bug])
        # Roughly half of generated buggy programs fail verification, so over-generate.
        tasks += [batch(bug) for _ in range(-(-round(missing * 1.15) // gem.programs_per_call))]
    log.info("program batches to run: %d", len(tasks))
    await asyncio.gather(*tasks)


# ---------------------------------------------------------------- items with context


def _ctx(chunk: dict[str, Any]) -> ContextChunk:
    return ContextChunk(
        chunk.get("source_name", ""), chunk.get("section", ""), str(chunk.get("page_or_slide", "")), chunk["text"]
    )


def _context_for(
    query: str, mode: str, chunks_by_id: dict[str, dict[str, Any]], k: int
) -> tuple[list[dict[str, Any]], float]:
    from oslm.rag.rerank import rerank
    from oslm.rag.retrieve import build_search_query, dense_search

    q = build_search_query(query, mode)
    top = rerank(q, dense_search(q), k=10)
    picked = [
        chunks_by_id[t.chunk_id]
        for t in top
        if t.chunk_id in chunks_by_id and chunks_by_id[t.chunk_id]["n_tokens"] <= MAX_CHUNK_TOKENS
    ][:k]
    return picked, (top[0].rerank_score or 0.0) if top else 0.0


def _mode_for(question: str, intended: str) -> str:
    r = route(question)
    if r.mode == "empty" or (r.ambiguous and intended in CONTENT_MODES + ("greeting", "off_topic", "os_out_of_scope")):
        return intended
    return r.mode


def build_items(
    questions: list[dict[str, Any]], programs: list[dict[str, Any]], out: Path, plan: dict[str, int], seed: int
) -> int:
    """Attaches real retrieval context, the router's mode and variant notes. Local only (no Gemini)."""
    rng = random.Random(seed + 2)
    cfg = load_config()
    k = cfg.sft["context_chunks"]
    all_chunks = load_chunks()
    chunks_by_id = {c["chunk_id"]: c for c in all_chunks}
    small = [c for c in all_chunks if 80 <= c["n_tokens"] <= MAX_CHUNK_TOKENS]
    done = ids_in(out)
    rows: list[dict[str, Any]] = []

    def finish(item: dict[str, Any], picked: list[dict[str, Any]] | None, note: str) -> None:
        item["chunk_ids"] = None if picked is None else [c["chunk_id"] for c in picked]
        context = None if picked is None else [_ctx(c) for c in picked]
        item["user_message"] = build_user_message(
            item["question"], item["mode"], context, topics=item.get("topics", ())
        )
        item["note"] = note
        rows.append(item)

    for q in questions:
        if q["id"] in done:
            continue
        cat = q["category"]
        if cat == "meta":
            mode = _mode_for(q["question"], q["style"])
            item = {"id": q["id"], "category": "meta", "variant": q["style"], "question": q["question"], "mode": mode}
            note = "Keep it to one or two friendly lines." if mode in ("greeting", "off_topic") else ""
            if mode in CONTENT_MODES:
                picked, _ = _context_for(q["question"], mode, chunks_by_id, k)
                finish(item, picked, note)
            else:
                finish(item, [] if rng.random() < 0.5 else None, note)
            continue
        if cat == "not_covered":
            mode = _mode_for(q["question"], "concept")
            item = {"id": q["id"], "category": cat, "question": q["question"], "mode": mode}
            if mode not in CONTENT_MODES:
                item["variant"] = "routed_" + mode
                finish(item, [], "")
            elif rng.random() < 0.25:
                item["variant"] = "not_in_notes_mode"
                item["mode"] = "not_in_notes"
                item["topics"] = [c["section"] or c["title"] for c in rng.sample(small, 3)]
                finish(item, [], "")
            else:
                item["variant"] = "irrelevant_context"
                picked, _ = _context_for(q["question"], mode, chunks_by_id, k)
                finish(
                    item,
                    picked,
                    "The context does NOT answer this question. Decline in one sentence, name the "
                    "closest covered topics using the source labels, give no citations and no answer "
                    "from your own knowledge.",
                )
            continue

        mode = _mode_for(q["question"], cat)
        item = {"id": q["id"], "category": cat, "question": q["question"], "mode": mode}
        roll = rng.random()
        if roll < 0.10:
            item["variant"] = "no_context"
            finish(item, None, "")
            continue
        picked, _ = _context_for(q["question"], mode, chunks_by_id, k)
        if roll < 0.32 and len(picked) >= 2:
            used_docs = {c["doc_id"] for c in picked}
            distractor = rng.choice([c for c in small if c["doc_id"] not in used_docs])
            picked = picked[: k - 1]
            pos = rng.randrange(len(picked) + 1)
            picked.insert(pos, distractor)
            item["variant"] = "distractor"
            finish(
                item, picked, f"Context chunk [{pos + 1}] is unrelated to the question: ignore it and never cite it."
            )
        else:
            item["variant"] = "standard"
            finish(item, picked, "")

    for p in programs:
        pid = "bug-" + p["id"]
        if pid in done or not p["verified"]:
            continue
        phrasing = rng.choice(BUG_PHRASINGS)
        lang = "c" if rng.random() < 0.85 else ""
        question = (
            f"{phrasing}\n```{lang}\n{p['code']}\n```"
            if rng.random() < 0.8
            else f"```{lang}\n{p['code']}\n```\n{phrasing}"
        )
        mode = _mode_for(question, "bug_find")
        item = {
            "id": pid,
            "category": "bug_find",
            "question": question,
            "mode": mode,
            "variant": "buggy" if p["has_bug"] else "correct_code",
            "bug_type": p["bug_type"],
            "has_bug": p["has_bug"],
        }
        if p["has_bug"]:
            note = (
                f"VERIFIED: this code HAS a synchronization bug ({p['bug_type'].replace('_', ' ')}): {p['explanation']} "
                f"Buggy lines (approximate): {p['buggy_lines']}. The checker observed: {p['checker']}. "
                "Start with 'Verdict: bug found'."
            )
        else:
            note = (
                f"VERIFIED: this code is CORRECT (ThreadSanitizer and Helgrind clean): {p['explanation']} "
                "Start with 'Verdict: no synchronization bug' and explain which interleavings are safe; do not invent bugs."
            )
        if rng.random() < 0.1:
            item["variant"] += "_no_context"
            finish(item, None, note)
        else:
            picked, _ = _context_for(question, mode, chunks_by_id, k)
            finish(item, picked, note)

    from oslm.sft.store import append_jsonl as _append

    _append(out, rows)
    log.info("built %d items (%s)", len(rows), dict(Counter(r["category"] for r in rows)))
    return len(rows)


# ---------------------------------------------------------------- answers


async def generate_answers(gem: Gemini | LocalQwen, items: list[dict[str, Any]], out: Path) -> None:
    batch_size, small_size = gem.answers_per_call, gem.bug_answers_per_call
    done = ids_in(out)
    todo = [it for it in items if it["id"] not in done]
    system = ANSWER_SYSTEM_TEMPLATE.format(system_prompt=SYSTEM_PROMPT)
    log.info("answers to generate: %d", len(todo))

    async def run(batch: list[dict[str, Any]]) -> None:
        payload = [{"id": it["id"], "user_message": it["user_message"], "note": it.get("note", "")} for it in batch]
        if any(it.get("history") for it in batch):
            for p, it in zip(payload, batch, strict=True):
                p["earlier_conversation"] = it.get("history", [])
        prompt = "Items:\n" + json.dumps(payload, ensure_ascii=False, indent=1)
        try:
            data = await gem.generate_json(prompt, system=system, temperature=0.6)
        except GeminiError as exc:
            log.warning("answer batch failed: %s", exc)
            return
        replies = {r.get("id"): r.get("reply", "") for r in items_of(data, "replies") if isinstance(r, dict)}
        rows = []
        for it in batch:
            reply = (replies.get(it["id"]) or "").strip()
            if reply:
                rows.append({**it, "answer": reply})
        append_jsonl(out, rows)

    # Bug reviews carry long code, so they go in smaller batches.
    small = [it for it in todo if it["category"] == "bug_find" or it.get("history")]
    rest = [it for it in todo if it not in small]
    batches = [rest[i : i + batch_size] for i in range(0, len(rest), batch_size)]
    batches += [small[i : i + small_size] for i in range(0, len(small), small_size)]
    await asyncio.gather(*(run(b) for b in batches))


async def build_multiturn(answers: list[dict[str, Any]], out: Path, plan: dict[str, int], seed: int) -> int:
    """Follow-up turns: first exchange from generated data, rewrite via the same Ollama call the pipeline uses."""
    from oslm.rag.generate import OllamaClient, model_for_setup, rewrite_followup

    rng = random.Random(seed + 3)
    cfg = load_config()
    chunks_by_id = {c["chunk_id"]: c for c in load_chunks()}
    have = len(read_jsonl(out))
    need = max(0, plan["multiturn"] - have)
    if not need:
        return 0
    pool = [a for a in answers if a["category"] in FOLLOWUPS and a.get("variant") == "standard"]
    rng.shuffle(pool)
    client = OllamaClient()
    rows = []
    try:
        for base in pool[:need]:
            follow = rng.choice(FOLLOWUPS[base["category"]])
            history = [{"role": "user", "content": base["question"]}, {"role": "assistant", "content": base["answer"]}]
            rewritten = await rewrite_followup(client, model_for_setup("base"), follow, history)
            r = route(follow, has_history=True)
            mode = refine_followup(rewritten) if r.mode == "concept" else r.mode
            picked, _ = _context_for(rewritten, mode, chunks_by_id, cfg.sft["context_chunks"])
            msg = build_user_message(rewritten, mode, [_ctx(c) for c in picked], followup=True)
            rows.append(
                {
                    "id": _hid("mt", base["id"], follow),
                    "category": "multiturn",
                    "variant": base["category"],
                    "question": follow,
                    "rewritten": rewritten,
                    "mode": mode,
                    "history": history,
                    "chunk_ids": [c["chunk_id"] for c in picked],
                    "user_message": msg,
                    "note": "The student is following up on the earlier conversation; answer in that context without repeating it.",
                }
            )
    finally:
        await client.aclose()
    append_jsonl(out, rows)
    return len(rows)
