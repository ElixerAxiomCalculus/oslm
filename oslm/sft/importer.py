"""Imports externally written (question, answer) pairs from CSV into the SFT pipeline.

Each row gets the same treatment as generated data: the router's mode, real retrieval context in the inference
layout, and automatic [n] citations where an answer sentence is clearly supported by a context chunk.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from oslm.config import load_config
from oslm.rag.prompts import build_user_message
from oslm.sft.generate import CONTENT_MODES, _context_for, _ctx, _mode_for, load_chunks
from oslm.sft.store import append_jsonl, ids_in, read_jsonl, write_jsonl

log = logging.getLogger(__name__)

CATEGORIES = {
    "concept",
    "code_write",
    "trace",
    "bug_find",
    "not_covered",
    "off_topic",
    "os_out_of_scope",
    "greeting",
    "followup",
}
_FOLLOWUP = re.compile(r"Earlier\s*[—–-]+\s*Student:\s*(.+?)\s*\|\s*Tutor:\s*(.+?)\s*\n\s*Now:\s*(.+)", re.S)
ALIASES = {
    "follow_up": "followup",
    "multiturn": "followup",
    "code": "code_write",
    "bug": "bug_find",
    "find_the_bug": "bug_find",
    "out_of_scope": "os_out_of_scope",
    "decline": "not_covered",
    "offtopic": "off_topic",
}
# Cosine between an answer sentence and a chunk (bge-large, normalized) above which the chunk supports it.
CITE_THRESHOLD = 0.80
_SENTENCE_END = re.compile(r"(?<=[.!?])(\s+)")
_C_FENCE = re.compile(r"(```[cC][ \t]*\n)(.*?)(```)", re.S)
_CK_DEFINITION = re.compile(r"^\s*static\s+void\s+ck\s*\([^\n]*\}\s*$", re.M)
_LEGACY_MACRO = re.compile(r"^\s*#define\s+(?:SO|P|V)\s*\([^\n]*\n?", re.M)
_PV_FUNCTION_START = re.compile(r"^\s*static\s+void\s+(?:P|V)\s*\([^)]*\)\s*\{", re.M)
_HEADER_USE = {
    "semaphore.h": r"\bsem_(?:t|init|wait|post|destroy|trywait)\b",
    "stdatomic.h": r"\b(?:_Atomic|atomic_|memory_order)",
    "stdio.h": r"\b(?:printf|fprintf|perror|puts|fputs)\s*\(",
    "stdlib.h": r"\b(?:abort|exit|malloc|calloc|realloc|free|EXIT_)\b",
    "assert.h": r"\bassert\s*\(",
    "errno.h": r"\b(?:errno|EINTR|EAGAIN|ETIMEDOUT)\b",
    "time.h": r"\b(?:timespec|clock_gettime|nanosleep|time_t|CLOCK_)\b",
    "string.h": r"\b(?:memcpy|memset|strcpy|strcmp|strlen)\s*\(",
    "stdbool.h": r"\b(?:bool|true|false)\b",
    "stdint.h": r"\b(?:u?int(?:8|16|32|64)_t|SIZE_MAX)\b",
    "limits.h": r"\b(?:INT_|LONG_|CHAR_|UINT_)\w*",
    "sched.h": r"\b(?:sched_|cpu_set_t)\w*",
    "signal.h": r"\b(?:signal|sigaction|SIG\w+)\b",
}


def _rewrite_calls(text: str, name: str, replacement: str) -> str:
    """Rewrite helper(expr) while respecting nested parentheses."""
    pattern = re.compile(rf"\b{re.escape(name)}\s*\(")
    while match := pattern.search(text):
        depth, end = 1, match.end()
        while end < len(text) and depth:
            depth += (text[end] == "(") - (text[end] == ")")
            end += 1
        if depth:
            break
        inner = text[match.end() : end - 1]
        text = text[: match.start()] + replacement.format(expr=inner) + text[end:]
    return text


def _remove_pv_functions(text: str) -> tuple[str, bool]:
    """Remove one-line or multi-line P/V helpers without consuming neighbouring functions."""
    found = False
    while match := _PV_FUNCTION_START.search(text):
        found = True
        depth, end = 1, match.end()
        while end < len(text) and depth:
            depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        if depth:
            break
        if end < len(text) and text[end] == "\n":
            end += 1
        text = text[: match.start()] + text[end:]
    return text, found


def _compact_c_block(code: str) -> str:
    code = re.sub(r"^\s*#define\s+_XOPEN_SOURCE\s+[^\n]+\n", "", code, flags=re.M)
    code = _CK_DEFINITION.sub("", code)
    code, pointer_pv = _remove_pv_functions(code)
    code = _LEGACY_MACRO.sub("", code)
    code = _rewrite_calls(code, "ck", "assert(({expr}) == 0)")
    address = "{expr}" if pointer_pv else "&({expr})"
    code = _rewrite_calls(code, "P", f"assert(sem_wait({address}) == 0)")
    code = _rewrite_calls(code, "V", f"assert(sem_post({address}) == 0)")
    code = _rewrite_calls(code, "SO", "assert(({expr}) != -1)")
    if "assert(" in code and not re.search(r"^\s*#include\s*<assert\.h>", code, re.M):
        code = "#include <assert.h>\n" + code
    body = re.sub(r"^\s*#include\s*<[^>]+>\s*$", "", code, flags=re.M)
    for header, token in _HEADER_USE.items():
        if not re.search(token, body):
            code = re.sub(rf"^\s*#include\s*<{re.escape(header)}>\s*\n?", "", code, flags=re.M)
    return re.sub(r"\n{3,}", "\n\n", code).strip() + "\n"


def clean_external_answer(answer: str, category: str) -> str:
    """Remove the repetitive code scaffold that the first adapter memorised."""
    if category != "code_write":
        return answer

    def compact(match: re.Match[str]) -> str:
        return match.group(1) + _compact_c_block(match.group(2)) + match.group(3)

    answer = _C_FENCE.sub(compact, answer)
    answer = answer.replace("gcc -Wall -pthread prog.c", "gcc -Wall -Wextra -pthread prog.c")
    answer = answer.replace("gcc -fsanitize=thread -g -pthread prog.c", "gcc -fsanitize=thread -g -O1 -pthread prog.c")
    answer = re.sub(
        r"\s*The second build requires a compiler/platform with ThreadSanitizer support\."
        r"(?: The example uses finite work and no intentional delays; runtime assumes ordinary scheduler progress\.)?",
        "",
        answer,
    )
    return answer.strip()


def _norm_category(raw: str) -> str:
    c = re.sub(r"[\s-]+", "_", raw.strip().lower())
    c = ALIASES.get(c, c)
    if c not in CATEGORIES:
        raise ValueError(f"unknown category {raw!r} (use one of {sorted(CATEGORIES)})")
    return c


def read_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows, problems = [], []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = {"category", "question", "answer"} - {h.strip().lower() for h in reader.fieldnames or []}
        if missing:
            raise SystemExit(f"{path.name} is missing columns: {sorted(missing)}")
        for n, raw in enumerate(reader, start=2):
            # CRLF inside cells would hide ``` fences from the code detectors downstream.
            row = {k.strip().lower(): (v or "").replace("\r\n", "\n").strip() for k, v in raw.items() if k}
            if not row["question"] or not row["answer"]:
                problems.append(f"line {n}: empty question or answer")
                continue
            try:
                row["category"] = _norm_category(row["category"])
            except ValueError as exc:
                problems.append(f"line {n}: {exc}")
                continue
            row["answer"] = clean_external_answer(row["answer"], row["category"])
            hb = row.get("has_bug", "").lower()
            row["has_bug"] = True if hb in ("true", "yes", "1") else False if hb in ("false", "no", "0") else None
            rows.append(row)
    return rows, problems


def add_citations(answer: str, chunks: list[dict[str, Any]]) -> str:
    """Appends [n] to prose sentences that closely match context chunk n; code blocks are left alone."""
    if not chunks or re.search(r"\[\d\]", answer):
        return answer
    from oslm.index.embed import embed_passages

    parts = answer.split("```")
    prose_idx = list(range(0, len(parts), 2))
    sentences: list[tuple[int, int, str]] = []
    pieces: dict[int, list[str]] = {}
    for i in prose_idx:
        split = _SENTENCE_END.split(parts[i])
        pieces[i] = split
        for j in range(0, len(split), 2):
            s = split[j]
            # Only real statements: skip table rows, headings, questions and very short fragments.
            if len(s.split()) >= 6 and not s.lstrip().startswith(("|", "#")) and not s.rstrip().endswith("?"):
                sentences.append((i, j, s))
    if not sentences:
        return answer
    chunk_vecs = embed_passages([c["text"] for c in chunks])
    sent_vecs = embed_passages([s for _, _, s in sentences])
    sims = sent_vecs @ chunk_vecs.T
    for (i, j, s), row in zip(sentences, sims, strict=True):
        best = int(np.argmax(row))
        if row[best] >= CITE_THRESHOLD:
            stripped = s.rstrip()
            trail = s[len(stripped) :]
            if stripped and stripped[-1] in ".!":
                pieces[i][j] = f"{stripped[:-1]} [{best + 1}]{stripped[-1]}{trail}"
            else:
                pieces[i][j] = f"{stripped} [{best + 1}]{trail}"
    for i in prose_idx:
        parts[i] = "".join(pieces[i])
    return "```".join(parts)


def parse_followup(question: str) -> tuple[list[dict[str, str]], str] | None:
    """Splits 'Earlier — Student: … | Tutor: …\\nNow: …' into chat history and the follow-up message."""
    m = _FOLLOWUP.search(question)
    if not m:
        return None
    history = [{"role": "user", "content": m.group(1).strip()}, {"role": "assistant", "content": m.group(2).strip()}]
    return history, m.group(3).strip()


async def _rewrite_all(followups: list[tuple[list[dict[str, str]], str]]) -> list[str]:
    """Standalone questions via the same Ollama rewrite call the live pipeline uses."""
    from oslm.rag.generate import OllamaClient, model_for_setup, rewrite_followup

    client = OllamaClient()
    try:
        return [await rewrite_followup(client, model_for_setup("base"), msg, hist) for hist, msg in followups]
    finally:
        await client.aclose()


def import_csv(path: Path, replace: bool = False) -> dict[str, Any]:
    import asyncio

    from oslm.rag.router import is_check_answer, refine_followup, route

    cfg = load_config()
    out = cfg.path("sft") / "raw" / "answers.jsonl"
    rows, problems = read_rows(path)
    if replace and out.exists():
        questions = {r["question"] for r in rows}
        questions.update(parsed[1] for r in rows if (parsed := parse_followup(r["question"])))
        existing = [
            it
            for it in read_jsonl(out)
            if it.get("source_csv") != path.name
            and not (
                str(it.get("variant", "")).startswith("external")
                and (it.get("source_question") in questions or it.get("question") in questions)
            )
        ]
        write_jsonl(out, existing)
    done = ids_in(out)
    chunks_by_id = {c["chunk_id"]: c for c in load_chunks()}
    k = cfg.sft["context_chunks"]
    new = []
    cited = 0

    pending = []
    for row in rows:
        row["id"] = "ext-" + hashlib.sha1((path.name + "||" + row["question"]).encode("utf-8")).hexdigest()[:12]
        if row["category"] == "followup" and row["id"] not in done:
            parsed = parse_followup(row["question"])
            if parsed is None:
                problems.append(
                    f"followup without 'Earlier — Student: … | Tutor: … / Now: …': {row['question'][:60]!r}"
                )
                row["skip"] = True
                continue
            row["history"], row["followup_msg"] = parsed
            pending.append(row)
    rewrites = asyncio.run(_rewrite_all([(r["history"], r["followup_msg"]) for r in pending])) if pending else []
    for row, rewritten in zip(pending, rewrites, strict=True):
        row["rewritten"] = rewritten

    for row in rows:
        item_id = row["id"]
        if item_id in done or row.get("skip"):
            continue
        cat = row["category"]
        followup = cat == "followup"
        check_answer = False
        if followup:
            r = route(row["followup_msg"], history=row["history"])
            check_answer = is_check_answer(row["followup_msg"], row["history"])
            mode = refine_followup(row["rewritten"]) if r.mode == "concept" and not check_answer else r.mode
            search_text = row["rewritten"]
        else:
            intended = cat if cat in CONTENT_MODES + ("off_topic", "os_out_of_scope", "greeting") else "concept"
            mode = _mode_for(row["question"], intended)
            search_text = row["question"]
        picked: list[dict[str, Any]] | None = []
        answer = row["answer"]
        variant = "external"
        if mode in CONTENT_MODES:
            picked, _ = _context_for(search_text, mode, chunks_by_id, k)
            if cat != "not_covered":
                answer = add_citations(answer, picked)
                cited += answer != row["answer"]
            # Showing notes next to an answer that never cites them would teach the model to ignore the notes,
            # so explanations without a supported citation become no-notes (ft setup) examples instead.
            if mode in ("concept", "trace", "code_write") and cat != "not_covered" and answer == row["answer"]:
                picked = None
                variant = "external_no_context"
        context = [_ctx(c) for c in picked] if picked is not None else None
        category = {"bug_find": "bug_find", "followup": "multiturn"}.get(
            cat, "meta" if cat in ("off_topic", "os_out_of_scope", "greeting") else cat
        )
        item = {
            "id": item_id,
            "category": category,
            "variant": variant,
            "question": row["followup_msg"] if followup else row["question"],
            "mode": mode,
            "chunk_ids": [c["chunk_id"] for c in picked] if picked is not None else None,
            "user_message": build_user_message(
                search_text,
                mode,
                context,
                followup=followup,
                check_answer=check_answer,
            ),
            "note": "",
            "answer": answer,
            "source_csv": path.name,
            "source_question": row["question"],
        }
        if followup:
            item["history"] = row["history"]
            item["rewritten"] = row["rewritten"]
        if cat == "bug_find" and row["has_bug"] is not None:
            item["has_bug"] = row["has_bug"]
        new.append(item)
    append_jsonl(out, new)
    stats = {
        "rows": len(rows),
        "imported": len(new),
        "skipped_existing": len(rows) - len(new),
        "auto_cited": cited,
        "problems": problems[:20],
        "n_problems": len(problems),
    }
    log.info("import: %s", stats)
    return stats
