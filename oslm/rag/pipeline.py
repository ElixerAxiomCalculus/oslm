from __future__ import annotations

import asyncio
import html
import logging
import re
import secrets
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from oslm.config import load_config
from oslm.rag.citations import CitationStream, cited_numbers
from oslm.rag.generate import (
    OllamaClient,
    OllamaError,
    auxiliary_model,
    classify_ambiguous,
    decoding_options,
    model_for_setup,
    needs_normalization,
    normalize_question,
    response_language,
    rewrite_followup,
    search_query,
)
from oslm.rag.prompts import DEFAULT_TOPICS, ContextChunk, build_code_repair_messages, build_messages
from oslm.rag.retrieve import Retrieved, build_search_query
from oslm.rag.router import Route, contains_code, is_check_answer, refine_followup, route
from oslm.sft.importer import clean_external_answer

log = logging.getLogger(__name__)

SETUPS = ("base", "base_rag", "ft", "ft_rag")
RETRIEVAL_MODES = {"concept", "code_write", "bug_find", "trace"}
GROUNDED_ONLY_MODES = {"concept", "trace"}

EMPTY_REPLY = (
    "Looks like that message was empty. Ask me anything about process synchronization — for example "
    '*"What is a race condition?"* or *"Why must `pthread_cond_wait` be called in a `while` loop?"*'
)

LANGUAGE_FALLBACK_PREFIX = {
    "Hindi using Devanagari script": "क्षमा करें—तकनीकी सटीकता के लिए मैं अभी English में उत्तर दे रहा हूँ।\n\n",
    "Tamil using Tamil script": "மன்னிக்கவும்—தொழில்நுட்பத் துல்லியத்திற்காக நான் இப்போது English-ல் பதிலளிக்கிறேன்.\n\n",
}

PETERSON_SAFE_REPLY = (
    "Peterson's algorithm is a software mutual-exclusion protocol for exactly two threads.\n\n"
    "For thread `i`, with the other thread `j`, the entry section is:\n\n"
    "1. Set `flag[i] = true` to announce interest.\n"
    "2. Set `turn = j` to let the peer win a simultaneous tie.\n"
    "3. Wait while `flag[j] && turn == j`.\n\n"
    "The thread then enters the critical section and sets `flag[i] = false` in its exit section. Under atomic "
    "sequentially consistent accesses, this provides mutual exclusion, progress, and bounded waiting. Plain or "
    "`volatile` C variables are not sufficient; a portable C implementation should use C11 atomics with suitable "
    "memory ordering. It busy-waits and applies only to two threads.\n\n"
    "Check your understanding: if `flag[j]` is false, can `turn == j` by itself stop thread `i`?"
)

_C_BLOCK = re.compile(r"```(?:c|C)?[ \t]*\r?\n(.*?)```", re.S)


@dataclass
class Event:
    type: str
    data: dict[str, Any]


@dataclass
class ChatRequest:
    message: str
    setup: str = "ft_rag"
    history: list[dict[str, str]] = field(default_factory=list)
    session_id: str = ""
    verify_code: bool = True


@dataclass
class PreparedQuestion:
    route: Route
    question: str
    answer_language: str | None = None
    normalized: bool = False
    check_answer: bool = False


def complete_c_programs(text: str) -> list[str]:
    return [m.group(1) for m in _C_BLOCK.finditer(text) if re.search(r"\bint\s+main\s*\(", m.group(1))]


def _clean_generated(text: str, n_sources: int) -> tuple[str, CitationStream]:
    stream = CitationStream(n_sources)
    stream.feed(text)
    stream.flush()
    return stream.text, stream


def _repair_feedback(programs: list[str], reports: list[Any]) -> str | None:
    if not programs:
        return "No complete C program with int main(void) was present in a ```c block."
    failures = [f"Program {i}: {r.summary}" for i, r in enumerate(reports, start=1) if r.status != "clean"]
    for i, report in enumerate(reports, start=1):
        findings = [finding.message for finding in getattr(report, "findings", [])]
        if findings:
            failures.append(f"Program {i} runtime findings: {'; '.join(findings[:8])}")
        warnings = [finding.message for finding in getattr(report, "compiler", []) if finding.kind == "warning"]
        if warnings:
            failures.append(f"Program {i} compiler warnings: {'; '.join(warnings[:5])}")
    return "\n".join(failures) or None


def _is_peterson_question(text: str) -> bool:
    return bool(re.search(r"\bpeterson(?:['\u2019]s|s)?\b", text, re.I))


def _is_generic_peterson_explanation(question: str) -> bool:
    return bool(
        len(re.findall(r"\w+", question)) <= 14
        and re.search(r"\b(what|explain|describe|overview|algorithm|algo)\b", question, re.I)
    )


def _peterson_answer_is_wrong(text: str, question: str) -> bool:
    compact = re.sub(r"\s+", " ", text.lower())
    explicit_error = bool(
        re.search(r"flags? (?:are|is) read before writ", compact)
        or re.search(r"reads? flag\s*\[(?:b|j|other|peer)\].{0,80}then writes? (?:its own )?flag", compact)
    )
    if explicit_error:
        return True
    if _is_generic_peterson_explanation(question):
        own_flag_first = bool(re.search(r"flag\s*\[\s*i\s*\]\s*=\s*true", compact))
        offers_turn = bool(re.search(r"turn\s*=\s*j", compact))
        waits_on_both = "flag[j]" in compact and "turn == j" in compact
        return not (own_flag_first and offers_turn and waits_on_both)
    return False


def _canonical_code_fallback(question: str) -> tuple[str, str] | None:
    """Return a small audited fallback for common examples the local model may not repair."""
    if not (re.search(r"\bproducer", question, re.I) and re.search(r"\bconsumer", question, re.I)):
        return None
    if not re.search(r"\b(semaph|bounded\s+buffer|buffer)\w*", question, re.I):
        return None
    return (
        "producer_consumer",
        """Use `empty_slots` and `full_slots` for capacity, and one mutex for the buffer plus both indexes.

```c
#include <pthread.h>
#include <semaphore.h>
#include <stdio.h>

#define BUFFER_SIZE 5
#define NUM_PRODUCERS 2
#define NUM_CONSUMERS 2
#define ITEMS_PER_PRODUCER 10
#define ITEMS_PER_CONSUMER ((NUM_PRODUCERS * ITEMS_PER_PRODUCER) / NUM_CONSUMERS)

static int buffer[BUFFER_SIZE];
static int head;
static int tail;
static sem_t empty_slots;
static sem_t full_slots;
static pthread_mutex_t buffer_lock = PTHREAD_MUTEX_INITIALIZER;

static void *producer(void *arg) {
    int id = *(int *)arg;
    for (int i = 0; i < ITEMS_PER_PRODUCER; ++i) {
        int item = id * 100 + i;
        sem_wait(&empty_slots);
        pthread_mutex_lock(&buffer_lock);
        buffer[tail] = item;
        tail = (tail + 1) % BUFFER_SIZE;
        pthread_mutex_unlock(&buffer_lock);
        sem_post(&full_slots);
    }
    return NULL;
}

static void *consumer(void *arg) {
    int id = *(int *)arg;
    for (int i = 0; i < ITEMS_PER_CONSUMER; ++i) {
        sem_wait(&full_slots);
        pthread_mutex_lock(&buffer_lock);
        int item = buffer[head];
        head = (head + 1) % BUFFER_SIZE;
        pthread_mutex_unlock(&buffer_lock);
        sem_post(&empty_slots);
        printf("Consumer %d consumed %d\\n", id, item);
    }
    return NULL;
}

int main(void) {
    pthread_t producers[NUM_PRODUCERS];
    pthread_t consumers[NUM_CONSUMERS];
    int producer_ids[NUM_PRODUCERS];
    int consumer_ids[NUM_CONSUMERS];

    sem_init(&empty_slots, 0, BUFFER_SIZE);
    sem_init(&full_slots, 0, 0);
    for (int i = 0; i < NUM_PRODUCERS; ++i) {
        producer_ids[i] = i;
        pthread_create(&producers[i], NULL, producer, &producer_ids[i]);
    }
    for (int i = 0; i < NUM_CONSUMERS; ++i) {
        consumer_ids[i] = i;
        pthread_create(&consumers[i], NULL, consumer, &consumer_ids[i]);
    }
    for (int i = 0; i < NUM_PRODUCERS; ++i)
        pthread_join(producers[i], NULL);
    for (int i = 0; i < NUM_CONSUMERS; ++i)
        pthread_join(consumers[i], NULL);

    sem_destroy(&empty_slots);
    sem_destroy(&full_slots);
    pthread_mutex_destroy(&buffer_lock);
    return 0;
}
```

- `empty_slots` reserves capacity and `full_slots` announces completed items.
- `buffer_lock` protects the buffer and the separate `head` and `tail` indexes.
- No thread blocks on a capacity semaphore while holding the mutex.

Compile with `gcc -Wall -Wextra -pthread prog.c` and test with `gcc -fsanitize=thread -g -O1 -pthread prog.c`.""",
    )


def _is_generic_producer_consumer_request(question: str) -> bool:
    return len(re.findall(r"\w+", question)) <= 10 and _canonical_code_fallback(question) is not None


def _withheld_code_reply(feedback: str, attempts: int) -> str:
    concise = re.sub(r"\s+", " ", feedback).strip()[:600]
    return (
        f"I couldn't produce a version that passed the synchronization checker after {attempts} repair attempts, "
        "so I withheld the unsafe C draft instead of presenting it as a solution. "
        f"Last checker result: {concise}"
    )


def _canonical_check_reply(message: str, history: list[dict[str, str]]) -> str | None:
    """Handle narrow factual checks where a small judge has shown repeatable grading errors."""
    previous = next((h.get("content", "") for h in reversed(history) if h.get("role") == "assistant"), "")
    previous_lower = previous.lower()
    answer = html.unescape(message).lower()
    peterson_memory_check = (
        _is_peterson_question(previous)
        and bool(re.search(r"\b(read|write|access|memory|atomic|consistent)", previous_lower))
        and "?" in previous
    )
    denies_atomicity = bool(re.search(r"\b(?:not|non[- ]|do not|don't)\b.{0,24}\batomic", answer))
    has_atomicity = bool(re.search(r"\batomic\w*\b", answer))
    has_sc = "sequential consistency" in answer or "sequentially consistent" in answer or bool(
        re.search(r"\bseq(?:uential)?[_ -]?cst\b", answer)
    )
    if peterson_memory_check and has_atomicity and has_sc and not denies_atomicity:
        return (
            "Correct: Peterson's `flag` and `turn` accesses must be atomic and sequentially consistent, or use "
            "another memory-order scheme with an equivalent proof. Plain or `volatile` C variables do not provide "
            "those guarantees."
        )
    return None


def bug_review_hints(text: str) -> list[str]:
    """Return high-confidence facts that keep a small model focused during code review."""
    hints: list[str] = []
    for match in re.finditer(r"pthread_mutex_lock\s*\(\s*&(?P<lock>\w+)\s*\)\s*;", text):
        lock = match.group("lock")
        tail = text[match.end() :]
        unlock = re.search(rf"pthread_mutex_unlock\s*\(\s*&{re.escape(lock)}\s*\)\s*;", tail)
        segment = tail[: unlock.start()] if unlock else tail
        if re.search(r"\breturn\b", segment):
            hints.append(f"A return path can leave mutex '{lock}' locked.")
            break
    if re.search(r"\bif\s*\([^)]*\)\s*(?:\{\s*)?pthread_cond_wait\s*\(", text, re.S):
        hints.append("A condition-variable wait is guarded by if; it must re-check its predicate in a while loop.")
    if re.search(
        r"sem_wait\s*\(\s*&mutex\s*\)\s*;[\s\S]{0,200}?sem_wait\s*\(\s*&(empty|full)\s*\)", text
    ):
        hints.append("The code waits for an item/slot while holding mutex, creating hold-and-wait deadlock.")
    return hints


def source_card(n: int, r: Retrieved) -> dict[str, Any]:
    m = r.metadata
    return {
        "n": n,
        "chunk_id": r.chunk_id,
        "source_name": m.get("source_name", ""),
        "title": m.get("title", ""),
        "section": m.get("section", ""),
        "page_or_slide": m.get("page_or_slide", ""),
        "url": m.get("url", ""),
        "trust_tier": m.get("trust_tier", 3),
        "score": round(r.rerank_score if r.rerank_score is not None else r.dense_score, 4),
        "preview": r.text[:280],
    }


def nearest_topics(candidates: list[Retrieved], limit: int = 4) -> list[str]:
    topics: list[str] = []
    for c in candidates:
        label = c.metadata.get("section") or c.metadata.get("title") or ""
        label = re.sub(r"^\d+(\.\d+)*\s+", "", label).strip()
        if label and label.lower() not in (t.lower() for t in topics):
            topics.append(label)
        if len(topics) >= limit:
            break
    return topics


class Retriever:
    """Dense retrieval + cross-encoder rerank; separated so tests can substitute a fake."""

    def search(self, query: str) -> tuple[list[Retrieved], list[Retrieved]]:
        from oslm.rag.rerank import rerank
        from oslm.rag.retrieve import dense_search

        candidates = dense_search(query)
        return rerank(query, list(candidates)), candidates


class Pipeline:
    def __init__(
        self, client: OllamaClient | None = None, retriever: Retriever | None = None, checker: Any = None
    ) -> None:
        self.cfg = load_config()
        self.client = client or OllamaClient()
        self.retriever = retriever or Retriever()
        self.checker = checker

    async def _collect(self, model: str, messages: list[dict[str, str]], options: dict[str, Any]) -> tuple[str, int]:
        parts: list[str] = []
        eval_count = 0
        async for chunk in self.client.chat_stream(model, messages, options):
            parts.append(chunk.get("message", {}).get("content", ""))
            if chunk.get("done"):
                eval_count += chunk.get("eval_count", 0)
        return "".join(parts), eval_count

    async def _route(self, req: ChatRequest) -> PreparedQuestion:
        history = req.history[-6:]
        question = req.message
        language = response_language(req.message)
        normalized = False
        check_answer = is_check_answer(req.message, history)
        aux_model = auxiliary_model()
        # Normalizing a long code-review request can discard the code and change its
        # route. Preserve code verbatim; prompts already isolate it from instructions.
        if needs_normalization(req.message) and not contains_code(req.message):
            question = await normalize_question(self.client, aux_model, req.message)
            normalized = question != req.message
        r = route(question, history=history)
        if r.followup:
            if check_answer:
                # The student's proposed answer is the evidence to grade. Rewriting it as a
                # standalone question discards that evidence and encourages lesson replay.
                question = html.unescape(req.message).strip()
            else:
                question = await rewrite_followup(self.client, aux_model, req.message, history)
                if r.mode == "concept":
                    r.mode = refine_followup(question)
        elif r.ambiguous:
            label = await classify_ambiguous(self.client, aux_model, question)
            r.mode = {"sync": "concept", "os": "os_out_of_scope", "greeting": "greeting", "other": "off_topic"}[label]
            r.needs_retrieval = r.mode == "concept"
            r.reason += f"; llm={label}"
        return PreparedQuestion(r, question, language, normalized, check_answer)

    async def run(self, req: ChatRequest) -> AsyncIterator[Event]:
        t0 = time.perf_counter()
        setup = req.setup if req.setup in SETUPS else self.cfg["setup"]
        use_rag = setup.endswith("_rag")
        requested_model = model_for_setup(setup)

        try:
            prepared = await self._route(req)
            r, question = prepared.route, prepared.question
        except OllamaError as exc:
            yield Event("error", {"message": str(exc), "kind": "ollama_unreachable"})
            return

        # The teaching adapter was trained on English course data. For a detected
        # non-English request, use the multilingual base model for the final answer
        # as well as normalization; otherwise the adapter can loop or emit unrelated C.
        language_fallback = prepared.answer_language is not None
        normalized_fallback = prepared.normalized and setup.startswith("ft")
        task_fallback = setup.startswith("ft") and r.mode in ("code_write", "bug_find")
        canonical_check = _canonical_check_reply(req.message, req.history) if prepared.check_answer else None
        answer_check_fallback = setup.startswith("ft") and prepared.check_answer and canonical_check is None
        algorithm_fallback = setup.startswith("ft") and _is_peterson_question(question)
        canonical_peterson = algorithm_fallback and _is_generic_peterson_explanation(question)
        base_fallback = (
            language_fallback
            or normalized_fallback
            or task_fallback
            or answer_check_fallback
            or algorithm_fallback
        )
        model = auxiliary_model() if base_fallback else requested_model
        fallback_reason = None
        if language_fallback:
            fallback_reason = "english_accuracy_fallback"
        elif normalized_fallback:
            fallback_reason = "normalized_prompt_base"
        elif task_fallback:
            fallback_reason = "code_safety_base"
        elif canonical_check:
            fallback_reason = "answer_check_canonical"
        elif answer_check_fallback:
            fallback_reason = "answer_check_base"
        elif canonical_peterson:
            fallback_reason = "peterson_canonical"
        elif algorithm_fallback:
            fallback_reason = "algorithm_accuracy_base"

        if r.mode == "empty":
            yield Event("meta", {"mode": "empty", "setup": setup, "rewritten_query": None, "followup": False})
            yield Event("sources", {"sources": []})
            yield Event("token", {"text": EMPTY_REPLY})
            yield Event("done", {"timings": {"total_ms": round((time.perf_counter() - t0) * 1000)}, "mode": "empty"})
            return

        mode = r.mode
        chunks: list[ContextChunk] | None = None if not use_rag else []
        sources: list[dict[str, Any]] = []
        topics: list[str] = []
        best_score: float | None = None
        t_retrieval = time.perf_counter()

        search_query_used: str | None = None
        if use_rag and r.needs_retrieval and mode in RETRIEVAL_MODES:
            query = build_search_query(question, mode)
            top, candidates = await asyncio.to_thread(self.retriever.search, query)
            best_score = top[0].rerank_score if top else None
            threshold = self.cfg.retrieval["relevance_threshold"]
            if best_score is None or best_score < threshold:
                # Typos and instruction-heavy phrasing confuse the cross-encoder; retry with a cleaned query.
                try:
                    search_query_used = await search_query(self.client, auxiliary_model(), question)
                except OllamaError:
                    search_query_used = None
                if search_query_used and search_query_used.lower() != query.lower():
                    top2, candidates2 = await asyncio.to_thread(self.retriever.search, search_query_used)
                    best2 = top2[0].rerank_score if top2 else None
                    if best2 is not None and (best_score is None or best2 > best_score):
                        top, candidates, best_score = top2, candidates2, best2
            if best_score is None or best_score < threshold:
                topics = nearest_topics(candidates) or [DEFAULT_TOPICS]
                if mode in GROUNDED_ONLY_MODES:
                    mode = "not_in_notes"
                top = []
            chunks = [
                ContextChunk(
                    t.metadata.get("source_name", ""),
                    t.metadata.get("section", ""),
                    str(t.metadata.get("page_or_slide", "")),
                    t.text,
                )
                for t in top
            ]
            sources = [source_card(i, t) for i, t in enumerate(top, start=1)]
        retrieval_ms = round((time.perf_counter() - t_retrieval) * 1000)

        review_evidence = bug_review_hints(req.message) if mode == "bug_find" else []
        if mode == "bug_find" and self.checker is not None:
            submitted_programs = complete_c_programs(req.message)
            if submitted_programs:
                submitted_reports = await self.checker.check_many(submitted_programs)
                review_evidence.extend(
                    f"Automated checker: {report.status} — {report.summary}" for report in submitted_reports
                )

        yield Event(
            "meta",
            {
                "mode": mode,
                "route": r.mode,
                "setup": setup,
                "model": model,
                "model_fallback": fallback_reason,
                "followup": r.followup,
                "rewritten_query": question if question != req.message else None,
                "best_rerank_score": best_score,
                "reason": r.reason,
                "search_query": search_query_used,
                "normalized": prepared.normalized,
                "answer_language": prepared.answer_language,
                "review_evidence": review_evidence,
            },
        )
        yield Event("sources", {"sources": sources})

        history = [] if r.mode in ("greeting", "off_topic") else req.history[-4:]
        messages = build_messages(
            question,
            mode,
            chunks,
            history=history,
            followup=r.followup,
            topics=topics,
            answer_language=None if language_fallback else prepared.answer_language,
            check_answer=prepared.check_answer,
        )
        if review_evidence:
            messages[-1]["content"] += (
                "\n\n<review_evidence>\n"
                + "\n".join(f"- {item}" for item in review_evidence)
                + "\n</review_evidence>\nUse this automated evidence as facts in the review; do not contradict it "
                "or invent a different bug."
            )
        ttft_ms: float | None = None
        t_gen = time.perf_counter()
        eval_count = 0
        verify_ms = 0
        reports: list[Any] = []
        repair_count = 0
        code_fallback: str | None = None
        code_withheld = False
        interactive_seed = self.cfg["seed"] if not req.session_id else secrets.randbelow(2**31)
        option_overrides: dict[str, Any] = {"seed": interactive_seed}
        if prepared.check_answer:
            option_overrides["temperature"] = 0.0
        options = decoding_options(mode, option_overrides)
        language_prefix = LANGUAGE_FALLBACK_PREFIX.get(prepared.answer_language or "", "")
        try:
            buffered_code = mode == "code_write" and req.verify_code and self.checker is not None
            if buffered_code:
                initial_fallback = (
                    _canonical_code_fallback(question) if _is_generic_producer_consumer_request(question) else None
                )
                if initial_fallback:
                    code_fallback, answer = initial_fallback
                    answer, stream = _clean_generated(answer, len(sources))
                else:
                    raw, eval_count = await self._collect(model, messages, options)
                    raw = clean_external_answer(raw, "code_write")
                    answer, stream = _clean_generated(raw, len(sources))
                answer = language_prefix + answer
                programs = complete_c_programs(answer)
                t_v = time.perf_counter()
                reports = await self.checker.check_many(programs) if programs else []
                feedback = _repair_feedback(programs, reports)
                max_repairs = int(self.cfg.decoding.get("code_repair_attempts", 1))
                while feedback and repair_count < max_repairs:
                    repair_count += 1
                    repair_messages = build_code_repair_messages(messages, answer, feedback)
                    raw, used = await self._collect(
                        model,
                        repair_messages,
                        decoding_options(mode, {"seed": (interactive_seed + repair_count) % (2**31)}),
                    )
                    eval_count += used
                    raw = clean_external_answer(raw, "code_write")
                    answer, stream = _clean_generated(raw, len(sources))
                    answer = language_prefix + answer
                    programs = complete_c_programs(answer)
                    reports = await self.checker.check_many(programs) if programs else []
                    feedback = _repair_feedback(programs, reports)
                if feedback and code_fallback is None:
                    known_fallback = _canonical_code_fallback(question)
                    if known_fallback:
                        code_fallback, fallback_answer = known_fallback
                        fallback_programs = complete_c_programs(fallback_answer)
                        fallback_reports = await self.checker.check_many(fallback_programs)
                        fallback_feedback = _repair_feedback(fallback_programs, fallback_reports)
                        if not fallback_feedback:
                            answer, stream = _clean_generated(language_prefix + fallback_answer, len(sources))
                            programs, reports, feedback = fallback_programs, fallback_reports, None
                        else:
                            reports, feedback = fallback_reports, fallback_feedback
                if feedback:
                    code_withheld = True
                    answer, stream = _clean_generated(
                        language_prefix + _withheld_code_reply(feedback, repair_count), len(sources)
                    )
                    programs = []
                verify_ms = round((time.perf_counter() - t_v) * 1000)
                ttft_ms = (time.perf_counter() - t0) * 1000
                yield Event("token", {"text": answer})
            elif canonical_check:
                answer, stream = _clean_generated(canonical_check, len(sources))
                ttft_ms = (time.perf_counter() - t0) * 1000
                yield Event("token", {"text": answer})
            elif canonical_peterson:
                answer, stream = _clean_generated(PETERSON_SAFE_REPLY, len(sources))
                ttft_ms = (time.perf_counter() - t0) * 1000
                yield Event("token", {"text": answer})
            elif prepared.check_answer or algorithm_fallback:
                raw, eval_count = await self._collect(model, messages, options)
                answer, stream = _clean_generated(raw, len(sources))
                if algorithm_fallback and _peterson_answer_is_wrong(answer, question):
                    answer, stream = _clean_generated(PETERSON_SAFE_REPLY, len(sources))
                answer = language_prefix + answer
                ttft_ms = (time.perf_counter() - t0) * 1000
                yield Event("token", {"text": answer})
            else:
                stream = CitationStream(len(sources))
                if language_prefix:
                    ttft_ms = (time.perf_counter() - t0) * 1000
                    yield Event("token", {"text": language_prefix})
                async for chunk in self.client.chat_stream(model, messages, options):
                    delta = chunk.get("message", {}).get("content", "")
                    if delta:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t0) * 1000
                        out = stream.feed(delta)
                        if out:
                            yield Event("token", {"text": out})
                    if chunk.get("done"):
                        eval_count = chunk.get("eval_count", 0)
                tail = stream.flush()
                if tail:
                    yield Event("token", {"text": tail})
                answer = language_prefix + stream.text
        except OllamaError as exc:
            yield Event(
                "error", {"message": str(exc), "kind": "ollama_unreachable" if "reach" in str(exc) else "ollama_error"}
            )
            return
        gen_ms = (time.perf_counter() - t_gen) * 1000

        programs = (
            complete_c_programs(answer)
            if mode == "code_write" and req.verify_code and self.checker is not None
            else []
        )
        if programs and not reports:
            t_v = time.perf_counter()
            reports = await self.checker.check_many(programs)
            verify_ms = round((time.perf_counter() - t_v) * 1000)
        for idx in range(len(programs)):
            yield Event("verification", {"block_index": idx, "status": "pending"})
        for idx, report in enumerate(reports):
            yield Event("verification", {"block_index": idx, "status": "done", "report": report.to_dict()})

        valid_cites, invalid_cites = stream.counts()
        yield Event(
            "done",
            {
                "mode": mode,
                "cited": cited_numbers(answer, len(sources)),
                "citations": {"valid": valid_cites, "invalid": invalid_cites},
                "repair_attempts": repair_count,
                "code_fallback": code_fallback,
                "code_withheld": code_withheld,
                "timings": {
                    "ttft_ms": round(ttft_ms) if ttft_ms is not None else None,
                    "retrieval_ms": retrieval_ms,
                    "generation_ms": round(gen_ms),
                    "verification_ms": verify_ms,
                    "total_ms": round((time.perf_counter() - t0) * 1000),
                    "tokens": eval_count,
                },
            },
        )


async def answer_text(pipeline: Pipeline, req: ChatRequest) -> tuple[str, dict[str, Any]]:
    """Runs the pipeline to completion and returns (answer, collected event data) — used by eval."""
    parts: list[str] = []
    info: dict[str, Any] = {"verification": []}
    async for ev in pipeline.run(req):
        if ev.type == "token":
            parts.append(ev.data["text"])
        elif ev.type == "verification":
            if ev.data.get("status") == "done":
                info["verification"].append(ev.data["report"])
        else:
            info[ev.type] = ev.data
    return "".join(parts), info
