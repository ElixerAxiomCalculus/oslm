from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from oslm.config import load_config
from oslm.rag.citations import CitationStream, cited_numbers
from oslm.rag.generate import (
    OllamaClient,
    OllamaError,
    classify_ambiguous,
    decoding_options,
    model_for_setup,
    rewrite_followup,
    search_query,
)
from oslm.rag.prompts import DEFAULT_TOPICS, ContextChunk, build_messages
from oslm.rag.retrieve import Retrieved, build_search_query
from oslm.rag.router import Route, refine_followup, route

log = logging.getLogger(__name__)

SETUPS = ("base", "base_rag", "ft", "ft_rag")
RETRIEVAL_MODES = {"concept", "code_write", "bug_find", "trace"}
GROUNDED_ONLY_MODES = {"concept", "trace"}

EMPTY_REPLY = (
    "Looks like that message was empty. Ask me anything about process synchronization — for example "
    '*"What is a race condition?"* or *"Why must `pthread_cond_wait` be called in a `while` loop?"*'
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


def complete_c_programs(text: str) -> list[str]:
    return [m.group(1) for m in _C_BLOCK.finditer(text) if re.search(r"\bint\s+main\s*\(", m.group(1))]


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

    async def _route(self, req: ChatRequest, model: str) -> tuple[Route, str]:
        history = req.history[-6:]
        r = route(req.message, has_history=bool(history))
        question = req.message
        if r.followup:
            question = await rewrite_followup(self.client, model, req.message, history)
            r.mode = refine_followup(question) if r.mode == "concept" else r.mode
        elif r.ambiguous:
            label = await classify_ambiguous(self.client, model, req.message)
            r.mode = {"sync": "concept", "os": "os_out_of_scope", "greeting": "greeting", "other": "off_topic"}[label]
            r.needs_retrieval = r.mode == "concept"
            r.reason += f"; llm={label}"
        return r, question

    async def run(self, req: ChatRequest) -> AsyncIterator[Event]:
        t0 = time.perf_counter()
        setup = req.setup if req.setup in SETUPS else self.cfg["setup"]
        use_rag = setup.endswith("_rag")
        model = model_for_setup(setup)

        try:
            r, question = await self._route(req, model)
        except OllamaError as exc:
            yield Event("error", {"message": str(exc), "kind": "ollama_unreachable"})
            return

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
                    search_query_used = await search_query(self.client, model, question)
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

        yield Event(
            "meta",
            {
                "mode": mode,
                "route": r.mode,
                "setup": setup,
                "model": model,
                "followup": r.followup,
                "rewritten_query": question if question != req.message else None,
                "best_rerank_score": best_score,
                "reason": r.reason,
                "search_query": search_query_used,
            },
        )
        yield Event("sources", {"sources": sources})

        history = [] if r.mode in ("greeting", "off_topic") else req.history[-4:]
        messages = build_messages(question, mode, chunks, history=history, followup=r.followup, topics=topics)
        stream = CitationStream(len(sources))
        ttft_ms: float | None = None
        t_gen = time.perf_counter()
        eval_count = 0
        try:
            async for chunk in self.client.chat_stream(model, messages, decoding_options(mode)):
                delta = chunk.get("message", {}).get("content", "")
                if delta:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000
                    out = stream.feed(delta)
                    if out:
                        yield Event("token", {"text": out})
                if chunk.get("done"):
                    eval_count = chunk.get("eval_count", 0)
        except OllamaError as exc:
            yield Event(
                "error", {"message": str(exc), "kind": "ollama_unreachable" if "reach" in str(exc) else "ollama_error"}
            )
            return
        tail = stream.flush()
        if tail:
            yield Event("token", {"text": tail})
        answer = stream.text
        gen_ms = (time.perf_counter() - t_gen) * 1000

        verify_ms = 0
        programs = complete_c_programs(answer) if req.verify_code and self.checker is not None else []
        if programs:
            t_v = time.perf_counter()
            for idx in range(len(programs)):
                yield Event("verification", {"block_index": idx, "status": "pending"})
            reports = await self.checker.check_many(programs)
            for idx, report in enumerate(reports):
                yield Event("verification", {"block_index": idx, "status": "done", "report": report.to_dict()})
            verify_ms = round((time.perf_counter() - t_v) * 1000)

        valid_cites, invalid_cites = stream.counts()
        yield Event(
            "done",
            {
                "mode": mode,
                "cited": cited_numbers(answer, len(sources)),
                "citations": {"valid": valid_cites, "invalid": invalid_cites},
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
