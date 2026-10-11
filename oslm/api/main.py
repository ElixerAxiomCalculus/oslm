from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from oslm.api.schemas import ChatBody, CheckCodeBody, HealthResponse, SetupInfo, SourceDetail
from oslm.checker.report import CheckReport, precise_lines
from oslm.checker.wsl_bridge import CheckerBridge
from oslm.config import load_config
from oslm.rag.generate import OllamaClient, OllamaError, model_for_setup
from oslm.rag.pipeline import SETUPS, ChatRequest, Event, Pipeline

log = logging.getLogger("oslm.api")

SETUP_LABELS = {
    "base": "Base model",
    "base_rag": "Base model + notes (RAG)",
    "ft": "Fine-tuned",
    "ft_rag": "Fine-tuned + notes (RAG)",
}
EXPLAIN_TTL_S = 600


class AppState:
    def __init__(self) -> None:
        self.cfg = load_config()
        self.client = OllamaClient()
        self.checker = CheckerBridge()
        self.pipeline = Pipeline(client=self.client, checker=self.checker)
        self.pending_explanations: dict[str, tuple[float, str, CheckReport, str]] = {}
        self.warm_tasks: list[asyncio.Task[Any]] = []


def _preload_retrieval() -> None:
    from oslm.rag.rerank import get_reranker
    from oslm.rag.retrieve import collection_size

    if collection_size() > 0:
        from oslm.index.embed import get_embedder

        get_embedder()
        get_reranker()


async def _warm_ollama(state: AppState) -> None:
    model = model_for_setup(state.cfg["setup"])
    try:
        await state.client._client.post("/api/generate", json={"model": model, "keep_alive": "30m"}, timeout=180)
    except Exception as exc:
        log.warning("Ollama warmup failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    state = AppState()
    app.state.oslm = state
    state.warm_tasks = [
        asyncio.create_task(state.checker.warmup()),
        asyncio.create_task(asyncio.to_thread(_preload_retrieval)),
        asyncio.create_task(_warm_ollama(state)),
    ]
    yield
    for task in state.warm_tasks:
        task.cancel()
    await state.client.aclose()


app = FastAPI(title="OSlm", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=load_config().api["cors_origins"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(RequestValidationError)
async def friendly_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    messages = [str(e.get("msg", "")).removeprefix("Value error, ") for e in exc.errors()]
    return JSONResponse(status_code=422, content={"error": "; ".join(messages) or "Invalid request."})


def _state(request: Request) -> AppState:
    return request.app.state.oslm


def _sse(events: AsyncIterator[Event]) -> EventSourceResponse:
    async def gen() -> AsyncIterator[dict[str, str]]:
        try:
            async for ev in events:
                yield {"event": ev.type, "data": json.dumps(ev.data, ensure_ascii=False)}
        except Exception as exc:
            log.exception("stream failed")
            yield {"event": "error", "data": json.dumps({"message": f"Internal error: {exc}", "kind": "internal"})}

    return EventSourceResponse(gen(), ping=15)


@app.get("/api/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    from oslm.rag.retrieve import collection_size

    state = _state(request)
    try:
        tags = await state.client.tags()
        ollama = "up"
    except OllamaError:
        tags, ollama = [], "down"
    checker = {True: "online", False: "offline", None: "unknown"}[state.checker.available]
    index_size = await asyncio.to_thread(collection_size)
    return HealthResponse(
        status="ok" if ollama == "up" else "degraded",
        setup=state.cfg["setup"],
        models={"base": state.cfg.ollama["base_tag"], "ft": state.cfg.ollama["ft_tag"]},
        ollama=ollama,
        ollama_models=tags,
        index_size=index_size,
        checker=checker,
    )


def _tag_present(tag: str, tags: list[str]) -> bool:
    return any(t == tag or t == f"{tag}:latest" or t.split(":")[0] == tag for t in tags)


@app.get("/api/setups", response_model=list[SetupInfo])
async def setups(request: Request) -> list[SetupInfo]:
    state = _state(request)
    try:
        tags = await state.client.tags()
    except OllamaError:
        tags = []
    return [
        SetupInfo(
            name=s,  # type: ignore[arg-type]
            label=SETUP_LABELS[s],
            model=model_for_setup(s),
            rag=s.endswith("_rag"),
            fine_tuned=s.startswith("ft"),
            available=_tag_present(model_for_setup(s), tags),
        )
        for s in SETUPS
    ]


@app.post("/api/chat")
async def chat(body: ChatBody, request: Request) -> EventSourceResponse:
    state = _state(request)
    req = ChatRequest(
        message=body.message,
        setup=body.setup or state.cfg["setup"],
        history=[t.model_dump() for t in body.history],
        session_id=body.session_id,
    )
    return _sse(state.pipeline.run(req))


def _explain_question(code: str, report: CheckReport) -> str:
    tools_by_kind: dict[str, list[str]] = {}
    for f in report.findings:
        tools_by_kind.setdefault(f.kind, []).append(f.tool)
    findings = (
        "\n".join(
            f"- {kind} (reported by {', '.join(dict.fromkeys(tools))})"
            + (f" at line(s) {', '.join(map(str, lines))}" if (lines := precise_lines(report.findings, kind)) else "")
            for kind, tools in tools_by_kind.items()
        )
        or "- no findings"
    )
    return (
        f"```c\n{code.strip()}\n```\n\n"
        f"The automatic checker (gcc + ThreadSanitizer + Helgrind) reported: {report.summary}\n{findings}\n\n"
        "Explain these results to me as my teacher: what they mean for this code, the interleaving that causes "
        "any problem, and the minimal fix. If the checker found nothing, confirm by reasoning about the interleavings."
    )


@app.post("/api/check-code")
async def check_code(body: CheckCodeBody, request: Request) -> dict[str, Any]:
    state = _state(request)
    report = await state.checker.check(body.code)
    explain_url = None
    if body.explain and report.status in ("clean", "issues"):
        now = time.monotonic()
        state.pending_explanations = {k: v for k, v in state.pending_explanations.items() if now - v[0] < EXPLAIN_TTL_S}
        token = secrets.token_urlsafe(16)
        state.pending_explanations[token] = (now, body.code, report, body.setup or state.cfg["setup"])
        explain_url = f"/api/check-code/explain/{token}"
    return {"report": report.to_dict(), "explain_url": explain_url}


@app.get("/api/check-code/explain/{token}")
async def explain_check(token: str, request: Request) -> EventSourceResponse:
    state = _state(request)
    item = state.pending_explanations.pop(token, None)
    if item is None:
        raise HTTPException(404, "Explanation expired or already used.")
    _, code, report, setup = item
    req = ChatRequest(message=_explain_question(code, report), setup=setup, verify_code=False)
    return _sse(state.pipeline.run(req))


@app.get("/api/sources/{chunk_id:path}", response_model=SourceDetail)
async def source(chunk_id: str) -> SourceDetail:
    from oslm.rag.retrieve import get_chunk

    chunk = await asyncio.to_thread(get_chunk, chunk_id)
    if chunk is None:
        raise HTTPException(404, "Source not found.")
    m = chunk.metadata
    return SourceDetail(
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        source_name=m.get("source_name", ""),
        title=m.get("title", ""),
        section=m.get("section", ""),
        page_or_slide=str(m.get("page_or_slide", "")),
        url=m.get("url", ""),
        trust_tier=int(m.get("trust_tier", 3)),
        has_code=bool(m.get("has_code", False)),
    )
