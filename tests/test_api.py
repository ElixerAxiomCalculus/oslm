from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from oslm.api import main as api
from oslm.checker.report import CheckReport, Finding
from oslm.rag.pipeline import ChatRequest, Event


class StubPipeline:
    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    async def run(self, req: ChatRequest) -> AsyncIterator[Event]:
        self.requests.append(req)
        yield Event("meta", {"mode": "concept", "setup": req.setup, "rewritten_query": None, "followup": False})
        yield Event("sources", {"sources": []})
        yield Event("token", {"text": "Hello "})
        yield Event("token", {"text": "world"})
        yield Event("done", {"timings": {"total_ms": 1}, "mode": "concept", "cited": []})


class StubChecker:
    available = True

    async def check(self, code: str) -> CheckReport:
        return CheckReport(
            "issues",
            "data race (line 3)",
            tsan_clean=False,
            helgrind_clean=False,
            findings=[Finding("tsan", "data race", "race", [3], ["thread T1"])],
        )

    async def warmup(self) -> bool:
        return True


class StubClient:
    async def tags(self) -> list[str]:
        return ["qwen3:4b-instruct-2507-q4_K_M"]

    async def aclose(self) -> None:
        return None


@pytest.fixture
async def http() -> AsyncIterator[tuple[httpx.AsyncClient, Any]]:
    state = api.AppState.__new__(api.AppState)
    state.cfg = api.load_config()
    state.client = StubClient()
    state.checker = StubChecker()
    state.pipeline = StubPipeline()
    state.pending_explanations = {}
    state.warm_tasks = []
    api.app.state.oslm = state
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, state


def parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for block in body.replace("\r\n", "\n").split("\n\n"):
        name = data = None
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()
        if name and data:
            events.append((name, json.loads(data)))
    return events


async def test_chat_streams_sse(http: tuple[httpx.AsyncClient, Any]) -> None:
    client, state = http
    resp = await client.post(
        "/api/chat",
        json={
            "session_id": "s1",
            "message": "What is a lock?",
            "setup": "base",
            "history": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(resp.text)
    assert [e[0] for e in events] == ["meta", "sources", "token", "token", "done"]
    assert "".join(d["text"] for n, d in events if n == "token") == "Hello world"
    assert state.pipeline.requests[0].setup == "base" and state.pipeline.requests[0].history[0]["content"] == "hi"


async def test_chat_rejects_long_message_with_friendly_error(http: tuple[httpx.AsyncClient, Any]) -> None:
    client, _ = http
    resp = await client.post("/api/chat", json={"message": "x" * 8001})
    assert resp.status_code == 422
    assert "under 8,000" in resp.json()["error"]


async def test_chat_rejects_unknown_setup(http: tuple[httpx.AsyncClient, Any]) -> None:
    client, _ = http
    assert (await client.post("/api/chat", json={"message": "hi", "setup": "gpt5"})).status_code == 422


async def test_chat_defaults_setup_from_config(http: tuple[httpx.AsyncClient, Any]) -> None:
    client, state = http
    await client.post("/api/chat", json={"message": "hi"})
    assert state.pipeline.requests[0].setup == state.cfg["setup"]


async def test_health_and_setups(http: tuple[httpx.AsyncClient, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = http
    import oslm.rag.retrieve as retrieve

    monkeypatch.setattr(retrieve, "collection_size", lambda: 42)
    health = (await client.get("/api/health")).json()
    assert health["ollama"] == "up" and health["index_size"] == 42 and health["checker"] == "online"
    setups = {s["name"]: s for s in (await client.get("/api/setups")).json()}
    assert setups["base_rag"]["available"] and setups["base_rag"]["rag"]
    assert not setups["ft"]["available"] and setups["ft"]["fine_tuned"]


async def test_check_code_and_explanation(http: tuple[httpx.AsyncClient, Any]) -> None:
    client, state = http
    resp = (await client.post("/api/check-code", json={"code": "int main(void){}"})).json()
    assert resp["report"]["status"] == "issues" and resp["report"]["has_sync_bug"]
    assert "raw" not in resp["report"]
    stream = await client.get(resp["explain_url"])
    assert [e[0] for e in parse_sse(stream.text)][-1] == "done"
    question = state.pipeline.requests[-1].message
    assert "data race" in question and state.pipeline.requests[-1].verify_code is False
    assert (await client.get(resp["explain_url"])).status_code == 404


async def test_cors_limited_to_vite_origin(http: tuple[httpx.AsyncClient, Any]) -> None:
    client, _ = http
    ok = await client.options(
        "/api/chat", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"}
    )
    bad = await client.options(
        "/api/chat", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"}
    )
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "access-control-allow-origin" not in bad.headers
