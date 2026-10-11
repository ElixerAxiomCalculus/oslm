from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from oslm.config import load_config


class OllamaError(RuntimeError):
    pass


def model_for_setup(setup: str) -> str:
    tags = load_config().ollama
    return tags["ft_tag"] if setup.startswith("ft") else tags["base_tag"]


def decoding_options(mode: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = load_config()
    dec = cfg.decoding
    options: dict[str, Any] = {
        "temperature": dec["temperature"],
        "top_p": dec["top_p"],
        "repeat_penalty": dec["repeat_penalty"],
        "num_predict": dec["max_tokens_code"] if mode == "code_write" else dec["max_tokens"],
        "num_ctx": cfg.ollama["num_ctx"],
        "seed": cfg["seed"],
    }
    options.update(overrides or {})
    return options


class OllamaClient:
    def __init__(self, base_url: str | None = None, timeout: float = 300.0) -> None:
        self.base_url = (base_url or load_config().ollama["url"]).rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(timeout, connect=5.0))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def tags(self) -> list[str]:
        try:
            resp = await self._client.get("/api/tags")
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaError(f"cannot reach Ollama at {self.base_url}: {exc}") from exc
        return [m["name"] for m in resp.json().get("models", [])]

    async def chat_stream(
        self, model: str, messages: list[dict[str, str]], options: dict[str, Any], keep_alive: str = "30m"
    ) -> AsyncIterator[dict[str, Any]]:
        """Yields Ollama chat chunks; the final chunk has done=True and timing fields."""
        payload = {"model": model, "messages": messages, "stream": True, "options": options, "keep_alive": keep_alive}
        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    raise OllamaError(f"Ollama returned {resp.status_code}: {body[:300]}")
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    if "error" in chunk:
                        raise OllamaError(chunk["error"])
                    yield chunk
        except httpx.HTTPError as exc:
            raise OllamaError(f"cannot reach Ollama at {self.base_url}: {exc}") from exc

    async def chat(self, model: str, messages: list[dict[str, str]], options: dict[str, Any]) -> str:
        parts: list[str] = []
        async for chunk in self.chat_stream(model, messages, options):
            parts.append(chunk.get("message", {}).get("content", ""))
        return "".join(parts)


REWRITE_SYSTEM = (
    "You rewrite a student's latest message into one standalone question about process synchronization, "
    "using the earlier conversation to resolve words like 'it', 'that', 'why?', 'now in C'. "
    "Keep the student's intent (explain, write code, trace, find a bug). Output only the rewritten question."
)

SEARCH_QUERY_SYSTEM = (
    "Turn the student's message into one short search query for course notes on process synchronization. "
    "Fix spelling, expand slang, and name the underlying concept explicitly (for example: race condition, lost update, "
    "critical section, semaphore, deadlock, condition variable, dining philosophers). Output only the query."
)

CLASSIFY_SYSTEM = (
    "Classify the student's message for an Operating Systems tutor that only teaches process synchronization. "
    "Answer with exactly one word:\n"
    "sync - about threads, race conditions, locks, semaphores, deadlock, monitors or classic synchronization problems "
    "(in any language, even with typos)\n"
    "os - another operating systems topic (scheduling, memory, paging, file systems, I/O)\n"
    "greeting - small talk or greeting\n"
    "other - anything else"
)


def _history_text(history: list[dict[str, str]], max_chars: int = 1200) -> str:
    lines = []
    for turn in history[-6:]:
        content = turn["content"]
        if len(content) > max_chars:
            content = content[:max_chars] + " …"
        lines.append(f"{turn['role'].upper()}: {content}")
    return "\n".join(lines)


def _aux_options(num_predict: int) -> dict[str, Any]:
    # Same num_ctx as answer generation: a different value makes Ollama reload the model (~seconds).
    return {"temperature": 0, "num_predict": num_predict, "num_ctx": load_config().ollama["num_ctx"]}


def _first_line(text: str) -> str:
    stripped = text.strip()
    return stripped.strip('"').splitlines()[0].strip().strip('"') if stripped else ""


async def rewrite_followup(client: OllamaClient, model: str, message: str, history: list[dict[str, str]]) -> str:
    if not history:
        return message
    prompt = f"Conversation:\n{_history_text(history)}\n\nLatest message: {message}\n\nStandalone question:"
    messages = [{"role": "system", "content": REWRITE_SYSTEM}, {"role": "user", "content": prompt}]
    return _first_line(await client.chat(model, messages, _aux_options(96))) or message


async def search_query(client: OllamaClient, model: str, message: str) -> str:
    messages = [{"role": "system", "content": SEARCH_QUERY_SYSTEM}, {"role": "user", "content": message[:2000]}]
    return _first_line(await client.chat(model, messages, _aux_options(48))) or message


async def classify_ambiguous(client: OllamaClient, model: str, message: str) -> str:
    messages = [{"role": "system", "content": CLASSIFY_SYSTEM}, {"role": "user", "content": message[:2000]}]
    out = (await client.chat(model, messages, _aux_options(4))).strip().lower()
    for label in ("sync", "os", "greeting", "other"):
        if out.startswith(label):
            return label
    return "sync"
