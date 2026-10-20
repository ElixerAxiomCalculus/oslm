from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from typing import Any

import httpx

from oslm.config import load_config


class OllamaError(RuntimeError):
    pass


def model_for_setup(setup: str) -> str:
    tags = load_config().ollama
    return tags["ft_tag"] if setup.startswith("ft") else tags["base_tag"]


def auxiliary_model() -> str:
    """Use the instruction-following base model for routing helpers.

    The teaching adapter is trained to answer lessons, not to emit classifier labels or
    standalone rewrites. Using it for control-plane calls can turn a label request into a
    canned lesson and silently poison routing.
    """
    return load_config().ollama["base_tag"]


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

NORMALIZE_SYSTEM = (
    "Extract the student's actual process-synchronization question. Remove rambling background and translate it "
    "to concise English when necessary, while preserving every requested sub-question and requested action "
    "(explain, compare, trace, review code, or write code). Output only the resulting question, on one line."
)

_STANDALONE_INTENT = re.compile(
    r"\b(what|why|how|when|where|which|define|explain|compare|describe|write|implement|show|trace|find|review|"
    r"can|could|does|do|is|are|should|would)\b",
    re.I,
)


def response_language(message: str) -> str | None:
    """Return the requested response language for scripts covered by the robustness probes."""
    if re.search(r"[\u0900-\u097f]", message):
        return "Hindi using Devanagari script"
    if re.search(r"[\u0b80-\u0bff]", message):
        return "Tamil using Tamil script"
    return None


def needs_normalization(message: str) -> bool:
    return response_language(message) is not None or len(re.findall(r"\w+", message, re.UNICODE)) > 120


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
    line = stripped.strip('"').splitlines()[0].strip().strip('"') if stripped else ""
    return re.sub(r"^(?:standalone\s+)?question\s*:\s*", "", line, flags=re.I)


def _valid_question(text: str) -> bool:
    lower = text.lower()
    return (
        3 <= len(text) <= 800
        and "check your understanding" not in lower
        and "common misconception" not in lower
        and bool(_STANDALONE_INTENT.search(text))
    )


def _followup_fallback(message: str, history: list[dict[str, str]]) -> str:
    previous = next((t.get("content", "") for t in reversed(history) if t.get("role") == "user"), "")
    previous = re.sub(r"\s+", " ", previous).strip()[:300]
    return f"Regarding the earlier question '{previous}': {message.strip()}" if previous else message


async def rewrite_followup(client: OllamaClient, model: str, message: str, history: list[dict[str, str]]) -> str:
    if not history:
        return message
    prompt = f"Conversation:\n{_history_text(history)}\n\nLatest message: {message}\n\nStandalone question:"
    messages = [{"role": "system", "content": REWRITE_SYSTEM}, {"role": "user", "content": prompt}]
    rewritten = _first_line(await client.chat(model, messages, _aux_options(96)))
    return rewritten if _valid_question(rewritten) else _followup_fallback(message, history)


async def normalize_question(client: OllamaClient, model: str, message: str) -> str:
    messages = [{"role": "system", "content": NORMALIZE_SYSTEM}, {"role": "user", "content": message[:8000]}]
    normalized = _first_line(await client.chat(model, messages, _aux_options(160)))
    return normalized if _valid_question(normalized) else message


async def search_query(client: OllamaClient, model: str, message: str) -> str:
    messages = [{"role": "system", "content": SEARCH_QUERY_SYSTEM}, {"role": "user", "content": message[:2000]}]
    return _first_line(await client.chat(model, messages, _aux_options(48))) or message


async def classify_ambiguous(client: OllamaClient, model: str, message: str) -> str:
    messages = [{"role": "system", "content": CLASSIFY_SYSTEM}, {"role": "user", "content": message[:2000]}]
    out = (await client.chat(model, messages, _aux_options(4))).strip().lower()
    for label in ("sync", "os", "greeting", "other"):
        if out.startswith(label):
            return label
    # An invalid label is not evidence that an arbitrary request is about synchronization.
    return "other"
