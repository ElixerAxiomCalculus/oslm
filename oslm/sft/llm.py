from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any

log = logging.getLogger(__name__)

# Free-tier Flash models only (user requirement); Pro models are never called.
DEFAULT_MODELS = "gemini-3.5-flash,gemini-3.8-flash,gemini-3.7-flash,gemini-flash-latest"
LITE_MODELS = "gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-flash-lite-latest,gemini-3.1-flash-lite-preview"


class GeminiError(RuntimeError):
    pass


def items_of(data: Any, key: str) -> list[Any]:
    """The list under `key`, tolerating a bare top-level list (Gemini sometimes drops the wrapper object)."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        value = data.get(key)
        if isinstance(value, list):
            return value
        lists = [v for v in data.values() if isinstance(v, list)]
        return lists[0] if len(lists) == 1 else []
    return []


def extract_json(text: str) -> Any:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1)
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if starts and min(starts) > 0:
        text = text[min(starts) :]
    return json.loads(text)


class Gemini:
    """Async google-genai wrapper: rotates across free models on quota/overload errors, JSON helpers."""

    # Free tier allows ~20 requests per model per day, so each request carries a lot of work.
    answers_per_call = 12
    bug_answers_per_call = 6
    excerpts_per_call = 3
    programs_per_call = 8
    judge_per_call = 12

    def __init__(self, models: str | None = None, concurrency: int = 4, lite: bool = False) -> None:
        from google import genai

        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key or key.startswith("your-"):
            raise GeminiError("GEMINI_API_KEY is not set in .env")
        names = (
            models
            or os.environ.get("GEMINI_LITE_MODELS" if lite else "GEMINI_MODELS")
            or (LITE_MODELS if lite else DEFAULT_MODELS)
        )
        self.models = [m.strip() for m in names.split(",") if m.strip() and "pro" not in m]
        self.client = genai.Client(api_key=key)
        self._limit = asyncio.Semaphore(concurrency)
        self._cooldown: dict[str, float] = {}
        self._exhausted: set[str] = set()
        self.calls: dict[str, int] = {}

    @property
    def all_exhausted(self) -> bool:
        return all(m in self._exhausted for m in self.models)

    def _pick(self) -> str | None:
        now = time.monotonic()
        live = [m for m in self.models if m not in self._exhausted]
        ready = [m for m in live if self._cooldown.get(m, 0) <= now]
        if ready:
            return ready[0]
        return min(live, key=lambda m: self._cooldown.get(m, 0)) if live else None

    def _penalise(self, model: str, exc: Exception) -> None:
        msg = str(exc)
        if "404" in msg or "NOT_FOUND" in msg:
            self._exhausted.add(model)
        elif "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            if "PerDay" in msg or "per day" in msg.lower():
                log.warning("daily quota exhausted for %s", model)
                self._exhausted.add(model)
            else:
                retry = re.search(r"retry in ([\d.]+)s|'retryDelay': '(\d+)s'", msg)
                wait = float(next(g for g in retry.groups() if g)) if retry else 30.0
                self._cooldown[model] = time.monotonic() + wait + 1
        else:
            self._cooldown[model] = time.monotonic() + 20

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.8,
        json_mode: bool = False,
        max_output_tokens: int = 24_000,
        attempts: int = 12,
        think: bool | None = None,  # noqa: ARG002 - local-model option, ignored by Gemini
    ) -> str:
        from google.genai import types

        # Thinking tokens count against max_output_tokens on these models, so the budget is generous.
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_instruction=system,
            response_mime_type="application/json" if json_mode else None,
        )
        async with self._limit:
            last: Exception | None = None
            for _ in range(attempts):
                model = self._pick()
                if model is None:
                    raise GeminiError(f"all free models exhausted for today ({', '.join(self.models)})")
                wait = self._cooldown.get(model, 0) - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(min(wait, 120))
                try:
                    self.calls[model] = self.calls.get(model, 0) + 1
                    resp = await self.client.aio.models.generate_content(model=model, contents=prompt, config=config)
                    text = resp.text or ""
                    if text.strip():
                        return text
                    raise GeminiError("empty response")
                except Exception as exc:
                    last = exc
                    log.warning("%s failed: %s", model, str(exc)[:200])
                    self._penalise(model, exc)
            raise GeminiError(f"giving up after {attempts} attempts: {last}")

    async def generate_json(self, prompt: str, system: str | None = None, temperature: float = 0.8, **kw: Any) -> Any:
        for attempt in range(3):
            text = await self.generate(prompt, system=system, temperature=temperature, json_mode=True, **kw)
            try:
                return extract_json(text)
            except json.JSONDecodeError:
                log.warning("invalid JSON from Gemini (attempt %d)", attempt + 1)
        raise GeminiError("Gemini did not return valid JSON")


class LocalQwen:
    """Same interface as Gemini, backed by a local Ollama model (default qwen3:8b with thinking).

    A local model is slower but has no quota, and it handles one item per request far more reliably than a batch.
    """

    answers_per_call = 1
    bug_answers_per_call = 1
    excerpts_per_call = 1
    programs_per_call = 2
    judge_per_call = 3

    def __init__(self, model: str | None = None, think: bool = True, concurrency: int = 1) -> None:
        from oslm.config import load_config

        cfg = load_config()
        self.model = model or os.environ.get("OSLM_LOCAL_MODEL") or cfg.sft.get("local_model", "qwen3:8b")
        self.think = think
        self.url = cfg.ollama["url"].rstrip("/") + "/api/chat"
        self.num_ctx = cfg.ollama["num_ctx"]
        self._limit = asyncio.Semaphore(concurrency)
        self.calls: dict[str, int] = {}
        self.all_exhausted = False

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.8,
        json_mode: bool = False,
        max_output_tokens: int = 6000,
        attempts: int = 3,
        think: bool | None = None,
    ) -> str:
        import httpx

        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": self.think if think is None else think,
            "keep_alive": "30m",
            # Thinking tokens share num_predict with the answer, hence the generous cap.
            "options": {
                "temperature": temperature,
                "num_ctx": self.num_ctx,
                "num_predict": min(max_output_tokens, 7000),
            },
        }
        if json_mode:
            body["format"] = "json"
        async with self._limit:
            last: Exception | None = None
            for _ in range(attempts):
                try:
                    self.calls[self.model] = self.calls.get(self.model, 0) + 1
                    async with httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=10.0)) as client:
                        resp = await client.post(self.url, json=body)
                    resp.raise_for_status()
                    text = resp.json().get("message", {}).get("content", "")
                    if text.strip():
                        return text
                    raise GeminiError("empty response")
                except Exception as exc:
                    last = exc
                    log.warning("%s failed: %s", self.model, str(exc)[:200])
                    await asyncio.sleep(5)
            raise GeminiError(f"local model failed after {attempts} attempts: {last}")

    async def generate_json(self, prompt: str, system: str | None = None, temperature: float = 0.8, **kw: Any) -> Any:
        for attempt in range(3):
            text = await self.generate(prompt, system=system, temperature=temperature, json_mode=True, **kw)
            try:
                return extract_json(text)
            except json.JSONDecodeError:
                log.warning("invalid JSON from %s (attempt %d)", self.model, attempt + 1)
        raise GeminiError(f"{self.model} did not return valid JSON")


def make_llm(lite: bool = False) -> Gemini | LocalQwen:
    """Generation backend from config `sft.provider` (env OSLM_SFT_PROVIDER overrides): "ollama" or "gemini"."""
    from oslm.config import load_config

    provider = os.environ.get("OSLM_SFT_PROVIDER") or load_config().sft.get("provider", "ollama")
    return Gemini(lite=lite) if provider == "gemini" else LocalQwen()
