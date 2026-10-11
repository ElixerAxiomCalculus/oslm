from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from oslm.config import load_config

MAX_CHARS = load_config().api["max_message_chars"]

Setup = Literal["base", "base_rag", "ft", "ft_rag"]


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=20_000)


class ChatBody(BaseModel):
    session_id: str = Field(default="", max_length=100)
    message: str
    setup: Setup | None = None
    history: list[HistoryTurn] = Field(default_factory=list, max_length=40)

    @field_validator("message")
    @classmethod
    def message_length(cls, v: str) -> str:
        if len(v) > MAX_CHARS:
            raise ValueError(
                f"That message is {len(v):,} characters — please keep it under {MAX_CHARS:,} "
                "(trim long code to the part you want checked)."
            )
        return v


class CheckCodeBody(BaseModel):
    code: str = Field(max_length=MAX_CHARS * 3)
    explain: bool = True
    setup: Setup | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    setup: str
    models: dict[str, str]
    ollama: Literal["up", "down"]
    ollama_models: list[str]
    index_size: int
    checker: Literal["online", "offline", "unknown"]


class SetupInfo(BaseModel):
    name: Setup
    label: str
    model: str
    rag: bool
    fine_tuned: bool
    available: bool


class SourceDetail(BaseModel):
    chunk_id: str
    text: str
    source_name: str
    title: str
    section: str
    page_or_slide: str
    url: str
    trust_tier: int
    has_code: bool
