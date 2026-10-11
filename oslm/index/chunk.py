from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from oslm.ingest.manifest import Document

TokenCounter = Callable[[str], int]

_MD_HEADING = re.compile(r"^(#{1,4}) +(.+?)\s*$")
_NUMBERED_HEADING = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})\s+([A-Z][^.!?]{2,80})$")
_MAN_HEADING = re.compile(r"^[A-Z][A-Z ]{2,30}$")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")


@dataclass
class Block:
    text: str
    loc: str | None
    is_code: bool = False
    is_heading: bool = False


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source_name: str
    title: str
    section: str
    page_or_slide: str
    url: str
    trust_tier: int
    has_code: bool
    topic_tags: list[str]
    text: str
    n_tokens: int = 0

    def chroma_metadata(self) -> dict[str, Any]:
        meta = asdict(self)
        meta.pop("text")
        meta["topic_tags"] = ";".join(self.topic_tags)
        return meta


@dataclass
class _Builder:
    blocks: list[Block] = field(default_factory=list)
    tokens: int = 0
    fresh: bool = False  # holds content beyond the overlap carried from the previous chunk

    def add(self, block: Block, size: int) -> None:
        self.blocks.append(block)
        self.tokens += size
        self.fresh = True


def heading_text(line: str) -> str | None:
    s = line.strip()
    if m := _MD_HEADING.match(s):
        return m.group(2).strip()
    if m := _NUMBERED_HEADING.match(s):
        return f"{m.group(1)} {m.group(2).strip()}"
    if _MAN_HEADING.match(s) and len(s.split()) <= 4:
        return s.title()
    return None


def to_blocks(doc: Document) -> list[Block]:
    """Splits a cleaned document into headings, atomic fenced code blocks and paragraphs."""
    blocks: list[Block] = []

    def flush_para(para: list[str], loc: str | None) -> None:
        if para and any(p.strip() for p in para):
            blocks.append(Block("\n".join(para).strip(), loc))
        para.clear()

    for seg in doc.segments:
        lines = seg.text.splitlines()
        i = 0
        para: list[str] = []
        while i < len(lines):
            line = lines[i]
            if line.strip().startswith("```"):
                flush_para(para, seg.loc)
                j = i + 1
                while j < len(lines) and not lines[j].strip().startswith("```"):
                    j += 1
                blocks.append(Block("\n".join(lines[i : j + 1]), seg.loc, is_code=True))
                i = j + 1
                continue
            if not line.strip():
                flush_para(para, seg.loc)
            elif (h := heading_text(line)) is not None and len(line) < 100:
                flush_para(para, seg.loc)
                blocks.append(Block(h, seg.loc, is_heading=True))
            else:
                para.append(line)
            i += 1
        flush_para(para, seg.loc)
    return blocks


def _split_long_text(text: str, limit: int, count: TokenCounter) -> list[str]:
    sentences = _SENTENCE.split(text)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if count(candidate) <= limit:
            current = candidate
            continue
        if current:
            pieces.append(current)
        if count(sentence) <= limit:
            current = sentence
        else:
            words = sentence.split()
            current = ""
            for word in words:
                candidate = f"{current} {word}".strip()
                if count(candidate) > limit and current:
                    pieces.append(current)
                    current = word
                else:
                    current = candidate
    if current:
        pieces.append(current)
    return pieces


def _overlap_tail(blocks: list[Block], overlap: int, count: TokenCounter) -> list[Block]:
    """Trailing prose (never code) from the previous chunk, about `overlap` tokens long."""
    for block in reversed(blocks):
        if block.is_code or block.is_heading:
            return []
        words = block.text.split()
        tail: list[str] = []
        for word in reversed(words):
            tail.insert(0, word)
            if count(" ".join(tail)) >= overlap:
                break
        return [Block(" ".join(tail), block.loc)]
    return []


def _loc_range(blocks: list[Block]) -> str:
    locs = list(dict.fromkeys(b.loc for b in blocks if b.loc))
    if not locs:
        return ""
    if len(locs) == 1:
        return locs[0]
    first, last = locs[0], locs[-1]
    prefix = re.match(r"^\D*", first).group(0) if first else ""
    return f"{first}-{last[len(prefix) :]}" if last.startswith(prefix) else f"{first}-{last}"


def chunk_document(doc: Document, count: TokenCounter, target: int = 450, overlap: int = 60) -> list[Chunk]:
    blocks = to_blocks(doc)
    section = ""
    raw_chunks: list[tuple[str, list[Block]]] = []
    builder = _Builder()

    def prefix_for(sec: str) -> str:
        return f"## {sec}\n\n" if sec else ""

    def emit(next_section: str) -> None:
        nonlocal builder
        if builder.fresh:
            raw_chunks.append((section, builder.blocks))
            tail = _overlap_tail(builder.blocks, overlap, count) if next_section == section else []
            builder = _Builder(list(tail), sum(count(b.text) for b in tail))
        else:
            builder = _Builder()

    for block in blocks:
        if block.is_heading:
            emit(block.text)
            section = block.text
            continue
        budget = target - count(prefix_for(section))
        size = count(block.text)
        if block.is_code and size > budget:
            emit(section)
            builder = _Builder()
            builder.add(block, size)
            emit("")  # oversized code is its own chunk, with no overlap carried forward
            continue
        if not block.is_code and size > budget:
            for piece in _split_long_text(block.text, budget - overlap, count):
                piece_size = count(piece)
                if builder.tokens + piece_size > budget:
                    emit(section)
                builder.add(Block(piece, block.loc), piece_size)
            continue
        if builder.tokens + size > budget:
            emit(section)
            if builder.tokens + size > budget:
                builder = _Builder()
        builder.add(block, size)
    emit("")

    chunks: list[Chunk] = []
    for i, (sec, chunk_blocks) in enumerate(raw_chunks):
        body = "\n\n".join(b.text for b in chunk_blocks)
        text = prefix_for(sec) + body
        chunks.append(
            Chunk(
                chunk_id=f"{doc.meta.doc_id}#{i:04d}",
                doc_id=doc.meta.doc_id,
                source_name=doc.meta.source_name,
                title=doc.meta.title,
                section=sec,
                page_or_slide=_loc_range(chunk_blocks),
                url=doc.meta.url_or_path,
                trust_tier=doc.meta.trust_tier,
                has_code=any(b.is_code for b in chunk_blocks),
                topic_tags=list(doc.meta.topic_tags),
                text=text,
                n_tokens=count(text),
            )
        )
    return chunks
