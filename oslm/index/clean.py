from __future__ import annotations

import re
import unicodedata
from collections import Counter

from oslm.ingest.manifest import Document, Segment

FENCE = "```"

_PAGE_NUMBER = re.compile(r"^\s*(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?\s*$", re.I)
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\s*)([a-z])")
_CODE_HINT = re.compile(
    r"(;\s*$|[{}]\s*$|^\s*#\s*(include|define)\b|\bpthread_\w+\s*\(|\bsem_\w+\s*\(|"
    r"^\s*(int|void|char|long|unsigned|static|struct|typedef|return|while|for|if|else)\b.*[;{)(]|"
    r"^\s*\}\s*(else|while)?|^\s*//|^\s*/\*|\*/\s*$|\+\+;|--;|==|!=|->)"
)
_FIGURE_LINE_NUMBER = re.compile(r"^\s*\d{1,3}\s{1,4}(?=\S)")


def _looks_like_code(line: str) -> bool:
    stripped = _FIGURE_LINE_NUMBER.sub("", line, count=1)
    return bool(stripped.strip()) and bool(_CODE_HINT.search(stripped))


def wrap_detected_code(text: str, min_lines: int = 3) -> str:
    """Wraps runs of C-looking lines that are not already fenced in ``` fences."""
    out: list[str] = []
    run: list[str] = []
    in_fence = False

    def flush() -> None:
        trailing: list[str] = []
        while run and not run[-1].strip():
            trailing.insert(0, run.pop())
        code_lines = sum(_looks_like_code(ln) for ln in run)
        if len(run) >= min_lines and code_lines / len(run) >= 0.5:
            out.extend([FENCE, *run, FENCE])
        else:
            out.extend(run)
        out.extend(trailing)
        run.clear()

    for line in text.splitlines():
        if line.strip().startswith(FENCE):
            flush()
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        brace_only = line.strip() in ("{", "}", "};")
        if _looks_like_code(line) or brace_only or (run and not line.strip() and len(run) < 200):
            run.append(line)
        else:
            flush()
            out.append(line)
    flush()
    return "\n".join(out)


def _split_fences(text: str) -> list[tuple[bool, str]]:
    parts: list[tuple[bool, str]] = []
    pattern = re.compile(r"```[^\n]*\n.*?\n```", re.S)
    pos = 0
    for m in pattern.finditer(text):
        if m.start() > pos:
            parts.append((False, text[pos : m.start()]))
        parts.append((True, m.group(0)))
        pos = m.end()
    if pos < len(text):
        parts.append((False, text[pos:]))
    return parts


def normalise_prose(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "").replace("​", "")
    text = _HYPHEN_BREAK.sub(r"\1\3", text)
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    # PDF extraction breaks lines mid-sentence; rejoin when the next line clearly continues it.
    text = re.sub(r"(?<=[^.!?:;\n])\n(?=[a-z(])", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalise_text(text: str) -> str:
    """Normalises prose while leaving fenced code byte-for-byte intact (minus trailing spaces)."""
    pieces = []
    for is_code, part in _split_fences(text):
        if is_code:
            pieces.append("\n".join(ln.rstrip() for ln in part.splitlines()))
        else:
            pieces.append(normalise_prose(part))
    return "\n\n".join(p for p in pieces if p.strip())


def repeated_lines(segments: list[Segment], min_pages: int = 4, frac: float = 0.3) -> set[str]:
    """Lines (headers/footers) that recur on a large fraction of pages."""
    if len(segments) < min_pages:
        return set()
    counts: Counter[str] = Counter()
    for seg in segments:
        lines = {ln.strip() for ln in seg.text.splitlines() if 0 < len(ln.strip()) < 100}
        counts.update(lines)
    threshold = max(3, int(len(segments) * frac))
    return {ln for ln, c in counts.items() if c >= threshold and not _looks_like_code(ln) and ln not in ("{", "}")}


def clean_segment(text: str, boilerplate: set[str], detect_code: bool) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip() not in boilerplate and not _PAGE_NUMBER.match(ln)]
    text = "\n".join(lines)
    if detect_code:
        text = wrap_detected_code(text)
    return normalise_text(text)


def clean_document(doc: Document) -> Document:
    boilerplate = repeated_lines(doc.segments)
    detect_code = doc.meta.type in ("pdf", "pptx", "txt")
    segments = [
        Segment(text=cleaned, loc=s.loc)
        for s in doc.segments
        if (cleaned := clean_segment(s.text, boilerplate, detect_code))
    ]
    return Document(doc.meta, segments)
