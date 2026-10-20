from __future__ import annotations

import re

_CITATION = re.compile(r"\[(\d{1,2}(?:\s*[,;]\s*\d{1,2})*)\]")
_CITATION_WITH_SPACE = re.compile(r"([ \t]?)\[(\d{1,2}(?:\s*[,;]\s*\d{1,2})*)\]")
_CITATION_PLACEHOLDER = re.compile(r"[ \t]*\[(?:n|source|citation)\]", re.I)
_CODE_SPAN = re.compile(r"(```.*?```|`[^`\r\n]*`)", re.S)
# A suffix that could still grow into a citation ("[1, 2"), a code fence ("``"), or whitespace that a
# dropped citation would otherwise leave dangling before punctuation.
_UNSAFE_TAIL = re.compile(r"([ \t]*\[(?:[\d,; ]{0,12}|[A-Za-z]{0,12})|[ \t]+|`{1,2}|\r)$")


def clean_citations(text: str, n_sources: int) -> str:
    """Keeps citation numbers in 1..n_sources; drops the bracket (and the space before it) if none remain."""

    def repl(m: re.Match[str]) -> str:
        nums = [int(x) for x in re.split(r"\s*[,;]\s*", m.group(2))]
        valid = [n for n in dict.fromkeys(nums) if 1 <= n <= n_sources]
        return m.group(1) + "".join(f"[{n}]" for n in valid) if valid else ""

    return _CITATION_PLACEHOLDER.sub("", _CITATION_WITH_SPACE.sub(repl, text))


def clean_outside_code(text: str, n_sources: int) -> str:
    # Array indexing such as buf[3] or inline `flag[0]` must not be treated as a citation.
    parts = _CODE_SPAN.split(text)
    return "".join(p if i % 2 else clean_citations(p, n_sources) for i, p in enumerate(parts))


def _prose_parts(text: str) -> list[str]:
    return _CODE_SPAN.split(text)[::2]


def _unclosed_code_start(text: str) -> int | None:
    """Index of an unfinished fenced/inline code span, so streaming holds it intact."""
    i = 0
    while i < len(text):
        if text.startswith("```", i):
            end = text.find("```", i + 3)
            if end < 0:
                return i
            i = end + 3
        elif text[i] == "`":
            end = text.find("`", i + 1)
            if end < 0 or "\n" in text[i + 1 : end]:
                return i
            i = end + 1
        else:
            i += 1
    return None


def cited_numbers(text: str, n_sources: int) -> list[int]:
    found: list[int] = []
    for prose in _prose_parts(text):
        for m in _CITATION.finditer(prose):
            for x in re.split(r"\s*[,;]\s*", m.group(1)):
                n = int(x)
                if 1 <= n <= n_sources and n not in found:
                    found.append(n)
    return found


class CitationStream:
    """Filters citations in a token stream; never emits half of a `[n]` or of a code fence."""

    def __init__(self, n_sources: int) -> None:
        self.n_sources = n_sources
        self.raw = ""
        self._emitted = 0

    def feed(self, delta: str) -> str:
        # The fine-tuned GGUF sometimes emits "\r\n" inside code; canonical "\n" keeps fences, code detection and
        # the sandbox working. A trailing "\r" is held back until we know whether "\n" follows.
        self.raw = (self.raw + delta).replace("\r\n", "\n")
        m = _UNSAFE_TAIL.search(self.raw)
        safe_end = m.start() if m else len(self.raw)
        open_code = _unclosed_code_start(self.raw[:safe_end])
        if open_code is not None:
            safe_end = min(safe_end, open_code)
        safe = self.raw[:safe_end]
        return self._emit(clean_outside_code(safe, self.n_sources))

    def flush(self) -> str:
        return self._emit(clean_outside_code(self.raw, self.n_sources))

    def _emit(self, processed: str) -> str:
        out = processed[self._emitted :]
        self._emitted = len(processed)
        return out

    @property
    def text(self) -> str:
        return clean_outside_code(self.raw, self.n_sources)

    def counts(self) -> tuple[int, int]:
        """(valid, invalid) citation numbers the model wrote, outside code."""
        valid = invalid = 0
        for prose in _prose_parts(self.raw):
            for m in _CITATION.finditer(prose):
                for x in re.split(r"\s*[,;]\s*", m.group(1)):
                    if 1 <= int(x) <= self.n_sources:
                        valid += 1
                    else:
                        invalid += 1
        return valid, invalid
