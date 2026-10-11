from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ManifestEntry:
    doc_id: str
    source_name: str
    url_or_path: str
    type: str
    license: str
    trust_tier: int
    title: str
    retrieved_at: str
    sha256: str
    topic_tags: list[str] = field(default_factory=list)


@dataclass
class Segment:
    """A page, slide or whole HTML document; `loc` is e.g. "p.3", "slide 4" or None."""

    text: str
    loc: str | None = None


@dataclass
class Document:
    meta: ManifestEntry
    segments: list[Segment]

    def to_json(self) -> dict[str, Any]:
        return {"meta": asdict(self.meta), "segments": [asdict(s) for s in self.segments]}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Document:
        return cls(ManifestEntry(**data["meta"]), [Segment(**s) for s in data["segments"]])

    @property
    def full_text(self) -> str:
        return "\n\n".join(s.text for s in self.segments)


def content_hash(segments: list[Segment]) -> str:
    h = hashlib.sha256()
    for s in segments:
        h.update(s.text.encode("utf-8"))
    return h.hexdigest()


def make_doc_id(source_name: str, title: str, key: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", f"{source_name} {title}".lower()).strip("-")[:48]
    return f"{slug}-{hashlib.sha1(key.encode('utf-8')).hexdigest()[:6]}"


class Manifest:
    """Append-only JSONL manifest plus one JSON file per document under `docs/`."""

    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = raw_dir
        self.path = raw_dir / "manifest.jsonl"
        self.docs_dir = raw_dir / "docs"
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.entries: dict[str, ManifestEntry] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = ManifestEntry(**json.loads(line))
                        self.entries[entry.doc_id] = entry

    def by_location(self, url_or_path: str) -> ManifestEntry | None:
        return next((e for e in self.entries.values() if e.url_or_path == url_or_path), None)

    def has_hash(self, sha256: str) -> bool:
        return any(e.sha256 == sha256 for e in self.entries.values())

    def doc_path(self, doc_id: str) -> Path:
        return self.docs_dir / f"{doc_id}.json"

    def add(self, doc: Document, replace: bool = False) -> bool:
        """Stores the document; returns False if identical content is already present (unless `replace`)."""
        existing = self.by_location(doc.meta.url_or_path)
        if existing and existing.sha256 == doc.meta.sha256 and not replace:
            return False
        if self.has_hash(doc.meta.sha256) and not existing:
            return False
        if existing:
            self.entries.pop(existing.doc_id)
            self.doc_path(existing.doc_id).unlink(missing_ok=True)
        self.entries[doc.meta.doc_id] = doc.meta
        self.doc_path(doc.meta.doc_id).write_text(json.dumps(doc.to_json(), ensure_ascii=False), encoding="utf-8")
        self._rewrite()
        return True

    def _rewrite(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for entry in self.entries.values():
                f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        tmp.replace(self.path)

    def load(self, doc_id: str) -> Document:
        return Document.from_json(json.loads(self.doc_path(doc_id).read_text(encoding="utf-8")))

    def documents(self) -> list[Document]:
        return [self.load(doc_id) for doc_id in self.entries if self.doc_path(doc_id).exists()]
