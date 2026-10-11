from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from oslm.index import build as build_mod
from oslm.index.build import build_chunks, fingerprint
from oslm.index.chunk import Chunk, chunk_document, heading_text, to_blocks
from oslm.index.clean import clean_document, normalise_text, repeated_lines, wrap_detected_code
from oslm.ingest.manifest import Document, Manifest, ManifestEntry, Segment, content_hash


def words(text: str) -> int:
    return len(text.split())


def make_doc(segments: list[Segment], doc_type: str = "pdf", doc_id: str = "doc-1") -> Document:
    meta = ManifestEntry(
        doc_id,
        "OSTEP",
        "https://x/threads-locks.pdf",
        doc_type,
        "free",
        1,
        "Locks",
        "now",
        content_hash(segments),
        ["locks"],
    )
    return Document(meta, segments)


C_PROGRAM = """#include <pthread.h>
static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
static int counter;
void *worker(void *arg) {
    pthread_mutex_lock(&m);
    counter = counter + 1;
    pthread_mutex_unlock(&m);
    return NULL;
}"""

PROSE = (
    "A lock provides mutual exclusion so that only one thread at a time can be inside the critical section. "
    "When a thread calls lock it either acquires the lock or waits until the holder releases it. "
) * 3


class TestClean:
    def test_removes_repeated_headers_and_page_numbers(self) -> None:
        segs = [Segment(f"OPERATING SYSTEMS [VERSION 1.10]\nBody text {i} about locks.\n{i + 1}") for i in range(6)]
        doc = clean_document(make_doc(segs))
        assert all("OPERATING SYSTEMS" not in s.text for s in doc.segments)
        assert all(not re.search(r"^\d+$", s.text, re.M) for s in doc.segments)
        assert "OPERATING SYSTEMS [VERSION 1.10]" in repeated_lines(segs)

    def test_dehyphenates_and_normalises_whitespace(self) -> None:
        assert normalise_text("synchroni-\nzation   is\n\n\n\nhard ﬁne") == "synchronization is\n\nhard fine"

    def test_code_whitespace_preserved(self) -> None:
        text = "Prose  here.\n\n```\nif (x)  {\n    y();\n}\n```"
        out = normalise_text(text)
        assert "if (x)  {\n    y();\n}" in out and "Prose here." in out

    def test_detects_unfenced_c_code(self) -> None:
        wrapped = wrap_detected_code("Figure 28.1 shows a lock.\n" + C_PROGRAM + "\nThe code above is simple.")
        assert wrapped.count("```") == 2
        inside = wrapped.split("```")[1]
        assert "pthread_mutex_lock(&m);" in inside and "Figure 28.1" not in inside

    def test_prose_not_wrapped(self) -> None:
        assert "```" not in wrap_detected_code(PROSE)


class TestChunking:
    def test_headings_detected(self) -> None:
        assert heading_text("# Locks") == "Locks"
        assert heading_text("28.2 Evaluating Locks") == "28.2 Evaluating Locks"
        assert heading_text("DESCRIPTION") == "Description"
        assert heading_text("This is a normal sentence that ends.") is None

    def test_no_chunk_splits_a_code_block(self) -> None:
        text = "\n\n".join([PROSE, "```c\n" + C_PROGRAM + "\n```", PROSE, "```c\n" + C_PROGRAM + "\n```", PROSE])
        chunks = chunk_document(make_doc([Segment(text, "p.1")], "html"), words, target=120, overlap=15)
        assert len(chunks) > 1
        for c in chunks:
            assert c.text.count("```") % 2 == 0
        assert sum(c.text.count("pthread_mutex_unlock(&m);") for c in chunks) == 2

    def test_oversized_code_block_is_its_own_chunk(self) -> None:
        big = "```c\n" + "\n".join(f"    x{i} = x{i} + 1;" for i in range(200)) + "\n```"
        chunks = chunk_document(make_doc([Segment(PROSE + "\n\n" + big + "\n\n" + PROSE)], "html"), words, 100, 10)
        code_chunks = [c for c in chunks if "x199" in c.text]
        assert len(code_chunks) == 1 and code_chunks[0].has_code
        assert code_chunks[0].text.count("```") == 2 and "x0 = x0" in code_chunks[0].text

    def test_token_lengths_within_bounds(self) -> None:
        text = "\n\n".join([PROSE] * 12 + ["One extremely long paragraph. " * 200])
        chunks = chunk_document(make_doc([Segment(text, "p.3")], "html"), words, target=120, overlap=20)
        assert all(c.n_tokens <= 120 for c in chunks)
        assert all(c.n_tokens >= 10 for c in chunks[:-1])

    def test_overlap_carries_tail(self) -> None:
        paras = [f"Paragraph {i} " + "word " * 50 + f"END{i}." for i in range(4)]
        chunks = chunk_document(make_doc([Segment("\n\n".join(paras))], "html"), words, target=70, overlap=10)
        assert any(f"END{i}." in chunks[i + 1].text for i in range(len(chunks) - 1))

    def test_section_prefix_and_metadata_complete(self) -> None:
        doc = make_doc([Segment("# Spin Locks\n\n" + PROSE, "p.4"), Segment(PROSE, "p.5")])
        chunks = chunk_document(doc, words, target=200, overlap=20)
        required = [
            "chunk_id",
            "doc_id",
            "source_name",
            "title",
            "section",
            "page_or_slide",
            "url",
            "trust_tier",
            "has_code",
            "topic_tags",
        ]
        for c in chunks:
            meta = c.chroma_metadata()
            assert all(k in meta for k in required)
            assert all(isinstance(v, (str, int, float, bool)) for v in meta.values())
            assert c.text.startswith("## Spin Locks")
            assert c.section == "Spin Locks"
        assert chunks[0].page_or_slide.startswith("p.4")

    def test_blocks_keep_page_location(self) -> None:
        blocks = to_blocks(make_doc([Segment("alpha", "p.1"), Segment("beta", "p.2")]))
        assert [(b.text, b.loc) for b in blocks] == [("alpha", "p.1"), ("beta", "p.2")]


class TestBuild:
    def test_reindex_is_noop(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        raw = tmp_path / "raw"
        m = Manifest(raw)
        segs = [Segment(PROSE + "\n\n```c\n" + C_PROGRAM + "\n```", "p.1")]
        m.add(make_doc(segs))
        chunks = build_chunks(raw, words, 120, 15)
        assert chunks and fingerprint(chunks, "e") == fingerprint(build_chunks(raw, words, 120, 15), "e")

        calls: list[int] = []

        class FakeCollection:
            def add(self, ids, embeddings, documents, metadatas):  # noqa: ANN001
                calls.append(len(ids))

        cfg = build_mod.load_config()
        monkeypatch.setitem(cfg["paths"], "chroma", str(tmp_path / "chroma"))
        (tmp_path / "chroma").mkdir()
        monkeypatch.setattr(build_mod, "get_collection", lambda create=False: FakeCollection())
        import oslm.index.embed as embed_mod

        monkeypatch.setattr(embed_mod, "embed_passages", lambda texts: np.zeros((len(texts), 4)))
        assert build_mod.build_index(chunks) is True
        assert build_mod.build_index(chunks) is False
        assert calls == [len(chunks)]
        assert json.loads((tmp_path / "chroma" / "fingerprint.json").read_text())["chunks"] == len(chunks)

    def test_chunk_roundtrip(self, tmp_path: Path) -> None:
        c = Chunk("d#0000", "d", "S", "T", "Sec", "p.1", "u", 1, True, ["a", "b"], "text", 1)
        build_mod.write_chunks([c], tmp_path / "c.jsonl")
        assert build_mod.read_chunks(tmp_path / "c.jsonl") == [c]
