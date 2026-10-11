from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path

from oslm.config import load_config
from oslm.index.chunk import Chunk, chunk_document
from oslm.index.clean import clean_document
from oslm.ingest.manifest import Manifest

log = logging.getLogger(__name__)


def build_chunks(raw_dir: Path, count, target: int, overlap: int) -> list[Chunk]:  # noqa: ANN001
    manifest = Manifest(raw_dir)
    chunks: list[Chunk] = []
    for doc in sorted(manifest.documents(), key=lambda d: d.meta.doc_id):
        chunks.extend(chunk_document(clean_document(doc), count, target=target, overlap=overlap))
    return chunks


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def read_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]


def fingerprint(chunks: list[Chunk], embedder: str) -> str:
    h = hashlib.sha256(embedder.encode("utf-8"))
    for c in chunks:
        h.update(c.chunk_id.encode("utf-8"))
        h.update(c.text.encode("utf-8"))
    return h.hexdigest()


def get_collection(create: bool = False):  # noqa: ANN201
    import chromadb

    cfg = load_config()
    client = chromadb.PersistentClient(path=str(cfg.path("chroma")))
    name = cfg.retrieval["collection"]
    if create:
        try:
            client.delete_collection(name)
        except Exception:
            log.debug("no existing collection to delete")
        return client.create_collection(name, metadata={"hnsw:space": "cosine"})
    return client.get_collection(name)


def build_index(chunks: list[Chunk], force: bool = False) -> bool:
    """Embeds chunks into Chroma; returns False when the stored index already matches."""
    from oslm.index.embed import embed_passages

    cfg = load_config()
    fp_path = cfg.path("chroma") / "fingerprint.json"
    fp = fingerprint(chunks, cfg.models["embedder"])
    if not force and fp_path.exists() and json.loads(fp_path.read_text(encoding="utf-8")).get("fingerprint") == fp:
        log.info("index up to date (%d chunks)", len(chunks))
        return False

    collection = get_collection(create=True)
    batch = 128
    for start in range(0, len(chunks), batch):
        part = chunks[start : start + batch]
        vectors = embed_passages([c.text for c in part])
        collection.add(
            ids=[c.chunk_id for c in part],
            embeddings=vectors.tolist(),
            documents=[c.text for c in part],
            metadatas=[c.chroma_metadata() for c in part],
        )
    fp_path.write_text(json.dumps({"fingerprint": fp, "chunks": len(chunks)}), encoding="utf-8")
    log.info("indexed %d chunks", len(chunks))
    return True


def run_index(force: bool = False, smoke: bool = False) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    chunks_path = cfg.path("chunks") / "chunks.jsonl"
    if not smoke:
        from oslm.index.embed import count_tokens

        chunks = build_chunks(
            cfg.path("raw"), count_tokens, cfg.chunking["target_tokens"], cfg.chunking["overlap_tokens"]
        )
        if not chunks:
            print("no documents in data/raw — run `oslm ingest` first")
            return 1
        write_chunks(chunks, chunks_path)
        sizes = sorted(c.n_tokens for c in chunks)
        print(
            f"{len(chunks)} chunks; tokens p50={sizes[len(sizes) // 2]} max={sizes[-1]}; "
            f"code chunks={sum(c.has_code for c in chunks)}"
        )
        build_index(chunks, force=force)
    from oslm.index.smoke import run_smoke

    return run_smoke()
