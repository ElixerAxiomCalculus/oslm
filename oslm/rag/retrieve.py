from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from oslm.config import load_config


@dataclass
class Retrieved:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    dense_score: float
    rerank_score: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return self.metadata.get("doc_id", "")


@lru_cache(maxsize=1)
def _collection():  # noqa: ANN202
    from oslm.index.build import get_collection

    return get_collection()


def collection_size() -> int:
    try:
        return _collection().count()
    except Exception:
        return 0


_API = re.compile(r"\b(pthread_\w+|sem_\w+)\b")


def build_search_query(question: str, mode: str) -> str:
    """For pasted code, search with the prose plus the synchronization calls used, not the raw code."""
    if mode != "bug_find":
        return question
    from oslm.rag.router import strip_code

    prose = strip_code(question).strip()
    calls = sorted(set(_API.findall(question)))
    return " ".join(filter(None, [prose[:500], "synchronization bug", " ".join(calls)]))


def dense_search(query: str, k: int | None = None) -> list[Retrieved]:
    from oslm.index.embed import embed_queries

    k = k or load_config().retrieval["top_k"]
    vector = embed_queries([query])[0].tolist()
    res = _collection().query(query_embeddings=[vector], n_results=k, include=["documents", "metadatas", "distances"])
    out: list[Retrieved] = []
    for cid, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True
    ):
        out.append(Retrieved(chunk_id=cid, text=doc, metadata=dict(meta), dense_score=1.0 - float(dist)))
    return out


def get_chunk(chunk_id: str) -> Retrieved | None:
    res = _collection().get(ids=[chunk_id], include=["documents", "metadatas"])
    if not res["ids"]:
        return None
    return Retrieved(
        chunk_id=res["ids"][0], text=res["documents"][0], metadata=dict(res["metadatas"][0]), dense_score=0.0
    )
