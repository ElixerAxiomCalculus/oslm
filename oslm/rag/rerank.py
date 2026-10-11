from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from oslm.config import load_config
from oslm.rag.retrieve import Retrieved

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    import torch
    from sentence_transformers import CrossEncoder

    from oslm.index.embed import load_cached_first

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = load_config().models["reranker"]
    model = load_cached_first(lambda local: CrossEncoder(name, device=device, max_length=512, local_files_only=local))
    if device == "cuda":
        model.model.half()
    return model


def rerank(query: str, candidates: list[Retrieved], k: int | None = None) -> list[Retrieved]:
    """Scores (query, passage) pairs with the cross-encoder; scores are sigmoid probabilities in [0, 1]."""
    if not candidates:
        return []
    k = k or load_config().retrieval["rerank_k"]
    scores = get_reranker().predict([(query, c.text) for c in candidates], batch_size=32, show_progress_bar=False)
    for cand, score in zip(candidates, scores, strict=True):
        cand.rerank_score = float(score)
    return sorted(candidates, key=lambda c: c.rerank_score or 0.0, reverse=True)[:k]
