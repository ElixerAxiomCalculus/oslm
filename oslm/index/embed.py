from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from oslm.config import load_config

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

T = TypeVar("T")


def load_cached_first(factory: Callable[[bool], T]) -> T:
    """Loads from the local HF cache without network checks; falls back to downloading on a cache miss."""
    try:
        return factory(True)
    except OSError:
        return factory(False)


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    device = _device()
    name = load_config().models["embedder"]
    model = load_cached_first(lambda local: SentenceTransformer(name, device=device, local_files_only=local))
    if device == "cuda":
        model.half()
    return model


@lru_cache(maxsize=1)
def get_tokenizer():  # noqa: ANN201 - transformers tokenizer type is not worth importing eagerly
    from transformers import AutoTokenizer

    name = load_config().models["hf_base"]
    return load_cached_first(lambda local: AutoTokenizer.from_pretrained(name, local_files_only=local))


def count_tokens(text: str) -> int:
    return len(get_tokenizer().encode(text, add_special_tokens=False))


def embed_passages(texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
    # BGE passages are embedded without the query instruction.
    return get_embedder().encode(
        list(texts), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=len(texts) > 64
    )


def embed_queries(queries: Sequence[str]) -> np.ndarray:
    instruction = load_config().models["bge_query_instruction"]
    return get_embedder().encode([instruction + q for q in queries], normalize_embeddings=True, show_progress_bar=False)
