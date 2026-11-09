"""Dense retrieval: a small embedding model and exact cosine search in numpy.

At about 13,000 passages an exact search is a single matrix product, so there is
no need for an approximate index or a vector database here.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from regrag.data import Passage
from regrag.retrievers.base import Hit


@dataclass(frozen=True)
class ModelSpec:
    name: str
    revision: str  # pinned Hugging Face commit, so the weights cannot change under us
    query_prefix: str = ""
    passage_prefix: str = ""
    max_seq_length: int = 512
    license: str = ""

    @property
    def key(self) -> str:
        return f"{self.name}@{self.revision[:12]}"


BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

MODELS: dict[str, ModelSpec] = {
    "bge-small": ModelSpec(
        name="BAAI/bge-small-en-v1.5",
        revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        query_prefix=BGE_QUERY_PREFIX,
        license="MIT",
    ),
    "bge-base": ModelSpec(
        name="BAAI/bge-base-en-v1.5",
        revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
        query_prefix=BGE_QUERY_PREFIX,
        license="MIT",
    ),
}


class Encoder(Protocol):
    key: str

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEncoder:
    """Wraps a sentence-transformers model. Embeddings are L2 normalised."""

    def __init__(
        self,
        spec: ModelSpec,
        *,
        device: str = "cpu",
        batch_size: int = 32,
        show_progress: bool = False,
    ):
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from sentence_transformers import SentenceTransformer  # heavy import, kept lazy

        self.spec = spec
        self.key = spec.key
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.model = SentenceTransformer(spec.name, revision=spec.revision, device=device)
        self.model.max_seq_length = spec.max_seq_length

    def _encode(self, texts: list[str], batch_size: int, progress: bool) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=progress,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([self.spec.query_prefix + t for t in texts], self.batch_size, False)

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(
            [self.spec.passage_prefix + t for t in texts], self.batch_size, self.show_progress
        )


def corpus_fingerprint(encoder_key: str, passages: Sequence[Passage]) -> str:
    digest = hashlib.sha256(encoder_key.encode())
    for p in passages:
        digest.update(b"\x00" + p.pid.encode() + b"\x00" + p.text.encode())
    return digest.hexdigest()[:16]


class DenseRetriever:
    def __init__(self, encoder: Encoder, *, cache_dir: Path | None = None, name: str = "dense"):
        self.name = name
        self.encoder = encoder
        self.cache_dir = cache_dir
        self.loaded_from_cache = False
        self._pids: list[str] = []
        self._matrix: np.ndarray | None = None

    def _cache_path(self, passages: Sequence[Passage]) -> Path | None:
        if self.cache_dir is None:
            return None
        slug = self.encoder.key.replace("/", "__").replace("@", "_")
        return self.cache_dir / f"{slug}_{corpus_fingerprint(self.encoder.key, passages)}.npy"

    def index(self, passages: Sequence[Passage]) -> None:
        if not passages:
            raise ValueError("cannot index an empty corpus")
        self._pids = [p.pid for p in passages]
        path = self._cache_path(passages)
        if path is not None and path.exists():
            self._matrix = np.load(path)
            self.loaded_from_cache = True
            return
        self._matrix = self.encoder.encode_passages([p.text for p in passages])
        self.loaded_from_cache = False
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, self._matrix)

    def search(self, query: str, k: int = 10) -> list[Hit]:
        if self._matrix is None:
            raise RuntimeError("call index() before search()")
        q = self.encoder.encode_queries([query])[0]
        scores = self._matrix @ q
        k = min(k, len(self._pids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [Hit(self._pids[int(i)], float(scores[i])) for i in top]
