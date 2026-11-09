"""Named retrieval systems, built once and shared, so the hybrid reuses the same indexes."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from regrag.data import Passage
from regrag.retrievers import (
    MODELS,
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    ModelSpec,
    Retriever,
    SentenceTransformerEncoder,
)
from regrag.retrievers.dense import Encoder


@dataclass(frozen=True)
class SystemSpec:
    name: str
    label: str
    description: str
    ablation: bool = False
    slow: bool = False  # left out of the default reproduce run on a laptop CPU


SPECS: dict[str, SystemSpec] = {
    s.name: s
    for s in [
        SystemSpec(
            "bm25",
            "BM25",
            "bm25s, Lucene scoring (k1=1.5, b=0.75), English stopwords, Snowball stemming",
        ),
        SystemSpec(
            "dense",
            "Dense, bge-small-en-v1.5",
            "BAAI/bge-small-en-v1.5 (33M parameters), exact cosine search over all passages",
        ),
        SystemSpec(
            "hybrid",
            "Hybrid, RRF of the two above",
            "Reciprocal rank fusion (k=60) of the top 100 from bm25 and dense",
        ),
        SystemSpec(
            "bm25-nostem",
            "BM25 without stemming",
            "Same as bm25 but without the stemmer",
            ablation=True,
        ),
        SystemSpec(
            "dense-base",
            "Dense, bge-base-en-v1.5",
            "BAAI/bge-base-en-v1.5 (110M parameters), exact cosine search",
            ablation=True,
            slow=True,
        ),
        SystemSpec(
            "hybrid-base",
            "Hybrid, RRF of BM25 and bge-base",
            "Reciprocal rank fusion (k=60) of the top 100 from bm25 and dense-base",
            ablation=True,
            slow=True,
        ),
    ]
}

MAIN_SYSTEMS = [name for name, spec in SPECS.items() if not spec.ablation]
ABLATIONS = [name for name, spec in SPECS.items() if spec.ablation]
DEFAULT_REPRODUCE = [name for name, spec in SPECS.items() if not spec.slow]

_DENSE_MODEL = {"dense": "bge-small", "dense-base": "bge-base"}
_HYBRID_PARTS = {"hybrid": ("bm25", "dense"), "hybrid-base": ("bm25", "dense-base")}

EncoderFactory = Callable[[ModelSpec], Encoder]


def default_encoder_factory(spec: ModelSpec) -> Encoder:
    return SentenceTransformerEncoder(spec, show_progress=True)


class Systems:
    """Builds and caches retrievers by name for one corpus."""

    def __init__(
        self,
        passages: Sequence[Passage],
        *,
        cache_dir: Path | None = None,
        encoder_factory: EncoderFactory = default_encoder_factory,
    ):
        self.passages = passages
        self.cache_dir = cache_dir
        self.encoder_factory = encoder_factory
        self.index_seconds: dict[str, float] = {}
        self.index_from_cache: dict[str, bool] = {}
        self._encoders: dict[str, Encoder] = {}
        self._built: dict[str, Retriever] = {}

    def _encoder(self, model: str) -> Encoder:
        if model not in self._encoders:
            self._encoders[model] = self.encoder_factory(MODELS[model])
        return self._encoders[model]

    def model_spec(self, name: str) -> ModelSpec | None:
        model = _DENSE_MODEL.get(name)
        return MODELS[model] if model else None

    def get(self, name: str) -> Retriever:
        if name not in SPECS:
            raise KeyError(f"unknown system {name!r}, expected one of {list(SPECS)}")
        if name in self._built:
            return self._built[name]
        if name in _HYBRID_PARTS:
            parts = [self.get(part) for part in _HYBRID_PARTS[name]]
            retriever: Retriever = HybridRetriever(parts, name=name)
            self.index_seconds[name] = sum(self.index_seconds[p] for p in _HYBRID_PARTS[name])
            self.index_from_cache[name] = any(self.index_from_cache[p] for p in _HYBRID_PARTS[name])
        else:
            if name in _DENSE_MODEL:
                retriever = DenseRetriever(
                    self._encoder(_DENSE_MODEL[name]), cache_dir=self.cache_dir, name=name
                )
            else:
                retriever = BM25Retriever(stem=(name == "bm25"), name=name)
            start = time.perf_counter()
            retriever.index(self.passages)
            self.index_seconds[name] = time.perf_counter() - start
            self.index_from_cache[name] = bool(getattr(retriever, "loaded_from_cache", False))
        self._built[name] = retriever
        return retriever
