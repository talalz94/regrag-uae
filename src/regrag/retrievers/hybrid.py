"""Hybrid retrieval with reciprocal rank fusion (Cormack et al., SIGIR 2009)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from regrag.data import Passage
from regrag.retrievers.base import Hit, Retriever


def reciprocal_rank_fusion(rankings: Sequence[Sequence[Hit]], *, k: int = 60) -> list[Hit]:
    """Fuse ranked lists by summing 1 / (k + rank). Ties go to the best single rank, then pid."""
    scores: dict[str, float] = defaultdict(float)
    best_rank: dict[str, int] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.pid] += 1.0 / (k + rank)
            best_rank[hit.pid] = min(best_rank.get(hit.pid, rank), rank)
    order = sorted(scores, key=lambda pid: (-scores[pid], best_rank[pid], pid))
    return [Hit(pid, scores[pid]) for pid in order]


class HybridRetriever:
    """Runs each retriever to `depth` and fuses the lists. RRF needs no score scaling."""

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        *,
        rrf_k: int = 60,
        depth: int = 100,
        name: str = "hybrid",
    ):
        if len(retrievers) < 2:
            raise ValueError("a hybrid needs at least two retrievers")
        self.name = name
        self.retrievers = list(retrievers)
        self.rrf_k = rrf_k
        self.depth = depth

    def index(self, passages: Sequence[Passage]) -> None:
        for retriever in self.retrievers:
            retriever.index(passages)

    def search(self, query: str, k: int = 10) -> list[Hit]:
        depth = max(self.depth, k)
        rankings = [r.search(query, depth) for r in self.retrievers]
        return reciprocal_rank_fusion(rankings, k=self.rrf_k)[:k]
