"""Hybrid retrieval with reciprocal rank fusion (Cormack et al., SIGIR 2009)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from fractions import Fraction

from regrag.data import Passage
from regrag.retrievers.base import Hit, Retriever, trec_order


def reciprocal_rank_fusion(rankings: Sequence[Sequence[Hit]], *, k: int = 60) -> list[Hit]:
    """Fuse ranked lists by summing 1 / (k + rank). Ties go to the larger pid, as in trec_eval.

    Each sum is kept as an exact fraction and rounded to a float once, so equal sums get
    exactly the same score. Adding floats one at a time can leave two equal sums one
    rounding step apart (1/84 + 1/90 and 1/63 + 1/140 are both 29/1260), and the order
    would then follow rounding noise instead of the tie rule.
    """
    totals: dict[str, Fraction] = defaultdict(Fraction)
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            totals[hit.pid] += Fraction(1, k + rank)
    return trec_order(Hit(pid, float(total)) for pid, total in totals.items())


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
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        depth = max(self.depth, k)
        rankings = [r.search(query, depth) for r in self.retrievers]
        return reciprocal_rank_fusion(rankings, k=self.rrf_k)[:k]
