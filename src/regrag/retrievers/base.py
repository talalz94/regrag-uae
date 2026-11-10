"""The interface every retriever implements, and the ranking order they all share."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from regrag.data import Passage


@dataclass(frozen=True, slots=True)
class Hit:
    pid: str
    score: float


@runtime_checkable
class Retriever(Protocol):
    name: str

    def index(self, passages: Sequence[Passage]) -> None:
        """Build the index. Must be called before `search`."""
        ...

    def search(self, query: str, k: int = 10) -> list[Hit]:
        """Return up to k hits in `trec_order`, with no duplicate pids."""
        ...


def trec_order(hits: Iterable[Hit]) -> list[Hit]:
    """Sort hits the way trec_eval does: by score, then by pid, both descending.

    trec_eval ignores the rank column of a run file and breaks score ties by the
    larger document id, so the metrics here and the saved runs use the same order.
    """
    return sorted(hits, key=lambda h: (h.score, h.pid), reverse=True)


def pid_ranks(pids: Sequence[str]) -> np.ndarray:
    """Position of each pid when all pids are sorted in descending order (the largest gets 0)."""
    order = np.argsort(np.array(pids))[::-1]
    ranks = np.empty(len(pids), dtype=np.int64)
    ranks[order] = np.arange(len(pids))
    return ranks


def top_k(scores: np.ndarray, k: int, ranks: np.ndarray) -> np.ndarray:
    """Indices of the k best scores, in `trec_order`.

    `ranks` comes from `pid_ranks`, so a tie at the cut goes to the larger pid and the
    result never depends on how numpy happens to order equal values.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if len(scores) == 0:
        return np.empty(0, dtype=np.int64)
    k = min(k, len(scores))
    kth = np.partition(scores, -k)[-k]
    candidates = np.flatnonzero(scores >= kth)
    order = np.lexsort((ranks[candidates], -scores[candidates]))
    return candidates[order[:k]]
