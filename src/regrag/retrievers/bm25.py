"""Lexical baseline: BM25 over stemmed tokens, using the bm25s library."""

from __future__ import annotations

from collections.abc import Sequence

import bm25s
import numpy as np

from regrag.data import Passage
from regrag.retrievers.base import Hit, pid_ranks, top_k
from regrag.text import tokenize


class BM25Retriever:
    def __init__(self, *, k1: float = 1.5, b: float = 0.75, stem: bool = True, name: str = "bm25"):
        self.name = name
        self.k1 = k1
        self.b = b
        self.stem = stem
        self._model: bm25s.BM25 | None = None
        self._pids: list[str] = []
        self._ranks = np.empty(0, dtype=np.int64)

    def index(self, passages: Sequence[Passage]) -> None:
        if not passages:
            raise ValueError("cannot index an empty corpus")
        self._pids = [p.pid for p in passages]
        self._ranks = pid_ranks(self._pids)
        tokens = [tokenize(p.text, stem=self.stem) for p in passages]
        self._model = bm25s.BM25(k1=self.k1, b=self.b, method="lucene")
        self._model.index(tokens, show_progress=False)

    def search(self, query: str, k: int = 10) -> list[Hit]:
        if self._model is None:
            raise RuntimeError("call index() before search()")
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        terms = tokenize(query, stem=self.stem)
        if not terms:
            return []
        scores = self._model.get_scores(terms)
        # A zero score means no query term matched, so it is not a real hit.
        matched = np.flatnonzero(scores > 0)
        top = matched[top_k(scores[matched], k, self._ranks[matched])]
        return [Hit(self._pids[int(i)], float(scores[i])) for i in top]
