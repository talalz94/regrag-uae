"""Lexical baseline: BM25 over stemmed tokens, using the bm25s library."""

from __future__ import annotations

from collections.abc import Sequence

import bm25s

from regrag.data import Passage
from regrag.retrievers.base import Hit
from regrag.text import tokenize


class BM25Retriever:
    def __init__(self, *, k1: float = 1.5, b: float = 0.75, stem: bool = True, name: str = "bm25"):
        self.name = name
        self.k1 = k1
        self.b = b
        self.stem = stem
        self._model: bm25s.BM25 | None = None
        self._pids: list[str] = []

    def index(self, passages: Sequence[Passage]) -> None:
        if not passages:
            raise ValueError("cannot index an empty corpus")
        self._pids = [p.pid for p in passages]
        tokens = [tokenize(p.text, stem=self.stem) for p in passages]
        self._model = bm25s.BM25(k1=self.k1, b=self.b, method="lucene")
        self._model.index(tokens, show_progress=False)

    def search(self, query: str, k: int = 10) -> list[Hit]:
        if self._model is None:
            raise RuntimeError("call index() before search()")
        terms = tokenize(query, stem=self.stem)
        if not terms:
            return []
        k = min(k, len(self._pids))
        ids, scores = self._model.retrieve([terms], k=k, show_progress=False)
        # A zero score means no query term matched, so it is not a real hit.
        return [
            Hit(self._pids[int(i)], float(s))
            for i, s in zip(ids[0], scores[0], strict=True)
            if s > 0
        ]
