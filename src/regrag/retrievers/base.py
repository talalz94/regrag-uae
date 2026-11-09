"""The interface every retriever implements."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

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
        """Return up to k hits, best first, with no duplicate pids."""
        ...
