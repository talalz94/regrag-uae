"""Ranking metrics with binary relevance.

Definitions follow trec_eval: recall@k divides by the number of gold passages,
and AP@k (map_cut) also divides by the number of gold passages, not by k.
Rankings must not contain duplicate ids.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence


def _check(relevant: Collection[str], k: int) -> set[str]:
    if k < 1:
        raise ValueError("k must be at least 1")
    gold = set(relevant)
    if not gold:
        raise ValueError("a question needs at least one gold passage")
    return gold


def recall_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    gold = _check(relevant, k)
    return len(gold.intersection(ranked[:k])) / len(gold)


def average_precision_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    gold = _check(relevant, k)
    hits = 0
    total = 0.0
    for rank, pid in enumerate(ranked[:k], start=1):
        if pid in gold:
            hits += 1
            total += hits / rank
    return total / len(gold)


def reciprocal_rank_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    gold = _check(relevant, k)
    for rank, pid in enumerate(ranked[:k], start=1):
        if pid in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    gold = _check(relevant, k)
    dcg = sum(
        1.0 / math.log2(rank + 1) for rank, pid in enumerate(ranked[:k], start=1) if pid in gold
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1))
    return dcg / ideal
