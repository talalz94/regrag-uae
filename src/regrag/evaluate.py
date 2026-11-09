"""Retrieval eval harness: runs a retriever over a split and scores it against gold passages."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from regrag.data import Question
from regrag.metrics import average_precision_at_k, ndcg_at_k, recall_at_k, reciprocal_rank_at_k
from regrag.retrievers.base import Hit, Retriever

RECALL_KS = (1, 5, 10, 20, 100)
CUTOFF = 10  # MAP, MRR and nDCG are reported at 10, like the published ObliQA baselines
DEPTH = 100


@dataclass
class EvalResult:
    system: str
    n_questions: int
    metrics: dict[str, float]
    slices: dict[str, dict[str, float]]
    latency_ms: dict[str, float]
    rankings: dict[str, list[Hit]] = field(repr=False)


def score_ranking(ranked: Sequence[str], gold: Sequence[str]) -> dict[str, float]:
    if len(set(ranked)) != len(ranked):
        raise ValueError("ranking contains duplicate passage ids")
    scores = {f"recall@{k}": recall_at_k(ranked, gold, k) for k in RECALL_KS}
    scores[f"map@{CUTOFF}"] = average_precision_at_k(ranked, gold, CUTOFF)
    scores[f"mrr@{CUTOFF}"] = reciprocal_rank_at_k(ranked, gold, CUTOFF)
    scores[f"ndcg@{CUTOFF}"] = ndcg_at_k(ranked, gold, CUTOFF)
    return scores


def _slice_name(n_gold: int) -> str:
    return str(n_gold) if n_gold < 3 else "3+"


def evaluate(
    retriever: Retriever,
    questions: Sequence[Question],
    *,
    depth: int = DEPTH,
    warmup: int = 3,
    on_progress: Callable[[int, int], None] | None = None,
) -> EvalResult:
    """Search every question one at a time, timing each search, and average the metrics.

    The first `warmup` searches run once before timing starts, so one time setup
    (model graph warmup, caches) does not land in the latency numbers.
    """
    if not questions:
        raise ValueError("no questions to evaluate")
    for q in questions[:warmup]:
        retriever.search(q.text, depth)

    per_query: list[dict[str, float]] = []
    slice_rows: dict[str, list[dict[str, float]]] = {}
    latencies: list[float] = []
    rankings: dict[str, list[Hit]] = {}
    for i, q in enumerate(questions, start=1):
        start = time.perf_counter()
        hits = retriever.search(q.text, depth)
        latencies.append((time.perf_counter() - start) * 1000.0)
        scores = score_ranking([h.pid for h in hits], q.gold)
        per_query.append(scores)
        slice_rows.setdefault(_slice_name(len(q.gold)), []).append(scores)
        rankings[q.qid] = hits
        if on_progress and (i % 250 == 0 or i == len(questions)):
            on_progress(i, len(questions))

    metric_names = list(per_query[0])
    metrics = {m: float(np.mean([row[m] for row in per_query])) for m in metric_names}
    slices = {
        name: {
            "n": float(len(rows)),
            **{m: float(np.mean([r[m] for r in rows])) for m in ("recall@10", "ndcg@10")},
        }
        for name, rows in sorted(slice_rows.items())
    }
    lat = np.asarray(latencies)
    latency = {
        "p50": float(np.percentile(lat, 50)),
        "p95": float(np.percentile(lat, 95)),
        "mean": float(lat.mean()),
    }
    return EvalResult(
        system=retriever.name,
        n_questions=len(questions),
        metrics=metrics,
        slices=slices,
        latency_ms=latency,
        rankings=rankings,
    )


def write_trec_run(path: Path, result: EvalResult) -> None:
    """Write the rankings in TREC run format: qid Q0 pid rank score system."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as out:
        for qid, hits in result.rankings.items():
            for rank, hit in enumerate(hits, start=1):
                out.write(f"{qid} Q0 {hit.pid} {rank} {hit.score:.6f} {result.system}\n")
