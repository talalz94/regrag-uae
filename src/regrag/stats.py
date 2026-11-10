"""Dataset statistics behind the numbers in docs/DATA_CARD.md. Run `regrag stats`."""

from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from regrag.data import (
    SPLITS,
    _gold_index,
    _read_json,
    _resolve_gold,
    load_corpus,
    load_questions,
    obliqa_dir,
)

RULE_NUMBER = re.compile(r"\b\d+(?:\.\d+)+\b")


def _words(text: str) -> int:
    return len(text.split())


def dataset_stats(data_dir: Path | None = None) -> dict:
    full = load_corpus(data_dir, drop_empty=False)
    corpus = [p for p in full if p.text.strip()]
    by_pid = {p.pid: p for p in corpus}
    index = _gold_index(corpus)
    shared = {key: n for key, n in Counter((p.doc_id, p.ref) for p in corpus).items() if n > 1}
    # Passages with exactly the same text get the same BM25 score, so only the tie order
    # can rank a gold passage above its twin.
    by_text: dict[str, list[str]] = defaultdict(list)
    for p in corpus:
        by_text[p.text].append(p.pid)
    twins = [pids for pids in by_text.values() if len(pids) > 1]
    words = [_words(p.text) for p in corpus]
    stats: dict = {
        "documents": len({p.doc_id for p in full}),
        "passages_in_documents": len(full),
        "empty_passages_dropped": len(full) - len(corpus),
        "passages_used": len(corpus),
        "references_shared_by_several_passages": len(shared),
        "passages_with_a_shared_reference": sum(shared.values()),
        "texts_shared_by_several_passages": len(twins),
        "passages_with_a_shared_text": sum(len(pids) for pids in twins),
        "passage_words": {
            "median": float(np.median(words)),
            "p95": float(np.percentile(words, 95, method="nearest")),
            "max": max(words),
            "total": sum(words),  # words are counted by splitting on whitespace
        },
        "splits": {},
    }
    for split in SPLITS:
        path = obliqa_dir(data_dir) / f"ObliQA_{split}.json"
        if not path.exists():  # for example after `regrag download --skip-train`
            stats["splits"][split] = "not downloaded"
            continue
        rows = _read_json(path)
        questions = load_questions(split, corpus, data_dir)
        text_differs = sum(
            1
            for row in rows
            for ref in row["Passages"]
            if by_pid[_resolve_gold(index, ref)].text != ref["Passage"]
        )
        # Gold references whose passage has a twin with the same text that is not gold
        # for the same question.
        twin_refs = [
            q.qid
            for q in questions
            for g in q.gold
            if any(pid not in q.gold for pid in by_text[by_pid[g].text])
        ]
        stats["splits"][split] = {
            "rows": len(rows),
            "questions": len(questions),
            "gold_passages_per_question": {
                str(n): count
                for n, count in sorted(Counter(len(q.gold) for q in questions).items())
            },
            "questions_with_gold_in_several_documents": sum(
                1 for q in questions if len({by_pid[g].doc_id for g in q.gold}) > 1
            ),
            "questions_citing_a_rule_number": sum(
                1 for q in questions if RULE_NUMBER.search(q.text)
            ),
            "median_question_words": float(statistics.median(_words(q.text) for q in questions)),
            "gold_refs_whose_text_differs_from_corpus": text_differs,
            "gold_refs_with_a_twin_that_is_not_gold": len(twin_refs),
            "questions_with_such_a_gold_ref": len(set(twin_refs)),
        }
    return stats
