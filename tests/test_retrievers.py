import dataclasses

import numpy as np
import pytest
from conftest import HashingEncoder

from regrag.data import Passage
from regrag.retrievers import (
    MODELS,
    BM25Retriever,
    DenseRetriever,
    Hit,
    HybridRetriever,
    reciprocal_rank_fusion,
)
from regrag.retrievers.base import pid_ranks, top_k, trec_order


@pytest.fixture
def bm25(corpus):
    retriever = BM25Retriever()
    retriever.index(corpus)
    return retriever


def _twins(text: str, *pids: str) -> list[Passage]:
    """Passages with one shared text, so every retriever gives them the same score."""
    passages = [Passage(pid=pid, doc_id=1, ref=f"1.{i}", text=text) for i, pid in enumerate(pids)]
    return [*passages, Passage(pid="other", doc_id=2, ref="9.9", text="Fees are due in January.")]


def test_bm25_ranks_the_gold_passage_first(bm25, test_questions):
    for q in test_questions:
        assert bm25.search(q.text, 3)[0].pid in q.gold


def test_bm25_returns_nothing_when_no_term_matches(bm25):
    assert bm25.search("the of and") == []
    assert bm25.search("zebra") == []


def test_bm25_handles_k_larger_than_corpus(bm25, corpus):
    hits = bm25.search("firm capital licence fee custodian", 1000)
    assert 0 < len(hits) <= len(corpus)
    assert len({h.pid for h in hits}) == len(hits)


def test_search_before_index_raises():
    with pytest.raises(RuntimeError):
        BM25Retriever().search("capital")


def test_k_below_one_is_rejected(bm25, corpus):
    dense = DenseRetriever(HashingEncoder())
    dense.index(corpus)
    for retriever in (bm25, dense, HybridRetriever([bm25, dense])):
        with pytest.raises(ValueError, match="at least 1"):
            retriever.search("custodian", 0)


def test_trec_order_sorts_by_score_then_by_the_larger_pid():
    hits = [Hit("a", 1.0), Hit("c", 0.5), Hit("b", 1.0)]
    assert [h.pid for h in trec_order(hits)] == ["b", "a", "c"]


def test_top_k_keeps_the_larger_pid_when_a_tie_crosses_the_cut():
    pids = ["a", "c", "b", "d"]
    scores = np.array([1.0, 2.0, 2.0, 2.0], dtype=np.float32)
    ranks = pid_ranks(pids)
    assert [pids[i] for i in top_k(scores, 2, ranks)] == ["d", "c"]
    assert [pids[i] for i in top_k(scores, 10, ranks)] == ["d", "c", "b", "a"]
    with pytest.raises(ValueError):
        top_k(scores, 0, ranks)


@pytest.mark.parametrize("pids", [("p-1", "p-2"), ("p-2", "p-1")])
def test_bm25_puts_the_larger_pid_first_among_identical_passages(pids):
    bm25 = BM25Retriever()
    bm25.index(_twins("Custodians must reconcile client holdings daily.", *pids))
    hits = bm25.search("custodian reconcile holdings", 3)
    assert [h.pid for h in hits] == ["p-2", "p-1"]
    assert hits[0].score == hits[1].score
    assert [h.pid for h in bm25.search("custodian reconcile holdings", 1)] == ["p-2"]


@pytest.mark.parametrize("pids", [("p-1", "p-2"), ("p-2", "p-1")])
def test_dense_puts_the_larger_pid_first_among_identical_passages(pids):
    # One word texts give one hot vectors, so the tied scores are exact on any machine.
    dense = DenseRetriever(HashingEncoder())
    dense.index(_twins("Custodians", *pids))
    hits = dense.search("custodian", 2)
    assert [h.pid for h in hits] == ["p-2", "p-1"]
    assert hits[0].score == hits[1].score
    assert [h.pid for h in dense.search("custodian", 1)] == ["p-2"]


def test_dense_ranks_the_gold_passage_first(corpus, test_questions):
    dense = DenseRetriever(HashingEncoder())
    dense.index(corpus)
    for q in test_questions:
        assert dense.search(q.text, 3)[0].pid in q.gold


def test_dense_embeddings_are_cached_per_corpus(corpus, tmp_path):
    encoder = HashingEncoder()
    DenseRetriever(encoder, cache_dir=tmp_path).index(corpus)
    again = DenseRetriever(encoder, cache_dir=tmp_path)
    again.index(corpus)
    assert again.loaded_from_cache
    assert encoder.passage_calls == 1
    changed = DenseRetriever(encoder, cache_dir=tmp_path)
    changed.index(corpus[:-1])
    assert not changed.loaded_from_cache
    assert encoder.passage_calls == 2


def test_passage_key_covers_every_setting_that_changes_a_passage_vector():
    spec = MODELS["bge-small"]
    assert dataclasses.replace(spec, max_seq_length=256).passage_key != spec.passage_key
    assert dataclasses.replace(spec, passage_prefix="passage: ").passage_key != spec.passage_key
    assert dataclasses.replace(spec, query_prefix="").passage_key == spec.passage_key


def test_dense_cache_is_not_reused_after_a_passage_setting_changes(corpus, tmp_path):
    encoder = HashingEncoder()
    encoder.cache_key = "test/hashing@0|max_seq_length=512"
    DenseRetriever(encoder, cache_dir=tmp_path).index(corpus)
    encoder.cache_key = "test/hashing@0|max_seq_length=256"
    changed = DenseRetriever(encoder, cache_dir=tmp_path)
    changed.index(corpus)
    assert not changed.loaded_from_cache
    assert encoder.passage_calls == 2


def test_rrf_sums_reciprocal_ranks_and_breaks_ties_like_trec_eval():
    first = [Hit("a", 9.0), Hit("b", 8.0), Hit("c", 7.0)]
    second = [Hit("b", 0.9), Hit("a", 0.8), Hit("d", 0.7)]
    fused = reciprocal_rank_fusion([first, second], k=60)
    # a and b tie, and so do c and d: the larger pid goes first, as in trec_eval
    assert [h.pid for h in fused] == ["b", "a", "d", "c"]
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_gives_equal_sums_the_same_score():
    # a at ranks 24 and 30, b at ranks 3 and 80: 1/84 + 1/90 and 1/63 + 1/140 are both
    # 29/1260, but adding the floats one at a time leaves a one rounding step above b
    first = [Hit(f"x{i}", 1.0) for i in range(100)]
    second = [Hit(f"y{i}", 1.0) for i in range(100)]
    first[23], first[2] = Hit("a", 1.0), Hit("b", 1.0)
    second[29], second[79] = Hit("a", 1.0), Hit("b", 1.0)
    fused = reciprocal_rank_fusion([first, second], k=60)
    score = {h.pid: h.score for h in fused}
    assert score["a"] == score["b"] == 29 / 1260
    order = [h.pid for h in fused]
    assert order.index("b") < order.index("a")  # a real tie, so the larger pid goes first


def test_rrf_prefers_passages_found_by_both():
    fused = reciprocal_rank_fusion([[Hit("solo", 1.0), Hit("both", 0.5)], [Hit("both", 1.0)]])
    assert fused[0].pid == "both"


def test_hybrid_needs_two_retrievers():
    with pytest.raises(ValueError):
        HybridRetriever([BM25Retriever()])


def test_hybrid_finds_every_gold_passage(corpus, test_questions):
    hybrid = HybridRetriever([BM25Retriever(), DenseRetriever(HashingEncoder())])
    hybrid.index(corpus)
    for q in test_questions:
        top = {h.pid for h in hybrid.search(q.text, 3)}
        assert set(q.gold) <= top
