import pytest
from conftest import HashingEncoder

from regrag.retrievers import (
    BM25Retriever,
    DenseRetriever,
    Hit,
    HybridRetriever,
    reciprocal_rank_fusion,
)


@pytest.fixture
def bm25(corpus):
    retriever = BM25Retriever()
    retriever.index(corpus)
    return retriever


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


def test_rrf_sums_reciprocal_ranks_and_breaks_ties_deterministically():
    first = [Hit("a", 9.0), Hit("b", 8.0), Hit("c", 7.0)]
    second = [Hit("b", 0.9), Hit("a", 0.8), Hit("d", 0.7)]
    fused = reciprocal_rank_fusion([first, second], k=60)
    assert [h.pid for h in fused] == ["a", "b", "c", "d"]
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)


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
