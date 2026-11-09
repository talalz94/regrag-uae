import math

import pytest

from regrag.metrics import average_precision_at_k, ndcg_at_k, recall_at_k, reciprocal_rank_at_k

RANKED = ["a", "x", "b", "y"]
GOLD = {"a", "b"}


def test_recall_at_k():
    assert recall_at_k(RANKED, GOLD, 1) == 0.5
    assert recall_at_k(RANKED, GOLD, 3) == 1.0


def test_average_precision_at_k():
    assert average_precision_at_k(RANKED, GOLD, 10) == pytest.approx((1 / 1 + 2 / 3) / 2)


def test_average_precision_divides_by_all_gold_like_trec_eval():
    assert average_precision_at_k(["x", "a"], {"a", "b", "c"}, 10) == pytest.approx((1 / 2) / 3)


def test_reciprocal_rank():
    assert reciprocal_rank_at_k(["x", "y", "a"], GOLD, 10) == pytest.approx(1 / 3)
    assert reciprocal_rank_at_k(["x", "y", "a"], GOLD, 2) == 0.0


def test_ndcg():
    expected = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(RANKED, GOLD, 10) == pytest.approx(expected)
    assert ndcg_at_k(["b", "a", "x"], GOLD, 10) == pytest.approx(1.0)


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        recall_at_k(RANKED, set(), 10)
    with pytest.raises(ValueError):
        ndcg_at_k(RANKED, GOLD, 0)
