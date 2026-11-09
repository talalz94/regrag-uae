from regrag.stats import dataset_stats


def test_stats_on_the_synthetic_rulebook(data_dir):
    stats = dataset_stats(data_dir)
    assert stats["documents"] == 2
    assert stats["passages_in_documents"] == 13
    assert stats["empty_passages_dropped"] == 1
    assert stats["passages_used"] == 12
    assert stats["references_shared_by_several_passages"] == 1
    assert stats["passages_with_a_shared_reference"] == 2
    dev, test, train = (stats["splits"][s] for s in ("dev", "test", "train"))
    assert dev["gold_passages_per_question"] == {"1": 3, "2": 1}
    assert test["questions_citing_a_rule_number"] == 1
    assert test["questions_with_gold_in_several_documents"] == 0
    assert (train["rows"], train["questions"]) == (3, 2)
    assert dev["gold_refs_whose_text_differs_from_corpus"] == 0
