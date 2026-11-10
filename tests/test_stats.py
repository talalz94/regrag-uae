import json
import shutil

from regrag.stats import dataset_stats


def test_stats_on_the_synthetic_rulebook(data_dir):
    stats = dataset_stats(data_dir)
    assert stats["documents"] == 2
    assert stats["passages_in_documents"] == 13
    assert stats["empty_passages_dropped"] == 1
    assert stats["passages_used"] == 12
    assert stats["references_shared_by_several_passages"] == 1
    assert stats["passages_with_a_shared_reference"] == 2
    assert stats["texts_shared_by_several_passages"] == 0
    assert stats["passage_words"]["total"] == 175
    dev, test, train = (stats["splits"][s] for s in ("dev", "test", "train"))
    assert dev["gold_passages_per_question"] == {"1": 3, "2": 1}
    assert test["questions_citing_a_rule_number"] == 1
    assert test["questions_with_gold_in_several_documents"] == 0
    assert (train["rows"], train["questions"]) == (3, 2)
    assert dev["gold_refs_whose_text_differs_from_corpus"] == 0
    assert test["gold_refs_with_a_twin_that_is_not_gold"] == 0


def test_stats_still_run_without_the_train_split(data_dir, tmp_path):
    copy = tmp_path / "mini"
    shutil.copytree(data_dir, copy)
    (copy / "obliqa" / "ObliQA_train.json").unlink()  # as after `regrag download --skip-train`
    stats = dataset_stats(copy)
    assert stats["splits"]["train"] == "not downloaded"
    assert stats["splits"]["dev"]["questions"] == 4


def test_stats_count_gold_passages_with_a_twin_that_is_not_gold(data_dir, tmp_path):
    copy = tmp_path / "mini"
    shutil.copytree(data_dir, copy)
    doc = copy / "obliqa" / "StructuredRegulatoryDocuments" / "2.json"
    rows = json.loads(doc.read_text(encoding="utf-8"))
    # p-2-4 is the gold passage of test question q-t1; give it a twin under another reference
    twin = dict(next(r for r in rows if r["ID"] == "p-2-4"), ID="p-2-6", PassageID="4.3")
    doc.write_text(json.dumps([*rows, twin]), encoding="utf-8")
    stats = dataset_stats(copy)
    assert stats["texts_shared_by_several_passages"] == 1
    assert stats["passages_with_a_shared_text"] == 2
    test = stats["splits"]["test"]
    assert test["gold_refs_with_a_twin_that_is_not_gold"] == 1
    assert test["questions_with_such_a_gold_ref"] == 1
    assert stats["splits"]["dev"]["gold_refs_with_a_twin_that_is_not_gold"] == 0
