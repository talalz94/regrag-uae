import json

import pytest
from conftest import HashingEncoder

from regrag.data import Question
from regrag.evaluate import evaluate, score_ranking, write_trec_run
from regrag.report import (
    END,
    START,
    ReadmeMarkersError,
    load_results,
    render,
    save_result,
    update_readme,
)
from regrag.retrievers import MODELS, BM25Retriever, Hit
from regrag.systems import DEFAULT_REPRODUCE, Systems, needs_torch

SYNTHETIC = {"repo": "synthetic test fixture, not ObliQA", "commit": None}


@pytest.fixture
def bm25_result(corpus, test_questions):
    retriever = BM25Retriever()
    retriever.index(corpus)
    return evaluate(retriever, test_questions)


def _save(result, results_dir, split="test", **extra):
    defaults = dict(
        n_passages=12, index_seconds=0.01, index_from_cache=False, model=None, dataset=SYNTHETIC
    )
    return save_result(result, split=split, results_dir=results_dir, **{**defaults, **extra})


class TiedRetriever:
    """Returns two passages with the same score, the smaller pid first."""

    name = "tied"

    def index(self, passages):
        pass

    def search(self, query, k=10):
        return [Hit("a", 1.0), Hit("b", 1.0)]


def test_evaluate_reports_metrics_slices_and_latency(bm25_result):
    assert bm25_result.n_questions == 3
    assert set(bm25_result.metrics) == {
        "recall@1",
        "recall@5",
        "recall@10",
        "recall@20",
        "recall@100",
        "map@10",
        "mrr@10",
        "ndcg@10",
    }
    assert all(0.0 <= v <= 1.0 for v in bm25_result.metrics.values())
    assert bm25_result.metrics["recall@10"] == 1.0
    assert set(bm25_result.slices) == {"1", "2"}
    assert bm25_result.slices["1"]["n"] == 2
    assert 0 <= bm25_result.latency_ms["p50"] <= bm25_result.latency_ms["p95"]


def test_tied_scores_are_ranked_like_trec_eval(tmp_path):
    question = Question(qid="q1", text="anything", group=1, gold=("a",))
    result = evaluate(TiedRetriever(), [question])
    # trec_eval puts the larger pid first among equal scores, so gold "a" is second
    assert result.metrics["mrr@10"] == 0.5
    assert [h.pid for h in result.rankings["q1"]] == ["b", "a"]
    path = tmp_path / "tied.trec"
    write_trec_run(path, result)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [line.split()[2] for line in lines] == ["b", "a"]


def test_duplicate_ids_in_a_ranking_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        score_ranking(["a", "a"], ["a"])


def test_trec_run_format(bm25_result, tmp_path):
    path = tmp_path / "runs" / "bm25.trec"
    write_trec_run(path, bm25_result)
    first = path.read_text(encoding="utf-8").splitlines()[0].split()
    assert len(first) == 6
    assert first[1] == "Q0" and first[3] == "1" and first[5] == "bm25"


def test_trec_run_keeps_exact_scores(bm25_result, tmp_path):
    path = tmp_path / "bm25.trec"
    write_trec_run(path, bm25_result)
    written = [float(line.split()[4]) for line in path.read_text(encoding="utf-8").splitlines()]
    expected = [hit.score for hits in bm25_result.rankings.values() for hit in hits]
    assert written == expected  # no rounding, so trec_eval sees exactly the same ties


def test_systems_share_indexes_between_single_and_hybrid(corpus):
    systems = Systems(corpus, encoder_factory=lambda spec: HashingEncoder())
    hybrid = systems.get("hybrid")
    assert hybrid.retrievers[0] is systems.get("bm25")
    assert hybrid.retrievers[1] is systems.get("dense")
    assert systems.index_seconds["hybrid"] == pytest.approx(
        systems.index_seconds["bm25"] + systems.index_seconds["dense"]
    )
    assert set(systems.load_seconds) == {"dense"}  # only dense systems load a model
    with pytest.raises(KeyError):
        systems.get("splade")


def test_a_hybrid_reports_the_model_of_its_dense_part(corpus):
    systems = Systems(corpus, encoder_factory=lambda spec: HashingEncoder())
    assert systems.model_spec("bm25") is None
    assert systems.model_spec("dense") == MODELS["bge-small"]
    assert systems.model_spec("hybrid") == MODELS["bge-small"]
    assert systems.model_spec("hybrid-base") == MODELS["bge-base"]


def test_systems_without_torch_come_first_in_the_default_run():
    assert DEFAULT_REPRODUCE == ["bm25", "bm25-nostem", "dense", "hybrid"]
    assert [needs_torch(name) for name in DEFAULT_REPRODUCE] == [False, False, True, True]
    assert needs_torch("dense-base") and needs_torch("hybrid-base")


def test_results_round_trip_into_the_readme(bm25_result, tmp_path):
    results_dir = tmp_path / "results"
    _save(bm25_result, results_dir)
    block = render(load_results(results_dir))
    assert "| BM25 | 100.0 |" in block
    assert "`uv run regrag eval --system bm25 --split test`" in block

    readme = tmp_path / "README.md"
    readme.write_text(f"# Title\n\n{START}\nSTALE TABLE\n{END}\n\nMore text\n", encoding="utf-8")
    assert update_readme(readme, block)
    text = readme.read_text(encoding="utf-8")
    assert "STALE TABLE" not in text and text.endswith(f"{END}\n\nMore text\n")
    assert not update_readme(readme, block)


def test_saved_result_records_the_dataset_model_settings_and_load_time(bm25_result, tmp_path):
    spec = MODELS["bge-small"]
    path = _save(bm25_result, tmp_path, model=spec, model_load_seconds=2.5)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["dataset"] == SYNTHETIC
    assert saved["model_load_seconds"] == 2.5
    assert saved["model"]["revision"] == spec.revision
    assert saved["model"]["max_seq_length"] == spec.max_seq_length
    assert saved["model"]["passage_prefix"] == spec.passage_prefix
    assert saved["model"]["query_prefix"] == spec.query_prefix


def test_render_refuses_results_from_two_machines(bm25_result, tmp_path):
    for split in ("test", "dev"):
        _save(bm25_result, tmp_path, split=split)
    results = load_results(tmp_path)
    results[("dev", "bm25")]["environment"]["cpu"] = "Another CPU"
    with pytest.raises(ValueError, match="more than one machine") as info:
        render(results)
    assert "Another CPU" in str(info.value)
    assert results[("test", "bm25")]["environment"]["cpu"] in str(info.value)


def test_render_gives_the_date_range_of_the_shown_results(bm25_result, tmp_path):
    for split in ("test", "dev"):
        _save(bm25_result, tmp_path, split=split)
    results = load_results(tmp_path)
    results[("test", "bm25")]["created_utc"] = "2026-11-15T08:00:00+00:00"
    results[("dev", "bm25")]["created_utc"] = "2026-11-09T20:07:01+00:00"
    assert ", 2026-11-09 to 2026-11-15 UTC. " in render(results)
    results[("dev", "bm25")]["created_utc"] = "2026-11-15T09:00:00+00:00"
    assert ", 2026-11-15 UTC. " in render(results)


def test_readme_without_markers_is_an_error(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# No markers\n", encoding="utf-8")
    with pytest.raises(ReadmeMarkersError, match="markers"):
        update_readme(readme, "x\n")


def test_markers_mentioned_in_prose_are_left_alone(tmp_path):
    readme = tmp_path / "README.md"
    prose = f"The table between `{START}` and `{END}` is generated."
    readme.write_text(f"{prose}\n\n{START}\nOLD TABLE\n{END}\n", encoding="utf-8")
    assert update_readme(readme, "NEW TABLE\n")
    assert readme.read_text(encoding="utf-8") == f"{prose}\n\n{START}\nNEW TABLE\n{END}\n"


def test_two_results_blocks_are_an_error(tmp_path):
    readme = tmp_path / "README.md"
    original = f"{START}\nONE\n{END}\n\n{START}\nTWO\n{END}\n"
    readme.write_text(original, encoding="utf-8")
    with pytest.raises(ReadmeMarkersError):
        update_readme(readme, "x\n")
    assert readme.read_text(encoding="utf-8") == original
