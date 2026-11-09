import pytest
from conftest import HashingEncoder

from regrag.evaluate import evaluate, score_ranking, write_trec_run
from regrag.report import END, START, load_results, render, save_result, update_readme
from regrag.retrievers import BM25Retriever
from regrag.systems import Systems


@pytest.fixture
def bm25_result(corpus, test_questions):
    retriever = BM25Retriever()
    retriever.index(corpus)
    return evaluate(retriever, test_questions)


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


def test_duplicate_ids_in_a_ranking_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        score_ranking(["a", "a"], ["a"])


def test_trec_run_format(bm25_result, tmp_path):
    path = tmp_path / "runs" / "bm25.trec"
    write_trec_run(path, bm25_result)
    first = path.read_text(encoding="utf-8").splitlines()[0].split()
    assert len(first) == 6
    assert first[1] == "Q0" and first[3] == "1" and first[5] == "bm25"


def test_systems_share_indexes_between_single_and_hybrid(corpus):
    systems = Systems(corpus, encoder_factory=lambda spec: HashingEncoder())
    hybrid = systems.get("hybrid")
    assert hybrid.retrievers[0] is systems.get("bm25")
    assert hybrid.retrievers[1] is systems.get("dense")
    assert systems.index_seconds["hybrid"] == pytest.approx(
        systems.index_seconds["bm25"] + systems.index_seconds["dense"]
    )
    with pytest.raises(KeyError):
        systems.get("splade")


def test_results_round_trip_into_the_readme(bm25_result, corpus, tmp_path):
    results_dir = tmp_path / "results"
    save_result(
        bm25_result,
        split="test",
        n_passages=len(corpus),
        results_dir=results_dir,
        index_seconds=0.01,
        index_from_cache=False,
        model=None,
    )
    block = render(load_results(results_dir))
    assert "| BM25 | 100.0 |" in block
    assert "`uv run regrag eval --system bm25 --split test`" in block

    readme = tmp_path / "README.md"
    readme.write_text(f"# Title\n\n{START}\nSTALE TABLE\n{END}\n\nMore text\n", encoding="utf-8")
    assert update_readme(readme, block)
    text = readme.read_text(encoding="utf-8")
    assert "STALE TABLE" not in text and text.endswith(f"{END}\n\nMore text\n")
    assert not update_readme(readme, block)


def test_readme_without_markers_is_an_error(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# No markers\n", encoding="utf-8")
    with pytest.raises(ValueError):
        update_readme(readme, "x\n")
