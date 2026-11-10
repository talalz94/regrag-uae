import builtins
import dataclasses
import json
import shutil

import httpx
import pytest
from conftest import HashingEncoder

from regrag.cli import main
from regrag.data import ChecksumError
from regrag.report import END, START
from regrag.retrievers import MODELS
from regrag.systems import Systems


def _results_args(tmp_path):
    return ["--results-dir", str(tmp_path / "results")]


def _out_args(tmp_path):
    return [*_results_args(tmp_path), "--runs-dir", str(tmp_path / "runs")]


def test_eval_saves_a_result_with_its_command(data_dir, tmp_path):
    argv = ["--data-dir", str(data_dir), "eval", "--system", "bm25", "--split", "test"]
    assert main(argv + _out_args(tmp_path)) == 0
    saved = json.loads((tmp_path / "results" / "test" / "bm25.json").read_text(encoding="utf-8"))
    assert saved["command"] == "uv run regrag eval --system bm25 --split test"
    assert saved["n_questions"] == 3
    # the dataset record comes from the fixture's manifest, so it never claims ObliQA
    assert saved["dataset"] == {"repo": "synthetic test fixture, not ObliQA", "commit": None}
    assert (tmp_path / "runs" / "test" / "bm25.trec").exists()


def test_a_saved_hybrid_result_records_its_embedding_model(data_dir, tmp_path, monkeypatch):
    def offline_systems(args, corpus):  # the hashing encoder stands in for bge-small
        return Systems(corpus, encoder_factory=lambda spec: HashingEncoder())

    monkeypatch.setattr("regrag.cli._systems", offline_systems)
    argv = ["--data-dir", str(data_dir), "eval", "--system", "hybrid", "--split", "test"]
    assert main(argv + _out_args(tmp_path)) == 0
    path = tmp_path / "results" / "test" / "hybrid.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["model"] == dataclasses.asdict(MODELS["bge-small"])


def test_eval_with_limit_saves_nothing(data_dir, tmp_path, capsys):
    argv = ["--data-dir", str(data_dir), "eval", "--system", "bm25", "--limit", "2"]
    assert main(argv + _out_args(tmp_path)) == 0
    assert "nothing was saved" in capsys.readouterr().out
    assert not (tmp_path / "results").exists()


def test_limit_below_one_is_rejected_and_saves_nothing(data_dir, tmp_path, capsys):
    argv = ["--data-dir", str(data_dir), "eval", "--system", "bm25", "--limit", "0"]
    with pytest.raises(SystemExit) as exit_info:
        main(argv + _out_args(tmp_path))
    assert exit_info.value.code == 2
    assert "must be 1 or more" in capsys.readouterr().err
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize(
    ("k", "message"),
    [("0", "must be 1 or more"), ("-1", "must be 1 or more"), ("two", "expected a whole number")],
)
def test_search_k_below_one_is_rejected(data_dir, capsys, k, message):
    with pytest.raises(SystemExit) as exit_info:
        main(["--data-dir", str(data_dir), "search", "custodian", "-k", k])
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


def test_reproduce_rewrites_the_readme(data_dir, tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"{START}\n{END}\n", encoding="utf-8")
    argv = ["--data-dir", str(data_dir), "reproduce", "--systems", "bm25", "bm25-nostem"]
    assert main([*argv, "--readme", str(readme), *_out_args(tmp_path)]) == 0
    text = readme.read_text(encoding="utf-8")
    assert "**Test split** (3 questions, 12 passages)" in text
    assert "**Dev split** (4 questions, 12 passages)" in text
    assert "BM25 without stemming" in text


def test_reproduce_runs_systems_without_torch_first(data_dir, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "regrag.cli._run_and_save",
        lambda args, systems, name, split, questions, n_passages: calls.append(name),
    )
    readme = tmp_path / "README.md"
    readme.write_text(f"{START}\n{END}\n", encoding="utf-8")
    systems = ["hybrid", "bm25-nostem", "dense", "bm25"]
    argv = ["--data-dir", str(data_dir), "reproduce", "--systems", *systems, "--splits", "test"]
    main([*argv, "--readme", str(readme), *_out_args(tmp_path)])
    assert calls == ["bm25-nostem", "bm25", "hybrid", "dense"]


def test_reproduce_checks_the_readme_before_any_eval(data_dir, tmp_path, capsys):
    readme = tmp_path / "README.md"
    readme.write_text("# No markers\n", encoding="utf-8")
    argv = ["--data-dir", str(data_dir), "reproduce", "--systems", "bm25"]
    assert main([*argv, "--readme", str(readme), *_out_args(tmp_path)]) == 1
    assert "markers" in capsys.readouterr().err
    assert not (tmp_path / "results").exists()


def test_report_on_a_readme_without_markers_is_a_clean_error(data_dir, tmp_path, capsys):
    argv = ["--data-dir", str(data_dir), "eval", "--system", "bm25"]
    assert main(argv + _out_args(tmp_path)) == 0
    readme = tmp_path / "README.md"
    readme.write_text("# No markers\n", encoding="utf-8")
    assert main(["report", "--readme", str(readme), *_results_args(tmp_path)]) == 1
    assert "markers" in capsys.readouterr().err


def test_report_offers_no_runs_dir_because_it_never_reads_runs(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["report", "--help"])
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--results-dir" in help_text
    assert "--runs-dir" not in help_text


def test_report_refuses_results_from_two_machines(data_dir, tmp_path, capsys):
    for split in ("dev", "test"):
        argv = ["--data-dir", str(data_dir), "eval", "--system", "bm25", "--split", split]
        assert main(argv + _out_args(tmp_path)) == 0
    path = tmp_path / "results" / "dev" / "bm25.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["environment"]["cpu"] = "Another CPU"
    path.write_text(json.dumps(saved), encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text(f"{START}\n{END}\n", encoding="utf-8")
    assert main(["report", "--readme", str(readme), *_results_args(tmp_path)]) == 1
    assert "more than one machine" in capsys.readouterr().err
    assert readme.read_text(encoding="utf-8") == f"{START}\n{END}\n"


def test_search_prints_citations(data_dir, capsys):
    argv = ["--data-dir", str(data_dir), "search", "custodian reconcile holdings"]
    assert main([*argv, "--system", "bm25", "-k", "2"]) == 0
    assert "[1] Doc 2, 4.2" in capsys.readouterr().out


def test_download_needs_consent_when_not_interactive(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert main(["--data-dir", str(tmp_path), "download"]) == 2
    assert "--yes" in capsys.readouterr().out
    assert not (tmp_path / "obliqa").exists()


@pytest.mark.parametrize("failure", [httpx.ConnectError("offline"), ChecksumError("x")])
def test_download_failures_are_clean_errors(tmp_path, monkeypatch, capsys, failure):
    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr("regrag.cli.download", fail)
    assert main(["--data-dir", str(tmp_path), "download", "--yes"]) == 1
    assert capsys.readouterr().err.startswith("error:")


def test_ctrl_c_at_the_consent_prompt_cancels_cleanly(tmp_path, monkeypatch, capsys):
    def interrupt(prompt=""):
        raise KeyboardInterrupt

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(builtins, "input", interrupt)
    assert main(["--data-dir", str(tmp_path), "download"]) == 130
    assert "Cancelled" in capsys.readouterr().err
    assert not (tmp_path / "obliqa").exists()


def test_closed_input_at_the_consent_prompt_downloads_nothing(tmp_path, monkeypatch, capsys):
    def closed(prompt=""):
        raise EOFError

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(builtins, "input", closed)
    assert main(["--data-dir", str(tmp_path), "download"]) == 1
    assert "Nothing was downloaded" in capsys.readouterr().out
    assert not (tmp_path / "obliqa").exists()


def test_missing_data_is_a_clean_error(tmp_path, capsys):
    assert main(["--data-dir", str(tmp_path), "eval", "--system", "bm25"]) == 1
    assert "regrag download" in capsys.readouterr().err


def test_an_unfinished_download_is_refused(data_dir, tmp_path, capsys):
    copy = tmp_path / "data"
    shutil.copytree(data_dir, copy)
    (copy / "obliqa" / "MANIFEST.json").unlink()
    assert main(["--data-dir", str(copy), "search", "custodian"]) == 1
    assert "regrag download" in capsys.readouterr().err
    assert main(["--data-dir", str(copy), "eval", "--system", "bm25", *_out_args(tmp_path)]) == 1
    assert "regrag download" in capsys.readouterr().err
    assert not (tmp_path / "results").exists()


def test_stats_prints_json(data_dir, capsys):
    assert main(["--data-dir", str(data_dir), "stats"]) == 0
    assert json.loads(capsys.readouterr().out)["passages_used"] == 12
