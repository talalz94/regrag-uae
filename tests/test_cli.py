import json

from regrag.cli import main
from regrag.report import END, START


def _out_args(tmp_path):
    return ["--results-dir", str(tmp_path / "results"), "--runs-dir", str(tmp_path / "runs")]


def test_eval_saves_a_result_with_its_command(data_dir, tmp_path):
    argv = ["--data-dir", str(data_dir), "eval", "--system", "bm25", "--split", "test"]
    assert main(argv + _out_args(tmp_path)) == 0
    saved = json.loads((tmp_path / "results" / "test" / "bm25.json").read_text(encoding="utf-8"))
    assert saved["command"] == "uv run regrag eval --system bm25 --split test"
    assert saved["n_questions"] == 3
    assert (tmp_path / "runs" / "test" / "bm25.trec").exists()


def test_eval_with_limit_saves_nothing(data_dir, tmp_path, capsys):
    argv = ["--data-dir", str(data_dir), "eval", "--system", "bm25", "--limit", "2"]
    assert main(argv + _out_args(tmp_path)) == 0
    assert "nothing was saved" in capsys.readouterr().out
    assert not (tmp_path / "results").exists()


def test_reproduce_rewrites_the_readme(data_dir, tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"{START}\n{END}\n", encoding="utf-8")
    argv = ["--data-dir", str(data_dir), "reproduce", "--systems", "bm25", "bm25-nostem"]
    assert main([*argv, "--readme", str(readme), *_out_args(tmp_path)]) == 0
    text = readme.read_text(encoding="utf-8")
    assert "**Test split** (3 questions, 12 passages)" in text
    assert "**Dev split** (4 questions, 12 passages)" in text
    assert "BM25 without stemming" in text


def test_search_prints_citations(data_dir, capsys):
    argv = ["--data-dir", str(data_dir), "search", "custodian reconcile holdings"]
    assert main([*argv, "--system", "bm25", "-k", "2"]) == 0
    assert "[1] Doc 2, 4.2" in capsys.readouterr().out


def test_download_needs_consent_when_not_interactive(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert main(["--data-dir", str(tmp_path), "download"]) == 2
    assert "--yes" in capsys.readouterr().out
    assert not (tmp_path / "obliqa").exists()


def test_missing_data_is_a_clean_error(tmp_path, capsys):
    assert main(["--data-dir", str(tmp_path), "eval", "--system", "bm25"]) == 1
    assert "regrag download" in capsys.readouterr().err
