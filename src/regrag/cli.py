"""Command line entry point.

regrag download              fetch ObliQA from its source repo (asks you to accept the notice)
regrag eval --system NAME    score one system on one split and save the result
regrag reproduce             run the default systems on dev and test, then rewrite the README table
regrag report                rewrite the README table from saved results
regrag search "question"     print the top passages for a question
regrag stats                 print the dataset statistics behind docs/DATA_CARD.md
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path

import httpx

from regrag.data import (
    NOTICE,
    SPLITS,
    ChecksumError,
    dataset_source,
    default_data_dir,
    download,
    load_corpus,
    load_questions,
)
from regrag.evaluate import EvalResult, evaluate, write_trec_run
from regrag.report import (
    ReadmeMarkersError,
    check_readme,
    load_results,
    render,
    save_result,
    update_readme,
)
from regrag.stats import dataset_stats
from regrag.systems import DEFAULT_REPRODUCE, SPECS, Systems, needs_torch


def _data_dir(args: argparse.Namespace) -> Path:
    return args.data_dir or default_data_dir()


def positive_int(value: str) -> int:
    """argparse type for a count of 1 or more."""
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {value!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, got {number}")
    return number


def _systems(args: argparse.Namespace, corpus) -> Systems:
    return Systems(corpus, cache_dir=_data_dir(args) / "cache")


def _progress(label: str):
    def report(done: int, total: int) -> None:
        print(f"  {label}: {done:,}/{total:,} questions", flush=True)

    return report


def _summary(result: EvalResult, split: str) -> str:
    m, lat = result.metrics, result.latency_ms
    return (
        f"{result.system:<12} {split:<5} R@10 {100 * m['recall@10']:.1f}  "
        f"MAP@10 {100 * m['map@10']:.1f}  MRR@10 {100 * m['mrr@10']:.1f}  "
        f"nDCG@10 {100 * m['ndcg@10']:.1f}  R@100 {100 * m['recall@100']:.1f}  "
        f"p50 {lat['p50']:.1f} ms  p95 {lat['p95']:.1f} ms"
    )


def _run_and_save(
    args: argparse.Namespace, systems: Systems, name: str, split: str, questions, n_passages: int
) -> EvalResult:
    retriever = systems.get(name)
    result = evaluate(retriever, questions, on_progress=_progress(f"{name} on {split}"))
    print(_summary(result, split), flush=True)
    path = save_result(
        result,
        split=split,
        n_passages=n_passages,
        results_dir=args.results_dir,
        index_seconds=systems.index_seconds[name],
        index_from_cache=systems.index_from_cache[name],
        model=systems.model_spec(name),
        dataset=dataset_source(_data_dir(args)),
        model_load_seconds=systems.load_seconds.get(name),
    )
    write_trec_run(args.runs_dir / split / f"{name}.trec", result)
    print(f"  saved {path.as_posix()}", flush=True)
    return result


def cmd_download(args: argparse.Namespace) -> int:
    print(NOTICE)
    if not args.yes:
        if not sys.stdin.isatty():
            print("Not an interactive terminal. Run it again with --yes to accept the notice.")
            return 2
        try:
            answer = input("Type yes to download: ")
        except EOFError:  # the input stream closed before an answer
            answer = ""
        if answer.strip().lower() != "yes":
            print("Cancelled. Nothing was downloaded.")
            return 1
    counts = {"downloaded": 0, "ok": 0}

    def on_file(rel: str, status: str) -> None:
        counts[status] += 1
        if status == "downloaded":
            print(f"  downloaded {rel}", flush=True)

    try:
        target = download(
            _data_dir(args), include_train=not args.skip_train, force=args.force, on_file=on_file
        )
    except httpx.HTTPError as err:
        print(
            f"error: could not fetch ObliQA ({err}). Check the network and run "
            "`regrag download` again; files already verified are kept.",
            file=sys.stderr,
        )
        return 1
    except ChecksumError as err:  # the message says what happened to the file
        print(f"error: {err}", file=sys.stderr)
        return 1
    print(
        f"Done: {counts['downloaded']} downloaded, {counts['ok']} already present and verified. "
        f"Files are in {target.as_posix()}"
    )
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    corpus = load_corpus(_data_dir(args))
    questions = load_questions(args.split, corpus, _data_dir(args))
    systems = _systems(args, corpus)
    if args.limit is not None:
        questions = questions[: args.limit]
        result = evaluate(systems.get(args.system), questions)
        print(_summary(result, args.split))
        print("  (--limit used, so nothing was saved)")
        return 0
    _run_and_save(args, systems, args.system, args.split, questions, len(corpus))
    return 0


def cmd_reproduce(args: argparse.Namespace) -> int:
    check_readme(args.readme)  # a wrong path or bad markers fail now, not after every eval
    corpus = load_corpus(_data_dir(args))
    print(f"Corpus: {len(corpus):,} passages", flush=True)
    splits = {s: load_questions(s, corpus, _data_dir(args)) for s in args.splits}
    systems = _systems(args, corpus)
    # Systems without torch run first, so their timings never share a process with it.
    # The sort is stable, so the given order is kept within each group.
    for name in sorted(args.systems, key=needs_torch):
        for split, questions in splits.items():
            _run_and_save(args, systems, name, split, questions, len(corpus))
    return cmd_report(args)


def cmd_report(args: argparse.Namespace) -> int:
    results = load_results(args.results_dir)
    if not results:
        print(f"No results in {args.results_dir}. Run `regrag reproduce` first.")
        return 1
    try:
        block = render(results)
        changed = update_readme(args.readme, block)
    except ValueError as err:  # results from several machines, or README markers missing
        print(f"error: {err}", file=sys.stderr)
        return 1
    print(f"{args.readme} {'updated' if changed else 'already up to date'}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    corpus = load_corpus(_data_dir(args))
    by_pid = {p.pid: p for p in corpus}
    hits = _systems(args, corpus).get(args.system).search(args.query, args.k)
    if not hits:
        print("No passages matched.")
    for rank, hit in enumerate(hits, start=1):
        passage = by_pid[hit.pid]
        print(f"\n[{rank}] {passage.citation}   score {hit.score:.3f}")
        body = textwrap.shorten(" ".join(passage.text.split()), width=500, placeholder=" ...")
        print(textwrap.fill(body, width=100, initial_indent="    ", subsequent_indent="    "))
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    print(json.dumps(dataset_stats(_data_dir(args)), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="regrag", description="Measured retrieval over ADGM rules (ObliQA)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="where ObliQA is stored (default: $REGRAG_DATA_DIR or ./data)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="fetch ObliQA from its source repo")
    p.add_argument("--yes", action="store_true", help="accept the dataset notice without a prompt")
    p.add_argument(
        "--skip-train",
        action="store_true",
        help=(
            "skip the 29 MB train split (if it is not already there, "
            "regrag stats shows it as not downloaded)"
        ),
    )
    p.add_argument("--force", action="store_true", help="download again even if files are present")
    p.set_defaults(func=cmd_download)

    def add_results_dir(p: argparse.ArgumentParser) -> None:
        p.add_argument("--results-dir", type=Path, default=Path("results"))

    def add_output_args(p: argparse.ArgumentParser) -> None:  # for commands that run evals
        add_results_dir(p)
        p.add_argument("--runs-dir", type=Path, default=Path("runs"))

    p = sub.add_parser("eval", help="score one system on one split")
    p.add_argument("--system", choices=list(SPECS), required=True)
    p.add_argument("--split", choices=SPLITS, default="test")
    p.add_argument(
        "--limit", type=positive_int, help="only the first N questions (nothing is saved)"
    )
    add_output_args(p)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("reproduce", help="run the default systems, then rewrite the README table")
    p.add_argument(
        "--systems",
        nargs="+",
        choices=list(SPECS),
        default=DEFAULT_REPRODUCE,
        help="default: all but the slow bge-base ablations (dense-base, hybrid-base)",
    )
    p.add_argument("--splits", nargs="+", choices=SPLITS, default=["dev", "test"])
    p.add_argument("--readme", type=Path, default=Path("README.md"))
    add_output_args(p)
    p.set_defaults(func=cmd_reproduce)

    p = sub.add_parser("report", help="rewrite the README table from saved results")
    p.add_argument("--readme", type=Path, default=Path("README.md"))
    add_results_dir(p)  # report reads only the results, never the run files
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("search", help="print the top passages for a question")
    p.add_argument("query")
    p.add_argument(
        "--system",
        choices=list(SPECS),
        default="bm25",
        help="default: bm25, the best and fastest system in the results table",
    )
    p.add_argument("-k", type=positive_int, default=5, help="how many passages to print")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("stats", help="print the dataset statistics behind the data card")
    p.set_defaults(func=cmd_stats)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # passages contain characters some consoles lack
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ReadmeMarkersError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
