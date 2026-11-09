"""Save eval results as JSON and render them into the README between two markers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from regrag import __version__
from regrag.data import OBLIQA_COMMIT, OBLIQA_REPO
from regrag.env import environment
from regrag.evaluate import EvalResult
from regrag.retrievers.dense import ModelSpec
from regrag.systems import ABLATIONS, MAIN_SYSTEMS, SPECS

START = "<!-- results:start -->"
END = "<!-- results:end -->"
API_COST_PER_1K = 0.0  # every system in this slice runs locally; no API calls


def eval_command(system: str, split: str) -> str:
    return f"uv run regrag eval --system {system} --split {split}"


def save_result(
    result: EvalResult,
    *,
    split: str,
    n_passages: int,
    results_dir: Path,
    index_seconds: float,
    index_from_cache: bool,
    model: ModelSpec | None,
) -> Path:
    spec = SPECS[result.system]
    payload = {
        "system": result.system,
        "label": spec.label,
        "description": spec.description,
        "split": split,
        "n_questions": result.n_questions,
        "n_passages": n_passages,
        "metrics": result.metrics,
        "slices_by_gold_passages": result.slices,
        "latency_ms": result.latency_ms,
        "index_seconds": index_seconds,
        "index_from_cache": index_from_cache,
        "api_cost_usd_per_1k_queries": API_COST_PER_1K,
        "command": eval_command(result.system, split),
        "model": (
            {"name": model.name, "revision": model.revision, "license": model.license}
            if model
            else None
        ),
        "dataset": {"repo": OBLIQA_REPO, "commit": OBLIQA_COMMIT},
        "environment": environment(),
        "regrag_version": __version__,
        "created_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }
    path = results_dir / split / f"{result.system}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def load_results(results_dir: Path) -> dict[tuple[str, str], dict]:
    results = {}
    for path in sorted(results_dir.glob("*/*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        results[(data["split"], data["system"])] = data
    return results


def _pct(value: float) -> str:
    return f"{100 * value:.1f}"


def _table(rows: list[dict]) -> list[str]:
    lines = [
        "| System | Recall@10 | MAP@10 | MRR@10 | nDCG@10 | Recall@100 "
        "| p50 ms | p95 ms | API cost per 1k queries | Reproduce |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        m, lat = r["metrics"], r["latency_ms"]
        lines.append(
            f"| {r['label']} | {_pct(m['recall@10'])} | {_pct(m['map@10'])} "
            f"| {_pct(m['mrr@10'])} | {_pct(m['ndcg@10'])} | {_pct(m['recall@100'])} "
            f"| {lat['p50']:.1f} | {lat['p95']:.1f} "
            f"| ${r['api_cost_usd_per_1k_queries']:.2f} | `{r['command']}` |"
        )
    return lines


def _slice_table(rows: list[dict]) -> list[str]:
    names = ["1", "2", "3+"]
    first = rows[0]["slices_by_gold_passages"]
    titles = {"1": "1 gold passage", "2": "2 gold passages", "3+": "3 to 6 gold passages"}
    header = " | ".join(f"{titles[n]} (n={int(first[n]['n']):,})" for n in names if n in first)
    lines = [f"| System | {header} |", "|---|" + "---:|" * sum(n in first for n in names)]
    for r in rows:
        s = r["slices_by_gold_passages"]
        cells = " | ".join(_pct(s[n]["recall@10"]) for n in names if n in s)
        lines.append(f"| {r['label']} | {cells} |")
    return lines


def _index_note(rows: list[dict]) -> str:
    parts = []
    for r in rows:
        if r["system"].startswith("hybrid"):
            continue
        seconds = r["index_seconds"]
        if r["index_from_cache"]:
            parts.append(f"{r['label']}: loaded from cache")
        elif seconds >= 120:
            parts.append(f"{r['label']}: {seconds / 60:.1f} min")
        else:
            parts.append(f"{r['label']}: {seconds:.1f} s")
    return "Index build time on the same machine: " + "; ".join(parts) + "."


def render(results: dict[tuple[str, str], dict]) -> str:
    out: list[str] = []
    for split in ("test", "dev"):
        rows = [results[(split, s)] for s in MAIN_SYSTEMS if (split, s) in results]
        if not rows:
            continue
        n_q, n_p = rows[0]["n_questions"], rows[0]["n_passages"]
        out += [f"**{split.title()} split** ({n_q:,} questions, {n_p:,} passages)", ""]
        out += [*_table(rows), ""]

    ablations = [results[("test", s)] for s in ABLATIONS if ("test", s) in results]
    if ablations:
        out += ["**Ablations, test split**", "", *_table(ablations), ""]

    sliced = [results[("test", s)] for s in MAIN_SYSTEMS if ("test", s) in results]
    if sliced:
        out += ["**Recall@10 by number of gold passages, test split**", ""]
        out += [*_slice_table(sliced), ""]

    if results:
        everything = list(results.values())
        sample = max(everything, key=lambda r: r["created_utc"])
        env = sample["environment"]
        timed = [results[("test", s)] for s in MAIN_SYSTEMS + ABLATIONS if ("test", s) in results]
        out += [
            f"Measured on {env['cpu']} ({env['logical_cpus']} logical CPUs, CPU only), "
            f"Python {env['python']}, {sample['created_utc'][:10]} UTC. "
            "Latency is one query at a time inside the retriever (query encoding, search "
            "and fusion), after 3 warmup queries. " + _index_note(timed),
            "",
            "Generated by `uv run regrag report` from `results/`. Do not edit by hand.",
        ]
    return "\n".join(out).rstrip() + "\n"


def update_readme(readme: Path, block: str) -> bool:
    """Replace the text between the results markers. Returns True if the file changed."""
    text = readme.read_text(encoding="utf-8")
    start, end = text.find(START), text.find(END)
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"{readme} needs {START} and {END} markers")
    updated = text[: start + len(START)] + "\n" + block + text[end:]
    if updated == text:
        return False
    readme.write_text(updated, encoding="utf-8", newline="\n")
    return True
