"""Download and load the ObliQA benchmark.

ObliQA has no license file, so this repo never hosts a copy. `regrag download`
fetches the files from the authors' GitHub repo at a pinned commit, checks the
SHA-256 of every file and keeps them in the local data directory, which git ignores.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx

from regrag._checksums import CHECKSUMS

OBLIQA_REPO = "RegNLP/ObliQADataset"
OBLIQA_COMMIT = "c1b2c3c04aee1f6ce75f76106ce13af672d48e02"
OBLIQA_URL = f"https://github.com/{OBLIQA_REPO}"
RAW_BASE_URL = f"https://raw.githubusercontent.com/{OBLIQA_REPO}/{OBLIQA_COMMIT}"
SPLITS = ("train", "dev", "test")
DOCS_DIR = "StructuredRegulatoryDocuments"

NOTICE = f"""\
ObliQA dataset notice
=====================
Source:  {OBLIQA_URL} (commit {OBLIQA_COMMIT[:12]})
Paper:   Gokhan et al., "RIRAG: Regulatory Information Retrieval and Answer
         Generation", arXiv:2409.05677
Content: questions generated with GPT-4 from 40 ADGM regulatory documents.

The dataset has no license file. This project does not host a copy. The files
are downloaded from the source repo to your machine for research use only.
Cite the paper if you use them, and contact the authors for any other use.
"""


class ChecksumError(RuntimeError):
    """A downloaded file does not match its pinned SHA-256."""


@dataclass(frozen=True, slots=True)
class Passage:
    """One regulatory passage, the unit that gets retrieved and cited."""

    pid: str  # the dataset's unique "ID" field
    doc_id: int  # "DocumentID", 1 to 40
    ref: str  # "PassageID", the rule reference inside the document, e.g. "11.2.1"
    text: str

    @property
    def citation(self) -> str:
        return f"Doc {self.doc_id}, {self.ref}"


@dataclass(frozen=True, slots=True)
class Question:
    qid: str
    text: str
    group: int
    gold: tuple[str, ...]  # pids of the passages the question was written from


def default_data_dir() -> Path:
    return Path(os.environ.get("REGRAG_DATA_DIR", "data"))


def obliqa_dir(data_dir: Path | None = None) -> Path:
    return (data_dir or default_data_dir()) / "obliqa"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(
    data_dir: Path | None = None,
    *,
    include_train: bool = True,
    force: bool = False,
    client: httpx.Client | None = None,
    base_url: str = RAW_BASE_URL,
    checksums: Mapping[str, str] = CHECKSUMS,
    on_file: Callable[[str, str], None] | None = None,
) -> Path:
    """Fetch ObliQA into `data_dir/obliqa`, verifying every file. Returns that folder.

    Files that are already present with the right checksum are skipped.
    """
    target = obliqa_dir(data_dir)
    own_client = client is None
    http = client or httpx.Client(timeout=60.0, follow_redirects=True)
    try:
        for rel in checksums:
            if not include_train and rel == "ObliQA_train.json":
                continue
            dest = target / rel
            expected = checksums[rel]
            if dest.exists() and not force and sha256_bytes(dest.read_bytes()) == expected:
                if on_file:
                    on_file(rel, "ok")
                continue
            response = http.get(f"{base_url}/{rel}")
            response.raise_for_status()
            data = response.content
            actual = sha256_bytes(data)
            if actual != expected:
                raise ChecksumError(f"{rel}: expected sha256 {expected}, got {actual}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            partial = dest.with_name(dest.name + ".part")
            partial.write_bytes(data)
            partial.replace(dest)
            if on_file:
                on_file(rel, "downloaded")
    finally:
        if own_client:
            http.close()
    return target


def _read_json(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `regrag download` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_corpus(data_dir: Path | None = None, *, drop_empty: bool = True) -> list[Passage]:
    """Load every passage of the 40 structured documents, in document order.

    Empty passages (section breaks in the source) are dropped by default. None of
    them is ever a gold passage, and they cannot be matched by any query.
    """
    docs = obliqa_dir(data_dir) / DOCS_DIR
    files = sorted(docs.glob("*.json"), key=lambda p: int(p.stem))
    if not files:
        raise FileNotFoundError(f"No ObliQA documents in {docs}. Run `regrag download` first.")
    passages: list[Passage] = []
    for path in files:
        for row in _read_json(path):
            text = row.get("Passage") or ""
            if drop_empty and not text.strip():
                continue
            passages.append(
                Passage(
                    pid=str(row["ID"]),
                    doc_id=int(row["DocumentID"]),
                    ref=str(row["PassageID"]),
                    text=text,
                )
            )
    return passages


def _gold_index(corpus: Iterable[Passage]) -> dict[tuple[int, str], list[Passage]]:
    index: dict[tuple[int, str], list[Passage]] = defaultdict(list)
    for passage in corpus:
        index[(passage.doc_id, passage.ref)].append(passage)
    return index


def _resolve_gold(index: Mapping[tuple[int, str], list[Passage]], ref: Mapping) -> str:
    """Map a question's gold passage to a corpus pid.

    Questions point at passages by (DocumentID, PassageID). That pair is unique for
    all but 15 keys in the corpus, so ties are broken by exact passage text.
    """
    key = (int(ref["DocumentID"]), str(ref["PassageID"]))
    candidates = index.get(key, [])
    if len(candidates) > 1:
        candidates = [p for p in candidates if p.text == ref.get("Passage")]
    if len(candidates) != 1:
        raise ValueError(
            f"Gold passage {key} matches {len(candidates)} corpus passages, expected exactly 1"
        )
    return candidates[0].pid


def load_questions(
    split: str, corpus: Sequence[Passage], data_dir: Path | None = None
) -> list[Question]:
    """Load one official split with its gold passages mapped to corpus pids."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}, expected one of {SPLITS}")
    rows = _read_json(obliqa_dir(data_dir) / f"ObliQA_{split}.json")
    index = _gold_index(corpus)
    questions: list[Question] = []
    seen: set[str] = set()
    for row in rows:
        qid = str(row["QuestionID"])
        if qid in seen:  # the train file repeats 3 ids; keep the first
            continue
        seen.add(qid)
        gold: list[str] = []
        for ref in row["Passages"]:
            pid = _resolve_gold(index, ref)
            if pid not in gold:
                gold.append(pid)
        questions.append(
            Question(qid=qid, text=str(row["Question"]), group=int(row["Group"]), gold=tuple(gold))
        )
    return questions
