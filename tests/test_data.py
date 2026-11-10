import hashlib
import json
import re

import httpx
import pytest

from regrag._checksums import CHECKSUMS
from regrag.data import (
    OBLIQA_COMMIT,
    OBLIQA_REPO,
    ChecksumError,
    dataset_source,
    download,
    load_corpus,
    load_questions,
)


def test_corpus_drops_empty_passages(corpus):
    assert len(corpus) == 12
    assert "p-1-4" not in {p.pid for p in corpus}


def test_corpus_can_keep_empty_passages(data_dir):
    assert len(load_corpus(data_dir, drop_empty=False)) == 13


def test_citation_uses_document_and_rule_reference(corpus):
    assert next(p for p in corpus if p.pid == "p-2-5").citation == "Doc 2, 4.2"


def test_gold_passages_map_to_corpus_ids(dev_questions):
    gold = {q.qid: q.gold for q in dev_questions}
    assert gold["q-d1"] == ("p-1-3",)
    assert gold["q-d2"] == ("p-2-2", "p-2-3")


def test_duplicate_rule_reference_is_resolved_by_text(dev_questions):
    # Two passages in document 1 share the reference "Guidance 1."
    assert next(q for q in dev_questions if q.qid == "q-d3").gold == ("p-1-8",)


def test_repeated_question_ids_and_gold_refs_are_dropped(corpus, data_dir):
    train = load_questions("train", corpus, data_dir)
    assert [q.qid for q in train] == ["q-tr1", "q-tr2"]
    assert train[1].gold == ("p-1-5",)


def test_unknown_gold_passage_raises(corpus, tmp_path):
    folder = tmp_path / "obliqa"
    folder.mkdir()
    bad = [
        {
            "QuestionID": "q1",
            "Question": "?",
            "Group": 1,
            "Passages": [{"DocumentID": 9, "PassageID": "9.9", "Passage": "missing"}],
        }
    ]
    (folder / "ObliQA_dev.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="matches 0 corpus passages"):
        load_questions("dev", corpus, tmp_path)


def test_missing_data_points_to_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="regrag download"):
        load_corpus(tmp_path)


def test_data_without_a_manifest_is_refused(tmp_path):
    (tmp_path / "obliqa").mkdir()
    with pytest.raises(FileNotFoundError, match="did not finish"):
        load_corpus(tmp_path)


def test_data_from_another_commit_is_refused(tmp_path):
    (tmp_path / "obliqa").mkdir()
    stale = {"repo": OBLIQA_REPO, "commit": "0" * 40}
    (tmp_path / "obliqa" / "MANIFEST.json").write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="regrag download"):
        dataset_source(tmp_path)


def test_the_fixture_manifest_does_not_claim_to_be_obliqa(data_dir):
    assert dataset_source(data_dir) == {
        "repo": "synthetic test fixture, not ObliQA",
        "commit": None,
    }


def test_unknown_split_raises(corpus, data_dir):
    with pytest.raises(ValueError, match="Unknown split"):
        load_questions("validation", corpus, data_dir)


def test_pinned_checksums_cover_every_file():
    assert len(CHECKSUMS) == 43
    assert {f"ObliQA_{s}.json" for s in ("train", "dev", "test")} <= set(CHECKSUMS)
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in CHECKSUMS.values())


def _mock_client(files: dict[str, bytes], calls: list[str]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        rel = request.url.path.split("/pinned/", 1)[1]
        calls.append(rel)
        return httpx.Response(200, content=files[rel])

    return httpx.Client(transport=httpx.MockTransport(handler))


FILES = {
    "ObliQA_train.json": b"[]",
    "ObliQA_dev.json": b'[{"q": 1}]',
    "StructuredRegulatoryDocuments/1.json": b"[]",
}
SUMS = {rel: hashlib.sha256(data).hexdigest() for rel, data in FILES.items()}


def test_download_writes_verified_files_and_skips_them_next_time(tmp_path):
    calls: list[str] = []
    client = _mock_client(FILES, calls)
    target = download(tmp_path, client=client, base_url="https://x.test/pinned", checksums=SUMS)
    assert (target / "StructuredRegulatoryDocuments" / "1.json").read_bytes() == b"[]"
    assert len(calls) == 3
    manifest = json.loads((target / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest == {"repo": OBLIQA_REPO, "commit": OBLIQA_COMMIT}
    assert dataset_source(tmp_path) == manifest
    download(tmp_path, client=client, base_url="https://x.test/pinned", checksums=SUMS)
    assert len(calls) == 3  # nothing fetched again


def test_download_can_skip_train(tmp_path):
    calls: list[str] = []
    download(
        tmp_path,
        include_train=False,
        client=_mock_client(FILES, calls),
        base_url="https://x.test/pinned",
        checksums=SUMS,
    )
    assert "ObliQA_train.json" not in calls
    assert not (tmp_path / "obliqa" / "ObliQA_train.json").exists()
    assert (tmp_path / "obliqa" / "MANIFEST.json").exists()


def test_skip_train_still_checks_a_train_file_that_is_already_there(tmp_path):
    pinned = {"base_url": "https://x.test/pinned", "checksums": SUMS}
    train = tmp_path / "obliqa" / "ObliQA_train.json"
    train.parent.mkdir(parents=True)
    train.write_bytes(FILES["ObliQA_train.json"])
    seen: dict[str, str] = {}
    calls: list[str] = []
    download(
        tmp_path,
        include_train=False,
        client=_mock_client(FILES, calls),
        on_file=lambda rel, status: seen.update({rel: status}),
        **pinned,
    )
    assert seen["ObliQA_train.json"] == "ok"
    assert "ObliQA_train.json" not in calls

    train.write_bytes(b'[{"edited": true}]')
    with pytest.raises(ChecksumError, match="--skip-train"):
        download(tmp_path, include_train=False, client=_mock_client(FILES, []), **pinned)
    assert not (tmp_path / "obliqa" / "MANIFEST.json").exists()
    assert train.read_bytes() == b'[{"edited": true}]'  # reported, never deleted


def test_download_rejects_a_changed_file(tmp_path):
    tampered = dict(FILES, **{"ObliQA_dev.json": b"[]  "})
    with pytest.raises(ChecksumError, match=r"ObliQA_dev\.json"):
        download(
            tmp_path,
            client=_mock_client(tampered, []),
            base_url="https://x.test/pinned",
            checksums=SUMS,
        )
    assert not (tmp_path / "obliqa" / "ObliQA_dev.json").exists()
    assert not (tmp_path / "obliqa" / "MANIFEST.json").exists()


def test_a_failed_download_removes_the_old_manifest(tmp_path):
    pinned = {"base_url": "https://x.test/pinned", "checksums": SUMS}
    download(tmp_path, client=_mock_client(FILES, []), **pinned)
    assert (tmp_path / "obliqa" / "MANIFEST.json").exists()
    tampered = dict(FILES, **{"ObliQA_dev.json": b"[]  "})
    with pytest.raises(ChecksumError):
        download(tmp_path, force=True, client=_mock_client(tampered, []), **pinned)
    assert not (tmp_path / "obliqa" / "MANIFEST.json").exists()
    with pytest.raises(FileNotFoundError, match="did not finish"):
        dataset_source(tmp_path)
