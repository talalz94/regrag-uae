"""Shared fixtures. Tests run on a tiny synthetic rulebook, never on ObliQA itself."""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from regrag.data import load_corpus, load_questions
from regrag.text import tokenize

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "mini"


class HashingEncoder:
    """Deterministic stand in for an embedding model: a hashed bag of stemmed words."""

    key = "test/hashing@0"

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.passage_calls = 0

    def _embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in tokenize(text):
            vector[zlib.crc32(token.encode()) % self.dim] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._embed(t) for t in texts])

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        self.passage_calls += 1
        return np.stack([self._embed(t) for t in texts])


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return FIXTURE_DIR


@pytest.fixture(scope="session")
def corpus(data_dir):
    return load_corpus(data_dir)


@pytest.fixture(scope="session")
def dev_questions(corpus, data_dir):
    return load_questions("dev", corpus, data_dir)


@pytest.fixture(scope="session")
def test_questions(corpus, data_dir):
    return load_questions("test", corpus, data_dir)
