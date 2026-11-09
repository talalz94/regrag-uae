"""Tokenizer for the lexical retriever."""

from __future__ import annotations

import re
from functools import lru_cache

import Stemmer
from bm25s.stopwords import STOPWORDS_EN

# Words and numbers, keeping dotted rule references such as "11.2.1" as one token,
# because questions often cite a rule by its number.
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*")

STOPWORDS: frozenset[str] = frozenset(STOPWORDS_EN)


@lru_cache(maxsize=1)
def _stemmer() -> Stemmer.Stemmer:
    return Stemmer.Stemmer("english")


def tokenize(text: str, *, stem: bool = True, stopwords: frozenset[str] = STOPWORDS) -> list[str]:
    """Lowercase, split, drop stopwords and (optionally) apply the Snowball stemmer."""
    tokens = [t for t in _TOKEN.findall(text.lower()) if t not in stopwords]
    if stem and tokens:
        tokens = _stemmer().stemWords(tokens)
    return tokens
