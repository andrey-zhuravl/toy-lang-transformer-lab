from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from src.vocabulary import Vocabulary


@dataclass
class BaseTokenizer:
    """Base tokenizer interface for the project."""

    vocab: Vocabulary

    def encode(self, text: str | Sequence[str]) -> list[int]:  # pragma: no cover - abstract
        raise NotImplementedError

    def decode(self, ids: Iterable[int]) -> list[str]:
        return self.vocab.decode(ids)
