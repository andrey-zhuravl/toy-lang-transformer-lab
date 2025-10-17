from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.vocabulary import Vocabulary

from .base import BaseTokenizer


@dataclass
class WordTokenizer(BaseTokenizer):
    """Whitespace tokenizer with optional lowercasing."""

    lowercase: bool = False

    def __init__(self, vocab: Vocabulary, lowercase: bool = False) -> None:
        super().__init__(vocab)
        self.lowercase = lowercase

    def encode(self, text: str | Sequence[str]) -> list[int]:
        if isinstance(text, str):
            tokens = text.strip().split()
        else:
            tokens = list(text)
        if self.lowercase:
            tokens = [token.lower() for token in tokens]
        return self.vocab.encode(tokens, add_bos=True, add_eos=True)
