from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .base import BaseTokenizer


@dataclass
class CharTokenizer(BaseTokenizer):
    """Character-level tokenizer."""

    def encode(self, text: str | Sequence[str]) -> list[int]:
        if isinstance(text, str):
            tokens = list(text)
        else:
            tokens = list("".join(text))
        return self.vocab.encode(tokens, add_bos=True, add_eos=True)
