from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .vocabulary import Vocabulary


def build_vocabulary_from_files(paths: Iterable[str | Path], lowercase: bool = False) -> Vocabulary:
    """Construct a :class:`Vocabulary` from newline-separated word lists."""

    tokens: list[str] = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                token = line.strip()
                if not token:
                    continue
                if lowercase:
                    token = token.lower()
                tokens.append(token)
    return Vocabulary(tokens=tokens)
