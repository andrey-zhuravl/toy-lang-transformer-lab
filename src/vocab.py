"""Utilities for managing vocabularies used by toy language datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

SPECIAL_TOKENS = {
    "pad": "<PAD>",
    "bos": "<BOS>",
    "eos": "<EOS>",
    "unk": "<UNK>",
    "span": "<SPAN>",
}


@dataclass
class Vocabulary:
    """Bidirectional mapping between tokens and ids.

    The class stores counts for optional pruning and exposes helpers for
    serialization. Special tokens are automatically added if not provided in the
    initial set.
    """

    tokens: Iterable[str] = field(default_factory=list)
    add_special_tokens: bool = True
    token_to_id: Dict[str, int] = field(init=False, default_factory=dict)
    id_to_token: List[str] = field(init=False, default_factory=list)
    counts: Dict[str, int] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.token_to_id = {}
        self.id_to_token = []
        if self.add_special_tokens:
            for token in SPECIAL_TOKENS.values():
                self._add_token(token)
        for token in self.tokens:
            self._add_token(token)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.id_to_token)

    def _add_token(self, token: str) -> int:
        if token not in self.token_to_id:
            idx = len(self.id_to_token)
            self.token_to_id[token] = idx
            self.id_to_token.append(token)
            self.counts[token] = 0
        return self.token_to_id[token]

    def add_tokens(self, tokens: Iterable[str]) -> None:
        for token in tokens:
            self._add_token(token)

    def add_token(self, token: str) -> int:
        return self._add_token(token)

    def get_id(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]
        return self.token_to_id[SPECIAL_TOKENS["unk"]]

    def get_token(self, idx: int) -> str:
        if idx < 0 or idx >= len(self.id_to_token):
            raise IndexError(f"Token id {idx} is out of bounds")
        return self.id_to_token[idx]

    def encode(self, tokens: Iterable[str], add_bos: bool = False, add_eos: bool = True) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.get_id(SPECIAL_TOKENS["bos"]))
        ids.extend(self.get_id(token) for token in tokens)
        if add_eos:
            ids.append(self.get_id(SPECIAL_TOKENS["eos"]))
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> List[str]:
        tokens: List[str] = []
        special = set(SPECIAL_TOKENS.values()) if skip_special else set()
        for idx in ids:
            token = self.get_token(idx)
            if token in special and skip_special:
                continue
            tokens.append(token)
        return tokens

    def register_occurrence(self, token: str) -> None:
        self.counts.setdefault(token, 0)
        self.counts[token] += 1

    def trim_by_frequency(self, min_count: int) -> "Vocabulary":
        preserved = [token for token, count in self.counts.items() if count >= min_count]
        return Vocabulary(tokens=preserved, add_special_tokens=self.add_special_tokens)

    def to_dict(self) -> Dict[str, object]:
        return {
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token,
            "counts": self.counts,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(tokens=[], add_special_tokens=False)
        vocab.token_to_id = {token: int(idx) for token, idx in data["token_to_id"].items()}
        vocab.id_to_token = list(data["id_to_token"])
        vocab.counts = {token: int(count) for token, count in data.get("counts", {}).items()}
        return vocab

    def ensure_special_tokens(self, tokens: Optional[Dict[str, str]] = None) -> None:
        tokens = tokens or SPECIAL_TOKENS
        for token in tokens.values():
            if token not in self.token_to_id:
                self._add_token(token)


def build_vocabulary_from_files(paths: Iterable[str | Path], lowercase: bool = False) -> Vocabulary:
    tokens = []
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
