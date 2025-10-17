from __future__ import annotations

import json
from pathlib import Path

from src.vocabulary import Vocabulary

from .base import BaseTokenizer
from .bpe import BPETokenizer
from .char import CharTokenizer
from .word import WordTokenizer


def save_tokenizer(tokenizer: BaseTokenizer, path: str | Path) -> None:
    data = {
        "vocab": tokenizer.vocab.to_dict(),
        "config": {
            "type": tokenizer.__class__.__name__,
            "kwargs": {},
        },
    }
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_tokenizer(path: str | Path) -> BaseTokenizer:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    vocab = Vocabulary(tokens=[], add_special_tokens=False)
    vocab.token_to_id = {token: int(idx) for token, idx in data["vocab"]["token_to_id"].items()}
    vocab.id_to_token = list(data["vocab"]["id_to_token"])
    vocab.counts = {token: int(count) for token, count in data["vocab"].get("counts", {}).items()}
    tokenizer_type = data["config"].get("type")
    if tokenizer_type == "WordTokenizer":
        return WordTokenizer(vocab=vocab)
    if tokenizer_type == "CharTokenizer":
        return CharTokenizer(vocab=vocab)
    if tokenizer_type == "BPETokenizer":
        return BPETokenizer(vocab=vocab)
    raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
