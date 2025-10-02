"""Tokenization utilities for toy language tasks."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from .vocab import SPECIAL_TOKENS, Vocabulary


@dataclass
class BaseTokenizer:
    vocab: Vocabulary

    def encode(self, text: str | Sequence[str]) -> List[int]:  # pragma: no cover - simple wrapper
        raise NotImplementedError

    def decode(self, ids: Iterable[int]) -> List[str]:
        return self.vocab.decode(ids)


@dataclass
class WordTokenizer(BaseTokenizer):
    lowercase: bool = False

    def encode(self, text: str | Sequence[str]) -> List[int]:
        if isinstance(text, str):
            tokens = text.strip().split()
        else:
            tokens = list(text)
        if self.lowercase:
            tokens = [token.lower() for token in tokens]
        return self.vocab.encode(tokens, add_bos=True, add_eos=True)


@dataclass
class CharTokenizer(BaseTokenizer):
    def encode(self, text: str | Sequence[str]) -> List[int]:
        if isinstance(text, str):
            tokens = list(text)
        else:
            tokens = list("".join(text))
        return self.vocab.encode(tokens, add_bos=True, add_eos=True)


@dataclass
class BPETokenizer(BaseTokenizer):
    merges: int = 1000

    def __post_init__(self) -> None:
        if len(self.vocab.id_to_token) <= len(SPECIAL_TOKENS):
            raise ValueError("BPE tokenizer requires a seed vocabulary")

    def train(self, corpus: Iterable[str]) -> None:
        vocab = Counter()
        for text in corpus:
            word = " ".join(list(text.strip())) + "</w>"
            vocab[word] += 1

        merges_done = 0
        while merges_done < self.merges:
            pairs = Counter()
            for word, freq in vocab.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pairs[(symbols[i], symbols[i + 1])] += freq
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            pattern = " ".join(best)
            replacement = "".join(best)
            new_vocab = Counter()
            for word, freq in vocab.items():
                new_word = word.replace(pattern, replacement)
                new_vocab[new_word] += freq
            vocab = new_vocab
            self.vocab.add_token(replacement)
            merges_done += 1

    def encode(self, text: str | Sequence[str]) -> List[int]:
        if isinstance(text, str):
            tokens = text.strip().split()
        else:
            tokens = list(text)
        bpe_tokens: List[str] = []
        for token in tokens:
            word = list(token) + ["</w>"]
            while len(word) > 1:
                pair_scores = {
                    (word[i], word[i + 1]): self.vocab.token_to_id.get(word[i] + word[i + 1], float("inf"))
                    for i in range(len(word) - 1)
                }
                best_pair = min(pair_scores, key=pair_scores.get)
                if pair_scores[best_pair] == float("inf"):
                    break
                i = 0
                new_word: List[str] = []
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                        new_word.append(word[i] + word[i + 1])
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                word = new_word
            bpe_tokens.extend([w for w in word if w != "</w>"])
        return self.vocab.encode(bpe_tokens, add_bos=True, add_eos=True)


def save_tokenizer(tokenizer: BaseTokenizer, path: str | Path) -> None:
    data = {
        "vocab": tokenizer.vocab.to_dict(),
        "config": {
            "type": tokenizer.__class__.__name__,
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
    tokenizer_type = data["config"]["type"]
    if tokenizer_type == "WordTokenizer":
        return WordTokenizer(vocab=vocab)
    if tokenizer_type == "CharTokenizer":
        return CharTokenizer(vocab=vocab)
    if tokenizer_type == "BPETokenizer":
        return BPETokenizer(vocab=vocab)
    raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
