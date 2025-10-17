from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from src.vocabulary import SPECIAL_TOKENS, Vocabulary

from .base import BaseTokenizer


@dataclass
class BPETokenizer(BaseTokenizer):
    """Simple byte-pair encoding tokenizer."""

    merges: int = 1000

    def __init__(self, vocab: Vocabulary, merges: int = 1000) -> None:
        if len(vocab.id_to_token) <= len(SPECIAL_TOKENS):
            raise ValueError("BPE tokenizer requires a seed vocabulary")
        super().__init__(vocab)
        self.merges = merges

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

    def encode(self, text: str | Sequence[str]) -> list[int]:
        if isinstance(text, str):
            tokens = text.strip().split()
        else:
            tokens = list(text)
        bpe_tokens: list[str] = []
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
                new_word: list[str] = []
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
