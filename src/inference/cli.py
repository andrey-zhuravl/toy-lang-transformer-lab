from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.analysis.loader import load_model
from src.tokenization import CharTokenizer, WordTokenizer
from src.vocabulary import Vocabulary

from .greedy_decoder import GreedyDecoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run greedy inference with a trained model")
    parser.add_argument("checkpoint", type=Path, help="Path to model checkpoint (.pt)")
    parser.add_argument("vocab", type=Path, help="Path to vocabulary JSON")
    parser.add_argument("prompt", type=str, help="Input prompt to feed into the model")
    parser.add_argument("--tokenizer", choices=["word", "char"], default="word")
    parser.add_argument("--max-length", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, _ = load_model(args.checkpoint)
    vocab = Vocabulary.load(args.vocab)
    tokenizer = WordTokenizer(vocab=vocab) if args.tokenizer == "word" else CharTokenizer(vocab=vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = GreedyDecoder(model=model, vocab=vocab, tokenizer=tokenizer, max_length=args.max_length, device=device)
    tokens = decoder.generate(args.prompt)
    print("Generated:", " ".join(tokens))


if __name__ == "__main__":  # pragma: no cover
    main()
