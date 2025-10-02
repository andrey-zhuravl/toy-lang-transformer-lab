"""Synthetic dataset generator for toy language experiments."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .vocab import Vocabulary, build_vocabulary_from_files


@dataclass
class Grammar:
    """Simple context-free grammar parser for rule-based generation."""

    rules: Dict[str, List[List[str]]] = field(default_factory=dict)
    start_symbol: str = "S"

    @classmethod
    def from_file(cls, path: str | Path) -> "Grammar":
        rules: Dict[str, List[List[str]]] = {}
        start_symbol = "S"
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("start:" ):
                start_symbol = line.split(":", 1)[1].strip()
                continue
            if "->" not in line:
                raise ValueError(f"Invalid grammar line: {line}")
            lhs, rhs = [part.strip() for part in line.split("->", 1)]
            expansions = [exp.strip().split() for exp in rhs.split("|")]
            rules.setdefault(lhs, []).extend(expansions)
        if start_symbol not in rules:
            raise ValueError("Start symbol must be defined in grammar rules")
        return cls(rules=rules, start_symbol=start_symbol)

    def sample(self, symbol: str | None = None) -> Tuple[str, Dict[str, List[str]]]:
        symbol = symbol or self.start_symbol
        production = random.choice(self.rules.get(symbol, [[symbol]]))
        sentence_parts: List[str] = []
        semantics: Dict[str, List[str]] = {symbol: []}
        for sym in production:
            if sym in self.rules:
                generated, sub_sem = self.sample(sym)
                sentence_parts.append(generated)
                semantics[symbol].append(sym)
                semantics.update(sub_sem)
            else:
                sentence_parts.append(sym)
                semantics[symbol].append(sym)
        return " ".join(sentence_parts), semantics


@dataclass
class GenerationConfig:
    task: str
    split: str
    num_samples: int
    output_path: Path
    dict_paths: Sequence[Path]
    grammar_path: Path
    seed: int = 13
    max_length: int | None = None
    include_oov: bool = False


class DataGenerator:
    def __init__(self, vocab: Vocabulary, grammar: Grammar, config: GenerationConfig) -> None:
        self.vocab = vocab
        self.grammar = grammar
        self.config = config
        random.seed(config.seed)

    def _write_samples(self, samples: Iterable[Dict[str, str]]) -> None:
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.output_path.open("w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    def _generate_lm(self) -> List[Dict[str, str]]:
        samples = []
        candidate_tokens = [token for token in self.vocab.id_to_token if not token.startswith("<")]
        if not candidate_tokens:
            candidate_tokens = list(self.vocab.id_to_token)
        for _ in range(self.config.num_samples):
            sentence, _ = self.grammar.sample()
            if self.config.max_length and len(sentence.split()) > self.config.max_length:
                continue
            next_token = random.choice(candidate_tokens)
            samples.append({"input": sentence, "output": next_token})
        return samples

    def _generate_parsing(self) -> List[Dict[str, str]]:
        samples = []
        for _ in range(self.config.num_samples):
            sentence, semantics = self.grammar.sample()
            if self.config.max_length and len(sentence.split()) > self.config.max_length:
                continue
            samples.append({"input": sentence, "output": json.dumps(semantics, ensure_ascii=False)})
        return samples

    def _generate_sem_to_nl(self) -> List[Dict[str, str]]:
        samples = []
        for _ in range(self.config.num_samples):
            sentence, semantics = self.grammar.sample()
            samples.append({"input": json.dumps(semantics, ensure_ascii=False), "output": sentence})
        return samples

    def _generate_copy(self) -> List[Dict[str, str]]:
        samples = []
        vocab_tokens = [token for token in self.vocab.id_to_token if token.isalpha()]
        for _ in range(self.config.num_samples):
            length = random.randint(2, 6)
            tokens = random.choices(vocab_tokens, k=length)
            input_seq = " ".join(tokens)
            if self.config.include_oov:
                tokens[-1] = tokens[-1] + "_OOV"
            samples.append({"input": input_seq, "output": " ".join(tokens)})
        return samples

    def generate(self) -> None:
        if self.config.task == "lm":
            samples = self._generate_lm()
        elif self.config.task == "parsing":
            samples = self._generate_parsing()
        elif self.config.task in {"sem_to_nl", "sem2nl"}:
            samples = self._generate_sem_to_nl()
        elif self.config.task == "copy":
            samples = self._generate_copy()
        else:
            raise ValueError(f"Unknown task: {self.config.task}")
        self._write_samples(samples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toy language dataset generator")
    parser.add_argument("--task", required=True, choices=["lm", "parsing", "sem_to_nl", "copy"])
    parser.add_argument("--split", required=True, choices=["train", "val", "test", "oov"])
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--dict", dest="dicts", action="append", default=["data/dicts/base.txt"], help="Paths to dictionary files")
    parser.add_argument("--grammar", default="data/grammar/rules.txt")
    parser.add_argument("--output", default=None, help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--include-oov", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or f"data/datasets/{args.split}.jsonl"
    dict_paths = [Path(path) for path in args.dicts if Path(path).exists()]
    if not dict_paths:
        raise FileNotFoundError("No dictionary files found")
    vocab = build_vocabulary_from_files(dict_paths)
    grammar = Grammar.from_file(args.grammar)
    config = GenerationConfig(
        task=args.task,
        split=args.split,
        num_samples=args.num_samples,
        output_path=Path(output),
        dict_paths=dict_paths,
        grammar_path=Path(args.grammar),
        seed=args.seed,
        max_length=args.max_length,
        include_oov=args.include_oov,
    )
    generator = DataGenerator(vocab=vocab, grammar=grammar, config=config)
    generator.generate()


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
