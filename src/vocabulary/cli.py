from __future__ import annotations

import argparse
from pathlib import Path

from .builder import build_vocabulary_from_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build vocabulary from dictionary files")
    parser.add_argument("--inputs", nargs="+", help="Paths to text files with one token per line")
    parser.add_argument("--output", required=True, help="Path to save resulting vocabulary JSON")
    parser.add_argument("--lowercase", action="store_true", help="Lowercase tokens before adding")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vocab = build_vocabulary_from_files(args.inputs, lowercase=args.lowercase)
    vocab.save(Path(args.output))


if __name__ == "__main__":  # pragma: no cover
    main()
