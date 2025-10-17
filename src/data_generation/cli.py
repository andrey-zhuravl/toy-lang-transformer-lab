from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import generate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic datasets")
    parser.add_argument(
        "--config",
        default=Path(__file__).resolve().parent / "config.yaml",
        type=Path,
        help="Path to generator configuration",
    )
    parser.add_argument(
        "--base-dir",
        default=Path(__file__).resolve().parents[2],
        type=Path,
        help="Project root containing data/ subdirectories",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_dataset(args.base_dir, args.config)


if __name__ == "__main__":  # pragma: no cover
    main()
