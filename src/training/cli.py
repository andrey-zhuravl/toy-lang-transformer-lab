from __future__ import annotations

import argparse

from .config import load_config
from .loop import train_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train toy transformer model")
    parser.add_argument("--config", default="src/config.yaml", help="Path to training YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_loop(config)


if __name__ == "__main__":  # pragma: no cover - CLI helper
    main()
