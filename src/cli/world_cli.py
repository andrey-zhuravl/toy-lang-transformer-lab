"""Command line interface for toy world dataset generation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from ..data_generation.world_tasks import load_and_build
from ..vocabulary import Vocabulary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toy world dataset generator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate world episodes and datasets")
    generate_parser.add_argument("--world-config", type=Path, required=True, help="Path to the world YAML config")
    generate_parser.add_argument("--data-config", type=Path, required=False, help="Path to the main data config")
    generate_parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for datasets")
    generate_parser.add_argument("--episodes", type=int, default=None, help="Number of episodes to sample")
    generate_parser.add_argument(
        "--export",
        choices=["qa", "parsing", "sem2nl", "effect", "lm", "all"],
        default="all",
        help="Which datasets to export",
    )
    generate_parser.add_argument("--seed", type=int, default=None, help="Random seed")

    vocab_parser = subparsers.add_parser("build-vocab", help="Build vocabulary from generated world datasets")
    vocab_parser.add_argument("--manifest", type=Path, required=True, help="Path to manifest_world.json")
    vocab_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the resulting vocabulary JSON (defaults to manifest value or alongside manifest)",
    )

    return parser.parse_args(argv)


def _load_data_config(path: Path | None) -> dict:
    if path is None:
        return {}
    try:
        import yaml
    except Exception:  # pragma: no cover - optional dependency
        return {}

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            return {}
        return data


def _resolve_vocab_output(data_config: dict, output_dir: Path) -> Path:
    raw_path = data_config.get("vocab_path")
    if raw_path:
        vocab_path = Path(raw_path)
        if not vocab_path.is_absolute():
            vocab_path = Path.cwd() / vocab_path
    else:
        vocab_path = output_dir.parent / "vocab.json"
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    return vocab_path


TOKEN_PATTERN = re.compile(r"[\w]+", re.UNICODE)


def _iter_strings(value: object) -> Iterable[str]:
    if value is None:
        return
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                yield key
            else:
                yield str(key)
            yield from _iter_strings(item)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_strings(item)
        return
    yield str(value)


def _tokens_from_value(value: object) -> List[str]:
    tokens: List[str] = []
    for text in _iter_strings(value):
        tokens.extend(match.group(0) for match in TOKEN_PATTERN.finditer(text))
    return tokens


def _collect_tokens_from_file(path: Path) -> List[str]:
    if not path.exists():
        return []
    tokens: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                tokens.extend(_tokens_from_value(payload.get("input")))
                tokens.extend(_tokens_from_value(payload.get("output")))
    return tokens


def _build_vocabulary_from_datasets(dataset_paths: Sequence[Path], vocab_path: Path) -> Optional[Path]:
    tokens: List[str] = []
    for dataset_path in dataset_paths:
        tokens.extend(_collect_tokens_from_file(dataset_path))
    if not tokens:
        return None
    vocabulary = Vocabulary(tokens=tokens)
    vocabulary.save(vocab_path)
    return vocab_path


def _gather_dataset_paths(datasets: Dict[str, Dict[str, Path]]) -> List[Path]:
    paths: List[Path] = []
    for split_mapping in datasets.values():
        for path in split_mapping.values():
            paths.append(path)
    return paths


def command_generate(args: argparse.Namespace) -> None:
    data_config = _load_data_config(args.data_config)
    default_tasks = ["world_qa", "world_parsing", "world_sem2nl", "world_effect"]
    tasks = list(data_config.get("world_tasks", [])) or [
        task for task in data_config.get("tasks", []) if task.startswith("world_")
    ]
    if not tasks:
        tasks = default_tasks
    if args.export and args.export != "all":
        export_mapping = {
            "qa": ["world_qa"],
            "parsing": ["world_parsing"],
            "sem2nl": ["world_sem2nl"],
            "effect": ["world_effect"],
            "lm": ["world_log_lm"],
        }
        tasks = export_mapping[args.export]

    splits = data_config.get("splits", {"train": 0.8, "val": 0.1, "test": 0.1})
    dataset_size = data_config.get("dataset_size", 100)
    num_episodes = args.episodes or dataset_size

    written = load_and_build(
        world_config_path=args.world_config,
        num_episodes=num_episodes,
        tasks=tasks,
        output_dir=args.out_dir,
        splits=splits,
        seed=args.seed,
    )

    manifest_path = args.out_dir / "manifest_world.json"
    dataset_paths = _gather_dataset_paths(written)
    vocab_path = _build_vocabulary_from_datasets(dataset_paths, _resolve_vocab_output(data_config, args.out_dir))

    manifest = {
        "world_config": str(args.world_config),
        "tasks": tasks,
        "splits": {key: float(value) for key, value in splits.items()},
        "datasets": {task: {split: str(path) for split, path in mapping.items()} for task, mapping in written.items()},
    }
    if vocab_path is not None:
        manifest["vocab_path"] = str(vocab_path)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def command_build_vocab(args: argparse.Namespace) -> None:
    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest file not found: {args.manifest}")
    with args.manifest.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    datasets = manifest.get("datasets", {})
    dataset_paths = []
    for task_mapping in datasets.values():
        for raw_path in task_mapping.values():
            path = Path(raw_path)
            if not path.is_absolute():
                path = args.manifest.parent / path
            dataset_paths.append(path)
    if not dataset_paths:
        raise ValueError("Manifest does not contain dataset paths to build vocabulary")

    if args.output is not None:
        output_path = args.output
    else:
        raw_output = manifest.get("vocab_path")
        if raw_output:
            output_path = Path(raw_output)
            if not output_path.is_absolute():
                output_path = args.manifest.parent / output_path
        else:
            output_path = args.manifest.parent / "vocab.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vocab_path = _build_vocabulary_from_datasets(dataset_paths, output_path)
    if vocab_path is None:
        raise ValueError("Failed to build vocabulary: dataset files did not yield any tokens")

    manifest["vocab_path"] = str(vocab_path)
    with args.manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "generate":
        command_generate(args)
    elif args.command == "build-vocab":
        command_build_vocab(args)
    else:  # pragma: no cover - defensive programming
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    main()
