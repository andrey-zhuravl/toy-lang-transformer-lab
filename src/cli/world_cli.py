"""Command line interface for toy world dataset generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

from ..data_generation.world_tasks import load_and_build
from ..vocabulary import build_vocabulary_from_files


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


def _collect_dict_paths(data_config: dict, output_dir: Path) -> List[Path]:
    configured_paths = data_config.get("dict_paths") or []
    paths: List[Path] = []
    for raw_path in configured_paths:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        if candidate.exists():
            paths.append(candidate)

    if not paths:
        default_dir = output_dir.parent / "dicts"
        if default_dir.exists():
            paths.extend(sorted(default_dir.glob("*.txt")))

    if not paths:
        fallback_dir = Path.cwd() / "data" / "dicts"
        if fallback_dir.exists():
            paths.extend(sorted(fallback_dir.glob("*.txt")))

    # Deduplicate while preserving order
    seen: set[Path] = set()
    unique_paths: List[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(path)
    return unique_paths


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


def _maybe_build_vocabulary(dict_paths: List[Path], vocab_path: Path) -> Optional[Path]:
    if not dict_paths:
        return None

    vocabulary = build_vocabulary_from_files(dict_paths)
    vocabulary.save(vocab_path)
    return vocab_path


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
    dict_paths = _collect_dict_paths(data_config, args.out_dir)
    vocab_path = _maybe_build_vocabulary(dict_paths, _resolve_vocab_output(data_config, args.out_dir))

    manifest = {
        "world_config": str(args.world_config),
        "tasks": tasks,
        "splits": {key: float(value) for key, value in splits.items()},
        "datasets": {task: {split: str(path) for split, path in mapping.items()} for task, mapping in written.items()},
    }
    if dict_paths:
        manifest["dict_paths"] = [str(path) for path in dict_paths]
    if vocab_path is not None:
        manifest["vocab_path"] = str(vocab_path)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "generate":
        command_generate(args)
    else:  # pragma: no cover - defensive programming
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    main()
