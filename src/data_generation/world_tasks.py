"""Dataset builders for the toy world generator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from ..world.episode import Episode, generate_episodes
from ..world.queries import (
    build_effect_examples,
    build_lm_examples,
    build_parsing_examples,
    build_qa_examples,
    build_sem2nl_examples,
)
from ..world.config import WorldConfig, load_world_config


DATASET_BUILDERS = {
    "world_qa": build_qa_examples,
    "world_parsing": build_parsing_examples,
    "world_sem2nl": build_sem2nl_examples,
    "world_effect": build_effect_examples,
    "world_log_lm": build_lm_examples,
}


def generate_world_episodes(world_config: WorldConfig, num_episodes: int, seed: Optional[int] = None) -> List[Episode]:
    return generate_episodes(world_config, num_episodes=num_episodes, seed=seed)


def build_datasets(
    episodes: Sequence[Episode],
    tasks: Iterable[str],
    output_dir: Path,
    splits: Mapping[str, float],
    seed: Optional[int] = None,
) -> Dict[str, Dict[str, Path]]:
    """Build datasets and write them to disk.

    Returns a mapping from dataset name to a mapping of split name -> path.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = None
    if seed is not None:
        import random

        rng = random.Random(seed)

    written_files: Dict[str, Dict[str, Path]] = {}
    for task in tasks:
        if task not in DATASET_BUILDERS:
            continue
        builder = DATASET_BUILDERS[task]
        examples = builder(episodes)
        if not examples:
            continue
        if rng:
            rng.shuffle(examples)

        split_paths: Dict[str, Path] = {}
        offset = 0
        total = len(examples)
        split_items = list(splits.items())
        for idx, (split_name, ratio) in enumerate(split_items):
            if idx == len(split_items) - 1:
                end = total
            else:
                end = offset + int(total * ratio)
            split_examples = examples[offset:end]
            offset = end
            path = output_dir / f"{task}_{split_name}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for example in split_examples:
                    handle.write(json.dumps(example, ensure_ascii=False) + "\n")
            split_paths[split_name] = path
        written_files[task] = split_paths
    return written_files


def load_and_build(
    world_config_path: Path,
    num_episodes: int,
    tasks: Iterable[str],
    output_dir: Path,
    splits: Mapping[str, float],
    seed: Optional[int] = None,
) -> Dict[str, Dict[str, Path]]:
    world_config = load_world_config(world_config_path)
    episodes = generate_world_episodes(world_config, num_episodes=num_episodes, seed=seed)
    return build_datasets(episodes, tasks=tasks, output_dir=output_dir, splits=splits, seed=seed)
