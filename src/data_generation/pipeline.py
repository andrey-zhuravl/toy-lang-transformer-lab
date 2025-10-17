from __future__ import annotations

import random
from pathlib import Path

from .config import GeneratorConfig
from .grammar import Grammar
from .lexicon import Lexicon
from .semantics import SemanticsGenerator
from .surface_realizer import SurfaceRealizer
from .writer import DatasetWriter


def generate_dataset(base_dir: Path, config_path: Path) -> None:
    config = GeneratorConfig.load(config_path)
    if config.random_seed is not None:
        random.seed(config.random_seed)

    lexicon = Lexicon(base_dir / "data/dicts")
    grammar = Grammar(base_dir / "data/grammar")
    semantics_gen = SemanticsGenerator(lexicon, grammar)
    realizer = SurfaceRealizer(grammar, lexicon, config.max_sentence_length)
    writer = DatasetWriter(base_dir / "data/datasets", config.tasks)

    generated = 0
    attempts = 0
    max_attempts = config.dataset_size * 10
    while generated < config.dataset_size and attempts < max_attempts:
        attempts += 1
        semantics = semantics_gen.generate()
        try:
            tokens = realizer.realize(semantics)
        except (ValueError, RuntimeError):
            continue
        if not tokens:
            continue
        writer.add_entry(tokens, semantics)
        generated += 1

    writer.flush()
