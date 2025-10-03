"""Synthetic dataset generation pipeline."""

from .cli import main
from .config import GeneratorConfig
from .pipeline import generate_dataset

__all__ = ["GeneratorConfig", "generate_dataset", "main"]
