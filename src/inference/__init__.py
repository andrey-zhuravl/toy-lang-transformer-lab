"""Inference helpers."""

from .cli import main
from .greedy_decoder import GreedyDecoder

__all__ = ["GreedyDecoder", "main"]
