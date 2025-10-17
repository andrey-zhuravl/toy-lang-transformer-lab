from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass
class AblationConfig:
    """Configuration for structured ablations."""

    disable_layers: List[int] | None = None
    disable_heads: Dict[int, List[int]] | None = None
    freeze_modules: Iterable[str] | None = None
    dropout_overrides: Dict[str, float] | None = None
