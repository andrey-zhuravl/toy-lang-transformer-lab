from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class TransformerConfig:
    """Configuration parameters for :class:`ToyTransformer`."""

    vocab_size: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    ffn_dim: int = 256
    dropout: float = 0.1
    architecture: str = "encoder_decoder"
    disabled_layers: Optional[List[int]] = None
    disabled_heads: Optional[Dict[int, List[int]]] = None
    max_seq_len: int = 128
