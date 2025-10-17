from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch

from src.models import ToyTransformer, TransformerConfig


def load_model(checkpoint_path: str | Path) -> Tuple[ToyTransformer, Dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    vocab_size = checkpoint.get("vocab_size") or model_cfg.get("vocab_size")
    if vocab_size is None:
        raise KeyError("Checkpoint must contain `vocab_size` field")
    transformer_cfg = TransformerConfig(
        vocab_size=vocab_size,
        d_model=model_cfg.get("d_model", 128),
        n_heads=model_cfg.get("n_heads", 4),
        n_layers=model_cfg.get("n_layers", 4),
        ffn_dim=model_cfg.get("ffn_dim", 256),
        dropout=model_cfg.get("dropout", 0.1),
        architecture=model_cfg.get("architecture", "encoder_decoder"),
        disabled_layers=model_cfg.get("disabled_layers"),
        disabled_heads=model_cfg.get("disabled_heads"),
        max_seq_len=data_cfg.get("max_seq_len", 128),
    )
    model = ToyTransformer(transformer_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config
