from __future__ import annotations

from src.models import ToyTransformer

from .config import AblationConfig
from .operations import apply_head_ablation, apply_layer_ablation, freeze_parameters, override_dropout


def apply_ablation(model: ToyTransformer, config: AblationConfig) -> ToyTransformer:
    if config.disable_layers:
        apply_layer_ablation(model, config.disable_layers)
    if config.disable_heads:
        apply_head_ablation(model, config.disable_heads)
    if config.freeze_modules:
        freeze_parameters(model, config.freeze_modules)
    if config.dropout_overrides:
        override_dropout(model, config.dropout_overrides)
    return model
