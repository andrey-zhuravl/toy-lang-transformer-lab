"""Utilities for performing architectural ablations on the toy transformer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch

from .model import ToyTransformer


@dataclass
class AblationConfig:
    disable_layers: List[int] | None = None
    disable_heads: Dict[int, List[int]] | None = None
    freeze_modules: Iterable[str] | None = None
    dropout_overrides: Dict[str, float] | None = None


def apply_layer_ablation(model: ToyTransformer, layers: Iterable[int]) -> None:
    for idx in layers:
        if 0 <= idx < len(model.encoder_layers):
            model.encoder_layers[idx].enabled = False
        if model.decoder_layers is not None and 0 <= idx < len(model.decoder_layers):
            model.decoder_layers[idx].enabled = False


def apply_head_ablation(model: ToyTransformer, disabled_heads: Dict[int, List[int]]) -> None:
    for layer_idx, heads in disabled_heads.items():
        if 0 <= layer_idx < len(model.encoder_layers):
            model.encoder_layers[layer_idx].self_attn.disable_heads(heads)
        if model.decoder_layers is not None and 0 <= layer_idx < len(model.decoder_layers):
            model.decoder_layers[layer_idx].self_attn.disable_heads(heads)
            model.decoder_layers[layer_idx].cross_attn.disable_heads(heads)


def freeze_parameters(model: ToyTransformer, module_names: Iterable[str]) -> None:
    for name, param in model.named_parameters():
        if any(name.startswith(module) for module in module_names):
            param.requires_grad = False


def override_dropout(model: ToyTransformer, overrides: Dict[str, float]) -> None:
    for name, module in model.named_modules():
        if name in overrides and hasattr(module, "p"):
            module.p = overrides[name]


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
