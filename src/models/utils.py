from __future__ import annotations

import torch
from torch import Tensor


def build_padding_mask(tokens: Tensor, pad_idx: int) -> Tensor:
    """Create mask marking padding tokens with ones."""

    return tokens.eq(pad_idx)


def generate_causal_mask(sz: int, device: torch.device | None = None) -> Tensor:
    """Upper triangular mask with ``-inf`` above the main diagonal."""

    return torch.triu(torch.ones(sz, sz, device=device) * float("-inf"), diagonal=1)
