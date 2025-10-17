from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class MaskedMultiheadAttention(nn.Module):
    """Multi-head attention layer with support for masking individual heads."""

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.head_mask = nn.Parameter(torch.ones(num_heads), requires_grad=False)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        batch_size, q_len, _ = query.size()
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        def reshape(x: Tensor) -> Tensor:
            return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        q = reshape(q)
        k = reshape(k)
        v = reshape(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            scores += attn_mask
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2).to(dtype=scores.dtype) * -1e9
            scores = scores + mask
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        head_mask = self.head_mask.view(1, self.num_heads, 1, 1).to(attn.device)
        attn = attn * head_mask

        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        output = self.out_proj(context)
        return output, attn

    def disable_heads(self, head_indices: list[int]) -> None:
        mask = self.head_mask.detach().clone()
        for idx in head_indices:
            if 0 <= idx < self.num_heads:
                mask[idx] = 0.0
        self.head_mask.copy_(mask)
