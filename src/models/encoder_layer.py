from __future__ import annotations

from typing import Optional, Tuple

from torch import Tensor, nn

from .config import TransformerConfig
from .masked_multihead_attention import MaskedMultiheadAttention


class TransformerEncoderLayer(nn.Module):
    """Single encoder block used by :class:`ToyTransformer`."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.self_attn = MaskedMultiheadAttention(config.d_model, config.n_heads, config.dropout)
        self.linear1 = nn.Linear(config.d_model, config.ffn_dim)
        self.linear2 = nn.Linear(config.ffn_dim, config.d_model)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()
        self.enabled = True

    def forward(self, src: Tensor, src_key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        if not self.enabled:
            zeros = src.new_zeros(
                src.size(0), self.self_attn.num_heads, src.size(1), src.size(1)
            )
            return src, zeros
        attn_output, attn_weights = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout(attn_output)
        src = self.norm1(src)
        ffn_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(ffn_output)
        src = self.norm2(src)
        return src, attn_weights
