from __future__ import annotations

from typing import Dict, Optional, Tuple

from torch import Tensor, nn

from .config import TransformerConfig
from .masked_multihead_attention import MaskedMultiheadAttention


class TransformerDecoderLayer(nn.Module):
    """Single decoder block used by :class:`ToyTransformer`."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.self_attn = MaskedMultiheadAttention(config.d_model, config.n_heads, config.dropout)
        self.cross_attn = MaskedMultiheadAttention(config.d_model, config.n_heads, config.dropout)
        self.linear1 = nn.Linear(config.d_model, config.ffn_dim)
        self.linear2 = nn.Linear(config.ffn_dim, config.d_model)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()
        self.enabled = True

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if not self.enabled:
            zero_self = tgt.new_zeros(tgt.size(0), self.self_attn.num_heads, tgt.size(1), tgt.size(1))
            zero_cross = tgt.new_zeros(
                tgt.size(0), self.cross_attn.num_heads, tgt.size(1), memory.size(1)
            )
            return tgt, {"self": zero_self, "cross": zero_cross}
        tgt2, self_attn = self.self_attn(
            tgt,
            tgt,
            tgt,
            key_padding_mask=tgt_key_padding_mask,
            attn_mask=tgt_mask,
        )
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        tgt2, cross_attn = self.cross_attn(
            tgt,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        ffn_output = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(ffn_output)
        tgt = self.norm3(tgt)
        return tgt, {"self": self_attn, "cross": cross_attn}
