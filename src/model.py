"""Minimal yet flexible transformer model for toy language research."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn


@dataclass
class TransformerConfig:
    vocab_size: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    ffn_dim: int = 256
    dropout: float = 0.1
    architecture: str = "encoder_decoder"  # or "encoder_only"
    disabled_layers: Optional[List[int]] = None
    disabled_heads: Optional[Dict[int, List[int]]] = None
    max_seq_len: int = 128


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MaskedMultiheadAttention(nn.Module):
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
        _, k_len, _ = key.size()
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

    def disable_heads(self, head_indices: List[int]) -> None:
        mask = self.head_mask.detach().clone()
        for idx in head_indices:
            if 0 <= idx < self.num_heads:
                mask[idx] = 0.0
        self.head_mask.copy_(mask)


class TransformerEncoderLayer(nn.Module):
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
            return src, torch.zeros(src.size(0), self.self_attn.num_heads, src.size(1), src.size(1), device=src.device)
        attn_output, attn_weights = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout(attn_output)
        src = self.norm1(src)
        ffn_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(ffn_output)
        src = self.norm2(src)
        return src, attn_weights


class TransformerDecoderLayer(nn.Module):
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
            shape = (tgt.size(0), self.self_attn.num_heads, tgt.size(1), tgt.size(1))
            cross_shape = (tgt.size(0), self.cross_attn.num_heads, tgt.size(1), memory.size(1))
            zero_attn = torch.zeros(shape, device=tgt.device)
            zero_cross = torch.zeros(cross_shape, device=tgt.device)
            return tgt, {"self": zero_attn, "cross": zero_cross}
        tgt2, self_attn = self.self_attn(tgt, tgt, tgt, key_padding_mask=tgt_key_padding_mask, attn_mask=tgt_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        tgt2, cross_attn = self.cross_attn(tgt, memory, memory, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        ffn_output = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(ffn_output)
        tgt = self.norm3(tgt)
        return tgt, {"self": self_attn, "cross": cross_attn}


class ToyTransformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_encoding = PositionalEncoding(config.d_model, config.dropout, config.max_seq_len)
        self.dropout = nn.Dropout(config.dropout)
        self.generator = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.n_layers)])
        if config.architecture == "encoder_decoder":
            self.decoder_layers = nn.ModuleList([TransformerDecoderLayer(config) for _ in range(config.n_layers)])
        else:
            self.decoder_layers = None

        # Weight tying
        self.generator.weight = self.embedding.weight

        self.apply(self._init_weights)
        self._apply_masks()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _apply_masks(self) -> None:
        disabled_layers = set(self.config.disabled_layers or [])
        for idx, layer in enumerate(self.encoder_layers):
            layer.enabled = idx not in disabled_layers
            if self.config.disabled_heads and idx in self.config.disabled_heads:
                layer.self_attn.disable_heads(self.config.disabled_heads[idx])
        if self.decoder_layers is not None:
            for idx, layer in enumerate(self.decoder_layers):
                layer.enabled = idx not in disabled_layers
                if self.config.disabled_heads and idx in self.config.disabled_heads:
                    heads = self.config.disabled_heads[idx]
                    layer.self_attn.disable_heads(heads)
                    layer.cross_attn.disable_heads(heads)

    def encode(self, src_tokens: Tensor, src_key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, List[Tensor]]:
        src_emb = self.embedding(src_tokens)
        src_emb = self.positional_encoding(src_emb)
        src = self.dropout(src_emb)
        attentions: List[Tensor] = []
        for layer in self.encoder_layers:
            src, attn = layer(src, src_key_padding_mask=src_key_padding_mask)
            attentions.append(attn)
        return src, attentions

    def decode(
        self,
        tgt_tokens: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Dict[str, Tensor]]]:
        if self.decoder_layers is None:
            raise RuntimeError("Decoder requested for encoder-only model")
        tgt_emb = self.embedding(tgt_tokens)
        tgt_emb = self.positional_encoding(tgt_emb)
        tgt = self.dropout(tgt_emb)
        attentions: List[Dict[str, Tensor]] = []
        for layer in self.decoder_layers:
            tgt, attn = layer(
                tgt,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            attentions.append(attn)
        return tgt, attentions

    def forward(
        self,
        src_tokens: Tensor,
        tgt_tokens: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor | List[Tensor] | List[Dict[str, Tensor]]]:
        memory, encoder_attn = self.encode(src_tokens, src_key_padding_mask=src_key_padding_mask)
        outputs: Dict[str, Tensor | List[Tensor] | List[Dict[str, Tensor]]] = {
            "encoder_states": memory,
            "encoder_attentions": encoder_attn,
        }
        if self.decoder_layers is not None and tgt_tokens is not None:
            decoder_states, decoder_attn = self.decode(
                tgt_tokens,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )
            logits = self.generator(decoder_states)
            outputs.update({
                "logits": logits,
                "decoder_states": decoder_states,
                "decoder_attentions": decoder_attn,
            })
        else:
            logits = self.generator(memory)
            outputs["logits"] = logits
        return outputs


def build_padding_mask(tokens: Tensor, pad_idx: int) -> Tensor:
    return tokens.eq(pad_idx)


def generate_causal_mask(sz: int, device: torch.device | None = None) -> Tensor:
    mask = torch.triu(torch.ones(sz, sz, device=device) * float("-inf"), diagonal=1)
    return mask
