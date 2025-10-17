from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import Tensor, nn

from .config import TransformerConfig
from .decoder_layer import TransformerDecoderLayer
from .encoder_layer import TransformerEncoderLayer
from .positional_encoding import PositionalEncoding


class ToyTransformer(nn.Module):
    """Configurable Transformer architecture for toy-language experiments."""

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

    def encode(self, src_tokens: Tensor, src_key_padding_mask: Optional[Tensor] = None) -> tuple[Tensor, List[Tensor]]:
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
    ) -> tuple[Tensor, List[Dict[str, Tensor]]]:
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
