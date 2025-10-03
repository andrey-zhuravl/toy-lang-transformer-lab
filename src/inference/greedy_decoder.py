from __future__ import annotations

from typing import List

import torch

from src.models import ToyTransformer, generate_causal_mask
from src.tokenization import BaseTokenizer
from src.vocabulary import SPECIAL_TOKENS, Vocabulary


class GreedyDecoder:
    """Simple greedy decoding helper for the toy transformer."""

    def __init__(
        self,
        model: ToyTransformer,
        vocab: Vocabulary,
        tokenizer: BaseTokenizer,
        max_length: int = 64,
        device: torch.device | None = None,
    ) -> None:
        self.model = model.to(device or torch.device("cpu"))
        self.model.eval()
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device or torch.device("cpu")
        self.pad_idx = vocab.get_id(SPECIAL_TOKENS["pad"])
        self.bos_idx = vocab.get_id(SPECIAL_TOKENS["bos"])
        self.eos_idx = vocab.get_id(SPECIAL_TOKENS["eos"])

    @torch.no_grad()
    def generate(self, prompt: str) -> List[str]:
        src_ids = self.tokenizer.encode(prompt)
        src_tensor = torch.tensor([src_ids], device=self.device)
        src_padding = src_tensor.eq(self.pad_idx)

        decoded = torch.tensor([[self.bos_idx]], device=self.device)
        for _ in range(self.max_length):
            tgt_mask = generate_causal_mask(decoded.size(1), device=self.device)
            outputs = self.model(
                src_tokens=src_tensor,
                tgt_tokens=decoded,
                src_key_padding_mask=src_padding,
                tgt_mask=tgt_mask,
            )
            logits = outputs["logits"][:, -1, :]
            next_token = torch.argmax(logits, dim=-1)
            decoded = torch.cat([decoded, next_token.unsqueeze(1)], dim=1)
            if next_token.item() == self.eos_idx:
                break

        token_ids = decoded.squeeze(0).tolist()
        token_ids = [tid for tid in token_ids if tid not in {self.bos_idx, self.pad_idx}]
        if token_ids and token_ids[-1] == self.eos_idx:
            token_ids = token_ids[:-1]
        return self.vocab.decode(token_ids, skip_special=True)
