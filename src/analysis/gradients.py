from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from src.models import ToyTransformer, generate_causal_mask


def compute_gradient_attributions(
    model: ToyTransformer,
    src_tokens: Tensor,
    tgt_tokens: Tensor,
    src_padding_mask: Tensor,
    tgt_padding_mask: Tensor,
    pad_idx: int,
) -> Tuple[Tensor, Tensor]:
    model.zero_grad()
    src_tokens = src_tokens.clone().detach().requires_grad_(True)
    tgt_tokens = tgt_tokens.clone().detach().requires_grad_(True)
    causal_mask = generate_causal_mask(tgt_tokens.size(1), device=tgt_tokens.device)
    outputs = model(
        src_tokens=src_tokens,
        tgt_tokens=tgt_tokens,
        src_key_padding_mask=src_padding_mask,
        tgt_mask=causal_mask,
        tgt_key_padding_mask=tgt_padding_mask,
    )
    logits = outputs["logits"]
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        tgt_tokens[:, 1:].reshape(-1),
        ignore_index=pad_idx,
    )
    loss.backward()
    return src_tokens.grad.detach(), tgt_tokens.grad.detach()


def integrated_gradients(
    model: ToyTransformer,
    baseline_src: Tensor,
    baseline_tgt: Tensor,
    src_tokens: Tensor,
    tgt_tokens: Tensor,
    src_padding_mask: Tensor,
    tgt_padding_mask: Tensor,
    pad_idx: int,
    steps: int = 50,
) -> Tuple[Tensor, Tensor]:
    src_integrated = torch.zeros_like(src_tokens, dtype=torch.float)
    tgt_integrated = torch.zeros_like(tgt_tokens, dtype=torch.float)
    for alpha in torch.linspace(0, 1, steps):
        src_interp = baseline_src + alpha * (src_tokens - baseline_src)
        tgt_interp = baseline_tgt + alpha * (tgt_tokens - baseline_tgt)
        src_interp.requires_grad_(True)
        tgt_interp.requires_grad_(True)
        causal_mask = generate_causal_mask(tgt_interp.size(1), device=tgt_interp.device)
        outputs = model(
            src_tokens=src_interp,
            tgt_tokens=tgt_interp,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )
        logits = outputs["logits"]
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            tgt_interp[:, 1:].reshape(-1),
            ignore_index=pad_idx,
        )
        loss.backward()
        src_integrated += src_interp.grad.detach()
        tgt_integrated += tgt_interp.grad.detach()
        model.zero_grad()
    src_attr = (src_tokens - baseline_src) * src_integrated / steps
    tgt_attr = (tgt_tokens - baseline_tgt) * tgt_integrated / steps
    return src_attr, tgt_attr
