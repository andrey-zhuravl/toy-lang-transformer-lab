from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
from torch import Tensor

from .sample import Sample


def pad_sequences(sequences: Iterable[List[int]], pad_idx: int) -> Tuple[Tensor, Tensor]:
    sequences = list(sequences)
    max_len = max(len(seq) for seq in sequences)
    batch = torch.full((len(sequences), max_len), pad_idx, dtype=torch.long)
    padding_mask = torch.ones((len(sequences), max_len), dtype=torch.bool)
    for i, seq in enumerate(sequences):
        batch[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        padding_mask[i, : len(seq)] = False
    return batch, padding_mask


def collate_fn(batch: List[Tuple[List[int], List[int]]], pad_idx: int) -> Sample:
    src_batch, src_mask = pad_sequences([item[0] for item in batch], pad_idx)
    tgt_batch, tgt_mask = pad_sequences([item[1] for item in batch], pad_idx)
    return Sample(src=src_batch, tgt=tgt_batch, src_padding_mask=src_mask, tgt_padding_mask=tgt_mask)
