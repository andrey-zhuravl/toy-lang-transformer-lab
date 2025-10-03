from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class Sample:
    """Batch of padded source/target sequences with masks."""

    src: Tensor
    tgt: Tensor
    src_padding_mask: Tensor
    tgt_padding_mask: Tensor
