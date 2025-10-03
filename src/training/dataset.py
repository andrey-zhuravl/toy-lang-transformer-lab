from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from torch.utils.data import Dataset

from src.tokenization import BaseTokenizer


class JsonlSeq2SeqDataset(Dataset):
    """Dataset backed by JSONL files with ``input``/``output`` fields."""

    def __init__(self, path: str | Path, tokenizer: BaseTokenizer, max_seq_len: int) -> None:
        self.samples: List[Dict[str, str]] = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                self.samples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def _truncate(self, ids: List[int]) -> List[int]:
        if self.max_seq_len and len(ids) > self.max_seq_len:
            return ids[: self.max_seq_len]
        return ids

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        sample = self.samples[idx]
        src_ids = self.tokenizer.encode(sample["input"])
        tgt_ids = self.tokenizer.encode(sample["output"])
        return self._truncate(src_ids), self._truncate(tgt_ids)
