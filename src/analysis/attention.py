from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import torch
from torch import Tensor

try:  # optional plotting deps
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:  # pragma: no cover - optional
    plt = None
    sns = None


def plot_attention_heatmap(
    attention: Tensor,
    tokens_x: List[str],
    tokens_y: List[str],
    title: str,
    save_path: Optional[str] = None,
) -> None:
    if plt is None or sns is None:
        raise RuntimeError("matplotlib/seaborn are required for attention visualisation")
    attention = attention.detach().cpu()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(attention, xticklabels=tokens_y, yticklabels=tokens_x, cmap="viridis", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Key")
    ax.set_ylabel("Query")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)


def attention_rollout(attn_weights: Iterable[Tensor]) -> Tensor:
    rollout = None
    for attn in attn_weights:
        attn = attn.mean(dim=1)
        if rollout is None:
            rollout = attn
        else:
            rollout = torch.matmul(rollout, attn)
    if rollout is None:
        raise ValueError("Empty attention weights")
    return rollout
