"""Analysis utilities for inspecting trained toy transformers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import Tensor

from .model import ToyTransformer, TransformerConfig, generate_causal_mask
from .vocab import Vocabulary

try:  # optional heavy dependencies
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:  # pragma: no cover - optional
    plt = None
    sns = None

try:  # dimensionality reduction
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
except Exception:  # pragma: no cover - optional
    PCA = None
    TSNE = None


def load_model(checkpoint_path: str | Path) -> Tuple[ToyTransformer, Dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    vocab_size = checkpoint.get("vocab_size") or model_cfg.get("vocab_size")
    if vocab_size is None:
        raise KeyError("Checkpoint must contain `vocab_size` field")
    transformer_cfg = TransformerConfig(
        vocab_size=vocab_size,
        d_model=model_cfg.get("d_model", 128),
        n_heads=model_cfg.get("n_heads", 4),
        n_layers=model_cfg.get("n_layers", 4),
        ffn_dim=model_cfg.get("ffn_dim", 256),
        dropout=model_cfg.get("dropout", 0.1),
        architecture=model_cfg.get("architecture", "encoder_decoder"),
        disabled_layers=model_cfg.get("disabled_layers"),
        disabled_heads=model_cfg.get("disabled_heads"),
        max_seq_len=data_cfg.get("max_seq_len", 128),
    )
    model = ToyTransformer(transformer_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def plot_attention_heatmap(attention: Tensor, tokens_x: List[str], tokens_y: List[str], title: str, save_path: Optional[str] = None) -> None:
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
        attn = attn.mean(dim=1)  # average heads
        if rollout is None:
            rollout = attn
        else:
            rollout = torch.matmul(rollout, attn)
    return rollout


def extract_embeddings(model: ToyTransformer) -> Tensor:
    return model.embedding.weight.detach().cpu()


def reduce_embeddings(embeddings: Tensor, method: str = "tsne") -> Tensor:
    if method == "pca":
        if PCA is None:
            raise RuntimeError("scikit-learn is required for PCA")
        reducer = PCA(n_components=2)
        return torch.tensor(reducer.fit_transform(embeddings.numpy()))
    if method == "tsne":
        if TSNE is None:
            raise RuntimeError("scikit-learn is required for t-SNE")
        reducer = TSNE(n_components=2, init="pca", learning_rate="auto")
        return torch.tensor(reducer.fit_transform(embeddings.numpy()))
    raise ValueError(f"Unknown reduction method: {method}")


def plot_embeddings(embeddings_2d: Tensor, vocab: Vocabulary, save_path: Optional[str] = None) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for embedding plots")
    fig, ax = plt.subplots(figsize=(8, 6))
    xs, ys = embeddings_2d[:, 0].numpy(), embeddings_2d[:, 1].numpy()
    ax.scatter(xs, ys)
    for i, token in enumerate(vocab.id_to_token):
        ax.annotate(token, (xs[i], ys[i]))
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)


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
