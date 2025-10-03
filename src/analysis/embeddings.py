from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

try:  # dimensionality reduction
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
except Exception:  # pragma: no cover - optional
    PCA = None
    TSNE = None

try:  # plotting
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional
    plt = None

from src.vocabulary import Vocabulary
from src.models import ToyTransformer


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
