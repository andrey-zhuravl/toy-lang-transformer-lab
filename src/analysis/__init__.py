"""Analysis helpers for inspecting trained transformers."""

from .attention import attention_rollout, plot_attention_heatmap
from .embeddings import extract_embeddings, plot_embeddings, reduce_embeddings
from .gradients import compute_gradient_attributions, integrated_gradients
from .loader import load_model

__all__ = [
    "attention_rollout",
    "plot_attention_heatmap",
    "extract_embeddings",
    "plot_embeddings",
    "reduce_embeddings",
    "compute_gradient_attributions",
    "integrated_gradients",
    "load_model",
]
