from __future__ import annotations

from typing import Dict

from torch.optim import AdamW

from src.models import ToyTransformer, TransformerConfig
from src.tokenization import CharTokenizer, WordTokenizer
from src.vocabulary import SPECIAL_TOKENS, Vocabulary


def setup_tokenizer(vocab: Vocabulary, tokenizer_type: str = "word") -> WordTokenizer | CharTokenizer:
    if tokenizer_type == "char":
        return CharTokenizer(vocab=vocab)
    return WordTokenizer(vocab=vocab)


def setup_model(config: Dict[str, object], vocab: Vocabulary) -> ToyTransformer:
    model_cfg = config.get("model", {})
    transformer_cfg = TransformerConfig(
        vocab_size=len(vocab),
        d_model=model_cfg.get("d_model", 128),
        n_heads=model_cfg.get("n_heads", 4),
        n_layers=model_cfg.get("n_layers", 4),
        ffn_dim=model_cfg.get("ffn_dim", 256),
        dropout=model_cfg.get("dropout", 0.1),
        architecture=model_cfg.get("architecture", "encoder_decoder"),
        disabled_layers=model_cfg.get("disabled_layers"),
        disabled_heads=model_cfg.get("disabled_heads"),
        max_seq_len=config.get("data", {}).get("max_seq_len", 128),
    )
    return ToyTransformer(transformer_cfg)


def setup_optimizer(model: ToyTransformer, config: Dict[str, object]) -> AdamW:
    training_cfg = config.get("training", {})
    return AdamW(
        model.parameters(),
        lr=training_cfg.get("learning_rate", 3e-4),
        weight_decay=training_cfg.get("weight_decay", 0.0),
    )


def get_padding_idx(vocab: Vocabulary) -> int:
    return vocab.get_id(SPECIAL_TOKENS["pad"])
