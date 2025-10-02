"""Training utilities for the toy transformer lab."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from .model import ToyTransformer, TransformerConfig, generate_causal_mask
from .tokenizer import CharTokenizer, WordTokenizer
from .vocab import SPECIAL_TOKENS, Vocabulary, build_vocabulary_from_files

try:  # optional dependency
    import wandb
except Exception:  # pragma: no cover - optional
    wandb = None


@dataclass
class Sample:
    src: Tensor
    tgt: Tensor
    src_padding_mask: Tensor
    tgt_padding_mask: Tensor


class JsonlSeq2SeqDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer: WordTokenizer | CharTokenizer, max_seq_len: int) -> None:
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


def load_config(path: str | Path) -> Dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_curriculum_fraction(epoch: int, total_epochs: int, curriculum_cfg: Dict[str, object]) -> float:
    if not curriculum_cfg.get("enabled", False):
        return 1.0
    schedule = curriculum_cfg.get("schedule", "linear")
    start = float(curriculum_cfg.get("start_fraction", 0.2))
    end = float(curriculum_cfg.get("end_fraction", 1.0))
    if schedule == "linear":
        if total_epochs <= 1:
            return end
        return start + (end - start) * (epoch / (total_epochs - 1))
    if schedule == "exponential":
        progress = epoch / max(total_epochs - 1, 1)
        return start * (end / max(start, 1e-3)) ** progress
    if schedule == "manual":
        boundaries = curriculum_cfg.get("manual_boundaries", [])
        fraction = start
        for entry in boundaries:
            boundary_epoch, boundary_fraction = entry
            if epoch >= boundary_epoch:
                fraction = boundary_fraction
        return fraction
    return end


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


def setup_optimizer(model: nn.Module, config: Dict[str, object]) -> AdamW:
    training_cfg = config.get("training", {})
    return AdamW(
        model.parameters(),
        lr=training_cfg.get("learning_rate", 3e-4),
        weight_decay=training_cfg.get("weight_decay", 0.0),
    )


def run_epoch(
    model: ToyTransformer,
    dataloader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    pad_idx: int,
    grad_clip: Optional[float],
    train: bool = True,
) -> Dict[str, float]:
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    total_loss = 0.0
    total_tokens = 0
    model.train(train)
    for batch in tqdm(dataloader, desc="train" if train else "eval", leave=False):
        src = batch.src.to(device)
        tgt = batch.tgt.to(device)
        src_padding = batch.src_padding_mask.to(device)
        tgt_padding = batch.tgt_padding_mask.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        tgt_padding_input = tgt_padding[:, :-1]
        causal_mask = generate_causal_mask(tgt_input.size(1), device=device)

        outputs = model(
            src_tokens=src,
            tgt_tokens=tgt_input,
            src_key_padding_mask=src_padding,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_padding_input,
        )
        logits = outputs["logits"]
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        tokens = (~tgt_padding[:, 1:]).sum().item()
        total_loss += loss.item() * tokens
        total_tokens += tokens
    ppl = torch.exp(torch.tensor(total_loss / max(total_tokens, 1))).item()
    return {"loss": total_loss / max(total_tokens, 1), "perplexity": ppl}


def train_loop(config: Dict[str, object]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})

    dict_paths = [Path(p) for p in data_cfg.get("dict_paths", ["data/dicts/base.txt"])]
    vocab = build_vocabulary_from_files(dict_paths)
    tokenizer_type = data_cfg.get("tokenizer", "word")
    tokenizer = setup_tokenizer(vocab, tokenizer_type)

    train_dataset = JsonlSeq2SeqDataset(data_cfg["train_path"], tokenizer, data_cfg.get("max_seq_len", 128))
    val_dataset = JsonlSeq2SeqDataset(data_cfg["val_path"], tokenizer, data_cfg.get("max_seq_len", 128))

    pad_idx = vocab.get_id(SPECIAL_TOKENS["pad"])

    model = setup_model(config, vocab).to(device)
    optimizer = setup_optimizer(model, config)

    log_dir = Path(training_cfg.get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    use_wandb = training_cfg.get("use_wandb", False) and wandb is not None
    if use_wandb:
        wandb.init(project=training_cfg.get("wandb_project", "toy-lang-transformer"), config=config)

    num_epochs = training_cfg.get("num_epochs", 10)
    grad_clip = training_cfg.get("gradient_clip")
    curriculum_cfg = training_cfg.get("curriculum", {})

    global_step = 0
    for epoch in range(num_epochs):
        fraction = compute_curriculum_fraction(epoch, num_epochs, curriculum_cfg)
        train_size = max(1, int(len(train_dataset) * fraction))
        subset_indices = torch.randperm(len(train_dataset))[:train_size]
        subset = torch.utils.data.Subset(train_dataset, subset_indices)
        train_loader = DataLoader(
            subset,
            batch_size=data_cfg.get("batch_size", 32),
            shuffle=True,
            collate_fn=lambda x: collate_fn(x, pad_idx),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=data_cfg.get("batch_size", 32),
            shuffle=False,
            collate_fn=lambda x: collate_fn(x, pad_idx),
        )

        train_metrics = run_epoch(model, train_loader, optimizer, device, pad_idx, grad_clip, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, pad_idx, grad_clip, train=False)

        writer.add_scalar("train/loss", train_metrics["loss"], epoch)
        writer.add_scalar("train/perplexity", train_metrics["perplexity"], epoch)
        writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        writer.add_scalar("val/perplexity", val_metrics["perplexity"], epoch)
        writer.add_scalar("curriculum/fraction", fraction, epoch)

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_perplexity": train_metrics["perplexity"],
                "val_loss": val_metrics["loss"],
                "val_perplexity": val_metrics["perplexity"],
                "curriculum_fraction": fraction,
            }, step=global_step)
        global_step += 1

        checkpoint_dir = Path(training_cfg.get("checkpoint_dir", "checkpoints"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"model_epoch_{epoch:03d}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "epoch": epoch,
            "vocab_size": len(vocab),
        }, checkpoint_path)

    writer.close()
    if use_wandb:
        wandb.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train toy transformer model")
    parser.add_argument("--config", default="src/config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_loop(config)


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
