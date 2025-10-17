from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.models import ToyTransformer, generate_causal_mask
from src.vocabulary import build_vocabulary_from_files

from .batching import collate_fn
from .curriculum import compute_curriculum_fraction
from .dataset import JsonlSeq2SeqDataset
from .factories import get_padding_idx, setup_model, setup_optimizer, setup_tokenizer
from .sample import Sample

try:  # optional dependency
    import wandb
except Exception:  # pragma: no cover - optional
    wandb = None


def run_epoch(
    model: ToyTransformer,
    dataloader: DataLoader[Sample],
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

    dict_paths = [Path(p) for p in data_cfg.get("dict_paths")]
    vocab = build_vocabulary_from_files(dict_paths)
    tokenizer_type = data_cfg.get("tokenizer", "word")
    tokenizer = setup_tokenizer(vocab, tokenizer_type)

    train_dataset = JsonlSeq2SeqDataset(data_cfg["train_path"], tokenizer, data_cfg.get("max_seq_len", 128))
    val_dataset = JsonlSeq2SeqDataset(data_cfg["val_path"], tokenizer, data_cfg.get("max_seq_len", 128))

    pad_idx = get_padding_idx(vocab)

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
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_perplexity": train_metrics["perplexity"],
                    "val_loss": val_metrics["loss"],
                    "val_perplexity": val_metrics["perplexity"],
                    "curriculum_fraction": fraction,
                },
                step=global_step,
            )
        global_step += 1

        checkpoint_dir = Path(training_cfg.get("checkpoint_dir", "checkpoints"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"model_epoch_{epoch:03d}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "epoch": epoch,
                "vocab_size": len(vocab),
            },
            checkpoint_path,
        )

    writer.close()
    if use_wandb:
        wandb.finish()
