#!/usr/bin/env python3
"""Train PyTorch TonicNet on TF2-format pickle datasets.

Usage:
    python train.py                         # train from scratch
    python train.py --weights ckpt.pt       # resume from checkpoint
    python train.py --epochs 150            # custom epoch count
    python train.py --overwrite             # overwrite existing best weights
"""

import argparse
import csv
import datetime
import math
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import VOCABULARY, SONG_START, SONG_END, MODEL_VERSION, load_checkpoint, TonicNet


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAD_VALUE = -1  # sentinel for masked positions
BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Data loading helpers (match TF2 train_jos.py exactly)
# ---------------------------------------------------------------------------

def compute_repetitions(sequence: list[int] | np.ndarray) -> list[int]:
    """Compute repetition counts per token position (matches TF2 training)."""
    repetitions = [0] * len(sequence)
    for index, element in enumerate(sequence):
        past_index = index - 5
        if past_index >= 0:
            past_element = sequence[past_index]
            if past_element == SONG_START:
                repetitions[index] = 0
            elif element == SONG_END:
                repetitions[index] = 0
            elif past_element == element:
                repetitions[index] = repetitions[past_index] + 1
            else:
                repetitions[index] = 0
            if repetitions[index] > 79:
                repetitions[index] = 79
    return repetitions


def compute_positions(sequence: list[int] | np.ndarray) -> list[int]:
    """Compute position indices per token (0-15 cycling every 5 tokens)."""
    return [0] + [index // 5 % 16 for index in range(len(sequence) - 1)]


def compute_countdown(sequence: list[int] | np.ndarray, c_tokens: int = 48) -> list[int]:
    """Compute bars-remaining countdown per token position.

    80 tokens per bar (5 tokens/timestep × 16 timesteps/bar).
    c[i] = clamp(total_bars - 1 - i // 80, 0, c_tokens - 1).
    """
    n = len(sequence)
    total_bars = math.ceil(n / 80)
    return [max(0, min(c_tokens - 1, total_bars - 1 - i // 80)) for i in range(n)]


def to_supervised(songs: list[np.ndarray]) -> tuple[list, list, list, list, list]:
    """Convert songs to (x, r, p, c, y) sequences for teacher forcing."""
    xs, rs, ps, cs, ys = [], [], [], [], []
    for song in songs:
        x = song[:-1]
        y = song[1:]
        r = compute_repetitions(x.tolist())
        p = compute_positions(x.tolist())
        c = compute_countdown(x.tolist())
        xs.append(torch.tensor(x, dtype=torch.long))
        rs.append(torch.tensor(r, dtype=torch.long))
        ps.append(torch.tensor(p, dtype=torch.long))
        cs.append(torch.tensor(c, dtype=torch.long))
        ys.append(torch.tensor(y, dtype=torch.long))
    return xs, rs, ps, cs, ys


def load_dataset(name: str) -> tuple[list, list, list, list, list]:
    """Load a pickle dataset and convert to supervised tensors."""
    assert name in ("train", "valid", "test"), name
    filename = f"dataset_{name}.p"
    if not os.path.exists(filename):
        sys.exit(f"ERROR: {filename} not found. Copy from TF2 directory first.")
    with open(filename, "rb") as f:
        encoded_songs = pickle.load(f)
    print(f"  {name}: {len(encoded_songs)} songs")
    return to_supervised(encoded_songs)


def collate_batch(
    xs: list[torch.Tensor],
    rs: list[torch.Tensor],
    ps: list[torch.Tensor],
    cs: list[torch.Tensor],
    ys: list[torch.Tensor],
    indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch of variable-length sequences.

    Returns (x, r, p, c, y, pad_mask) where pad_mask is True for valid positions.
    """
    batch_x = [xs[i] for i in indices]
    batch_r = [rs[i] for i in indices]
    batch_p = [ps[i] for i in indices]
    batch_c = [cs[i] for i in indices]
    batch_y = [ys[i] for i in indices]

    lengths = torch.tensor([len(s) for s in batch_x])

    x = nn.utils.rnn.pad_sequence(batch_x, batch_first=True, padding_value=0)
    r = nn.utils.rnn.pad_sequence(batch_r, batch_first=True, padding_value=0)
    p = nn.utils.rnn.pad_sequence(batch_p, batch_first=True, padding_value=0)
    c = nn.utils.rnn.pad_sequence(batch_c, batch_first=True, padding_value=0)
    y = nn.utils.rnn.pad_sequence(batch_y, batch_first=True, padding_value=PAD_VALUE)

    # pad_mask: True for valid positions, False for padding
    pad_mask = torch.arange(x.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
    return x, r, p, c, y, pad_mask


# ---------------------------------------------------------------------------
# Loss / accuracy with mask
# ---------------------------------------------------------------------------

def masked_loss_accuracy(
    logits: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, float]:
    """Cross-entropy loss and accuracy, ignoring positions where target == PAD_VALUE."""
    mask = targets != PAD_VALUE
    y_safe = targets.clamp(min=0)

    per_token_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), y_safe.view(-1), reduction="none"
    )
    per_token_loss = per_token_loss.view_as(targets)
    loss = (per_token_loss * mask).sum() / mask.sum()

    preds = logits.argmax(dim=-1)
    accuracy = ((preds == targets) & mask).sum().item() / mask.sum().item()
    return loss, accuracy


# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def lr_lambda(step: int, warmup_steps: int, total_steps: int, min_lr: float, max_lr: float) -> float:
    """Returns multiplier for LambdaLR (applied to base lr = max_lr)."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return max(min_lr / max_lr, cosine)


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train_epoch(
    model: TonicNet,
    xs: list[torch.Tensor],
    rs: list[torch.Tensor],
    ps: list[torch.Tensor],
    cs: list[torch.Tensor],
    ys: list[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    indices = np.random.permutation(len(xs)).tolist()
    n_batches = (len(xs) + BATCH_SIZE - 1) // BATCH_SIZE
    total_loss = 0.0
    total_acc = 0.0

    for b in range(n_batches):
        batch_idx = indices[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
        x, r, p, c, y, pad_mask = collate_batch(xs, rs, ps, cs, ys, batch_idx)
        x, r, p, c, y = x.to(device), r.to(device), p.to(device), c.to(device), y.to(device)
        pad_mask = pad_mask.to(device)

        logits = model(x, r, p, c, pad_mask=pad_mask)
        loss, acc = masked_loss_accuracy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_acc += acc
        print(f"\r  Batch {b+1}/{n_batches}", end="", flush=True)

    return total_loss / n_batches, total_acc / n_batches


@torch.no_grad()
def evaluate(
    model: TonicNet,
    xs: list[torch.Tensor],
    rs: list[torch.Tensor],
    ps: list[torch.Tensor],
    cs: list[torch.Tensor],
    ys: list[torch.Tensor],
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    n_batches = (len(xs) + BATCH_SIZE - 1) // BATCH_SIZE
    total_loss = 0.0
    total_acc = 0.0

    for b in range(n_batches):
        batch_idx = list(range(b * BATCH_SIZE, min((b + 1) * BATCH_SIZE, len(xs))))
        x, r, p, c, y, pad_mask = collate_batch(xs, rs, ps, cs, ys, batch_idx)
        x, r, p, c, y = x.to(device), r.to(device), p.to(device), c.to(device), y.to(device)
        pad_mask = pad_mask.to(device)

        logits = model(x, r, p, c, pad_mask=pad_mask)
        loss, acc = masked_loss_accuracy(logits, y)
        total_loss += loss.item()
        total_acc += acc

    return total_loss / n_batches, total_acc / n_batches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train PyTorch TonicNet")
    parser.add_argument("--weights", default=None,
                        help="Resume from checkpoint (.pt file)")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing best weights")
    parser.add_argument("--out", default="tonicnet-best.pt",
                        help="Output path for best checkpoint")
    args = parser.parse_args()

    if not args.overwrite and os.path.exists(args.out):
        sys.exit(f"ERROR: {args.out} already exists. Use --overwrite to retrain.")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    print("Loading datasets...")
    train_x, train_r, train_p, train_c, train_y = load_dataset("train")
    val_x, val_r, val_p, val_c, val_y = load_dataset("valid")
    print()

    model = TonicNet()
    if args.weights:
        state_dict = load_checkpoint(args.weights, device)
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded weights from {args.weights}")
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # AdamW with linear warmup + cosine decay
    max_lr = 3e-4
    min_lr = 1e-5
    warmup_steps = 500
    n_batches_per_epoch = (len(train_x) + BATCH_SIZE - 1) // BATCH_SIZE
    total_steps = args.epochs * n_batches_per_epoch

    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_lambda(step, warmup_steps, total_steps, min_lr, max_lr),
    )

    best_val_loss = float("inf")
    t0 = time.time()

    # CSV log
    log_fields = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "secs"]
    log_path = os.path.splitext(args.out)[0] + ".csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    log_writer.writeheader()
    log_file.flush()

    print(f"\nTraining for {args.epochs} epochs (batch_size={BATCH_SIZE}, "
          f"{n_batches_per_epoch} batches/epoch, {total_steps} total steps)")
    print(f"Logging to {log_path}\n")

    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.time()

        train_loss, train_acc = train_epoch(
            model, train_x, train_r, train_p, train_c, train_y, optimizer, scheduler, device)
        val_loss, val_acc = evaluate(
            model, val_x, val_r, val_p, val_c, val_y, device)

        current_lr = scheduler.get_last_lr()[0]
        epoch_secs = int(time.time() - epoch_t0)
        dt = datetime.timedelta(seconds=epoch_secs)
        print(f"\rEpoch {epoch:3d}  "
              f"loss={train_loss:.4f}  acc={100*train_acc:.2f}%  "
              f"val_loss={val_loss:.4f}  val_acc={100*val_acc:.2f}%  "
              f"lr={current_lr:.2e}  [{dt}]")

        log_writer.writerow({
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "train_acc": f"{train_acc:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "val_acc": f"{val_acc:.6f}",
            "lr": f"{current_lr:.6e}",
            "secs": epoch_secs,
        })
        log_file.flush()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"version": MODEL_VERSION, "state_dict": model.state_dict()}, args.out)
            print(f"  -> saved {args.out} (val_loss={val_loss:.4f})")

    log_file.close()
    total = datetime.timedelta(seconds=int(time.time() - t0))
    print(f"\nTraining complete in {total}  (log: {log_path})")


if __name__ == "__main__":
    main()
