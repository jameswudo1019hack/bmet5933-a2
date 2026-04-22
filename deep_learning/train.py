"""Two-stage training loop for the EfficientNet-B0 transfer model.

Stage 1: freeze backbone, train classification head only with Adam LR=1e-3.
Stage 2: unfreeze last N blocks, fine-tune with AdamW LR=1e-5 + weight decay.

Best checkpoint (by val macro-F1) is saved alongside a run_log.json
recording config, per-epoch metrics, and total wall time.

Designed to run identically on CUDA (Colab) and MPS (M1 Pro) and CPU.
For real training runs on Colab, use notebooks/colab_train.ipynb.
For local smoke tests, call via `python -m deep_learning.train --smoke`.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset

from deep_learning.config import (
    BATCH_SIZE,
    CHECKPOINT_FILENAME,
    EARLY_STOPPING_PATIENCE,
    NUM_WORKERS,
    STAGE1_EPOCHS,
    STAGE1_LR,
    STAGE2_EPOCHS,
    STAGE2_LR,
    STAGE2_UNFREEZE_BLOCKS,
    STAGE2_WEIGHT_DECAY,
    USE_CLASS_WEIGHTS,
)
from deep_learning.dataset import KidneyCTDataset
from deep_learning.model import (
    build_model,
    compute_class_weights,
    count_trainable,
    freeze_backbone,
    unfreeze_last_blocks,
)
from shared.config import CLASSES, RESULTS_DIR, SEED
from shared.data_efficiency import stratified_train_indices


def resolve_device(prefer: str | None = None) -> torch.device:
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_macro_f1(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    model.eval()
    ys: list[int] = []
    ps: list[int] = []
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        total_loss += float(loss_fn(logits, y).item()) * x.size(0)
        n += x.size(0)
        ps.extend(logits.argmax(dim=1).cpu().tolist())
        ys.extend(y.cpu().tolist())
    macro_f1 = float(f1_score(ys, ps, average="macro", zero_division=0))
    return macro_f1, total_loss / max(n, 1)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


def train(
    output_dir: Path,
    device: torch.device,
    smoke: bool = False,
    seed: int = SEED,
    train_frac: float = 1.0,
) -> dict:
    seed_all(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] device={device}  seed={seed}  smoke={smoke}  train_frac={train_frac}  out={output_dir}")

    train_ds = KidneyCTDataset("train")
    val_ds = KidneyCTDataset("val")
    if train_frac < 1.0:
        idxs = stratified_train_indices(train_frac, seed=SEED)
        train_ds = Subset(train_ds, idxs)
        print(f"[train] subsetting train to {len(idxs)} samples ({train_frac:.0%})")
    if smoke:
        # 64 train / 32 val samples is enough to verify the pipeline runs
        train_ds = Subset(train_ds, list(range(min(64, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(32)))

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model().to(device)

    if USE_CLASS_WEIGHTS:
        class_weights = compute_class_weights().to(device)
        print(f"[train] class weights: {class_weights.tolist()}")
    else:
        class_weights = None
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    log: dict = {
        "device": str(device),
        "seed": seed,
        "smoke": smoke,
        "train_frac": train_frac,
        "n_train_samples": len(train_ds),
        "classes": list(CLASSES),
        "config": {
            "batch_size": BATCH_SIZE,
            "stage1_epochs": STAGE1_EPOCHS if not smoke else 1,
            "stage1_lr": STAGE1_LR,
            "stage2_epochs": STAGE2_EPOCHS if not smoke else 1,
            "stage2_lr": STAGE2_LR,
            "stage2_unfreeze_blocks": STAGE2_UNFREEZE_BLOCKS,
            "stage2_weight_decay": STAGE2_WEIGHT_DECAY,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "use_class_weights": USE_CLASS_WEIGHTS,
        },
        "epochs": [],
    }

    best_val_f1 = -1.0
    best_epoch = -1
    epochs_since_improvement = 0
    ckpt_path = output_dir / CHECKPOINT_FILENAME
    t0 = time.time()

    # Stage 1: freeze backbone, train head
    freeze_backbone(model)
    print(f"[stage1] trainable params: {count_trainable(model):,}")
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=STAGE1_LR
    )
    stage1_epochs = 1 if smoke else STAGE1_EPOCHS
    for e in range(stage1_epochs):
        train_loss = run_epoch(model, train_loader, optimizer, loss_fn, device)
        val_f1, val_loss = evaluate_macro_f1(model, val_loader, device)
        print(
            f"[stage1 epoch {e + 1}/{stage1_epochs}] "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_macro_f1={val_f1:.4f}"
        )
        log["epochs"].append(
            {
                "stage": 1,
                "epoch": e + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1": val_f1,
            }
        )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = len(log["epochs"])
            torch.save(model.state_dict(), ckpt_path)

    # Stage 2: unfreeze last blocks, fine-tune
    unfreeze_last_blocks(model, STAGE2_UNFREEZE_BLOCKS)
    print(f"[stage2] trainable params: {count_trainable(model):,}")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=STAGE2_LR,
        weight_decay=STAGE2_WEIGHT_DECAY,
    )
    stage2_epochs = 1 if smoke else STAGE2_EPOCHS
    epochs_since_improvement = 0
    for e in range(stage2_epochs):
        train_loss = run_epoch(model, train_loader, optimizer, loss_fn, device)
        val_f1, val_loss = evaluate_macro_f1(model, val_loader, device)
        improved = val_f1 > best_val_f1
        if improved:
            best_val_f1 = val_f1
            best_epoch = len(log["epochs"]) + 1
            torch.save(model.state_dict(), ckpt_path)
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        print(
            f"[stage2 epoch {e + 1}/{stage2_epochs}] "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_macro_f1={val_f1:.4f}{'  *best*' if improved else ''}"
        )
        log["epochs"].append(
            {
                "stage": 2,
                "epoch": e + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1": val_f1,
                "improved": improved,
            }
        )
        if not smoke and epochs_since_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"[stage2] early stopping after {EARLY_STOPPING_PATIENCE} epochs without improvement")
            break

    log["best_val_macro_f1"] = best_val_f1
    log["best_epoch"] = best_epoch
    log["wall_time_sec"] = time.time() - t0
    log["checkpoint"] = str(ckpt_path)

    (output_dir / "run_log.json").write_text(json.dumps(log, indent=2))
    print(f"[train] done  best_val_macro_f1={best_val_f1:.4f}  wall={log['wall_time_sec']:.1f}s")
    return log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, help="override: cpu | mps | cuda")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR / "dl_run"))
    parser.add_argument("--smoke", action="store_true", help="tiny run to verify pipeline")
    parser.add_argument(
        "--train-frac",
        type=float,
        default=1.0,
        help="stratified fraction of the train split to use (for data-efficiency sweep)",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    train(
        Path(args.output_dir),
        device=device,
        smoke=args.smoke,
        train_frac=args.train_frac,
    )


if __name__ == "__main__":
    main()
