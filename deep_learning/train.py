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
    model_name: str = "efficientnet_b0",
    split_csv: str | None = None,
    dataset_root: str | None = None,
    image_size: int = 224,
    batch_size: int | None = None,
    stage2_weight_decay: float = STAGE2_WEIGHT_DECAY,
    stage2_unfreeze_blocks: int = STAGE2_UNFREEZE_BLOCKS,
    stage1_epochs_override: int | None = None,
    stage2_epochs_override: int | None = None,
    lr_schedule: str = "constant",
    warmup_epochs: int = 0,
) -> dict:
    seed_all(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = batch_size or BATCH_SIZE

    print(f"[train] model={model_name}  image_size={image_size}  batch_size={batch_size}")
    print(f"[train] device={device}  seed={seed}  smoke={smoke}  train_frac={train_frac}")
    print(f"[train] split_csv={split_csv or 'default (medium split.csv)'}")
    print(f"[train] lr_schedule={lr_schedule}  warmup_epochs={warmup_epochs}")
    print(f"[train] out={output_dir}")

    train_ds = KidneyCTDataset(
        "train", split_csv=split_csv, dataset_root=dataset_root, image_size=image_size
    )
    val_ds = KidneyCTDataset(
        "val", split_csv=split_csv, dataset_root=dataset_root, image_size=image_size
    )
    if train_frac < 1.0:
        idxs = stratified_train_indices(train_frac, seed=seed)
        train_ds = Subset(train_ds, idxs)
        print(f"[train] subsetting train to {len(idxs)} samples ({train_frac:.0%})")
    if smoke:
        train_ds = Subset(train_ds, list(range(min(64, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(32, len(val_ds)))))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(name=model_name, image_size=image_size).to(device)

    if USE_CLASS_WEIGHTS:
        class_weights = compute_class_weights(split_csv=split_csv).to(device)
        print(f"[train] class weights: {class_weights.tolist()}")
    else:
        class_weights = None
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # Resolve effective epoch counts (CLI override takes precedence over config)
    s1_target = stage1_epochs_override if stage1_epochs_override is not None else STAGE1_EPOCHS
    s2_target = stage2_epochs_override if stage2_epochs_override is not None else STAGE2_EPOCHS

    log: dict = {
        "device": str(device),
        "seed": seed,
        "smoke": smoke,
        "train_frac": train_frac,
        "n_train_samples": len(train_ds),
        "classes": list(CLASSES),
        "config": {
            "model_name": model_name,
            "image_size": image_size,
            "split_csv": str(split_csv) if split_csv else "split.csv",
            "batch_size": batch_size,
            "stage1_epochs": s1_target if not smoke else 1,
            "stage1_lr": STAGE1_LR,
            "stage2_epochs": s2_target if not smoke else 1,
            "stage2_lr": STAGE2_LR,
            "stage2_unfreeze_blocks": stage2_unfreeze_blocks,
            "stage2_weight_decay": stage2_weight_decay,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "use_class_weights": USE_CLASS_WEIGHTS,
            "lr_schedule": lr_schedule,
            "warmup_epochs": warmup_epochs,
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
    stage1_epochs = 1 if smoke else s1_target
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
    unfreeze_last_blocks(model, stage2_unfreeze_blocks)
    print(f"[stage2] trainable params: {count_trainable(model):,}")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=STAGE2_LR,
        weight_decay=stage2_weight_decay,
    )
    stage2_epochs = 1 if smoke else s2_target

    # Optional LR schedule (default "constant" preserves prior behaviour).
    # "cosine" applies linear warmup (LR 1% → 100% over warmup_epochs) then
    # cosine anneal from STAGE2_LR to STAGE2_LR/100 over the remaining epochs.
    scheduler = None
    if lr_schedule == "cosine" and not smoke:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        eta_min = STAGE2_LR / 100.0
        if warmup_epochs > 0 and warmup_epochs < stage2_epochs:
            warm = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
            cos = CosineAnnealingLR(optimizer, T_max=stage2_epochs - warmup_epochs, eta_min=eta_min)
            scheduler = SequentialLR(optimizer, schedulers=[warm, cos], milestones=[warmup_epochs])
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=stage2_epochs, eta_min=eta_min)
        print(f"[stage2] LR schedule: cosine, warmup={warmup_epochs} epochs, eta_min={eta_min:.2e}")
    elif lr_schedule == "constant":
        print(f"[stage2] LR schedule: constant @ {STAGE2_LR:.2e}")

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
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[stage2 epoch {e + 1}/{stage2_epochs}] "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_macro_f1={val_f1:.4f}  lr={current_lr:.2e}"
            f"{'  *best*' if improved else ''}"
        )
        log["epochs"].append(
            {
                "stage": 2,
                "epoch": e + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1": val_f1,
                "lr": current_lr,
                "improved": improved,
            }
        )
        if scheduler is not None:
            scheduler.step()
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
        "--train-frac", type=float, default=1.0,
        help="stratified fraction of the train split to use (for data-efficiency sweep)",
    )
    parser.add_argument(
        "--model", default="efficientnet_b0",
        choices=["efficientnet_b0", "convnextv2_base"],
        help="backbone architecture",
    )
    parser.add_argument(
        "--split-csv", default=None,
        help="path to the split CSV (default: split.csv). Use split_full.csv for full-dataset runs.",
    )
    parser.add_argument(
        "--dataset-root", default=None,
        help="optional dataset root override (default: shared.config.DATASET_ROOT)",
    )
    parser.add_argument(
        "--image-size", type=int, default=224,
        help="CNN input resolution (224 for EfficientNet-B0, 384 for ConvNeXt V2)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="override batch size (default comes from deep_learning.config)",
    )
    parser.add_argument(
        "--stage2-weight-decay", type=float, default=STAGE2_WEIGHT_DECAY,
        help="AdamW weight decay in stage 2 (default 1e-4; recommended 5e-2 for ConvNeXt V2)",
    )
    parser.add_argument(
        "--stage2-unfreeze-blocks", type=int, default=STAGE2_UNFREEZE_BLOCKS,
        help="number of last backbone stages to unfreeze in stage 2 "
             "(default 2 for EfficientNet-B0; use 1 for ConvNeXt V2 Base where stages are much larger)",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help=f"override the global SEED (default {SEED}). Use different seeds for ensemble runs.",
    )
    parser.add_argument(
        "--stage1-epochs", type=int, default=None,
        help=f"override STAGE1_EPOCHS (default {STAGE1_EPOCHS} from config)",
    )
    parser.add_argument(
        "--stage2-epochs", type=int, default=None,
        help=f"override STAGE2_EPOCHS (default {STAGE2_EPOCHS} from config)",
    )
    parser.add_argument(
        "--lr-schedule", default="constant", choices=["constant", "cosine"],
        help="stage-2 learning-rate schedule: 'constant' (default, original behaviour) "
             "or 'cosine' (linear warmup then cosine anneal to STAGE2_LR/100)",
    )
    parser.add_argument(
        "--warmup-epochs", type=int, default=0,
        help="cosine schedule only — number of linear-warmup epochs at the start of stage 2",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    train(
        Path(args.output_dir),
        device=device,
        smoke=args.smoke,
        seed=args.seed,
        train_frac=args.train_frac,
        model_name=args.model,
        split_csv=args.split_csv,
        dataset_root=args.dataset_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        stage2_weight_decay=args.stage2_weight_decay,
        stage2_unfreeze_blocks=args.stage2_unfreeze_blocks,
        stage1_epochs_override=args.stage1_epochs,
        stage2_epochs_override=args.stage2_epochs,
        lr_schedule=args.lr_schedule,
        warmup_epochs=args.warmup_epochs,
    )


if __name__ == "__main__":
    main()
