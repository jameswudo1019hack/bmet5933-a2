"""Tier 1 — TTA hflip on clean-dataset DL checkpoints (no retraining).

Loads the two best_model.pt files from Sprint 5 (EffNet-B0 and ConvNeXt V2,
both trained on the deduplicated Updated_Dataset), and re-predicts on the
n=1,888 test set with horizontal-flip TTA. Sprint 1 showed this gave EffNet
+0.84 pp on medium-leaky; we re-apply the same recipe here.

This is inference only — no retraining, no hyperparameter changes.

Inputs:
  Results/dl_run_full/best_model.pt          (EffNet-B0, clean)
  Results/convnextv2_full_run/best_model.pt  (ConvNeXt V2, clean)
  split_full.csv + Updated_Dataset/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone_unique

Outputs:
  Results/dl_run_full_tta_hflip/dl_predictions.npz
  Results/dl_run_full_tta_hflip/dl_results.json
  Results/convnextv2_full_run_tta_hflip/dl_predictions.npz
  Results/convnextv2_full_run_tta_hflip/dl_results.json

Usage:
  python -m analysis.tier1_tta_clean
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from deep_learning.dataset import build_transforms
from deep_learning.model import build_model
from deep_learning.train import resolve_device
from shared.config import REPO_ROOT, RESULTS_DIR, SPLIT_CSV_FULL
from shared.evaluate import evaluate, save_results, print_summary
from shared.preprocessing import load_image, load_split


DATASET_ROOT = REPO_ROOT / "Updated_Dataset" / "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone_unique"


class HflipTTADataset(Dataset):
    """Returns (2, 3, H, W) — original + hflip — per test image."""

    def __init__(self, image_size: int):
        self.df = load_split(
            "test",
            split_csv=str(SPLIT_CSV_FULL),
            dataset_root=str(DATASET_ROOT),
        )
        self.tf = build_transforms(train=False, image_size=image_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_256 = load_image(row["abs_path"])
        pil = Image.fromarray(img_256, mode="L")
        flip = pil.transpose(Image.FLIP_LEFT_RIGHT)
        t_orig = self.tf(np.asarray(pil))
        t_flip = self.tf(np.asarray(flip))
        return torch.stack([t_orig, t_flip], dim=0), int(row["class_idx"])


@torch.no_grad()
def tta_predict_hflip(
    checkpoint_path: Path, model_name: str, image_size: int,
    device: torch.device, batch_size: int = 16,
) -> dict[str, np.ndarray]:
    model = build_model(name=model_name, image_size=image_size).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    ds = HflipTTADataset(image_size=image_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_probs: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    for stacked, y in loader:
        B = stacked.shape[0]
        flat = stacked.view(B * 2, 3, image_size, image_size).to(device)
        logits = model(flat)
        probs = torch.softmax(logits, dim=1).view(B, 2, -1).mean(dim=1)
        all_probs.append(probs.cpu().numpy())
        all_true.append(y.numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_pred = y_prob.argmax(axis=1)
    y_true = np.concatenate(all_true, axis=0)
    return {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}


def run_one(checkpoint_path: Path, model_name: str, image_size: int,
            output_dir: Path, label: str, device: torch.device,
            batch_size: int = 16) -> None:
    print(f"\n=== TTA hflip — {label} ===")
    print(f"  checkpoint={checkpoint_path}")
    print(f"  model={model_name}  image_size={image_size}  batch={batch_size}  device={device}")

    preds = tta_predict_hflip(
        checkpoint_path, model_name=model_name, image_size=image_size,
        device=device, batch_size=batch_size,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "dl_predictions.npz",
             y_true=preds["y_true"], y_pred=preds["y_pred"], y_prob=preds["y_prob"])

    results = evaluate(preds["y_true"], preds["y_pred"], y_prob=preds["y_prob"],
                       model_name=f"{model_name}_tta_hflip_clean")
    print_summary(results)
    save_results(results, output_dir / "dl_results.json")
    print(f"  saved -> {output_dir}")


def main() -> None:
    device = resolve_device(None)
    print(f"[tta] device={device}")

    # EfficientNet-B0 — small model, batch 32 fine on MPS / CPU
    run_one(
        checkpoint_path=RESULTS_DIR / "dl_run_full" / "best_model.pt",
        model_name="efficientnet_b0",
        image_size=224,
        output_dir=RESULTS_DIR / "dl_run_full_tta_hflip",
        label="EfficientNet-B0 (clean)",
        device=device,
        batch_size=32,
    )

    # ConvNeXt V2 Base — large model at 384, drop batch to 8 on MPS to avoid OOM
    run_one(
        checkpoint_path=RESULTS_DIR / "convnextv2_full_run" / "best_model.pt",
        model_name="convnextv2_base",
        image_size=384,
        output_dir=RESULTS_DIR / "convnextv2_full_run_tta_hflip",
        label="ConvNeXt V2 Base (clean)",
        device=device,
        batch_size=8,
    )


if __name__ == "__main__":
    main()
