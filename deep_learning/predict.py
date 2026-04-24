"""Load a trained checkpoint and run inference on the held-out test set.

Emits a results JSON into Results/ (via shared.evaluate) containing every
metric specified in the Phase 0 design doc. Also saves raw predictions
(y_pred, y_prob) so downstream analyses (McNemar vs classical, data
efficiency curves) can read them without re-running inference.

Supports EfficientNet-B0 (primary, 224×224) and ConvNeXt V2 Base
(Sprint 2 supplementary, 384×384). Also accepts an alternative split CSV
(e.g. split_full.csv) and an alternative dataset root.

Usage:
  python -m deep_learning.predict --checkpoint path/to/best_model.pt
  python -m deep_learning.predict --checkpoint .../convnextv2.pt \\
      --model convnextv2_base --image-size 384 --split-csv split_full.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from deep_learning.config import (
    BATCH_SIZE,
    NUM_WORKERS,
    RESULTS_FILENAME,
)
from deep_learning.dataset import KidneyCTDataset
from deep_learning.model import build_model
from deep_learning.train import resolve_device
from shared.config import RESULTS_DIR
from shared.evaluate import evaluate, print_summary, save_results


@torch.no_grad()
def predict(
    checkpoint_path: Path,
    device: torch.device,
    model_name: str = "efficientnet_b0",
    split_csv: str | None = None,
    dataset_root: str | None = None,
    image_size: int = 224,
    batch_size: int | None = None,
) -> dict:
    model = build_model(name=model_name, image_size=image_size).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    test_ds = KidneyCTDataset(
        "test",
        train=False,
        split_csv=split_csv,
        dataset_root=dataset_root,
        image_size=image_size,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size or BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    all_probs: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        all_probs.append(probs)
        all_preds.append(preds)
        all_true.append(y.numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_true, axis=0)

    return {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to best_model.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model", default="efficientnet_b0",
                        choices=["efficientnet_b0", "convnextv2_base"])
    parser.add_argument("--model-name", default=None,
                        help="label for results JSON (defaults to --model)")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR / "dl_run"))
    parser.add_argument("--split-csv", default=None,
                        help="path to split CSV (default: split.csv)")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = predict(
        Path(args.checkpoint),
        device=device,
        model_name=args.model,
        split_csv=args.split_csv,
        dataset_root=args.dataset_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )

    np.savez(
        out_dir / "dl_predictions.npz",
        y_true=preds["y_true"],
        y_pred=preds["y_pred"],
        y_prob=preds["y_prob"],
    )

    results = evaluate(
        preds["y_true"],
        preds["y_pred"],
        y_prob=preds["y_prob"],
        model_name=args.model_name or args.model,
    )
    print_summary(results)
    save_results(results, out_dir / RESULTS_FILENAME)


if __name__ == "__main__":
    main()
