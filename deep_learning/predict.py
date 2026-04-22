"""Load a trained checkpoint and run inference on the held-out test set.

Emits a results JSON into Results/ (via shared.evaluate) containing every
metric specified in the Phase 0 design doc. Also saves raw predictions
(y_pred, y_prob) so downstream analyses (McNemar vs classical, data
efficiency curves) can read them without re-running inference.

Usage:
  python -m deep_learning.predict --checkpoint path/to/best_model.pt
"""
from __future__ import annotations

import argparse
import json
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
def predict(checkpoint_path: Path, device: torch.device) -> dict:
    model = build_model().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    test_ds = KidneyCTDataset("test", train=False)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
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
    parser.add_argument("--model-name", default="efficientnet_b0")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    device = resolve_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = predict(Path(args.checkpoint), device=device)

    # Save raw arrays for downstream paired tests (McNemar vs classical)
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
        model_name=args.model_name,
    )
    print_summary(results)
    save_results(results, out_dir / RESULTS_FILENAME)


if __name__ == "__main__":
    main()
