"""Test-Time Augmentation for the EfficientNet-B0 classifier.

Generates multiple augmented views of each test image, runs the trained
model on each view, and averages softmax outputs before taking argmax.

Design rationale (see Planning/DL_Improvements_Analysis.md §2.2):
  16 of 19 DL errors have the true class at rank 2, with mean confidence
  0.56 on wrong predictions vs 0.95 on correct. Averaging across augmented
  views is exactly the intervention that tips rank-2-but-borderline
  predictions toward the correct class. Upper bound is top-2 accuracy
  (99.6 %); TTA cannot exceed that without changing the model.

Augmentation set (6 views — justified in the sprint log):
  1. Original
  2. Horizontal flip
  3. Rotation +10°
  4. Rotation −10°
  5. Horizontal flip + rotation +10°
  6. Horizontal flip + rotation −10°

No vertical flip: CT anatomy is not top-bottom symmetric. ±10° rotation
matches the training-time distribution (±15°) so inference-time views
come from the same distribution the model saw.

Usage:
  python -m deep_learning.tta --checkpoint Results/dl_run/best_model.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from deep_learning.config import BATCH_SIZE, NUM_WORKERS, RESULTS_FILENAME
from deep_learning.dataset import build_transforms
from deep_learning.model import build_model
from deep_learning.train import resolve_device
from shared.config import RESULTS_DIR
from shared.evaluate import evaluate, print_summary, save_results
from shared.preprocessing import load_image, load_split


def _tta_views(img_256: np.ndarray, view_set: str = "full") -> list[np.ndarray]:
    """Return a list of augmented views of the shared 256x256 grayscale image.

    view_set:
      - "hflip": 2 views — original + horizontal flip
      - "rot":   3 views — original + rot+10° + rot−10°
      - "basic": 4 views — original + hflip + rot+10° + rot−10°
      - "full":  6 views — basic + hflip×rot+10° + hflip×rot−10°  (default)
    """
    pil = Image.fromarray(img_256, mode="L")
    pil_flip = pil.transpose(Image.FLIP_LEFT_RIGHT)
    if view_set == "hflip":
        return [np.asarray(pil), np.asarray(pil_flip)]
    if view_set == "rot":
        return [
            np.asarray(pil),
            np.asarray(pil.rotate(10, resample=Image.BILINEAR)),
            np.asarray(pil.rotate(-10, resample=Image.BILINEAR)),
        ]
    if view_set == "basic":
        return [
            np.asarray(pil),
            np.asarray(pil_flip),
            np.asarray(pil.rotate(10, resample=Image.BILINEAR)),
            np.asarray(pil.rotate(-10, resample=Image.BILINEAR)),
        ]
    if view_set == "full":
        return [
            np.asarray(pil),
            np.asarray(pil_flip),
            np.asarray(pil.rotate(10, resample=Image.BILINEAR)),
            np.asarray(pil.rotate(-10, resample=Image.BILINEAR)),
            np.asarray(pil_flip.rotate(10, resample=Image.BILINEAR)),
            np.asarray(pil_flip.rotate(-10, resample=Image.BILINEAR)),
        ]
    raise ValueError(f"unknown view_set: {view_set!r}")


_VIEW_COUNT = {"hflip": 2, "rot": 3, "basic": 4, "full": 6}


class TTADataset(Dataset):
    """Returns a stack of N TTA views (N, 3, 224, 224) per item."""

    def __init__(self, split_name: str = "test", view_set: str = "full"):
        self.df = load_split(split_name)
        self.cnn_transform = build_transforms(train=False)
        self.split_name = split_name
        self.view_set = view_set

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_256 = load_image(row["abs_path"])
        views = _tta_views(img_256, view_set=self.view_set)
        tensors = [self.cnn_transform(v) for v in views]
        stacked = torch.stack(tensors, dim=0)  # (N_views, 3, 224, 224)
        return stacked, int(row["class_idx"])


@torch.no_grad()
def tta_predict(
    checkpoint_path: Path,
    device: torch.device,
    split_name: str = "test",
    view_set: str = "full",
) -> dict[str, np.ndarray]:
    model = build_model().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    ds = TTADataset(split_name=split_name, view_set=view_set)
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    all_probs: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    n_views = _VIEW_COUNT[view_set]
    for stacked, y in loader:
        # stacked: (B, N, 3, 224, 224); flatten to (B*N, 3, 224, 224)
        B = stacked.shape[0]
        flat = stacked.view(B * n_views, 3, 224, 224).to(device)
        logits = model(flat)
        probs = torch.softmax(logits, dim=1).view(B, n_views, -1).mean(dim=1)
        all_probs.append(probs.cpu().numpy())
        all_true.append(y.numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_pred = y_prob.argmax(axis=1)
    y_true = np.concatenate(all_true, axis=0)
    return {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}


def main() -> None:
    parser = argparse.ArgumentParser(description="TTA inference for the DL classifier")
    parser.add_argument("--checkpoint", required=True, help="path to best_model.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default=None,
                        help="default: Results/dl_run_tta_<view_set>/")
    parser.add_argument("--model-name", default=None,
                        help="default: efficientnet_b0_tta_<view_set>")
    parser.add_argument("--views", default="full",
                        choices=["hflip", "rot", "basic", "full"],
                        help="TTA augmentation set")
    args = parser.parse_args()

    device = resolve_device(args.device)
    n_views = _VIEW_COUNT[args.views]
    print(f"[tta] device={device}  split={args.split}  views={args.views} "
          f"(n={n_views})  checkpoint={args.checkpoint}")

    out_dir = Path(args.output_dir) if args.output_dir else (
        RESULTS_DIR / f"dl_run_tta_{args.views}"
    )
    model_name = args.model_name or f"efficientnet_b0_tta_{args.views}"
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = tta_predict(
        Path(args.checkpoint),
        device=device,
        split_name=args.split,
        view_set=args.views,
    )

    np.savez(
        out_dir / f"dl_predictions_{args.split}.npz" if args.split != "test" else out_dir / "dl_predictions.npz",
        y_true=preds["y_true"],
        y_pred=preds["y_pred"],
        y_prob=preds["y_prob"],
    )

    # Only emit full results JSON for test split (val is instrumental)
    if args.split == "test":
        results = evaluate(
            preds["y_true"],
            preds["y_pred"],
            y_prob=preds["y_prob"],
            model_name=model_name,
        )
        print_summary(results)
        save_results(results, out_dir / RESULTS_FILENAME)
    else:
        print(f"[tta] wrote {args.split} predictions ({len(preds['y_true'])} rows)")


if __name__ == "__main__":
    main()
