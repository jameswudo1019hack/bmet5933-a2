"""Cross-architecture Grad-CAM comparison.

Produces a side-by-side figure showing what EfficientNet-B0 and
ConvNeXt V2 attend to on the same test images. Central interpretability
figure for the paper under framing v2 — the point is not the raw accuracy
differences between architectures, but *what each architecture looks at*.

Target layers:
  - EfficientNet-B0: model.features[-1]  (7x7 feature map at 224 input)
  - ConvNeXt V2 Base: model.stages[-1]  (12x12 feature map at 384 input)

Sample selection:
  - Both models came from the full-dataset split (split_full.csv),
    evaluated on the same 1867 test images.
  - Pick from the paired-disagreement buckets:
      both correct        -> control (where do models agree?)
      only EffNet wrong   -> where ConvNeXt V2's capacity helps
      only ConvNeXt wrong -> reverse direction

Usage:
    python -m deep_learning.gradcam_compare \\
        --effnet-ckpt Results/dl_run_full/best_model.pt \\
        --convnext-ckpt Results/convnextv2_full_run/best_model.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from deep_learning.dataset import build_transforms
from deep_learning.gradcam import GradCAM, overlay
from deep_learning.model import build_model
from deep_learning.train import resolve_device
from shared.config import CLASSES, RESULTS_DIR, SPLIT_CSV_FULL
from shared.preprocessing import load_image, load_split


# ─── Per-architecture target layer ────────────────────────────────────────

def _target_layer(model: torch.nn.Module):
    """Return the architecture's last conv feature-map module."""
    if hasattr(model, "features"):         # torchvision EfficientNet
        return model.features[-1]
    if hasattr(model, "stages"):           # timm ConvNeXt V2
        return model.stages[-1]
    raise ValueError(f"Unknown architecture {type(model).__name__}")


# ─── Sample selection from paired disagreements ───────────────────────────

def _pick_test_samples(
    eff_preds_path: Path, cvx_preds_path: Path, seed: int = 0
) -> list[dict]:
    """Return a list of sample dicts from the disagreement buckets.

    Each dict: {
      'split_idx': int (into test split DataFrame),
      'bucket': 'both_correct' | 'only_effnet_wrong' | 'only_convnext_wrong',
      'y_true': int, 'y_eff': int, 'y_cvx': int,
      'p_eff': np.ndarray (4,), 'p_cvx': np.ndarray (4,),
    }
    """
    eff = dict(np.load(eff_preds_path))
    cvx = dict(np.load(cvx_preds_path))
    assert np.array_equal(eff["y_true"], cvx["y_true"]), "y_true mismatch"
    y = eff["y_true"]

    eff_wrong = y != eff["y_pred"]
    cvx_wrong = y != cvx["y_pred"]

    buckets = {
        "both_correct":        np.where(~eff_wrong & ~cvx_wrong)[0],
        "only_effnet_wrong":   np.where(eff_wrong & ~cvx_wrong)[0],
        "only_convnext_wrong": np.where(~eff_wrong & cvx_wrong)[0],
    }

    rng = np.random.default_rng(seed)
    quotas = [
        ("both_correct", 2),
        ("only_effnet_wrong", 3),
        ("only_convnext_wrong", 1),
    ]
    chosen: list[dict] = []
    for bucket, n in quotas:
        idxs = buckets[bucket]
        if len(idxs) == 0:
            continue
        # Prefer samples where the models are highly confident in their (different) answers,
        # i.e. pick the ones where the models visibly disagree.
        if bucket.startswith("only_"):
            # sort by confidence of the wrong prediction
            wrong_model_preds = eff["y_pred"] if "effnet" in bucket else cvx["y_pred"]
            wrong_model_probs = eff["y_prob"] if "effnet" in bucket else cvx["y_prob"]
            conf = wrong_model_probs[idxs, wrong_model_preds[idxs]]
            order = idxs[np.argsort(-conf)]  # most confidently-wrong first
            pick = order[:n]
        else:
            pick = rng.choice(idxs, size=min(n, len(idxs)), replace=False)

        for i in pick:
            chosen.append({
                "split_idx": int(i),
                "bucket": bucket,
                "y_true": int(y[i]),
                "y_eff": int(eff["y_pred"][i]),
                "y_cvx": int(cvx["y_pred"][i]),
                "p_eff": eff["y_prob"][i],
                "p_cvx": cvx["y_prob"][i],
            })
    return chosen


# ─── Run Grad-CAM for one model on a list of samples ──────────────────────

@torch.no_grad()
def _load_and_eval_model(
    checkpoint_path: Path, arch: str, image_size: int, device: torch.device
) -> torch.nn.Module:
    model = build_model(name=arch, image_size=image_size).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    return model.eval()


def _gradcam_for_sample(
    model: torch.nn.Module,
    cam: GradCAM,
    raw_256: np.ndarray,
    image_size: int,
    class_idx: int,
    device: torch.device,
) -> np.ndarray:
    """Return a (image_size, image_size) Grad-CAM heatmap in [0, 1]."""
    # Use the non-augmenting transform at the architecture's target resolution
    tfm = build_transforms(train=False, image_size=image_size)
    x = tfm(raw_256).unsqueeze(0).to(device)
    heatmap = cam.compute(x, class_idx=class_idx)
    return heatmap


def _display_resize(raw_256: np.ndarray, size: int) -> np.ndarray:
    """Bilinear resize of uint8 grayscale to display size."""
    return np.asarray(
        F.interpolate(
            torch.tensor(raw_256, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        ).squeeze().numpy(),
        dtype=np.uint8,
    )


# ─── Figure composition ───────────────────────────────────────────────────

def build_figure(
    samples: list[dict],
    effnet_ckpt: Path,
    convnext_ckpt: Path,
    output_path: Path,
    device: torch.device,
    display_size: int = 384,
) -> Path:
    test_df = load_split("test", split_csv=str(SPLIT_CSV_FULL),
                         dataset_root=str(Path("Dataset/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone").resolve()))

    eff = _load_and_eval_model(effnet_ckpt, "efficientnet_b0", 224, device)
    cvx = _load_and_eval_model(convnext_ckpt, "convnextv2_base", 384, device)

    eff_cam = GradCAM(eff, _target_layer(eff))
    cvx_cam = GradCAM(cvx, _target_layer(cvx))

    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(11, 3.6 * n), squeeze=False)

    bucket_labels = {
        "both_correct":        "Both correct",
        "only_effnet_wrong":   "Only EffNet-B0 wrong",
        "only_convnext_wrong": "Only ConvNeXt V2 wrong",
    }

    for row_idx, s in enumerate(samples):
        row = test_df.iloc[s["split_idx"]]
        raw = load_image(row["abs_path"])  # 256x256 uint8
        display = _display_resize(raw, display_size)

        # Compute Grad-CAM for each architecture for its PREDICTED class
        eff_heatmap = _gradcam_for_sample(eff, eff_cam, raw, 224, s["y_eff"], device)
        cvx_heatmap = _gradcam_for_sample(cvx, cvx_cam, raw, 384, s["y_cvx"], device)

        # Upsample EffNet heatmap to 384 for a fair visual side-by-side
        eff_heatmap_384 = np.asarray(
            F.interpolate(
                torch.tensor(eff_heatmap, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                size=(display_size, display_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze().numpy()
        )
        eff_overlay = overlay(display, eff_heatmap_384)
        cvx_overlay = overlay(display, cvx_heatmap)

        # Column 1: original
        ax = axes[row_idx, 0]
        ax.imshow(display, cmap="gray")
        title = f"{bucket_labels[s['bucket']]}\ntrue = {CLASSES[s['y_true']]}"
        ax.set_title(title, fontsize=9)
        ax.axis("off")

        # Column 2: EffNet-B0 Grad-CAM
        ax = axes[row_idx, 1]
        ax.imshow(eff_overlay)
        p = float(s["p_eff"][s["y_eff"]])
        tick = "✓" if s["y_eff"] == s["y_true"] else "✗"
        ax.set_title(f"EffNet-B0 @ 224  →  {CLASSES[s['y_eff']]}  p={p:.2f}  {tick}",
                     fontsize=9)
        ax.axis("off")

        # Column 3: ConvNeXt V2 Grad-CAM
        ax = axes[row_idx, 2]
        ax.imshow(cvx_overlay)
        p = float(s["p_cvx"][s["y_cvx"]])
        tick = "✓" if s["y_cvx"] == s["y_true"] else "✗"
        ax.set_title(f"ConvNeXt V2 @ 384  →  {CLASSES[s['y_cvx']]}  p={p:.2f}  {tick}",
                     fontsize=9)
        ax.axis("off")

    fig.suptitle(
        "Cross-architecture Grad-CAM — what each DL model attends to on the same CT slices",
        fontsize=12, y=0.998,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--effnet-ckpt",
                        default=str(RESULTS_DIR / "dl_run_full" / "best_model.pt"))
    parser.add_argument("--convnext-ckpt",
                        default=str(RESULTS_DIR / "convnextv2_full_run" / "best_model.pt"))
    parser.add_argument("--effnet-preds",
                        default=str(RESULTS_DIR / "dl_run_full" / "dl_predictions.npz"))
    parser.add_argument("--convnext-preds",
                        default=str(RESULTS_DIR / "convnextv2_full_run" / "dl_predictions.npz"))
    parser.add_argument("--output",
                        default=str(RESULTS_DIR / "gradcam" / "cross_architecture.png"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"[gradcam_compare] device={device}")

    samples = _pick_test_samples(
        Path(args.effnet_preds), Path(args.convnext_preds), seed=args.seed
    )
    print(f"[gradcam_compare] selected {len(samples)} samples")
    for s in samples:
        t = CLASSES[s["y_true"]]
        e = CLASSES[s["y_eff"]]
        c = CLASSES[s["y_cvx"]]
        print(f"  {s['bucket']:<22}  true={t:<6}  eff={e:<6}  cvx={c:<6}  idx={s['split_idx']}")

    out = build_figure(
        samples,
        Path(args.effnet_ckpt),
        Path(args.convnext_ckpt),
        Path(args.output),
        device=device,
    )
    print(f"[gradcam_compare] wrote {out}")


if __name__ == "__main__":
    main()
