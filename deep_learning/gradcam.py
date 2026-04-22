"""Grad-CAM visualisation for the EfficientNet-B0 classifier.

Implements the Grad-CAM algorithm directly (no third-party dependency):
    L_GradCAM(c) = ReLU( sum_k  alpha_k^c * A^k )

where A^k is the activation map of the target layer's k-th channel and
alpha_k^c is the global-average-pooled gradient of class c's logit
w.r.t. A^k. The result is bilinearly upsampled to the input resolution
and overlaid on the grayscale CT slice.

Target layer: the final feature stage of EfficientNet-B0 (`features[-1]`),
which outputs a (batch, 1280, 7, 7) feature map for a 224x224 input.
This is the standard target for EfficientNet Grad-CAM.

Usage:
    python -m deep_learning.gradcam \\
        --checkpoint Results/dl_run/best_model.pt \\
        --output-dir Results/gradcam
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from deep_learning.dataset import build_transforms
from deep_learning.model import build_model
from deep_learning.train import resolve_device
from shared.config import CLASSES, CLASS_TO_IDX, RESULTS_DIR
from shared.preprocessing import load_image, load_split


class GradCAM:
    """Grad-CAM for a single target layer."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model.eval()
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inp, out: torch.Tensor) -> None:
        self.activations = out.detach()

    def _save_gradient(self, _module, _grad_in, grad_out) -> None:
        self.gradients = grad_out[0].detach()

    def compute(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        """Return a (H, W) Grad-CAM heatmap normalised to [0, 1]."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        score = logits[0, class_idx]
        score.backward()

        # alpha_k^c = global average pool of gradients over spatial dims
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)

        # Upsample to input resolution
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


def overlay(original_grayscale: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a [0,1] heatmap on top of a uint8 grayscale image. Returns RGB uint8."""
    base = np.stack([original_grayscale] * 3, axis=-1).astype(np.float32) / 255.0
    cmap = plt.get_cmap("jet")
    hm_rgba = cmap(heatmap)[..., :3]  # drop alpha channel
    blended = (1 - alpha) * base + alpha * hm_rgba
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def make_figure(
    model_checkpoint: Path,
    selections: list[dict],
    output_dir: Path,
    device: torch.device,
) -> Path:
    """Build a grid of (original | grad-cam) panels, one per selected image."""
    model = build_model().to(device)
    state = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(state)

    target_layer = model.features[-1]
    cam = GradCAM(model, target_layer)
    tfm = build_transforms(train=False)

    n = len(selections)
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n), squeeze=False)

    for i, sel in enumerate(selections):
        raw = load_image(sel["abs_path"])  # uint8, 256x256
        x = tfm(raw).unsqueeze(0).to(device)

        heatmap = cam.compute(x, class_idx=sel["pred_idx"])

        # Resize raw image to match heatmap resolution for clean overlay
        hm_224 = heatmap  # already 224x224 because x is 224x224
        raw_224 = np.asarray(
            torch.nn.functional.interpolate(
                torch.tensor(raw, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            ).squeeze().numpy(),
            dtype=np.uint8,
        )
        blended = overlay(raw_224, hm_224)

        axes[i, 0].imshow(raw_224, cmap="gray")
        axes[i, 0].set_title(sel["title_left"], fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(blended)
        axes[i, 1].set_title(sel["title_right"], fontsize=9)
        axes[i, 1].axis("off")

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "gradcam_panel.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def pick_samples(predictions_npz: Path) -> list[dict]:
    """Pick a representative set: one correct per class + four notable errors."""
    data = np.load(predictions_npz)
    y_true = data["y_true"]
    y_pred = data["y_pred"]
    y_prob = data["y_prob"]

    test_df = load_split("test").reset_index(drop=True)
    correct_mask = y_true == y_pred

    rng = np.random.default_rng(0)
    selections: list[dict] = []

    # One high-confidence correct per class
    for cls_idx, cls_name in enumerate(CLASSES):
        idxs = np.where(correct_mask & (y_true == cls_idx))[0]
        if len(idxs) == 0:
            continue
        # Pick the highest-confidence correct prediction
        confidences = y_prob[idxs, cls_idx]
        chosen = idxs[int(confidences.argmax())]
        row = test_df.iloc[int(chosen)]
        selections.append(
            {
                "abs_path": row["abs_path"],
                "pred_idx": int(y_pred[chosen]),
                "title_left": f"{cls_name}  (correct)",
                "title_right": f"Grad-CAM @ {cls_name}  p={y_prob[chosen, cls_idx]:.3f}",
            }
        )

    # Up to four representative errors, preferring high-confidence wrong predictions
    error_idxs = np.where(~correct_mask)[0]
    if len(error_idxs) > 0:
        wrong_conf = y_prob[error_idxs, y_pred[error_idxs]]
        order = error_idxs[np.argsort(-wrong_conf)]  # most confidently-wrong first
        for chosen in order[:4]:
            row = test_df.iloc[int(chosen)]
            true_cls = CLASSES[int(y_true[chosen])]
            pred_cls = CLASSES[int(y_pred[chosen])]
            selections.append(
                {
                    "abs_path": row["abs_path"],
                    "pred_idx": int(y_pred[chosen]),
                    "title_left": f"true {true_cls}  (ERROR)",
                    "title_right": f"Grad-CAM @ {pred_cls}  p={y_prob[chosen, int(y_pred[chosen])]:.3f}",
                }
            )

    return selections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--predictions", default=str(RESULTS_DIR / "dl_run" / "dl_predictions.npz"))
    parser.add_argument("--output-dir", default=str(RESULTS_DIR / "gradcam"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    predictions = Path(args.predictions)
    assert predictions.exists(), f"missing {predictions}; run deep_learning.predict first"

    selections = pick_samples(predictions)
    print(f"[gradcam] selected {len(selections)} panels")
    for s in selections:
        print(f"  {s['title_left']:<25}  |  {s['title_right']}")

    out = make_figure(
        Path(args.checkpoint),
        selections,
        Path(args.output_dir),
        device=device,
    )
    print(f"[gradcam] wrote {out}")

    # Also dump the selection manifest alongside for reproducibility
    manifest = [
        {k: v for k, v in s.items() if k != "pred_idx"} | {"pred_idx": s["pred_idx"]}
        for s in selections
    ]
    manifest_path = Path(args.output_dir) / "gradcam_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[gradcam] wrote {manifest_path}")


if __name__ == "__main__":
    main()
