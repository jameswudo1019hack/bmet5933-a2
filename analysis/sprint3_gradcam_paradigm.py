"""Sprint 3 — cross-paradigm Grad-CAM (classical-XGB vs EffNet-B0 vs ConvNeXt V2).

Extends the Sprint 2 cross-architecture Grad-CAM (`deep_learning/gradcam_compare.py`,
which compared EffNet vs ConvNeXt only) with classical-vs-DL disagreement buckets.

Buckets (3-way: classical-XGB / EffNet-B0-full / ConvNeXt V2-full predictions):
  classical_right_dl_wrong  : classical correct, both DL wrong
                              (where the classical paradigm uniquely succeeds)
  classical_wrong_dl_right  : classical wrong, both DL correct
                              (where the DL paradigm uniquely succeeds)
  all_three_wrong           : universal hard case
                              (the dataset's irreducible difficulty)

Within each bucket we prioritise samples by their relevance to the paradigm-stable
narrative: Cyst->Stone DL-only errors (where classical wins visually), and
Stone->Cyst classical-only errors (where DL wins visually).

Outputs:
  Results/gradcam/cross_paradigm_disagreement.png
  Results/gradcam/gradcam_paradigm_manifest.json

Usage:
  python -m analysis.sprint3_gradcam_paradigm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from deep_learning.gradcam import GradCAM, overlay
from deep_learning.gradcam_compare import (
    _display_resize,
    _gradcam_for_sample,
    _load_and_eval_model,
    _target_layer,
)
from deep_learning.train import resolve_device
from shared.config import CLASSES, REPO_ROOT, RESULTS_DIR, SPLIT_CSV_FULL
from shared.preprocessing import load_image, load_split


# ── Sample selection from 3-way classical / EffNet / ConvNeXt disagreement ──

def _pick_paradigm_samples(
    classical_preds_path: Path,
    eff_preds_path: Path,
    cvx_preds_path: Path,
    seed: int = 0,
) -> list[dict]:
    cls = dict(np.load(classical_preds_path))
    eff = dict(np.load(eff_preds_path))
    cvx = dict(np.load(cvx_preds_path))
    assert np.array_equal(cls["y_true"], eff["y_true"])
    assert np.array_equal(cls["y_true"], cvx["y_true"])

    y = cls["y_true"]
    cls_wrong = y != cls["y_pred"]
    eff_wrong = y != eff["y_pred"]
    cvx_wrong = y != cvx["y_pred"]

    classical_right_dl_wrong = np.where(~cls_wrong & eff_wrong & cvx_wrong)[0]
    classical_wrong_dl_right = np.where(cls_wrong & ~eff_wrong & ~cvx_wrong)[0]
    all_three_wrong          = np.where(cls_wrong & eff_wrong & cvx_wrong)[0]

    print(f"[buckets] classical_right_dl_wrong: {len(classical_right_dl_wrong)}")
    print(f"[buckets] classical_wrong_dl_right: {len(classical_wrong_dl_right)}")
    print(f"[buckets] all_three_wrong:          {len(all_three_wrong)}")

    rng = np.random.default_rng(seed)

    def _prioritise(idxs: np.ndarray, true_class: str | None,
                    pred_class_dl: str | None) -> np.ndarray:
        """Move samples matching (true, pred) to the front; rest in order."""
        if true_class is None:
            return idxs
        true_idx = CLASSES.index(true_class)
        if pred_class_dl is None:
            mask = y[idxs] == true_idx
        else:
            pred_idx = CLASSES.index(pred_class_dl)
            mask = (y[idxs] == true_idx) & (eff["y_pred"][idxs] == pred_idx)
        return np.concatenate([idxs[mask], idxs[~mask]])

    chosen: list[dict] = []

    # Up to 3 of "classical_right_dl_wrong", priority: Cyst -> Stone DL errors
    pri = _prioritise(classical_right_dl_wrong, true_class="Cyst", pred_class_dl="Stone")
    for i in pri[:3]:
        chosen.append({
            "split_idx": int(i),
            "bucket": "classical_right_dl_wrong",
            "y_true": int(y[i]),
            "y_cls": int(cls["y_pred"][i]),
            "y_eff": int(eff["y_pred"][i]),
            "y_cvx": int(cvx["y_pred"][i]),
            "p_cls": cls["y_prob"][i],
            "p_eff": eff["y_prob"][i],
            "p_cvx": cvx["y_prob"][i],
        })

    # Up to 2 of "classical_wrong_dl_right", priority: Stone -> Cyst classical errors
    pri = _prioritise(classical_wrong_dl_right, true_class="Stone", pred_class_dl=None)
    for i in pri[:2]:
        chosen.append({
            "split_idx": int(i),
            "bucket": "classical_wrong_dl_right",
            "y_true": int(y[i]),
            "y_cls": int(cls["y_pred"][i]),
            "y_eff": int(eff["y_pred"][i]),
            "y_cvx": int(cvx["y_pred"][i]),
            "p_cls": cls["y_prob"][i],
            "p_eff": eff["y_prob"][i],
            "p_cvx": cvx["y_prob"][i],
        })

    # 1 of "all_three_wrong"
    if len(all_three_wrong) > 0:
        i = int(all_three_wrong[0])
        chosen.append({
            "split_idx": i,
            "bucket": "all_three_wrong",
            "y_true": int(y[i]),
            "y_cls": int(cls["y_pred"][i]),
            "y_eff": int(eff["y_pred"][i]),
            "y_cvx": int(cvx["y_pred"][i]),
            "p_cls": cls["y_prob"][i],
            "p_eff": eff["y_prob"][i],
            "p_cvx": cvx["y_prob"][i],
        })

    return chosen


# ── Figure composition (3 columns: original / EffNet CAM / ConvNeXt CAM) ─────

def build_figure(
    samples: list[dict],
    effnet_ckpt: Path,
    convnext_ckpt: Path,
    output_path: Path,
    device: torch.device,
    display_size: int = 384,
) -> Path:
    test_df = load_split(
        "test",
        split_csv=str(SPLIT_CSV_FULL),
        dataset_root=str((REPO_ROOT / "Dataset" / "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone").resolve()),
    )

    eff = _load_and_eval_model(effnet_ckpt,   "efficientnet_b0", 224, device)
    cvx = _load_and_eval_model(convnext_ckpt, "convnextv2_base", 384, device)

    eff_cam = GradCAM(eff, _target_layer(eff))
    cvx_cam = GradCAM(cvx, _target_layer(cvx))

    bucket_labels = {
        "classical_right_dl_wrong": "Classical right, DL wrong",
        "classical_wrong_dl_right": "Classical wrong, DL right",
        "all_three_wrong":          "All three wrong (hard case)",
    }

    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(11, 3.6 * n), squeeze=False)

    for row_idx, s in enumerate(samples):
        row = test_df.iloc[s["split_idx"]]
        raw = load_image(row["abs_path"])  # 256x256 uint8
        display = _display_resize(raw, display_size)

        eff_heatmap = _gradcam_for_sample(eff, eff_cam, raw, 224, s["y_eff"], device)
        cvx_heatmap = _gradcam_for_sample(cvx, cvx_cam, raw, 384, s["y_cvx"], device)

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

        # Column 1: original + classical's verdict
        cls_tick = "✓" if s["y_cls"] == s["y_true"] else "✗"
        p_cls = float(s["p_cls"][s["y_cls"]])
        ax = axes[row_idx, 0]
        ax.imshow(display, cmap="gray")
        ax.set_title(
            f"{bucket_labels[s['bucket']]}\n"
            f"true = {CLASSES[s['y_true']]}  ·  "
            f"classical-XGB → {CLASSES[s['y_cls']]} ({p_cls:.2f}) {cls_tick}",
            fontsize=9,
        )
        ax.axis("off")

        # Column 2: EffNet-B0 Grad-CAM
        eff_tick = "✓" if s["y_eff"] == s["y_true"] else "✗"
        p_eff = float(s["p_eff"][s["y_eff"]])
        ax = axes[row_idx, 1]
        ax.imshow(eff_overlay)
        ax.set_title(
            f"EffNet-B0 @ 224  →  {CLASSES[s['y_eff']]}  p={p_eff:.2f}  {eff_tick}",
            fontsize=9,
        )
        ax.axis("off")

        # Column 3: ConvNeXt V2 Grad-CAM
        cvx_tick = "✓" if s["y_cvx"] == s["y_true"] else "✗"
        p_cvx = float(s["p_cvx"][s["y_cvx"]])
        ax = axes[row_idx, 2]
        ax.imshow(cvx_overlay)
        ax.set_title(
            f"ConvNeXt V2 @ 384  →  {CLASSES[s['y_cvx']]}  p={p_cvx:.2f}  {cvx_tick}",
            fontsize=9,
        )
        ax.axis("off")

    fig.suptitle(
        "Cross-paradigm Grad-CAM — classical-XGBoost vs DL on the same CT slices",
        fontsize=12, y=0.998,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_manifest(samples: list[dict], path: Path) -> None:
    out = []
    for s in samples:
        out.append({
            "split_idx": s["split_idx"],
            "bucket": s["bucket"],
            "y_true": CLASSES[s["y_true"]],
            "classical_xgb_pred": CLASSES[s["y_cls"]],
            "classical_xgb_prob_predicted": float(s["p_cls"][s["y_cls"]]),
            "effnetb0_pred": CLASSES[s["y_eff"]],
            "effnetb0_prob_predicted": float(s["p_eff"][s["y_eff"]]),
            "convnextv2_pred": CLASSES[s["y_cvx"]],
            "convnextv2_prob_predicted": float(s["p_cvx"][s["y_cvx"]]),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"samples": out}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classical-preds",
                        default=str(RESULTS_DIR / "classical_run_full" / "classical_predictions.npz"))
    parser.add_argument("--effnet-ckpt",
                        default=str(RESULTS_DIR / "dl_run_full" / "best_model.pt"))
    parser.add_argument("--convnext-ckpt",
                        default=str(RESULTS_DIR / "convnextv2_full_run" / "best_model.pt"))
    parser.add_argument("--effnet-preds",
                        default=str(RESULTS_DIR / "dl_run_full" / "dl_predictions.npz"))
    parser.add_argument("--convnext-preds",
                        default=str(RESULTS_DIR / "convnextv2_full_run" / "dl_predictions.npz"))
    parser.add_argument("--output",
                        default=str(RESULTS_DIR / "gradcam" / "cross_paradigm_disagreement.png"))
    parser.add_argument("--manifest",
                        default=str(RESULTS_DIR / "gradcam" / "gradcam_paradigm_manifest.json"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"[gradcam_paradigm] device={device}")

    samples = _pick_paradigm_samples(
        Path(args.classical_preds),
        Path(args.effnet_preds),
        Path(args.convnext_preds),
        seed=args.seed,
    )
    print(f"[gradcam_paradigm] selected {len(samples)} samples")
    for s in samples:
        t = CLASSES[s["y_true"]]
        c = CLASSES[s["y_cls"]]
        e = CLASSES[s["y_eff"]]
        v = CLASSES[s["y_cvx"]]
        print(f"  {s['bucket']:<26}  true={t:<6}  cls={c:<6}  eff={e:<6}  cvx={v:<6}  idx={s['split_idx']}")

    out = build_figure(
        samples,
        Path(args.effnet_ckpt),
        Path(args.convnext_ckpt),
        Path(args.output),
        device=device,
    )
    _save_manifest(samples, Path(args.manifest))
    print(f"[gradcam_paradigm] wrote {out}")
    print(f"[gradcam_paradigm] manifest -> {args.manifest}")


if __name__ == "__main__":
    main()
