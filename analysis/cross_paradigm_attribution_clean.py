"""Cross-paradigm attribution figure on the CLEAN deduplicated test set.

For 6 selected test images, produces a 4-column figure:
  original CT slice | Classical occlusion saliency | EffNet-B0 Grad-CAM | ConvNeXt V2 Grad-CAM

Classical attribution is **occlusion sensitivity** (Zeiler & Fergus 2014) — slides
a patch across the image, recomputes the 138-dim feature vector, runs the
deployed SVM, and records the drop in confidence for the predicted class as
a spatial map. This is the model-agnostic equivalent of Grad-CAM.

DL attribution is true Grad-CAM (Selvaraju et al. 2017) on the last conv
feature map of each backbone.

Inputs:
  Results/classical_run_full/classical_predictions.npz
  Results/classical_run_full/classical_pipeline.pkl
  Results/dl_run_full/{dl_predictions.npz, best_model.pt}
  Results/convnextv2_full_run/{dl_predictions.npz, best_model.pt}

Outputs:
  Results/gradcam/cross_paradigm_clean.png
  Results/gradcam/cross_paradigm_clean_manifest.json

Usage:
  python -m analysis.cross_paradigm_attribution_clean
"""
from __future__ import annotations

import json
import pickle
import warnings
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from PIL import Image

from classical.features import extract_features
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


DATASET_ROOT = REPO_ROOT / "Updated_Dataset" / "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone_unique"
OUT_PNG = RESULTS_DIR / "gradcam" / "cross_paradigm_clean.png"
OUT_JSON = RESULTS_DIR / "gradcam" / "cross_paradigm_clean_manifest.json"


# ── Image selection ─────────────────────────────────────────────────────────

def _pick_clean_samples(seed: int = 0) -> list[dict]:
    cls = dict(np.load(RESULTS_DIR / "classical_run_full" / "classical_predictions.npz"))
    eff = dict(np.load(RESULTS_DIR / "dl_run_full" / "dl_predictions.npz"))
    cvx = dict(np.load(RESULTS_DIR / "convnextv2_full_run" / "dl_predictions.npz"))
    assert np.array_equal(cls["y_true"], eff["y_true"])
    assert np.array_equal(cls["y_true"], cvx["y_true"])
    y = cls["y_true"]
    cl_right = cls["y_pred"] == y
    ef_right = eff["y_pred"] == y
    cn_right = cvx["y_pred"] == y

    rng = np.random.default_rng(seed)

    buckets = {
        "all_correct":              np.where(cl_right & ef_right & cn_right)[0],
        "classical_right_dl_wrong": np.where(cl_right & ~ef_right & ~cn_right)[0],
        "dl_right_classical_wrong": np.where(~cl_right & ef_right & cn_right)[0],
        "all_wrong":                np.where(~cl_right & ~ef_right & ~cn_right)[0],
    }
    for k, v in buckets.items():
        print(f"  bucket {k}: {len(v)} images")

    # Quota: 1 all-correct (control), 2 classical-right-DL-wrong, 2 DL-right-classical-wrong, 1 all-wrong
    # Per-bucket class-priority targets to ensure visual diversity:
    quotas = [
        ("all_correct",              1, ["Tumor"]),                       # Tumor is uniformly easy on clean → good control
        ("classical_right_dl_wrong", 2, ["Cyst", "Stone"]),                # Sprint 3 finding: DL uniquely makes Cyst→Stone errors
        ("dl_right_classical_wrong", 2, ["Stone", "Cyst"]),                # Stone-class wins for DL
        ("all_wrong",                1, ["Stone"]),                        # universally hard case, usually Stone
    ]
    chosen: list[dict] = []
    for bucket, n, priority_classes in quotas:
        idxs = buckets[bucket]
        if len(idxs) == 0:
            print(f"  [SKIP] {bucket} bucket is empty")
            continue
        # Re-order idxs so priority classes come first
        priority_idx_lists = []
        for cname in priority_classes:
            cid = CLASSES.index(cname)
            cls_idxs = idxs[y[idxs] == cid]
            priority_idx_lists.append(cls_idxs)
        priority_idx_lists.append(idxs[~np.isin(y[idxs], [CLASSES.index(c) for c in priority_classes])])
        ordered = np.concatenate(priority_idx_lists)
        pick = ordered[:n]
        for i in pick:
            chosen.append({
                "split_idx": int(i),
                "bucket": bucket,
                "y_true": int(y[i]),
                "y_cls": int(cls["y_pred"][i]),
                "y_eff": int(eff["y_pred"][i]),
                "y_cvx": int(cvx["y_pred"][i]),
                "p_cls": cls["y_prob"][i],
                "p_eff": eff["y_prob"][i],
                "p_cvx": cvx["y_prob"][i],
            })
    return chosen


# ── Classical occlusion saliency ────────────────────────────────────────────

def _classical_predict_proba(img_uint8: np.ndarray, scaler, classifier) -> np.ndarray:
    """Extract 138-dim features, scale, predict_proba."""
    feats = extract_features(img_uint8).reshape(1, -1)
    feats_sc = scaler.transform(feats)
    return classifier.predict_proba(feats_sc)[0]


def _occlude_position(img_uint8: np.ndarray, y: int, x: int, patch_size: int,
                       scaler, classifier, baseline_conf: float, target_class: int
                       ) -> tuple[int, int, float]:
    """Occlude a patch at (y, x) with gray (128), predict, return drop in
    target_class confidence."""
    occluded = img_uint8.copy()
    h, w = occluded.shape
    y2 = min(y + patch_size, h)
    x2 = min(x + patch_size, w)
    occluded[y:y2, x:x2] = 128
    probs = _classical_predict_proba(occluded, scaler, classifier)
    drop = float(baseline_conf - probs[target_class])
    return (y, x, drop)


def classical_occlusion_saliency(img_uint8: np.ndarray, scaler, classifier,
                                 target_class: int,
                                 patch_size: int = 48, stride: int = 24,
                                 n_jobs: int = -1) -> np.ndarray:
    """Slide a patch over the image and record the prediction-confidence drop.

    Returns a (256, 256) float saliency map normalised to [0, 1]."""
    h, w = img_uint8.shape
    baseline = _classical_predict_proba(img_uint8, scaler, classifier)
    baseline_conf = float(baseline[target_class])

    positions = [(y, x)
                 for y in range(0, h - patch_size // 2, stride)
                 for x in range(0, w - patch_size // 2, stride)]

    results = Parallel(n_jobs=n_jobs)(
        delayed(_occlude_position)(img_uint8, y, x, patch_size, scaler, classifier,
                                    baseline_conf, target_class)
        for y, x in positions
    )

    saliency = np.zeros((h, w), dtype=np.float64)
    counts = np.zeros((h, w), dtype=np.float64)
    for y, x, drop in results:
        y2 = min(y + patch_size, h)
        x2 = min(x + patch_size, w)
        # Positive drop = patch was important (occluding it hurt the prediction)
        saliency[y:y2, x:x2] += max(drop, 0.0)
        counts[y:y2, x:x2] += 1
    saliency = saliency / np.maximum(counts, 1)
    # Normalise to [0, 1] for visualisation
    if saliency.max() > 0:
        saliency = saliency / saliency.max()
    return saliency


def _overlay_classical(display_img_uint8: np.ndarray, saliency: np.ndarray,
                       display_size: int = 384) -> np.ndarray:
    """Same overlay function as Grad-CAM (red-hot heatmap on grayscale)."""
    # Resize saliency to display size
    sal_resized = np.asarray(
        F.interpolate(
            torch.tensor(saliency, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            size=(display_size, display_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze().numpy()
    )
    return overlay(display_img_uint8, sal_resized)


# ── Figure ──────────────────────────────────────────────────────────────────

def build_figure(samples: list[dict], device: torch.device,
                 display_size: int = 384) -> None:
    test_df = load_split(
        "test",
        split_csv=str(SPLIT_CSV_FULL),
        dataset_root=str(DATASET_ROOT),
    )

    # Load classical
    with open(RESULTS_DIR / "classical_run_full" / "classical_pipeline.pkl", "rb") as f:
        warnings.filterwarnings("ignore", message="Trying to unpickle")
        cl_pipe = pickle.load(f)
    cl_scaler = cl_pipe["scaler"]
    cl_clf = cl_pipe["classifier"]

    # Load DL models
    eff = _load_and_eval_model(
        RESULTS_DIR / "dl_run_full" / "best_model.pt",
        "efficientnet_b0", 224, device,
    )
    cvx = _load_and_eval_model(
        RESULTS_DIR / "convnextv2_full_run" / "best_model.pt",
        "convnextv2_base", 384, device,
    )
    eff_cam = GradCAM(eff, _target_layer(eff))
    cvx_cam = GradCAM(cvx, _target_layer(cvx))

    bucket_labels = {
        "all_correct":              "All correct (control)",
        "classical_right_dl_wrong": "Classical right, DL wrong",
        "dl_right_classical_wrong": "DL right, Classical wrong",
        "all_wrong":                "All wrong",
    }

    n = len(samples)
    fig, axes = plt.subplots(n, 4, figsize=(15, 3.8 * n), squeeze=False)

    manifest: list[dict] = []

    for row_idx, s in enumerate(samples):
        row = test_df.iloc[s["split_idx"]]
        raw = load_image(row["abs_path"])  # 256x256 uint8
        display = _display_resize(raw, display_size)

        # Classical attribution: occlusion saliency for the predicted class
        print(f"\n[row {row_idx + 1}/{n}] split_idx={s['split_idx']}  "
              f"bucket={s['bucket']}  true={CLASSES[s['y_true']]}")
        print("  computing classical occlusion saliency …")
        cls_sal = classical_occlusion_saliency(
            raw, cl_scaler, cl_clf, target_class=s["y_cls"],
            patch_size=48, stride=24, n_jobs=-1,
        )
        cls_overlay = _overlay_classical(display, cls_sal, display_size=display_size)

        # DL Grad-CAMs
        print("  computing Grad-CAMs …")
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

        cls_tick = "✓" if s["y_cls"] == s["y_true"] else "✗"
        eff_tick = "✓" if s["y_eff"] == s["y_true"] else "✗"
        cvx_tick = "✓" if s["y_cvx"] == s["y_true"] else "✗"

        # Column 0: original
        ax = axes[row_idx, 0]
        ax.imshow(display, cmap="gray")
        ax.set_title(
            f"{bucket_labels[s['bucket']]}\n"
            f"true = {CLASSES[s['y_true']]}",
            fontsize=10,
        )
        ax.axis("off")

        # Column 1: classical occlusion saliency
        ax = axes[row_idx, 1]
        ax.imshow(cls_overlay)
        ax.set_title(
            f"Classical SVM (occlusion sensitivity)\n"
            f"→ {CLASSES[s['y_cls']]}  p={float(s['p_cls'][s['y_cls']]):.2f}  {cls_tick}",
            fontsize=10,
        )
        ax.axis("off")

        # Column 2: EffNet Grad-CAM
        ax = axes[row_idx, 2]
        ax.imshow(eff_overlay)
        ax.set_title(
            f"EffNet-B0 Grad-CAM\n"
            f"→ {CLASSES[s['y_eff']]}  p={float(s['p_eff'][s['y_eff']]):.2f}  {eff_tick}",
            fontsize=10,
        )
        ax.axis("off")

        # Column 3: ConvNeXt V2 Grad-CAM
        ax = axes[row_idx, 3]
        ax.imshow(cvx_overlay)
        ax.set_title(
            f"ConvNeXt V2 Grad-CAM\n"
            f"→ {CLASSES[s['y_cvx']]}  p={float(s['p_cvx'][s['y_cvx']]):.2f}  {cvx_tick}",
            fontsize=10,
        )
        ax.axis("off")

        manifest.append({
            "split_idx": s["split_idx"],
            "bucket": s["bucket"],
            "y_true": CLASSES[s["y_true"]],
            "classical_pred": CLASSES[s["y_cls"]],
            "classical_prob_predicted": float(s["p_cls"][s["y_cls"]]),
            "effnetb0_pred": CLASSES[s["y_eff"]],
            "effnetb0_prob_predicted": float(s["p_eff"][s["y_eff"]]),
            "convnextv2_pred": CLASSES[s["y_cvx"]],
            "convnextv2_prob_predicted": float(s["p_cvx"][s["y_cvx"]]),
        })

    fig.suptitle(
        "Cross-paradigm attribution on the deduplicated test set — "
        "Classical SVM (occlusion sensitivity) vs DL Grad-CAM (EffNet-B0 + ConvNeXt V2)",
        fontsize=11, y=1.005,
    )
    plt.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved -> {OUT_PNG}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"samples": manifest}, indent=2))
    print(f"saved -> {OUT_JSON}")


def main() -> None:
    device = resolve_device(None)
    print(f"device = {device}")

    print("\n=== picking samples from clean-test disagreement buckets ===")
    samples = _pick_clean_samples(seed=0)
    print(f"\nSelected {len(samples)} samples:")
    for s in samples:
        print(f"  idx={s['split_idx']}  bucket={s['bucket']:<28}  "
              f"true={CLASSES[s['y_true']]}  "
              f"cls={CLASSES[s['y_cls']]}  eff={CLASSES[s['y_eff']]}  "
              f"cvx={CLASSES[s['y_cvx']]}")

    print("\n=== building figure (classical occlusion ~60s per row, "
          f"~{60*len(samples)}s total + ~{15*len(samples)}s DL) ===")
    build_figure(samples, device=device)


if __name__ == "__main__":
    main()
