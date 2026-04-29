"""Diagnostic 4 — DL learning curves from existing per-epoch run logs.

Both EfficientNet-B0-full and ConvNeXt V2-full's training scripts logged
train_loss, val_loss, val_macro_f1 per epoch in their run_log.json files
(no retraining required — we just parse + plot).

What we plot:
  - 4-panel figure: (top-left) EffNet-B0 train+val loss over epochs
                    (top-right) EffNet-B0 val macro-F1 over epochs
                    (bottom-left) ConvNeXt V2 train+val loss over epochs
                    (bottom-right) ConvNeXt V2 val macro-F1 over epochs
  - Stage-1 -> stage-2 boundary marked
  - Best-epoch marked
  - Verdict per pipeline annotated

Verdict heuristic per pipeline (looking at val_loss late in training):
  - "DL overfit": val_loss starts climbing while train_loss continues to fall
  - "saturation/leakage": val_loss tracks train_loss closely and both plateau
  - "early stopping triggered cleanly": val_macro_f1 saturated, training stopped
    on a plateau

Outputs:
  Results/diagnostics/dl_learning_curves.json   (parsed series + verdicts)
  Results/diagnostics/dl_learning_curves.png    (4-panel figure)

Usage:
  python -m analysis.diag_dl_learning_curves
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from shared.config import RESULTS_DIR


EFF_LOG = RESULTS_DIR / "dl_run_full" / "run_log.json"
CVX_LOG = RESULTS_DIR / "convnextv2_full_run" / "run_log.json"
OUT_DIR = RESULTS_DIR / "diagnostics"


def _parse_run_log(path: Path) -> dict:
    log = json.loads(path.read_text())
    epochs = log["epochs"]
    train_loss = np.array([e["train_loss"] for e in epochs], dtype=float)
    val_loss   = np.array([e["val_loss"]   for e in epochs], dtype=float)
    val_macro_f1 = np.array([e["val_macro_f1"] for e in epochs], dtype=float)
    stages = np.array([e["stage"] for e in epochs], dtype=int)
    epoch_idx = np.arange(1, len(epochs) + 1)

    # Stage-1 -> stage-2 transition (last index of stage 1)
    stage1_end = int(np.where(stages == 1)[0][-1] + 1)
    best_epoch_idx = int(np.argmax(val_macro_f1))
    best_epoch = int(epoch_idx[best_epoch_idx])
    best_val_f1 = float(val_macro_f1[best_epoch_idx])

    # Late-training overfit signal: in stage 2, does val_loss rise after a min?
    stage2_start = stage1_end
    stage2_train = train_loss[stage2_start:]
    stage2_val   = val_loss[stage2_start:]
    if len(stage2_val) >= 5:
        val_min_idx_in_s2 = int(np.argmin(stage2_val))
        val_min = float(stage2_val[val_min_idx_in_s2])
        val_end = float(stage2_val[-1])
        val_rebound = val_end - val_min
        train_rebound = float(stage2_train[-1] - stage2_train[val_min_idx_in_s2])

        if val_rebound > 0.05 and train_rebound < 0:
            verdict = (
                "DL overfit signature: stage-2 val loss rebounded "
                f"by {val_rebound:.3f} from minimum while train loss continued to fall."
            )
        elif val_rebound > 0.02:
            verdict = (
                f"Mild stage-2 val-loss rebound ({val_rebound:+.3f}) — early-stopping "
                "patience absorbed the worst of it but model is at the edge of overfitting."
            )
        else:
            verdict = (
                f"Saturation: val loss plateau in stage 2 (rebound {val_rebound:+.3f}), "
                "no overfit rebound detected."
            )
    else:
        verdict = "stage 2 too short to diagnose"

    return {
        "config":         log.get("config", {}),
        "epoch_idx":      epoch_idx.tolist(),
        "stages":         stages.tolist(),
        "train_loss":     train_loss.tolist(),
        "val_loss":       val_loss.tolist(),
        "val_macro_f1":   val_macro_f1.tolist(),
        "stage1_end":     stage1_end,
        "best_epoch":     best_epoch,
        "best_val_f1":    best_val_f1,
        "n_epochs_total": int(len(epochs)),
        "verdict":        verdict,
    }


def _plot_pipeline(ax_loss, ax_f1, parsed: dict, name: str) -> None:
    epochs = np.array(parsed["epoch_idx"])
    train_loss = np.array(parsed["train_loss"])
    val_loss   = np.array(parsed["val_loss"])
    val_f1     = np.array(parsed["val_macro_f1"])
    stage1_end = parsed["stage1_end"]
    best_epoch = parsed["best_epoch"]
    best_val_f1 = parsed["best_val_f1"]

    # ── Loss panel ─────────────────────────────────────────────────────────
    ax_loss.plot(epochs, train_loss, color="#4c72b0", linewidth=1.6, label="train loss")
    ax_loss.plot(epochs, val_loss,   color="#c44e52", linewidth=1.6, label="val loss")
    ax_loss.axvline(stage1_end + 0.5, color="black", linestyle="--", linewidth=1,
                    alpha=0.6, label=f"stage 1→2 boundary")
    ax_loss.axvline(best_epoch, color="#55a868", linestyle=":", linewidth=1.4,
                    label=f"best val-F1 epoch={best_epoch}")
    ax_loss.set_xlabel("epoch (cumulative across stages)")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title(f"{name} — train + val loss")
    ax_loss.legend(fontsize=8, loc="upper right")
    ax_loss.spines["top"].set_visible(False)
    ax_loss.spines["right"].set_visible(False)
    ax_loss.grid(axis="y", linestyle="--", alpha=0.4)

    # ── F1 panel ───────────────────────────────────────────────────────────
    ax_f1.plot(epochs, val_f1, color="#dd8452", linewidth=1.6, label="val macro-F1")
    ax_f1.axvline(stage1_end + 0.5, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax_f1.axvline(best_epoch, color="#55a868", linestyle=":", linewidth=1.4)
    ax_f1.scatter([best_epoch], [best_val_f1], color="#55a868", s=80, zorder=5,
                  label=f"best={best_val_f1:.4f}")
    ax_f1.set_xlabel("epoch (cumulative across stages)")
    ax_f1.set_ylabel("val macro-F1")
    ax_f1.set_title(f"{name} — val macro-F1")
    ax_f1.legend(fontsize=8, loc="lower right")
    ax_f1.spines["top"].set_visible(False)
    ax_f1.spines["right"].set_visible(False)
    ax_f1.grid(axis="y", linestyle="--", alpha=0.4)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    eff = _parse_run_log(EFF_LOG)
    cvx = _parse_run_log(CVX_LOG)

    print("[EfficientNet-B0 full]")
    print(f"  total epochs={eff['n_epochs_total']}  stage1_end={eff['stage1_end']}  "
          f"best_epoch={eff['best_epoch']}  best_val_f1={eff['best_val_f1']:.4f}")
    print(f"  verdict: {eff['verdict']}")
    print()
    print("[ConvNeXt V2 Base full]")
    print(f"  total epochs={cvx['n_epochs_total']}  stage1_end={cvx['stage1_end']}  "
          f"best_epoch={cvx['best_epoch']}  best_val_f1={cvx['best_val_f1']:.4f}")
    print(f"  verdict: {cvx['verdict']}")

    summary = {
        "efficientnet_b0_full": eff,
        "convnextv2_base_full": cvx,
    }
    out_json = OUT_DIR / "dl_learning_curves.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\n[save] {out_json}")

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    _plot_pipeline(axes[0, 0], axes[0, 1], eff, "EfficientNet-B0 (full)")
    _plot_pipeline(axes[1, 0], axes[1, 1], cvx, "ConvNeXt V2 Base (full)")

    fig.suptitle(
        "Deep-learning training curves on the full-dataset split (per-epoch logs)\n"
        f"EffNet verdict: {eff['verdict']}\n"
        f"ConvNeXt verdict: {cvx['verdict']}",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    out_png = OUT_DIR / "dl_learning_curves.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_png}")


if __name__ == "__main__":
    main()
