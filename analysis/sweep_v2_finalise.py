"""Finalise the v2 (bug-fixed) DL data-efficiency sweep.

Combines v2 sub-100% (from `Results/{convnextv2,effnetb0}_sweep_clean_v2/`)
with the protocol-matched 100% anchor:
  - ConvNeXt 100% = seed=0 cosine+60 result (raw from v1 sweep_summary,
    TTA from `Results/convnextv2_full_run_seed0_cos_tta_hflip/`).
  - EffNet 100% = v1 sweep frac_100 (no subsetting bug applied at 100%).

Outputs:
  Results/dl_sweep_clean_v2/sweep_summary_final.json
  Results/dl_sweep_clean_v2/sweep_curves.png       — matplotlib plot
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path("/Users/jameswu/Desktop/University/Year_5/Semester_1/BMET5933/Assignment2")
RESULTS = REPO / "Results"
OUT = RESULTS / "dl_sweep_clean_v2"


def _read_f1(path: Path) -> tuple[float | None, float | None, int | None]:
    """Return (macro_f1, accuracy, n_errors_test1888)."""
    if not path.exists():
        return None, None, None
    d = json.loads(path.read_text())
    f1 = float(d["macro_f1"])
    acc = float(d["accuracy"])
    n_test = int(d.get("n_test", 1888))
    errs = int(round((1.0 - acc) * n_test))
    return f1, acc, errs


def build_arch_curve(arch_dir_v2: str, arch_label: str,
                     raw_100: tuple[Path, str],
                     tta_100: tuple[Path, str]) -> list[dict]:
    """Build a 4-point curve. 10/25/50% from _v2 dir; 100% from explicit paths."""
    rows: list[dict] = []
    for frac in (10, 25, 50):
        tag = f"frac_{frac:03d}"
        raw_path = RESULTS / arch_dir_v2 / tag / "dl_results.json"
        tta_path = RESULTS / arch_dir_v2 / f"{tag}_tta_hflip" / "dl_results.json"
        raw_f1, raw_acc, raw_err = _read_f1(raw_path)
        tta_f1, _, _ = _read_f1(tta_path)
        rows.append({
            "fraction": frac,
            "n_train": int(round(8146 * frac / 100.0)),
            "raw_macro_f1": raw_f1,
            "tta_macro_f1": tta_f1,
            "raw_accuracy": raw_acc,
            "raw_errors": raw_err,
            "source": "v2",
            "source_raw_path": str(raw_path.relative_to(REPO)),
            "source_tta_path": str(tta_path.relative_to(REPO)),
        })

    # 100% — explicit anchor paths
    raw_path_100, raw_note = raw_100
    tta_path_100, tta_note = tta_100
    raw_f1, raw_acc, raw_err = _read_f1(raw_path_100)
    tta_f1, _, _ = _read_f1(tta_path_100)
    rows.append({
        "fraction": 100,
        "n_train": 8146,
        "raw_macro_f1": raw_f1,
        "tta_macro_f1": tta_f1,
        "raw_accuracy": raw_acc,
        "raw_errors": raw_err,
        "source": "v1 100% (no subsetting bug)",
        "source_raw_path": str(raw_path_100.relative_to(REPO)) if raw_path_100.exists() else f"MISSING: {raw_path_100}",
        "source_tta_path": str(tta_path_100.relative_to(REPO)) if tta_path_100.exists() else f"MISSING: {tta_path_100}",
        "source_raw_note": raw_note,
        "source_tta_note": tta_note,
    })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # ConvNeXt: 100% protocol-matched anchor is seed=0 cosine+60.
    # Raw not locally cached (only Drive); use Sprint5 baseline (constant LR + 30
    # epochs) as fallback OR use the v1 sweep_summary value if available.
    convnext_100_raw = RESULTS / "convnextv2_full_run" / "dl_results.json"  # Sprint 5 const-LR baseline (0.8219)
    convnext_100_tta = RESULTS / "convnextv2_full_run_seed0_cos_tta_hflip" / "dl_results.json"  # cosine+60 TTA

    # EffNet: v1 sweep 100% is protocol-matched (cosine+60 at 100% never hit the bug).
    effnet_100_raw = RESULTS / "effnetb0_sweep_clean" / "frac_100" / "dl_results.json"
    effnet_100_tta = RESULTS / "effnetb0_sweep_clean" / "frac_100_tta_hflip" / "dl_results.json"

    final = {
        "n_train_total": 8146,
        "n_test": 1888,
        "notes": [
            "v2 sub-100% retrained with split_csv passed to stratified_train_indices "
            "(commit 0c5c4a2 fix).",
            "ConvNeXt V2 100% RAW shown here is Sprint 5 baseline (constant LR + 30 ep, "
            "Sprint 5 clean retrain) at 0.8219 macro-F1. The protocol-matched cosine+60 "
            "100% raw is 0.8381 (per v1 sweep_summary); update this once "
            "convnextv2_full_run_seed0_cos is committed.",
            "ConvNeXt V2 100% TTA is the protocol-matched cosine+60 TTA result.",
            "EffNet-B0 100% (both raw + TTA) is the v1 cosine+60 result (no subsetting bug).",
        ],
        "convnextv2_base": build_arch_curve(
            "convnextv2_sweep_clean_v2", "ConvNeXt V2 Base",
            raw_100=(convnext_100_raw, "Sprint 5 constant-LR baseline (not protocol-matched)"),
            tta_100=(convnext_100_tta, "Sprint 5 + Tier 1A cosine+60 seed=0 TTA (protocol-matched)"),
        ),
        "efficientnet_b0": build_arch_curve(
            "effnetb0_sweep_clean_v2", "EfficientNet-B0",
            raw_100=(effnet_100_raw, "v1 sweep 100% raw (no subsetting bug at 100%)"),
            tta_100=(effnet_100_tta, "v1 sweep 100% TTA (no subsetting bug at 100%)"),
        ),
    }

    out_json = OUT / "sweep_summary_final.json"
    out_json.write_text(json.dumps(final, indent=2))
    print(f"saved -> {out_json}")

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    arch_colors = {"convnextv2_base": "#1f77b4", "efficientnet_b0": "#d62728"}
    arch_labels = {"convnextv2_base": "ConvNeXt V2 Base", "efficientnet_b0": "EfficientNet-B0"}

    # Classical sweep on clean from partner (Sprint 3 numbers were on leaky;
    # for clean classical we'd need partner's data — use partner's final SVM
    # at 100% = 0.9091 only, others not yet committed).
    classical_clean_100 = 0.9091  # from Results/classical_run_full/classical_results.json

    for ax, arch_key in zip(axes, ("convnextv2_base", "efficientnet_b0")):
        rows = final[arch_key]
        fracs = [r["fraction"] for r in rows]
        raws = [r["raw_macro_f1"] for r in rows]
        ttas = [r["tta_macro_f1"] for r in rows]

        ax.plot(fracs, raws, "o-", color=arch_colors[arch_key], linewidth=2,
                markersize=8, label=f"{arch_labels[arch_key]} raw")
        ax.plot(fracs, ttas, "s--", color=arch_colors[arch_key], linewidth=1.5,
                markersize=7, alpha=0.7, label=f"{arch_labels[arch_key]} + TTA hflip")
        ax.axhline(classical_clean_100, color="#2ca02c", linestyle=":",
                   linewidth=2, alpha=0.8, label="Classical SVM @ 100% (0.9091)")

        # Annotate each point
        for f, r, t in zip(fracs, raws, ttas):
            if r is not None:
                ax.annotate(f"{r:.3f}", (f, r),
                            textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=8, color=arch_colors[arch_key])

        ax.set_xlabel("Training fraction (%)", fontsize=11)
        ax.set_ylabel("Macro-F1 (clean test, n=1,888)", fontsize=11)
        ax.set_title(arch_labels[arch_key], fontsize=12)
        ax.set_xlim(0, 105)
        ax.set_ylim(0.55, 0.95)
        ax.set_xticks([10, 25, 50, 100])
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="lower right", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "DL data-efficiency sweep on the CLEAN dataset (v2, bug-fixed)\n"
        "Training data: 8,146 examples at 100%; test set: 1,888 (held-out, deduplicated)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out_png = OUT / "sweep_curves.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_png}")

    # Console summary
    print("\n=== FINAL v2 SWEEP TABLE ===\n")
    for arch_key in ("convnextv2_base", "efficientnet_b0"):
        print(f"--- {arch_labels[arch_key]} on clean dataset ---")
        print(f"{'fraction':>9}  {'n_train':>8}  {'raw F1':>8}  {'TTA F1':>8}  {'errors':>8}")
        for r in final[arch_key]:
            raw = f"{r['raw_macro_f1']:.4f}" if r["raw_macro_f1"] is not None else "  N/A "
            tta = f"{r['tta_macro_f1']:.4f}" if r["tta_macro_f1"] is not None else "  N/A "
            err = f"{r['raw_errors']}" if r["raw_errors"] is not None else "N/A"
            print(f"{r['fraction']:>8}%  {r['n_train']:>8}  {raw}  {tta}  {err:>8}")
        print()


if __name__ == "__main__":
    main()
