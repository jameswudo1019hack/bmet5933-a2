"""Diagnostic 2 — XGBoost learning curves on the deployed full-classical pipeline.

XGBoost has a built-in `evals_result_` that records loss and error per
boosting round when an `eval_set` is supplied to `fit()`. The original
classical/train.py did not use this, so we refit here with `eval_set` and
read out the per-round metrics.

What we plot:
  - train_mlogloss   vs val_mlogloss   over n_estimators
  - train_merror     vs val_merror     over n_estimators

What it diagnoses:
  - If val_mlogloss minimum occurs at n_estimators < 200, then the deployed
    XGB at n_estimators=200 is over-trained (more trees did not improve val).
  - If train_merror -> 0 while val_merror plateaus far above 0, that is the
    textbook classical-overfit signature.
  - If both train and val curves converge together at low values, that is
    the saturation pattern (probably patient leakage at dataset level, not
    classifier overfitting).

We fit on the same StandardScaler + PCA(50) projections as the deployed
pipeline so the curves are comparable to the deployed model's metrics.

Outputs:
  Results/diagnostics/xgb_learning_curves.json   (per-round series + verdict)
  Results/diagnostics/xgb_learning_curves.png    (4-panel figure)

Usage:
  python -m analysis.diag_xgb_learning_curves
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

from shared.config import RESULTS_DIR, SEED


PIPELINE_PATH = RESULTS_DIR / "classical_run_full" / "classical_pipeline.pkl"
FEATURES_DIR = RESULTS_DIR / "classical_features_full"
OUT_DIR = RESULTS_DIR / "diagnostics"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load deployed pipeline (we want its scaler + PCA, not its classifier)
    with open(PIPELINE_PATH, "rb") as f:
        pipeline = pickle.load(f)
    scaler = pipeline["scaler"]
    pca = pipeline["pca"]
    best_params = pipeline["best_params"]
    print(f"[load] deployed best_params = {best_params}")

    # Load cached features
    train_npz = np.load(FEATURES_DIR / "train_frac100.npz")
    val_npz   = np.load(FEATURES_DIR / "val.npz")
    test_npz  = np.load(FEATURES_DIR / "test.npz")
    X_train, y_train = train_npz["X"], train_npz["y"]
    X_val,   y_val   = val_npz["X"],   val_npz["y"]
    X_test,  y_test  = test_npz["X"],  test_npz["y"]
    print(f"[load] train={X_train.shape}  val={X_val.shape}  test={X_test.shape}")

    # NB: deployed pipeline's scaler+PCA were fit on train+val combined for the
    # final classifier. For a clean train-vs-val learning curve we want a
    # scaler+PCA fit on train ONLY so the val signal is genuinely held-out.
    # This matches what train.py does in its grid-search phase before final retrain.
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from classical.config import PCA_N_COMPONENTS

    train_scaler = StandardScaler()
    X_train_sc = train_scaler.fit_transform(X_train)
    pca_n = min(PCA_N_COMPONENTS, X_train_sc.shape[0] - 1, X_train_sc.shape[1])
    train_pca = PCA(n_components=pca_n, svd_solver="full", random_state=SEED)
    X_train_pca = train_pca.fit_transform(X_train_sc)
    X_val_pca = train_pca.transform(train_scaler.transform(X_val))
    X_test_pca = train_pca.transform(train_scaler.transform(X_test))
    print(f"[fit] train-only scaler+PCA: 108 -> {X_train_pca.shape[1]}")

    # Use a wider n_estimators ceiling than the deployed 200 to see if val keeps improving
    # past the deployed point. (If it plateaus at n=200, deployed is fine. If it overfit
    # *before* n=200, deployed is over-trained.)
    n_estimators_ceiling = 400
    sample_weights = compute_sample_weight("balanced", y_train)

    print(f"[fit] training XGB with eval_set, n_estimators ceiling = {n_estimators_ceiling}")
    clf = xgb.XGBClassifier(
        objective="multi:softprob",
        eval_metric=["mlogloss", "merror"],
        verbosity=0,
        random_state=SEED,
        n_jobs=-1,
        learning_rate=best_params["learning_rate"],
        max_depth=best_params["max_depth"],
        n_estimators=n_estimators_ceiling,
    )
    clf.fit(
        X_train_pca, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_train_pca, y_train), (X_val_pca, y_val)],
        verbose=False,
    )

    er = clf.evals_result()
    # XGBoost names eval sets validation_0 (first) and validation_1 (second)
    train_mlogloss = np.array(er["validation_0"]["mlogloss"])
    val_mlogloss   = np.array(er["validation_1"]["mlogloss"])
    train_merror   = np.array(er["validation_0"]["merror"])
    val_merror     = np.array(er["validation_1"]["merror"])

    rounds = np.arange(1, len(train_mlogloss) + 1)

    # Diagnostics
    deployed_n = best_params["n_estimators"]   # 200
    val_min_round = int(np.argmin(val_mlogloss)) + 1
    val_min_logloss = float(val_mlogloss.min())
    val_at_deployed = float(val_mlogloss[deployed_n - 1])
    train_at_deployed = float(train_mlogloss[deployed_n - 1])
    train_merror_at_end = float(train_merror[-1])
    val_merror_at_end = float(val_merror[-1])
    train_merror_at_deployed = float(train_merror[deployed_n - 1])
    val_merror_at_deployed = float(val_merror[deployed_n - 1])

    # Test-set predictions at deployed point + at val-best point
    # (Refit with the val-best n_estimators to avoid using "best_iteration" semantics)
    def _eval_test_at(n_trees: int) -> dict:
        c = xgb.XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            verbosity=0,
            random_state=SEED,
            n_jobs=-1,
            learning_rate=best_params["learning_rate"],
            max_depth=best_params["max_depth"],
            n_estimators=n_trees,
        )
        c.fit(X_train_pca, y_train, sample_weight=sample_weights)
        from sklearn.metrics import f1_score, accuracy_score
        y_pred = c.predict(X_test_pca)
        return {
            "n_estimators": n_trees,
            "test_macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
        }

    test_at_deployed = _eval_test_at(deployed_n)
    test_at_val_best = _eval_test_at(val_min_round)

    # Verdict heuristics
    overfit_gap = train_merror_at_deployed - val_merror_at_deployed   # negative or near-zero is fine
    early_minimum = val_min_round < deployed_n - 5
    if early_minimum and (val_at_deployed - val_min_logloss) > 0.02:
        verdict = (
            f"Over-trained: val mlogloss minimum at round {val_min_round} but "
            f"deployed at n_estimators={deployed_n}. Suggest using val_min_round."
        )
    elif overfit_gap > 0.05 and val_merror_at_deployed > 0.02:
        verdict = (
            "Classical-overfit signature: train merror much lower than val merror; "
            "model has memorised some train structure without generalising."
        )
    elif train_merror_at_end < 0.01 and val_merror_at_end < 0.02:
        verdict = (
            "Saturation pattern: both train and val converge to near-zero error. "
            "Consistent with dataset-level signal (or leakage), not classifier overfit."
        )
    else:
        verdict = "No strong overfit signal; train-val gap is small."

    summary = {
        "deployed_n_estimators": deployed_n,
        "n_estimators_ceiling_evaluated": int(n_estimators_ceiling),
        "best_params": best_params,
        "train_mlogloss_at_deployed": train_at_deployed,
        "val_mlogloss_at_deployed":   val_at_deployed,
        "val_mlogloss_minimum":       val_min_logloss,
        "val_mlogloss_minimum_round": val_min_round,
        "train_merror_at_deployed":   train_merror_at_deployed,
        "val_merror_at_deployed":     val_merror_at_deployed,
        "train_merror_at_end":        train_merror_at_end,
        "val_merror_at_end":          val_merror_at_end,
        "deployed_overfit_gap_merror": float(overfit_gap),
        "test_metric_at_deployed":    test_at_deployed,
        "test_metric_at_val_best":    test_at_val_best,
        "verdict":                    verdict,
        "series": {
            "rounds":         rounds.tolist(),
            "train_mlogloss": train_mlogloss.tolist(),
            "val_mlogloss":   val_mlogloss.tolist(),
            "train_merror":   train_merror.tolist(),
            "val_merror":     val_merror.tolist(),
        },
    }
    out_json = OUT_DIR / "xgb_learning_curves.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\n[save] {out_json}")
    print(f"[verdict] {verdict}")

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    ax.plot(rounds, train_mlogloss, color="#4c72b0", linewidth=1.6, label="train mlogloss")
    ax.plot(rounds, val_mlogloss,   color="#c44e52", linewidth=1.6, label="val mlogloss")
    ax.axvline(deployed_n, color="black", linestyle="--", linewidth=1,
               label=f"deployed n_estimators={deployed_n}")
    ax.axvline(val_min_round, color="#55a868", linestyle=":", linewidth=1,
               label=f"val-best round={val_min_round}")
    ax.set_xlabel("n_estimators (boosting rounds)")
    ax.set_ylabel("multi-class logloss")
    ax.set_title("XGBoost mlogloss curves\n"
                 f"deployed val mlogloss={val_at_deployed:.4f}  ·  "
                 f"val minimum={val_min_logloss:.4f}")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax = axes[1]
    ax.plot(rounds, train_merror, color="#4c72b0", linewidth=1.6, label="train merror")
    ax.plot(rounds, val_merror,   color="#c44e52", linewidth=1.6, label="val merror")
    ax.axvline(deployed_n, color="black", linestyle="--", linewidth=1,
               label=f"deployed n={deployed_n}")
    ax.set_xlabel("n_estimators (boosting rounds)")
    ax.set_ylabel("multi-class error rate")
    ax.set_title(f"XGBoost merror curves\n"
                 f"train_merror@deployed={train_merror_at_deployed:.4f}  ·  "
                 f"val_merror@deployed={val_merror_at_deployed:.4f}  ·  "
                 f"gap={overfit_gap:+.4f}")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(
        "XGBoost (full-classical) learning curves\n"
        f"verdict: {verdict}",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    out_png = OUT_DIR / "xgb_learning_curves.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_png}")

    print("\n" + "=" * 60)
    print("Test-set check at two operating points:")
    print(f"  deployed (n={deployed_n}):       "
          f"macro-F1={test_at_deployed['test_macro_f1']:.4f}  "
          f"acc={test_at_deployed['test_accuracy']:.4f}")
    print(f"  val-best (n={val_min_round}):    "
          f"macro-F1={test_at_val_best['test_macro_f1']:.4f}  "
          f"acc={test_at_val_best['test_accuracy']:.4f}")
    diff = test_at_val_best["test_macro_f1"] - test_at_deployed["test_macro_f1"]
    print(f"  delta (val-best − deployed): {diff:+.4f}")


if __name__ == "__main__":
    main()
