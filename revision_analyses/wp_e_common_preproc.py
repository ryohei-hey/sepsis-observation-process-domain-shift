"""
WP-E: Common-preprocessing sensitivity   [R1C6, R4C7]

The primary LR uses IterativeImputer (MICE); XGBoost uses SimpleImputer(median).
Reviewers ask whether the LR-more-sensitive-than-XGB pattern is an imputation
artifact. We refit LR with median imputation (matching XGBoost) for Models 1-5
(Models 6-7 use pre-imputed diff features identically in both algorithms) and
compare ΔAUROC across: LR(MICE, primary) vs LR(median) vs XGBoost(median).

Output -> outputs/outputs_for_common_preproc/STable_common_preproc.md
"""
import pickle
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score

import repro
import refit_util as ru

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(__file__).resolve().parent / "cache" / "primary_predictions.pkl"
OUT = ROOT / "outputs" / "outputs_for_common_preproc"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    with open(CACHE, "rb") as f:
        C = pickle.load(f)
    yte, yex = C["y_test"], C["y_ext"]

    S = repro.build(train_xgb=False, verbose=False, fit_primary=False)

    rows = []
    for mk in [f"Model {i}" for i in range(1, 8)]:
        num, cat = S.features[mk]
        # LR median (Models 1-5 median; Models 6-7 pre-imputed diff -> imputer None)
        if mk in ("Model 6", "Model 7"):
            Xtr, Xte, Xex = S.X_train_diff, S.X_test_diff, S.X_ext_diff
            pte_med, pex_med = ru.fit_predict_lr(Xtr, S.y_train, Xte, Xex, num, cat, imputer=None)
        else:
            pte_med, pex_med = ru.fit_predict_lr(S.X_train, S.y_train, S.X_test, S.X_ext, num, cat, imputer="median")

        # deltas
        lr_mice_d = roc_auc_score(yex, C["lr"][mk]["ext"]) - roc_auc_score(yte, C["lr"][mk]["test"])
        lr_med_d = roc_auc_score(yex, pex_med) - roc_auc_score(yte, pte_med)
        xgb_d = roc_auc_score(yex, C["xgb"][mk]["ext"]) - roc_auc_score(yte, C["xgb"][mk]["test"])
        rows.append({
            "Model": f"{mk} ({C['names'][mk]})",
            "ΔAUROC LR (MICE, primary)": f"{lr_mice_d:+.3f}",
            "ΔAUROC LR (median)": f"{lr_med_d:+.3f}",
            "ΔAUROC XGBoost (median)": f"{xgb_d:+.3f}",
            "LR(median) more degraded than XGB?": "Yes" if lr_med_d < xgb_d else "No",
        })
        print(f"  {mk}: LR-MICE {lr_mice_d:+.3f} | LR-median {lr_med_d:+.3f} | XGB {xgb_d:+.3f}")

    df = pd.DataFrame(rows)
    with open(OUT / "STable_common_preproc.md", "w", encoding="utf-8") as f:
        f.write("### S.Table. Common-preprocessing sensitivity: LR imputation matched to XGBoost\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n1. Primary LR uses IterativeImputer (MICE); XGBoost uses SimpleImputer(median). "
                "LR(median) refits LR with median imputation to match XGBoost's pipeline (Models 1-5).\n")
        f.write("2. Models 6-7 use pre-imputed diff features (identical for both algorithms), so no imputation change applies.\n")
        f.write("3. Δ = external − internal AUROC. If LR remains more degraded than XGBoost under a common "
                "imputation pipeline, the algorithm difference is not an imputation artifact.\n")
    print(f"\nWritten -> {OUT}")


if __name__ == "__main__":
    main()
