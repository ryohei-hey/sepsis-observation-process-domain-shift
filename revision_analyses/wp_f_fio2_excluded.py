"""
WP-F: FiO2 / A-a gradient excluded sensitivity   [R4C2, R1C6]

Refits LR for all 7 specifications after removing FiO2 and the FiO2-derived A-a
gradient (all _latest/_min/_max/_count/_diff variants) to test whether the
greater external degradation of the variability specifications (Models 6-7)
persists without FiO2 -- i.e. whether the variability-non-portability conclusion
is an artifact of the FiO2 recording-convention difference (MIMIC mean 92.5% vs
eICU 59.4%; see disclosure re: itemid 220277 SpO2 contamination).

Output -> outputs/outputs_for_fio2_excluded/STable_fio2_excluded.md
"""
import pickle
from pathlib import Path
import pandas as pd

import repro
import refit_util as ru

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(__file__).resolve().parent / "cache" / "primary_predictions.pkl"
OUT = ROOT / "outputs" / "outputs_for_fio2_excluded"
OUT.mkdir(parents=True, exist_ok=True)


def drop_fio2(feats):
    return [f for f in feats if "fio2" not in f and "aa_grad" not in f]


def main():
    with open(CACHE, "rb") as f:
        C = pickle.load(f)
    yte, yex = C["y_test"], C["y_ext"]
    S = repro.build(train_xgb=False, verbose=False, fit_primary=False)

    rows = []
    for mk in [f"Model {i}" for i in range(1, 8)]:
        num, cat = S.features[mk]
        num_x = drop_fio2(num)
        n_removed = len(num) - len(num_x)
        if mk in ("Model 6", "Model 7"):
            pte, pex = ru.fit_predict_lr(S.X_train_diff, S.y_train, S.X_test_diff, S.X_ext_diff, num_x, cat, imputer=None)
        elif mk == "Model 1":
            pte, pex = ru.fit_predict_lr(S.X_train, S.y_train, S.X_test, S.X_ext, num_x, cat, imputer="iterative")
        else:
            pte, pex = ru.fit_predict_lr(S.X_train, S.y_train, S.X_test, S.X_ext, num_x, cat, imputer="iterative")
        r = ru.shift_row(yte, pte, yex, pex)
        # primary (with FiO2) delta for reference
        from sklearn.metrics import roc_auc_score
        prim_d = roc_auc_score(yex, C["lr"][mk]["ext"]) - roc_auc_score(yte, C["lr"][mk]["test"])
        rows.append({
            "Model": f"{mk} ({C['names'][mk]})",
            "FiO2/A-a vars removed": n_removed,
            "ΔAUROC primary (with FiO2)": f"{prim_d:+.3f}",
            "ΔAUROC (FiO2/A-a excluded)": r["ΔAUROC (95% CI)"],
            "Ext calib slope (excl.)": r["Ext calib slope"],
        })
        print(f"  {mk}: primary ΔAUROC {prim_d:+.3f} -> excl {r['_dAUROC']:+.3f} (slope {r['Ext calib slope']})")

    df = pd.DataFrame(rows)
    with open(OUT / "STable_fio2_excluded.md", "w", encoding="utf-8") as f:
        f.write("### S.Table. Sensitivity analysis excluding FiO2 and the A-a gradient (logistic regression)\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n1. FiO2 and the FiO2-derived A-a gradient (all latest/min/max/count/diff variants) were removed "
                "from every specification and the models refit.\n")
        f.write("2. Δ = external − internal AUROC (95% CI from bootstrap, B = 2000). Calibration slope in external, "
                "unpenalized. Ideal slope = 1.\n")
        f.write("3. If the variability specifications (Models 6-7) still show the largest external degradation and "
                "lowest calibration slope after removing FiO2, the variability-non-portability finding is not driven "
                "by the FiO2 recording-convention difference.\n")
    print(f"\nWritten -> {OUT}")


if __name__ == "__main__":
    main()
