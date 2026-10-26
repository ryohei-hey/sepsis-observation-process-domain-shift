"""
WP-C2 (R2, Reviewer 3 Major 3): one extra rung on the latest-anchored ablation ladder.

Reviewer 3 notes that Model 1 -> Model 2 changes four things at once (aggregate score ->
latest values, plus age, urine output and the comorbidity category), and that urine
output and comorbidity are demonstrably non-equivalent across the databases. The
requested rung is "latest values without the comorbidity category and urine output".

This script fits, with logistic regression only (the ladder is LR-only; the XGB cache is
never touched):
  (a) latest                       -- reproduction check against S2 Table (B) row 1
  (b) latest, no comorbidity/uop   -- the new rung (18 latest physiologic values + age)
  (c) latest, no comorbidity       -- decomposition: which of the two carries it
  (d) latest, no uop
  (e) latest, no age/comorbidity/uop -- 18 physiologic latest values only

Output -> outputs/outputs_for_placebo_count/STable_ablation_rung_r2.md
"""
from pathlib import Path
import pandas as pd

import repro
import refit_util as ru

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "outputs_for_placebo_count"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    S = repro.build(train_xgb=False, verbose=False, fit_primary=False)
    yte, yex = S.y_test.values, S.y_ext.values
    latest = list(repro.MODEL2_NUM)
    phys = [c for c in latest if c not in ("age", "uop")]

    rungs = [
        ("latest (reproduction of S2 Table row)", latest, ["comorbidity"]),
        ("latest, no comorbidity, no urine output", [c for c in latest if c != "uop"], []),
        ("latest, no comorbidity", latest, []),
        ("latest, no urine output", [c for c in latest if c != "uop"], ["comorbidity"]),
        ("18 latest physiologic values only (no age, urine output, comorbidity)", phys, []),
    ]
    rows = []
    for name, num, cat in rungs:
        pte, pex = ru.fit_predict_lr(S.X_train, S.y_train, S.X_test, S.X_ext, num, cat, imputer="iterative")
        r = ru.shift_row(yte, pte, yex, pex)
        n_pred = len(num) + (1 if cat else 0)
        rows.append({"Specification": name, "# Predictors": n_pred,
                     "AUROC int": r["AUROC int"], "AUROC ext": r["AUROC ext"],
                     "ΔAUROC (95% CI)": r["ΔAUROC (95% CI)"],
                     "Ext calib slope": r["Ext calib slope"], "Ext CITL": r["Ext CITL"]})
        print(f"  {name}: {r['AUROC int']}/{r['AUROC ext']} Δ{r['_dAUROC']:+.3f} "
              f"slope {r['Ext calib slope']} CITL {r['Ext CITL']}", flush=True)
    df = pd.DataFrame(rows)
    with open(OUT / "STable_ablation_rung_r2.md", "w", encoding="utf-8") as f:
        f.write("### S.Table addendum (R2). Latest-value rung without the harmonization-fragile "
                "non-physiologic predictors (logistic regression)\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n1. # Predictors counts the comorbidity category as one predictor before one-hot expansion.\n")
        f.write("2. Δ = external − internal AUROC (95% CI bootstrap, B = 2000). External calibration slope "
                "unpenalized; CITL from the offset model.\n")
    print(f"\nWritten -> {OUT / 'STable_ablation_rung_r2.md'}")
    print("@@@WPC2_DONE")


if __name__ == "__main__":
    main()
