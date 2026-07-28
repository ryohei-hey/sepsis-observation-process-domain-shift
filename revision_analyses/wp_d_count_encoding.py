"""
WP-D: Count-encoding & zero-count sensitivity + count descriptives   [R1C3, R3C12]

Using the Latest strategy (Model 2 base -> Model 3 latest+count), re-specify the
count block as:
  (i)   raw counts (= Model 3, reference)
  (ii)  binary measured/not-measured (count > 0)
  (iii) log-counts (log1p)
  (iv)  winsorized counts (capped at train 99th percentile)
and, for zero-count meaning:
  (v)   vital-sign counts only (zero ~ documentation gap)
  (vi)  laboratory counts only (zero ~ clinical decision)
Refit LR each time; report internal/external AUROC, ΔAUROC (95% CI), ext slope.

Also: variable-level count distributions by database, and within-cohort
count-mortality associations (point-biserial correlation).

Output -> outputs/outputs_for_count_encoding/
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr

import repro
import refit_util as ru

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "outputs_for_count_encoding"
OUT.mkdir(parents=True, exist_ok=True)
COUNTS = repro.COUNT_FEATURES
VITAL = repro.VITAL_COUNTS
LAB = repro.LAB_COUNTS


def encode_counts(X, cols, mode, caps=None):
    Xc = X.copy()
    if mode == "raw":
        pass
    elif mode == "binary":
        Xc[cols] = (Xc[cols] > 0).astype(float)
    elif mode == "log":
        Xc[cols] = np.log1p(Xc[cols])
    elif mode == "winsor":
        for c in cols:
            Xc[c] = Xc[c].clip(upper=caps[c])
    return Xc


def main():
    S = repro.build(train_xgb=False, verbose=False, fit_primary=False)
    yte, yex = S.y_test.values, S.y_ext.values
    base = repro.MODEL2_NUM
    cat = repro.MODEL2_CAT
    caps = {c: S.X_train[c].quantile(0.99) for c in COUNTS}

    # ---- encoding sensitivity ----
    specs = [
        ("Latest only (no count, M2)", base, None, None),
        ("Latest + raw counts (M3)", base + COUNTS, "raw", COUNTS),
        ("Latest + binary measured/not", base + COUNTS, "binary", COUNTS),
        ("Latest + log counts", base + COUNTS, "log", COUNTS),
        ("Latest + winsorized counts (99th)", base + COUNTS, "winsor", COUNTS),
        ("Latest + vital counts only", base + VITAL, "raw", VITAL),
        ("Latest + lab counts only", base + LAB, "raw", LAB),
    ]
    rows = []
    for name, num, mode, cols in specs:
        if mode is None:
            Xtr, Xte, Xex = S.X_train, S.X_test, S.X_ext
        else:
            Xtr = encode_counts(S.X_train, cols, mode, caps)
            Xte = encode_counts(S.X_test, cols, mode, caps)
            Xex = encode_counts(S.X_ext, cols, mode, caps)
        pte, pex = ru.fit_predict_lr(Xtr, S.y_train, Xte, Xex, num, cat, imputer="iterative")
        r = ru.shift_row(yte, pte, yex, pex)
        rows.append({"Count encoding": name, "# num predictors": len(num),
                     "AUROC int": r["AUROC int"], "AUROC ext": r["AUROC ext"],
                     "ΔAUROC (95% CI)": r["ΔAUROC (95% CI)"], "Ext calib slope": r["Ext calib slope"]})
        print(f"  {name}: {r['AUROC int']}/{r['AUROC ext']} Δ{r['_dAUROC']:+.3f} slope {r['Ext calib slope']}")
    df_enc = pd.DataFrame(rows)

    # ---- count distributions by database ----
    mimic = pd.concat([S.X_train, S.X_test])
    drows = []
    for c in COUNTS:
        for lab, X in [("MIMIC (internal)", mimic), ("eICU (external)", S.X_ext)]:
            v = X[c].dropna()
            drows.append({"Count feature": c, "Cohort": lab, "Mean": round(v.mean(), 1),
                          "Median": round(v.median(), 1),
                          "IQR": f"{v.quantile(.25):.0f}-{v.quantile(.75):.0f}",
                          "Zero %": f"{(v == 0).mean()*100:.1f}", "P99": round(v.quantile(.99), 1)})
    df_dist = pd.DataFrame(drows)

    # ---- count-mortality association (point-biserial) within each cohort ----
    arows = []
    mimic_y = pd.concat([S.y_train, S.y_test]).values
    for c in COUNTS:
        rm = pointbiserialr(mimic_y, mimic[c].values)[0]
        re = pointbiserialr(yex, S.X_ext[c].values)[0]
        arows.append({"Count feature": c,
                      "corr(count, mortality) MIMIC": f"{rm:+.3f}",
                      "corr(count, mortality) eICU": f"{re:+.3f}"})
    df_assoc = pd.DataFrame(arows)

    with open(OUT / "STable_count_encoding.md", "w", encoding="utf-8") as f:
        f.write("### S.Table. Count-encoding sensitivity (logistic regression, Latest strategy)\n\n")
        f.write(df_enc.to_markdown(index=False))
        f.write("\n\n1. Δ = external − internal AUROC (95% CI bootstrap, B = 2000). Ext slope unpenalized (ideal = 1).\n")
        f.write("2. Binary = measured/not-measured; log = log1p; winsor = capped at the training 99th percentile.\n")
        f.write("3. Vital-only vs lab-only isolate whether the count effect is driven by universally-monitored "
                "vitals (zero ~ documentation gap) or selectively-ordered labs (zero ~ clinical decision).\n")
    with open(OUT / "STable_count_distributions.md", "w", encoding="utf-8") as f:
        f.write("### S.Table. Measurement-count distributions by database\n\n")
        f.write(df_dist.to_markdown(index=False))
    with open(OUT / "STable_count_mortality_association.md", "w", encoding="utf-8") as f:
        f.write("### S.Table. Within-cohort count-mortality association (point-biserial correlation)\n\n")
        f.write(df_assoc.to_markdown(index=False))
    print(f"\nWritten -> {OUT}")


if __name__ == "__main__":
    main()
