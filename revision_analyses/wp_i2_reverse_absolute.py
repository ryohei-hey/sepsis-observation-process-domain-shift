"""
WP-I2 (R2, Reviewer 3 Major 5): absolute internal and external AUROC in both directions.

Reviewer 3 asks whether the forward/reverse asymmetry could reflect a lower reverse
internal baseline (the eICU 40% split is itself a multi-site average). This script
reports, for LOGISTIC REGRESSION ONLY (deterministic; the XGB cache is never re-tuned):

  forward : internal (MIMIC 40%) and external (eICU) AUROC from cache/primary_predictions.pkl
  reverse : internal (eICU 40%) and external (MIMIC) AUROC from a fresh repro.build(dev_source='eicu')

and checks that the reverse LR ΔAUROC reproduces STable_forward_vs_reverse.md to 3 decimals.

Output -> outputs/outputs_for_reverse_validation_refreshed/STable_bidirectional_absolute_lr.md
"""
import pickle
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

import repro

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(__file__).resolve().parent / "cache" / "primary_predictions.pkl"
OUT = ROOT / "outputs" / "outputs_for_reverse_validation_refreshed"
MODELS = [f"Model {i}" for i in range(1, 8)]


def main():
    with open(CACHE, "rb") as f:
        C = pickle.load(f)
    fyi, fye = C["y_test"], C["y_ext"]

    R = repro.build(train_xgb=False, verbose=True, fit_primary=True, dev_source="eicu")
    ryi, rye = R.y_test.values, R.y_ext.values

    prev = pd.read_table(OUT / "STable_forward_vs_reverse.md", sep="|", skiprows=2, header=None,
                         skipinitialspace=True, engine="python")
    rows = []
    for mk in MODELS:
        f_int = roc_auc_score(fyi, C["lr"][mk]["test"]); f_ext = roc_auc_score(fye, C["lr"][mk]["ext"])
        r_int = roc_auc_score(ryi, R.predictions[mk]["test"]); r_ext = roc_auc_score(rye, R.predictions[mk]["ext"])
        rows.append({"Model ID": mk, "Feature set": R.names[mk],
                     "Forward internal AUROC (MIMIC-IV 40%)": f"{f_int:.3f}",
                     "Forward external AUROC (eICU-CRD)": f"{f_ext:.3f}",
                     "Forward ΔAUROC": f"{f_ext - f_int:+.3f}",
                     "Reverse internal AUROC (eICU-CRD 40%)": f"{r_int:.3f}",
                     "Reverse external AUROC (MIMIC-IV)": f"{r_ext:.3f}",
                     "Reverse ΔAUROC": f"{r_ext - r_int:+.3f}"})
        print(f"  {mk}: fwd {f_int:.3f}->{f_ext:.3f} ({f_ext-f_int:+.3f}) | rev {r_int:.3f}->{r_ext:.3f} ({r_ext-r_int:+.3f})", flush=True)
    df = pd.DataFrame(rows)
    with open(OUT / "STable_bidirectional_absolute_lr.md", "w", encoding="utf-8") as f:
        f.write("### S.Table addendum (R2). Absolute internal and external AUROC in both directions (logistic regression)\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n1. Forward: MIMIC-IV development (60%), MIMIC-IV internal validation (40%), eICU-CRD external. "
                "Reverse: eICU-CRD development (60%), eICU-CRD internal validation (40%), MIMIC-IV external.\n")
        f.write("2. Reverse ΔAUROC must match STable_forward_vs_reverse.md (logistic regression rows) to 3 decimals.\n")
    print("\nPreviously reported LR reverse ΔAUROC (S12):")
    print(prev.iloc[:7, 1:5].to_string(index=False, header=False))
    print(f"\nWritten -> {OUT / 'STable_bidirectional_absolute_lr.md'}")
    print("@@@WPI2_DONE")


if __name__ == "__main__":
    main()
