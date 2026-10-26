"""
WP-I4 (R3, Reviewer 3 Major 1): within-database comparison of the locally developed
and the transported model, both algorithms.

Reviewer 3 notes that ΔAUROC (external − internal) mixes the loss from transporting a
model with the difference in how well the two populations can be discriminated. Within
each database the two can be separated by comparing
  - the model developed in that database (its 40% held-out split), with
  - the model transported from the other database (the whole database).
The patients overlap but are not identical (40% split vs whole database).

No model is fitted: forward predictions come from the frozen cache, reverse
predictions from cache/reverse_predictions.pkl (wp_i3_reverse_absolute_xgb.py).

Output -> outputs/outputs_for_reverse_validation_refreshed/STable_within_database_r3.md
"""
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "outputs_for_reverse_validation_refreshed"
MODELS = [f"Model {i}" for i in range(1, 8)]
PAIRS = [("Model 3", "Model 2"), ("Model 5", "Model 4"), ("Model 7", "Model 6")]


def main():
    with open(HERE / "cache" / "primary_predictions.pkl", "rb") as f:
        F = pickle.load(f)
    with open(HERE / "cache" / "reverse_predictions.pkl", "rb") as f:
        R = pickle.load(f)

    rows, loss = [], {}
    for akey, algo in (("lr", "Logistic regression"), ("xgb", "XGBoost")):
        for mk in MODELS:
            mimic_local = roc_auc_score(F["y_test"], F[akey][mk]["test"])   # MIMIC model, MIMIC 40%
            eicu_transp = roc_auc_score(F["y_ext"], F[akey][mk]["ext"])     # MIMIC model, whole eICU
            eicu_local = roc_auc_score(R["y_test"], R[akey][mk]["test"])    # eICU model, eICU 40%
            mimic_transp = roc_auc_score(R["y_ext"], R[akey][mk]["ext"])    # eICU model, whole MIMIC
            loss[(akey, mk)] = (eicu_transp - eicu_local, mimic_transp - mimic_local)
            rows.append({
                "Algorithm": algo, "Model ID": mk,
                "eICU-CRD: local (40%)": f"{eicu_local:.3f}",
                "eICU-CRD: transported from MIMIC-IV": f"{eicu_transp:.3f}",
                "eICU-CRD: transported − local": f"{eicu_transp - eicu_local:+.3f}",
                "MIMIC-IV: local (40%)": f"{mimic_local:.3f}",
                "MIMIC-IV: transported from eICU-CRD": f"{mimic_transp:.3f}",
                "MIMIC-IV: transported − local": f"{mimic_transp - mimic_local:+.3f}",
                "Local MIMIC-IV − local eICU-CRD": f"{mimic_local - eicu_local:+.3f}",
                "Check: fwd Δ + rev Δ − both losses": f"{(eicu_transp - mimic_local) + (mimic_transp - eicu_local) - sum(loss[(akey, mk)]):+.1e}",
            })

    inc = []
    for akey, algo in (("lr", "Logistic regression"), ("xgb", "XGBoost")):
        for a, b in PAIRS:
            inc.append({"Algorithm": algo, "Contrast": f"{a} vs {b}",
                        "Count increment, eICU-CRD": f"{loss[(akey, a)][0] - loss[(akey, b)][0]:+.3f}",
                        "Count increment, MIMIC-IV": f"{loss[(akey, a)][1] - loss[(akey, b)][1]:+.3f}"})

    with open(OUT / "STable_within_database_r3.md", "w", encoding="utf-8") as f:
        f.write("### Within-database comparison of locally developed and transported models (R3)\n\n")
        f.write(pd.DataFrame(rows).to_markdown(index=False, disable_numparse=True))
        f.write("\n\n### Count increment in the transported-minus-local difference\n\n")
        f.write(pd.DataFrame(inc).to_markdown(index=False, disable_numparse=True))
        f.write("\n\n1. Local: model developed in that database, evaluated on its 40% held-out split. Transported: model "
                "developed in the other database, evaluated on the whole database. The patients overlap but are not identical.\n"
                "2. Forward ΔAUROC + reverse ΔAUROC equals the sum of the two transported − local differences (identity; check column).\n")
    print(open(OUT / "STable_within_database_r3.md", encoding="utf-8").read())


if __name__ == "__main__":
    main()
